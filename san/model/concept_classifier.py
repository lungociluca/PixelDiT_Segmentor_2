"""Open-vocabulary mask classification from PixelDiT concept attention.

Training-free, after ConceptAttention (Helbling et al., arXiv:2502.04320):
every class name becomes one "concept" token that is run through the text
stream of the frozen PixelDiT next to the image (see PixDiT_T2I.forward). The
similarity between image tokens and concept tokens in the attention output
space gives a per-patch score for every class; a mask proposal is classified
by averaging those scores over the mask.
"""
import logging
import math
import re
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


# Dataset class names that are glued words or abbreviations. Values may hold
# comma-separated synonyms (split by SPLIT_SYNONYMS like any other name).
NAME_FIXES = {
    "aeroplane": "aeroplane, airplane",
    "diningtable": "dining table",
    "pottedplant": "potted plant",
    "motorbike": "motorbike, motorcycle",
    "tvmonitor": "tv monitor, television",
    "tv": "tv, television",
    "playingfield": "playing field",
    "waterdrops": "water drops",
}
# COCO-Stuff qualifiers that are not part of the object's name.
_NAME_SUFFIXES = ("-merged", "-other", "-stuff")


def clean_class_name(name: str) -> str:
    """Normalise one dataset class name (may return comma-separated synonyms)."""
    name = name.strip()
    key = name.lower()
    if key in NAME_FIXES:
        return NAME_FIXES[key]
    changed = True
    while changed:
        changed = False
        for sfx in _NAME_SUFFIXES:
            if key.endswith(sfx):
                key = key[: -len(sfx)]
                name = name[: -len(sfx)]
                changed = True
    if key in NAME_FIXES:
        return NAME_FIXES[key]
    return re.sub(r"\s+", " ", name.replace("_", " ").replace("-", " ")).strip()


def class_name_variants(name: str, clean: bool = True, split: bool = True) -> List[str]:
    """The concept strings of one class: cleaned and split into synonyms,
    duplicates removed, order kept (the first is the main name)."""
    full = clean_class_name(name) if clean else name.strip()
    parts = [p.strip() for p in full.split(",")] if split else [full]
    out = []
    for p in parts:
        if p and p.lower() not in {o.lower() for o in out}:
            out.append(p)
    return out or [name.strip()]


def instruction_prefix(default_prompt: str) -> str:
    """The instruction part of the default prompt, ending with 'User Prompt: '."""
    idx = default_prompt.rfind("User Prompt:")
    if idx < 0:
        raise ValueError("default_prompt has no 'User Prompt:' to cut the instruction at.")
    return default_prompt[:idx] + "User Prompt: "


def _encode_in_template(tokenizer, text_encoder, name, template, pool, device):
    """Pooled hidden states of `name`'s own tokens inside `template` -> ((D,), start
    token index of the name, whether the name's tokens were located)."""
    prefix, suffix = template.split("{target}")
    # Tokens before the name (special tokens included). rstrip: SentencePiece
    # attaches the space before the name to the name's first token.
    prefix_ids = tokenizer(prefix.rstrip()).input_ids
    start = len(prefix_ids)
    n_suffix = len(tokenizer(suffix, add_special_tokens=False).input_ids) if suffix.strip() else 0
    tokens = tokenizer(template.format(target=name), return_tensors="pt").to(device)
    ids = tokens.input_ids[0].tolist()
    n = len(ids)
    s, e = start, n - n_suffix
    found = ids[:start] == prefix_ids and s < e
    if not found:
        # The tokenizer merged the name with the template; fall back to every
        # token after the leading special tokens.
        s, e = min(len(tokenizer("").input_ids), n - 1), n
    hidden = text_encoder(tokens.input_ids, tokens.attention_mask)[0][0]  # (n, D)
    hidden = hidden[s:e].float()
    return (hidden.mean(0) if pool == "mean" else hidden[-1]), start, found


@torch.no_grad()
def encode_concepts(
    tokenizer,
    text_encoder,
    names: List[str],
    template="A picture of a {target}",
    pool: str = "mean",
    device: str = "cuda",
    context: str = "",
    clean_names: bool = False,
    split_synonyms: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Text-encoder vectors for the concepts of every class.

    Each class name is turned into one or more variants (cleaned / split into
    synonyms, see class_name_variants); each variant is encoded inside every
    template (a str or a list of str, each with one '{target}', prefixed by
    `context`) and the hidden states of the variant's own tokens are pooled
    ("mean", or "last" = its final token; the encoder is causal, so the last
    token has seen the whole name), then averaged over the templates.

    Returns (N, D) float32 embeddings (N >= K variants), the (N,) class index of
    every variant, and the token index at which the name starts in the first
    template, used as the text position of every concept.
    """
    if pool not in ("mean", "last"):
        raise ValueError(f"MODEL.CONCEPT.POOL must be 'mean' or 'last', got {pool!r}")
    templates = [template] if isinstance(template, str) else list(template)
    if not templates:
        raise ValueError("MODEL.CONCEPT.TEMPLATE(S): no template given.")
    for t in templates:
        if t.count("{target}") != 1:
            raise ValueError(f"Concept templates need exactly one '{{target}}', got {t!r}")
    templates = [context + t for t in templates]

    embs, classes = [], []
    start = None
    warned = set()
    for k, name in enumerate(names):
        for variant in class_name_variants(name, clean=clean_names, split=split_synonyms):
            acc = []
            for t in templates:
                e, s, found = _encode_in_template(tokenizer, text_encoder, variant, t, pool, device)
                if start is None:
                    start = s
                if not found and t not in warned:
                    logger.warning(
                        "Could not locate the tokens of %r in %r; pooling over the whole "
                        "prompt for such names.", variant, t,
                    )
                    warned.add(t)
                acc.append(e)
            embs.append(torch.stack(acc).mean(0))
            classes.append(k)
    return torch.stack(embs, dim=0), torch.tensor(classes, dtype=torch.long), start


def reduce_synonyms(
    scores: torch.Tensor, variant_class: torch.Tensor, num_classes: int, mode: str = "max"
) -> torch.Tensor:
    """Combine the per-variant scores (..., N) of a class's synonyms into one
    score per class (..., K). Identity when every class has a single variant."""
    if variant_class.numel() == num_classes:
        return scores
    if mode not in ("max", "mean"):
        raise ValueError(f"MODEL.CONCEPT.SYNONYM_REDUCE must be 'max' or 'mean', got {mode!r}")
    idx = variant_class.to(scores.device).expand(*scores.shape[:-1], -1)
    out = scores.new_zeros(*scores.shape[:-1], num_classes)
    return out.scatter_reduce(
        -1, idx, scores, reduce="amax" if mode == "max" else "mean", include_self=False
    )


class ConceptLoRA(nn.Module):
    """Trainable low-rank update of the PixelDiT text-stream attention weights
    (qkv_y and proj_y of every block), applied to the concept tokens only.

    Attention is one-way (concepts read the image, nothing reads the concepts),
    so the image stream and the prompt stream are untouched. No parameter is
    tied to a class: a class enters only through its text embedding, so the
    adapter applies unchanged to class names never seen in training. The up
    projections start at zero, i.e. training starts from the frozen model.
    """

    def __init__(self, dim: int, num_blocks: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        if rank < 1:
            raise ValueError(f"MODEL.CONCEPT.LORA.RANK must be >= 1, got {rank}")
        self.scale = alpha / rank
        self.qkv_down = nn.ModuleList(nn.Linear(dim, rank, bias=False) for _ in range(num_blocks))
        self.qkv_up = nn.ModuleList(nn.Linear(rank, 3 * dim, bias=False) for _ in range(num_blocks))
        self.proj_down = nn.ModuleList(nn.Linear(dim, rank, bias=False) for _ in range(num_blocks))
        self.proj_up = nn.ModuleList(nn.Linear(rank, dim, bias=False) for _ in range(num_blocks))
        for up in list(self.qkv_up) + list(self.proj_up):
            nn.init.zeros_(up.weight)

    def _delta(self, down: nn.Linear, up: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        # In the adapter's own dtype (fp32), whatever dtype the frozen backbone runs in.
        with torch.autocast(x.device.type, enabled=False):
            d = up(down(x.to(down.weight.dtype))) * self.scale
        return d.to(x.dtype)

    def qkv(self, block_id: int, x: torch.Tensor) -> torch.Tensor:
        """Added to qkv_y(x) of block `block_id` (0-based): (..., dim) -> (..., 3 * dim)."""
        return self._delta(self.qkv_down[block_id], self.qkv_up[block_id], x)

    def proj(self, block_id: int, x: torch.Tensor) -> torch.Tensor:
        """Added to proj_y(x) of block `block_id` (0-based): (..., dim) -> (..., dim)."""
        return self._delta(self.proj_down[block_id], self.proj_up[block_id], x)


class ConceptMaskClassifier(nn.Module):
    """Per-patch concept scores -> class probabilities for every mask proposal.

    temperature:   divides the raw similarities before the softmax over classes.
    pixel_softmax: True  - softmax over the classes at every patch and layer,
                           averaged over layers, then averaged over the mask
                           (the saliency maps of the paper);
                   False - raw similarities averaged over layers and over the
                           mask, then one softmax per mask.
    binarize_masks: pool over the thresholded mask instead of its sigmoid.
    learned:       trained head (used with ConceptLoRA). The layers are combined
                   with learned weights, the similarities are multiplied by a
                   learned scale instead of divided by `temperature`, and the
                   softmax is taken once per mask (pixel_softmax is ignored).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        pixel_softmax: bool = True,
        binarize_masks: bool = False,
        mask_threshold: float = 0.5,
        learned: bool = False,
        num_layers: int = 1,
        init_logit_scale: float = 20.0,
    ):
        super().__init__()
        self.learned = learned
        if learned:
            pixel_softmax = False
            self.layer_logits = nn.Parameter(torch.zeros(num_layers))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale)))
        if temperature <= 0:
            raise ValueError(f"MODEL.CONCEPT.TEMPERATURE must be positive, got {temperature}")
        self.temperature = temperature
        self.pixel_softmax = pixel_softmax
        self.binarize_masks = binarize_masks
        self.mask_threshold = mask_threshold

    def pixel_scores(self, scores: Dict[int, torch.Tensor]) -> torch.Tensor:
        """{layer: (B, P, K)} -> (B, P, K): class probabilities per patch when
        pixel_softmax, otherwise scaled similarities (logits)."""
        if self.learned:
            layers = sorted(scores)
            assert len(layers) == self.layer_logits.numel(), (layers, self.layer_logits.numel())
            w = self.layer_logits.softmax(dim=0)
            s = sum(w[j] * scores[k].float() for j, k in enumerate(layers))
            return s * self.logit_scale.exp().clamp(max=100.0)
        maps = [s.float() / self.temperature for s in scores.values()]
        if self.pixel_softmax:
            maps = [m.softmax(dim=-1) for m in maps]
        return torch.stack(maps, dim=0).mean(0)

    def pixel_map(self, scores: Dict[int, torch.Tensor], hw: Tuple[int, int]) -> torch.Tensor:
        """Class probabilities per patch as an image: (B, K, h, w)."""
        s = self.pixel_scores(scores)
        if not self.pixel_softmax:
            s = s.softmax(dim=-1)
        return s.transpose(1, 2).reshape(s.shape[0], s.shape[-1], hw[0], hw[1])

    @torch.no_grad()
    def forward(
        self,
        scores: Dict[int, torch.Tensor],
        mask_logits: torch.Tensor,
        hw: Tuple[int, int],
    ) -> torch.Tensor:
        """scores: {layer: (B, h*w, K)}, mask_logits: (B, Q, hm, wm) -> (B, Q, K)
        class probabilities (rows sum to 1, or to 0 for an empty binarized mask)."""
        s = self.pixel_scores(scores)  # (B, P, K)
        B, P, K = s.shape
        assert P == hw[0] * hw[1], (P, hw)
        if tuple(mask_logits.shape[-2:]) != tuple(hw):
            # Mask grid differs from the PixelDiT patch grid: resample the scores.
            s = F.interpolate(
                s.transpose(1, 2).reshape(B, K, hw[0], hw[1]),
                size=mask_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)
        m = mask_logits.float().sigmoid().flatten(2)  # (B, Q, P)
        if self.binarize_masks:
            m = (m > self.mask_threshold).float()
        return self.pool(s, m)

    def pool(self, pixel_scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Average `pixel_scores` (..., P, K), as returned by pixel_scores(), with
        per-patch mask weights (..., Q, P) -> (..., Q, K) class probabilities."""
        pooled = self.pool_logits(pixel_scores, weights)
        if not self.pixel_softmax:
            pooled = pooled.softmax(dim=-1) * (weights.sum(-1, keepdim=True) > 0)
        return pooled

    def pool_logits(self, pixel_scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted average of `pixel_scores` over each mask, before any softmax
        over the classes: the class logits of the masks when not pixel_softmax."""
        weights = weights.float()
        area = weights.sum(-1, keepdim=True)
        return torch.einsum("...qp,...pk->...qk", weights, pixel_scores.float()) / area.clamp(min=1e-6)


class MaskEmbedHead(nn.Module):
    """Trained open-vocabulary mask classifier on PixelDiT image features
    (ODISE-style "category label supervision").

    Every mask is an average of the image tokens it covers, taken from several
    PixelDiT blocks of the concept-scoring pass (pooled per block, LayerNorm,
    combined with learned block weights) and mapped by an MLP to an embedding.
    The class names are mapped to the same space from their frozen text-encoder
    vectors (the concept embeddings: cleaned names, synonyms, template ensemble)
    by a class-agnostic projection; logits = learned scale * cosine similarity.
    No parameter is tied to a class, so new class names need no retraining.
    """

    def __init__(
        self,
        feat_dim: int,
        text_dim: int,
        num_layers: int,
        embed_dim: int = 768,
        hidden_dim: int = 1024,
        text_proj: str = "linear",
        init_logit_scale: float = 20.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if text_proj not in ("linear", "mlp"):
            raise ValueError(f"MODEL.CONCEPT.MASK_HEAD.TEXT_PROJ must be 'linear' or 'mlp', got {text_proj!r}")
        self.layer_norms = nn.ModuleList(nn.LayerNorm(feat_dim) for _ in range(num_layers))
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.img_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        # The text-encoder hidden states have a large, uneven scale: normalise first.
        text_layers = [nn.LayerNorm(text_dim)]
        if text_proj == "linear":
            text_layers.append(nn.Linear(text_dim, embed_dim))
        else:
            text_layers += [nn.Linear(text_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim)]
        self.text_proj = nn.Sequential(*text_layers)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale)))

    def mask_embeddings(self, features: Dict[int, torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """features: {layer: (P, C)} of one image, weights: (Q, P) mask weights
        on the patch grid -> (Q, embed_dim) unit-norm mask embeddings."""
        layers = sorted(features)
        assert len(layers) == len(self.layer_norms), (layers, len(self.layer_norms))
        weights = weights.float()
        area = weights.sum(-1, keepdim=True).clamp(min=1e-6)
        w = self.layer_logits.softmax(0)
        pooled = 0
        for j, k in enumerate(layers):
            # Pool first (cheap: Q x C instead of P x C), then normalise.
            f = (weights @ features[k].float()) / area
            pooled = pooled + w[j] * self.layer_norms[j](f)
        return F.normalize(self.img_mlp(pooled), dim=-1)

    def text_embeddings(self, concepts: torch.Tensor) -> torch.Tensor:
        """(N, text_dim) concept vectors -> (N, embed_dim) unit-norm embeddings."""
        return F.normalize(self.text_proj(concepts.float()), dim=-1)

    def logits(self, features, weights, concepts, variant_class, num_classes, reduce="max"):
        """-> (Q, K) class logits of the masks (synonyms combined per class)."""
        img = self.mask_embeddings(features, weights)
        txt = self.text_embeddings(concepts)
        s = img @ txt.t()  # (Q, N)
        s = reduce_synonyms(s, variant_class, num_classes, reduce)
        return s * self.logit_scale.exp().clamp(max=100.0)


def ensemble_probs(concept_probs: torch.Tensor, head_logits: torch.Tensor, alpha: float) -> torch.Tensor:
    """Geometric ensemble p ∝ p_concept^(1 - alpha) * p_head^alpha over the classes.
    Rows of concept_probs that are all zero (empty masks) stay zero."""
    if alpha <= 0:
        return concept_probs
    head = head_logits.float().log_softmax(-1)
    if alpha >= 1:
        out = head.exp()
    else:
        out = ((1 - alpha) * concept_probs.float().clamp(min=1e-8).log() + alpha * head).softmax(-1)
    return out * (concept_probs.sum(-1, keepdim=True) > 0)
