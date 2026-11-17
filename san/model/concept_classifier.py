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
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_concepts(
    tokenizer,
    text_encoder,
    names: List[str],
    template: str = "A picture of a {target}",
    pool: str = "mean",
    device: str = "cuda",
) -> Tuple[torch.Tensor, int]:
    """One text-encoder vector per class name.

    Each name is encoded inside `template` and the hidden states of the name's
    own tokens are pooled ("mean", or "last" = its final token; the encoder is
    causal, so the last token has seen the whole name).

    Returns (K, D) float32 embeddings and the token index at which the name
    starts in the template, which is used as the text position of every concept.
    """
    if pool not in ("mean", "last"):
        raise ValueError(f"MODEL.CONCEPT.POOL must be 'mean' or 'last', got {pool!r}")
    if template.count("{target}") != 1:
        raise ValueError(f"MODEL.CONCEPT.TEMPLATE needs exactly one '{{target}}', got {template!r}")
    prefix, suffix = template.split("{target}")
    # Tokens before the name (special tokens included). rstrip: SentencePiece
    # attaches the space before the name to the name's first token.
    prefix_ids = tokenizer(prefix.rstrip()).input_ids
    start = len(prefix_ids)
    n_suffix = len(tokenizer(suffix, add_special_tokens=False).input_ids) if suffix.strip() else 0

    embs = []
    warned = False
    for name in names:
        tokens = tokenizer(template.format(target=name), return_tensors="pt").to(device)
        ids = tokens.input_ids[0].tolist()
        n = len(ids)
        s, e = start, n - n_suffix
        if ids[:start] != prefix_ids or not s < e:
            # The tokenizer merged the name with the template; fall back to
            # every token after the leading special tokens.
            if not warned:
                logger.warning(
                    "Could not locate the tokens of %r in %r; pooling over the whole prompt "
                    "for such names.", name, template,
                )
                warned = True
            s, e = min(len(tokenizer("").input_ids), n - 1), n
        hidden = text_encoder(tokens.input_ids, tokens.attention_mask)[0][0]  # (n, D)
        hidden = hidden[s:e].float()
        embs.append(hidden.mean(0) if pool == "mean" else hidden[-1])
    return torch.stack(embs, dim=0), start


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
