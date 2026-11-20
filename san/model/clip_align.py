"""PixelDiT -> CLIP modality alignment for open-vocabulary mask classification.

Training (MODEL.CONCEPT.ALIGN.ENABLED):
  * Teacher: a frozen CLIP ViT whose attention is restricted to the inside of
    the ground-truth segments (CorrCLIP-style "scope reconstruction", arXiv:
    2411.10086, with GT masks in place of SAM masks). The semantic dataset
    mapper gives one segment per class present, so a patch only attends to
    patches of its own class. The last block is ClearCLIP-style: query-query
    attention, no residual, no FFN. Dense features are projected into CLIP's
    joint space and pooled over every segment -> one target embedding per
    segment.
  * Student: ClipAlignAdapter, a light MLP on the clean-image tokens of
    PixelDiT blocks MODEL.CONCEPT.ALIGN.LAYERS (AGO-style modality adapter,
    arXiv:2504.10117: separate branch, 2 layers, cosine loss). Its dense output
    is pooled over the same segment (GT mask, or the predicted mask matched to
    it in "joint" mode) and pulled to the teacher embedding.

Inference: no CLIP image encoder and no GT. The adapter's features are pooled
over the predicted masks and compared with CLIP text embeddings of the class
names (computed once at construction); the resulting class probabilities are
ensembled with the existing PixelDiT concept classifier.
"""
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .concept_classifier import class_name_variants, reduce_synonyms

logger = logging.getLogger(__name__)

# open_clip.constants.OPENAI_DATASET_MEAN / STD (the same for the LAION models).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------- CLIP


def load_clip(model_name: str, pretrained: str):
    """Frozen open_clip model (fp32, eval) and its tokenizer."""
    import open_clip

    model = open_clip.create_model(model_name, pretrained=pretrained)
    model.eval().requires_grad_(False)
    return model, open_clip.get_tokenizer(model_name)


@torch.no_grad()
def encode_clip_text(
    model,
    tokenizer,
    names: List[str],
    templates: Sequence[str],
    device,
    clean_names: bool = True,
    split_synonyms: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CLIP text embeddings of every class variant (same cleaning / synonym
    split as the concept classifier), averaged over the templates.
    Returns (N, D) unit-norm float32 embeddings and the (N,) class index of
    each variant."""
    if not templates:
        raise ValueError("MODEL.CONCEPT.ALIGN: no text templates.")
    model = model.to(device)
    embs, classes = [], []
    for k, name in enumerate(names):
        for variant in class_name_variants(name, clean=clean_names, split=split_synonyms):
            tokens = tokenizer([t.format(target=variant) for t in templates]).to(device)
            e = F.normalize(model.encode_text(tokens).float(), dim=-1).mean(0)
            embs.append(F.normalize(e, dim=0))
            classes.append(k)
    return torch.stack(embs).cpu(), torch.tensor(classes, dtype=torch.long)


def _interp_pos_embed(pos: torch.Tensor, grid: int) -> torch.Tensor:
    """(1 + g0*g0, C) CLIP positional embedding -> (1 + grid*grid, C)."""
    cls_pos, patch_pos = pos[:1], pos[1:]
    g0 = int(round(math.sqrt(patch_pos.shape[0])))
    if g0 == grid:
        return pos
    patch_pos = patch_pos.reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos.float(), size=(grid, grid), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid * grid, -1).to(pos.dtype)
    return torch.cat([cls_pos, patch_pos], dim=0)


class ScopedClipTeacher:
    """Frozen CLIP ViT image tower with attention restricted to GT segments.

    Not an nn.Module on purpose: it is kept outside the SAN module so that it is
    not saved in checkpoints, not wrapped by DDP and not seen by the optimizer.

    input_size:   side of the square CLIP input (the padded network input is
                  resized to it; 672 with a /16 model = the 42x42 grid of CorrCLIP).
    scope_layers: number of final blocks whose attention is restricted (>= 1;
                  the last one is always the ClearCLIP-style block).
    query_query:  last block uses q q^T similarities (ClearCLIP / CorrCLIP
                  baseline) instead of q k^T.
    min_coverage: a patch belongs to a segment if that segment covers at least
                  this fraction of it (otherwise it only attends to itself).
    chunk:        images per teacher forward (bounds the attention memory).
    """

    def __init__(
        self,
        visual: nn.Module,
        input_size: int = 672,
        scope_layers: int = 1,
        query_query: bool = True,
        min_coverage: float = 0.5,
        chunk: int = 4,
    ):
        for attr in ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post", "proj"):
            if not hasattr(visual, attr):
                raise ValueError(
                    f"MODEL.CONCEPT.ALIGN needs an open_clip ViT image tower (missing '{attr}')."
                )
        if getattr(visual, "input_patchnorm", False) or getattr(visual, "attn_pool", None) is not None:
            raise ValueError("MODEL.CONCEPT.ALIGN: patch-norm / attention-pool CLIP towers are not supported.")
        self.visual = visual.eval().requires_grad_(False)
        self.patch = visual.conv1.kernel_size[0]
        if input_size % self.patch:
            raise ValueError(f"MODEL.CONCEPT.ALIGN.INPUT_SIZE {input_size} must be a multiple of {self.patch}.")
        self.input_size = input_size
        self.grid = input_size // self.patch
        n = len(visual.transformer.resblocks)
        if not 1 <= scope_layers <= n:
            raise ValueError(f"MODEL.CONCEPT.ALIGN.SCOPE_LAYERS must be in 1..{n}, got {scope_layers}")
        self.scope_layers = scope_layers
        self.query_query = query_query
        self.min_coverage = min_coverage
        self.chunk = max(1, int(chunk))
        self.device = None
        self._pos = None
        self.embed_dim = visual.proj.shape[1]

    def to(self, device):
        if self.device != torch.device(device):
            self.visual.to(device)
            self.device = torch.device(device)
            with torch.no_grad():
                self._pos = _interp_pos_embed(self.visual.positional_embedding.float(), self.grid)
        return self

    # -- attention helpers (manual, so the open_clip tensor layout does not matter)
    @staticmethod
    def _attention(blk, x, mask=None, query_query=False):
        """x: (B, L, C) -> (B, L, C) multi-head attention output of `blk`.
        mask: optional (B, 1, L, L) bool, True = may attend."""
        attn = blk.attn
        B, L, C = x.shape
        H = attn.num_heads
        q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, L, H, C // H).transpose(1, 2) for t in (q, k, v))
        if query_query:
            k = q
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, L, C)
        return attn.out_proj(out)

    def _block(self, blk, x, mask=None):
        ls_1 = getattr(blk, "ls_1", nn.Identity())
        ls_2 = getattr(blk, "ls_2", nn.Identity())
        ln_attn = getattr(blk, "ln_attn", nn.Identity())
        x = x + ls_1(ln_attn(self._attention(blk, blk.ln_1(x), mask)))
        return x + ls_2(blk.mlp(blk.ln_2(x)))

    def _scope_mask(self, coverage: List[torch.Tensor]) -> torch.Tensor:
        """Per image (S_i, g*g) segment coverage -> (B, 1, 1+g*g, 1+g*g) bool mask.
        Patches attend to patches of their own segment (and to themselves); the
        class token attends to everything; no patch attends to the class token."""
        P = self.grid * self.grid
        L = P + 1
        masks = []
        for cov in coverage:
            seg = torch.full((P,), -1, dtype=torch.long, device=self.device)
            if cov.numel():
                best, idx = cov.max(dim=0)
                seg = torch.where(best >= self.min_coverage, idx, seg)
            m = torch.zeros(L, L, dtype=torch.bool, device=self.device)
            m[1:, 1:] = (seg[:, None] == seg[None, :]) & (seg[:, None] >= 0)
            m[0, :] = True
            m.fill_diagonal_(True)
            masks.append(m)
        return torch.stack(masks)[:, None]

    @torch.no_grad()
    def region_embeddings(self, images: torch.Tensor, seg_masks: List[torch.Tensor]) -> List[torch.Tensor]:
        """images: (B, 3, H, W) in [-1, 1] (padded network input); seg_masks: per
        image (S_i, H, W) binary GT segments on the same padded grid.
        Returns per image (S_i, D) unit-norm CLIP embeddings of the segments."""
        assert self.device is not None, "call .to(device) first"
        out: List[torch.Tensor] = []
        for c0 in range(0, images.shape[0], self.chunk):
            out += self._forward_chunk(images[c0 : c0 + self.chunk], seg_masks[c0 : c0 + self.chunk])
        return out

    def _forward_chunk(self, images, seg_masks):
        g, S = self.grid, self.input_size
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        x = ((images.float() + 1) / 2).clamp(0, 1)
        x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False, antialias=True)
        x = (x - mean) / std
        # Coverage of every CLIP patch by every segment.
        coverage = [
            F.adaptive_avg_pool2d(m.float()[None], (g, g))[0].flatten(1) if m.numel() else
            torch.zeros(0, g * g, device=self.device)
            for m in seg_masks
        ]
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if self.device.type != "cuda":
            dtype = torch.float32
        v = self.visual
        with torch.autocast(self.device.type, dtype=dtype, enabled=dtype != torch.float32):
            t = v.conv1(x.to(v.conv1.weight.dtype)).flatten(2).transpose(1, 2)  # (B, g*g, C)
            cls = v.class_embedding.to(t.dtype).expand(t.shape[0], 1, -1)
            t = torch.cat([cls, t], dim=1) + self._pos.to(t.dtype)
            t = v.ln_pre(t)
            blocks = v.transformer.resblocks
            n = len(blocks)
            mask = self._scope_mask(coverage)
            for i in range(n - 1):
                t = self._block(blocks[i], t, mask if i >= n - self.scope_layers else None)
            last = blocks[n - 1]
            # ClearCLIP-style last block: attention output only (no residual, no FFN).
            t = self._attention(last, last.ln_1(t), mask, query_query=self.query_query)
            dense = v.ln_post(t[:, 1:]) @ v.proj  # (B, g*g, D)
        dense = dense.float()
        res = []
        for i, cov in enumerate(coverage):
            if cov.numel() == 0:
                res.append(dense.new_zeros(0, dense.shape[-1]))
                continue
            pooled = cov @ dense[i] / cov.sum(-1, keepdim=True).clamp(min=1e-6)
            res.append(F.normalize(pooled, dim=-1))
        return res


# ------------------------------------------------------------------ student


class ClipAlignAdapter(nn.Module):
    """Dense modality adapter: PixelDiT image tokens of several blocks
    (LayerNorm per block, learned softmax block weights) -> 2-layer MLP ->
    CLIP joint space. No parameter is tied to a class."""

    def __init__(
        self,
        feat_dim: int,
        clip_dim: int,
        num_layers: int,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_norms = nn.ModuleList(nn.LayerNorm(feat_dim) for _ in range(num_layers))
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
        )

    def dense(self, features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """{layer: (B, P, C)} -> (B, P, clip_dim) aligned (unnormalised) features."""
        layers = sorted(features)
        assert len(layers) == len(self.layer_norms), (layers, len(self.layer_norms))
        w = self.layer_logits.softmax(0)
        x = sum(w[j] * self.layer_norms[j](features[k].float()) for j, k in enumerate(layers))
        return self.mlp(x)

    @staticmethod
    def pool(dense: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """dense: (P, D) of one image, weights: (Q, P) mask weights on the patch
        grid -> (Q, D) unit-norm mask embeddings (masked average pooling)."""
        weights = weights.float()
        pooled = weights @ dense.float() / weights.sum(-1, keepdim=True).clamp(min=1e-6)
        return F.normalize(pooled, dim=-1)


def clip_logits(
    embeddings: torch.Tensor,
    text: torch.Tensor,
    variant_class: torch.Tensor,
    num_classes: int,
    scale: float,
    reduce: str = "max",
) -> torch.Tensor:
    """(Q, D) unit-norm mask embeddings x (N, D) text variants -> (Q, K) logits."""
    s = embeddings.float() @ text.float().t()
    return reduce_synonyms(s, variant_class, num_classes, reduce) * scale


def combine_with_align(
    concept_probs: torch.Tensor, align_logits: Optional[torch.Tensor], mode: str, alpha: float
) -> torch.Tensor:
    """Ensemble (Q, K) concept probabilities with the aligned-CLIP logits.
    "geometric": p ∝ p_concept^(1 - alpha) * p_align^alpha.
    "entropy":   per mask, the distribution with the lower entropy (AGO's
                 open-world identifier).
    Rows of concept_probs that are all zero (empty masks) stay zero."""
    if align_logits is None:
        return concept_probs
    nonempty = concept_probs.sum(-1, keepdim=True) > 0
    pa = align_logits.float().softmax(-1)
    if mode == "entropy":
        pc = concept_probs.float()
        pc_n = pc / pc.sum(-1, keepdim=True).clamp(min=1e-8)
        h_c = -(pc_n * pc_n.clamp(min=1e-8).log()).sum(-1, keepdim=True)
        h_a = -(pa * pa.clamp(min=1e-8).log()).sum(-1, keepdim=True)
        out = torch.where(h_a < h_c, pa, pc_n)
    elif mode == "geometric":
        if alpha <= 0:
            return concept_probs
        if alpha >= 1:
            out = pa
        else:
            log_pc = concept_probs.float().clamp(min=1e-8).log()
            out = ((1 - alpha) * log_pc + alpha * pa.clamp(min=1e-8).log()).softmax(-1)
    else:
        raise ValueError(f"MODEL.CONCEPT.ALIGN.ENSEMBLE must be 'geometric' or 'entropy', got {mode!r}")
    return out * nonempty
