"""Forward-time probe that measures vision-language alignment inside PixelDiT.

The MMDiT blocks in `t2i/pixdit_core/pixeldit_t2i.py` call

    segment_data["vl_probe"](layer_id, space, image_feats, text_feats)

for every feature space (see SPACES). Image features are (B, N_patches, D) or
(B, N_patches, heads, head_dim); text features have the same layout over the
300 text tokens. The probe never keeps full feature maps: per image it reduces
them right away to

* image-level class scores (mean of the top-k patch similarities),
* oracle-mask scores (mean-pooled features of every ground-truth segment vs
  every class text vector),
* localization AUROC of each class similarity map inside vs outside its mask,

which `t2i.vl_stats.metrics.RawStore` aggregates over the dataset.

Which batch item provides the text vector and which provides the image
features for each class is set per forward pass with `set_roles`, so the same
probe supports class prompts batched with a separate "default prompt" pass
(the way SAN conditions the backbone) or image features taken from the same
pass as the class prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import key_name

# Spaces reported by the hooks in pixeldit_t2i.py.
SPACES = (
    "q",         # q_x vs q_y              (after per-head RMSNorm, before RoPE)
    "k",         # k_x vs k_y
    "v",         # v_x vs v_y
    "qk",        # q_x vs k_y              image queries -> text keys (what joint attention computes)
    "kq",        # k_x vs q_y              text queries -> image keys
    "attn_out",  # proj_x(attn) vs proj_y(attn), before the adaLN gate
    "block",     # residual streams x, y at the output of MMDiTBlockT2I
)
HEAD_SPACES = ("q", "k", "v", "qk", "kq")


# --------------------------------------------------------------------------- #
# Ground truth at patch resolution
# --------------------------------------------------------------------------- #
@dataclass
class PatchGT:
    num_classes: int
    hw: Tuple[int, int]               # patch grid (Hp, Wp)
    frac: torch.Tensor                # (C, N) fraction of each patch covered by class c
    ignore_frac: torch.Tensor         # (N,) fraction of ignore pixels
    presence: np.ndarray              # (C,) 1 present / 0 absent / -1 negligible
    seg_classes: List[int]            # class of every segment (one per present class)
    seg_area: np.ndarray              # (S,) area in patches
    seg_weights: torch.Tensor         # (S, N) pooling weights (rows sum to 1)
    seg_pos: torch.Tensor             # (S, N) bool, patches inside the object
    seg_neg: torch.Tensor             # (S, N) bool, labelled patches outside the object


def build_patch_gt(gt: torch.Tensor, num_classes: int, patch: int, *, background_id: Optional[int],
                   ignore_id: int = 255, purity: float = 0.5, min_area: float = 1.0) -> PatchGT:
    """`gt` is an (H, W) long tensor with H, W multiples of `patch`.

    Labels: 0..C-1 classes, `background_id` (may be None) for background, `ignore_id` ignored.
    A class is *present* if it covers at least `min_area` patches, *absent* if it has no pixel.
    """
    H, W = gt.shape
    assert H % patch == 0 and W % patch == 0, "GT must be resized to a multiple of the patch size"
    C = num_classes
    lab = gt.clone().long()
    # channels: 0..C-1 classes, C background, C+1 ignore
    lab[lab == ignore_id] = C + 1
    if background_id is not None:
        lab[lab == background_id] = C
    lab[(lab < 0) | (lab > C + 1)] = C + 1
    onehot = F.one_hot(lab, C + 2).permute(2, 0, 1).float()[None]
    frac = F.avg_pool2d(onehot, kernel_size=patch, stride=patch)[0]       # (C+2, Hp, Wp)
    Hp, Wp = frac.shape[-2:]
    frac = frac.flatten(1)                                                 # (C+2, N)
    cls_frac, ignore_frac = frac[:C], frac[C + 1]

    area = cls_frac.sum(1)                                                 # in patches
    pixel_count = torch.bincount(lab.flatten(), minlength=C + 2)[:C]
    presence = np.full((C,), -1, np.int8)
    presence[(area >= min_area).cpu().numpy()] = 1
    presence[(pixel_count == 0).cpu().numpy()] = 0

    seg_classes = [int(c) for c in np.nonzero(presence == 1)[0]]
    N = Hp * Wp
    if seg_classes:
        f = cls_frac[seg_classes]                                          # (S, N)
        weights = f / f.sum(1, keepdim=True).clamp_min(1e-8)
        pos = f >= purity
        empty = ~pos.any(1)
        if empty.any():   # thin objects: fall back to their best-covered patches
            pos[empty] = f[empty] >= f[empty].amax(1, keepdim=True)
        neg = (f == 0) & (ignore_frac[None] < 0.5)
    else:
        weights = torch.zeros((0, N), device=gt.device)
        pos = torch.zeros((0, N), dtype=torch.bool, device=gt.device)
        neg = torch.zeros((0, N), dtype=torch.bool, device=gt.device)
    return PatchGT(
        num_classes=C, hw=(Hp, Wp), frac=cls_frac, ignore_frac=ignore_frac, presence=presence,
        seg_classes=seg_classes, seg_area=area[seg_classes].cpu().numpy() if seg_classes else np.zeros(0),
        seg_weights=weights, seg_pos=pos, seg_neg=neg,
    )


def torch_auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    neg = torch.sort(neg.float()).values
    pos = pos.float()
    less = torch.searchsorted(neg, pos, right=False).double()
    less_eq = torch.searchsorted(neg, pos, right=True).double()
    return float((less + 0.5 * (less_eq - less)).sum() / (pos.numel() * neg.numel()))


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
class VLAlignmentProbe:
    def __init__(self, num_classes: int, spaces: Iterable[str] = SPACES, layers: Optional[Iterable[int]] = None,
                 per_head_spaces: Iterable[str] = (), topk_frac: float = 0.1):
        self.C = int(num_classes)
        self.spaces = set(spaces)
        unknown = self.spaces - set(SPACES)
        if unknown:
            raise ValueError(f"unknown spaces {sorted(unknown)}; choose from {SPACES}")
        self.layers = None if layers is None else set(int(l) for l in layers)
        self.per_head_spaces = set(per_head_spaces) & set(HEAD_SPACES)
        self.topk_frac = float(topk_frac)
        self.gt: Optional[PatchGT] = None
        self.text_roles: List[List[Tuple[int, torch.Tensor]]] = []
        self.image_roles: List[List[int]] = []

    # ------------------------------------------------------------- per image
    def start_image(self, gt: PatchGT):
        self.gt = gt
        self.text_vecs: Dict[str, torch.Tensor] = {}
        self.results: Dict[str, Dict[str, np.ndarray]] = {}

    def set_roles(self, text_roles: Sequence[Sequence[Tuple[int, torch.Tensor]]],
                  image_roles: Sequence[Sequence[int]]):
        """text_roles[b] = [(class_id, token_indices), ...] read from batch item b;
        image_roles[b] = class ids whose image features come from batch item b."""
        assert len(text_roles) == len(image_roles)
        self.text_roles = [list(r) for r in text_roles]
        self.image_roles = [list(r) for r in image_roles]

    def end_image(self) -> Dict[str, Dict[str, np.ndarray]]:
        out, self.results = self.results, {}
        self.text_vecs = {}
        return out

    # ----------------------------------------------------------------- hook
    @torch.no_grad()
    def __call__(self, layer: int, space: str, img: torch.Tensor, txt: torch.Tensor):
        if space not in self.spaces or (self.layers is not None and int(layer) not in self.layers):
            return
        if self.gt is None:
            raise RuntimeError("VLAlignmentProbe.start_image() was not called")
        img = img.detach()
        txt = txt.detach()
        variants = [(space, img.flatten(2), txt.flatten(2))]
        if space in self.per_head_spaces and img.dim() == 4:
            for h in range(img.shape[2]):
                variants.append((f"{space}@h{h:02d}", img[:, :, h], txt[:, :, h]))
        for name, I, T in variants:
            key = key_name(layer, name)
            self._collect_text(key, T)
            self._score_images(key, I)

    def _collect_text(self, key: str, T: torch.Tensor):
        if not any(self.text_roles):
            return
        buf = self.text_vecs.get(key)
        if buf is None:
            buf = torch.full((self.C, T.shape[-1]), float("nan"), device=T.device, dtype=torch.float32)
            self.text_vecs[key] = buf
        for b, roles in enumerate(self.text_roles):
            for c, idx in roles:
                buf[c] = T[b, idx.to(T.device)].float().mean(0)

    def _result(self, key: str) -> Dict[str, np.ndarray]:
        r = self.results.get(key)
        if r is None:
            S = len(self.gt.seg_classes)
            r = {
                "img_scores": np.full((self.C,), np.nan, np.float32),
                "seg_scores": np.full((S, self.C), np.nan, np.float32),
                "loc_auroc": np.full((S,), np.nan, np.float32),
                "loc_gap": np.full((S,), np.nan, np.float32),
            }
            self.results[key] = r
        return r

    def _score_images(self, key: str, I: torch.Tensor):
        gt = self.gt
        for b, classes in enumerate(self.image_roles):
            if not classes:
                continue
            tv = self.text_vecs.get(key)
            if tv is None or torch.isnan(tv[classes]).any():
                raise RuntimeError(f"{key}: text vectors for {classes} must be collected before the image pass")
            Ib = I[b].float()                                          # (N, D)
            if Ib.shape[0] != gt.frac.shape[1]:
                raise RuntimeError(f"{key}: {Ib.shape[0]} image tokens but GT has {gt.frac.shape[1]} patches")
            Tn = F.normalize(tv[classes], dim=-1)                      # (k, D)
            maps = F.normalize(Ib, dim=-1) @ Tn.T                     # (N, k) cosine maps
            K = max(1, int(round(self.topk_frac * maps.shape[0])))
            img_scores = maps.topk(K, dim=0).values.mean(0)            # (k,)

            r = self._result(key)
            r["img_scores"][classes] = img_scores.cpu().numpy()
            if gt.seg_classes:
                pooled = F.normalize(gt.seg_weights @ Ib, dim=-1)      # (S, D)
                r["seg_scores"][:, classes] = (pooled @ Tn.T).cpu().numpy()
                col = {c: j for j, c in enumerate(classes)}
                for s, c in enumerate(gt.seg_classes):
                    if c not in col:
                        continue
                    m = maps[:, col[c]]
                    p, n = m[gt.seg_pos[s]], m[gt.seg_neg[s]]
                    r["loc_auroc"][s] = torch_auroc(p, n)
                    if p.numel() and n.numel():
                        r["loc_gap"][s] = float(p.mean() - n.mean())
