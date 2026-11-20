"""Region targets from the CLIP teacher.

(a) "crop":  square crop around the mask's bbox (with margin), background outside the
             mask filled with gray (CLIP mean colour) or a blurred copy of the image,
             resized to the model input size, encoded.
(b) "dense": dense CLIP tokens (MaskCLIP / ClearCLIP / SCLIP) mask-pooled with
             area-weighted downsampling of the mask to the token grid.
(c) "avg":   normalize(normalize(a) + normalize(b)); computed when loading, not cached.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from ..features.pooling import mask_area_weights, mask_pool
from .clip import ClipTeacher


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    from torchvision.transforms.functional import gaussian_blur

    k = int(2 * round(3 * sigma) + 1)
    return gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])


@torch.no_grad()
def crop_targets(
    teacher: ClipTeacher,
    image01: torch.Tensor,        # [3, H, W] in [0, 1] on device
    masks: torch.Tensor,          # [n, H, W] bool on device
    bboxes: torch.Tensor,         # [n, 4] x0, y0, x1, y1 (exclusive)
    margin: float = 0.1,
    background: str = "gray",     # gray | blur | none
    blur_sigma: float = 8.0,
    square: bool = True,
    batch_size: int = 128,
) -> torch.Tensor:
    S = teacher.image_size
    _, H, W = image01.shape
    gray = teacher.mean.view(3, 1, 1).to(image01.dtype)
    fill_img = _blur(image01[None], blur_sigma)[0] if background == "blur" else None
    crops, out = [], []
    for m, (x0, y0, x1, y1) in zip(masks, bboxes.tolist()):
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if square:
            side = max(bw, bh) * (1 + 2 * margin)
            hw_, hh_ = side / 2, side / 2
        else:
            hw_, hh_ = bw * (1 + 2 * margin) / 2, bh * (1 + 2 * margin) / 2
        hw_, hh_ = max(hw_, 4.0), max(hh_, 4.0)
        X0, X1 = int(round(cx - hw_)), int(round(cx + hw_))
        Y0, Y1 = int(round(cy - hh_)), int(round(cy + hh_))
        canvas = gray.expand(3, Y1 - Y0, X1 - X0).clone()     # outside the image: gray
        ix0, iy0, ix1, iy1 = max(X0, 0), max(Y0, 0), min(X1, W), min(Y1, H)
        patch = image01[:, iy0:iy1, ix0:ix1]
        if background != "none":
            mm = m[iy0:iy1, ix0:ix1].to(patch.dtype)[None]
            bg = gray.expand_as(patch) if fill_img is None else fill_img[:, iy0:iy1, ix0:ix1]
            patch = patch * mm + bg * (1 - mm)
        canvas[:, iy0 - Y0:iy1 - Y0, ix0 - X0:ix1 - X0] = patch
        crops.append(F.interpolate(canvas[None], size=(S, S), mode="bicubic", align_corners=False,
                                   antialias=True).clamp(0, 1)[0])
        if len(crops) == batch_size:
            out.append(teacher.encode_crops(torch.stack(crops)))
            crops = []
    if crops:
        out.append(teacher.encode_crops(torch.stack(crops)))
    return torch.cat(out)


@torch.no_grad()
def dense_targets(
    teacher: ClipTeacher,
    image01: torch.Tensor,        # [3, H, W] in [0, 1] on device
    masks: torch.Tensor,          # [n, H, W]
    mode: str = "clearclip",
    short_side: int = 448,
    token_norm: bool = False,
    corrclip=None,                # cfg node teacher.dense.corrclip (mode == "corrclip")
    dino=None,                    # DinoSimilarity for corrclip.similarity == "dino"
) -> torch.Tensor:
    Hd, Wd = teacher.dense_input_size(image01.shape[-2:], short_side)
    x = F.interpolate(image01[None], size=(Hd, Wd), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    if mode == "corrclip":
        from .corrclip import corrclip_dense

        tok = corrclip_dense(teacher, x, masks, corrclip, dino)   # attention scoped to GT regions
    else:
        tok = teacher.dense(x, mode)                           # [h, w, D]
    if token_norm:
        tok = F.normalize(tok, dim=-1)
    wts = mask_area_weights(masks, tok.shape[:2])
    return F.normalize(mask_pool(tok, wts), dim=-1)


def combine_targets(crop: Optional[torch.Tensor], dense: Optional[torch.Tensor], kind: str) -> torch.Tensor:
    if kind == "crop":
        return F.normalize(crop.float(), dim=-1)
    if kind == "dense":
        return F.normalize(dense.float(), dim=-1)
    if kind == "avg":
        return F.normalize(F.normalize(crop.float(), dim=-1) + F.normalize(dense.float(), dim=-1), dim=-1)
    raise ValueError(f"Unknown teacher target {kind!r} (crop | dense | avg)")
