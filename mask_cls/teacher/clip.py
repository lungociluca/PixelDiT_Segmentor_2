"""OpenCLIP teacher (used at training/caching time only).

Supports
  * image embeddings of crops (`encode_crops`), any OpenCLIP model;
  * dense per-patch embeddings in CLIP space (`dense`):
      - ViT: last block replaced by one of
          "maskclip"  : value embedding only (no q-k attention), no residual, no FFN
          "clearclip" : q-q self-attention, no residual, no FFN (ClearCLIP)
          "sclip"     : q-q + k-k attention (SCLIP's correlative attention), no residual, no FFN
          "vanilla"   : unmodified block (for checks only; dense tokens are poor)
        positional embeddings are interpolated to the input grid.
      - ConvNeXt (timm trunk): per-location head norm + projection (FC-CLIP style).

`sanity_check()` verifies the re-implemented forward pass against
`model.encode_image` (OpenCLIP internals change between versions), so run it
once per model; tools/mask_cls/extract_teacher.py does.
"""
import logging
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("mask_cls")


def _find_normalize(preprocess) -> Tuple[List[float], List[float]]:
    for t in getattr(preprocess, "transforms", []):
        if t.__class__.__name__ == "Normalize":
            return list(t.mean), list(t.std)
    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    return list(OPENAI_DATASET_MEAN), list(OPENAI_DATASET_STD)


class ClipTeacher(nn.Module):
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai", device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        mean, std = _find_normalize(preprocess)
        self.register_buffer("mean", torch.tensor(mean, device=device).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, device=device).view(1, 3, 1, 1), persistent=False)
        size = self.model.visual.image_size
        self.image_size = int(size[0] if isinstance(size, (tuple, list)) else size)
        self.device_str = device
        self.dtype = dtype
        v = self.model.visual
        self.is_vit = hasattr(v, "conv1") and hasattr(v, "transformer")
        self.is_timm = hasattr(v, "trunk")
        if self.is_vit:
            ks = v.conv1.kernel_size
            self.patch = int(ks[0] if isinstance(ks, (tuple, list)) else ks)
            if getattr(v, "attn_pool", None) is not None:
                log.warning("Visual attention pooling (CoCa-style) is not supported for dense features.")
        elif self.is_timm:
            self.patch = 32
        self.embed_dim = int(self.model.text_projection.shape[1]) if hasattr(self.model, "text_projection") and \
            isinstance(self.model.text_projection, torch.Tensor) else None

    # ---------------------------------------------------------------- common
    def _norm(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01.to(self.device_str, torch.float32) - self.mean) / self.std

    def _autocast(self):
        return torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32)

    @torch.no_grad()
    def encode_crops(self, crops01: torch.Tensor) -> torch.Tensor:
        """crops in [0, 1], [n, 3, S, S] (S = model input size) -> L2-normalised [n, D]."""
        with self._autocast():
            e = self.model.encode_image(self._norm(crops01))
        return F.normalize(e.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str], batch_size: int = 256) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenizer(texts[i:i + batch_size]).to(self.device_str)
            with self._autocast():
                out.append(F.normalize(self.model.encode_text(tok).float(), dim=-1))
        return torch.cat(out)

    # ---------------------------------------------------------------- dense
    def dense_input_size(self, hw, short_side: int) -> Tuple[int, int]:
        H, W = hw
        s = short_side / float(min(H, W))
        p = self.patch
        return max(p, int(round(H * s / p)) * p), max(p, int(round(W * s / p)) * p)

    @torch.no_grad()
    def dense(self, image01: torch.Tensor, mode: str = "clearclip") -> torch.Tensor:
        """image in [0, 1], [1, 3, H, W] (H, W multiples of the patch/stride) -> [h, w, D] (not normalised)."""
        x = self._norm(image01)
        with self._autocast():
            if self.is_vit:
                out = self._vit_dense(x, mode)
            elif self.is_timm:
                out = self._timm_dense(x)
            else:
                raise NotImplementedError(f"Dense features not implemented for {type(self.model.visual)}")
        return out[0].float()

    def _vit_tokens_before_last(self, x):
        v = self.model.visual
        x = v.conv1(x)                                        # [B, C, gh, gw]
        B, C, gh, gw = x.shape
        x = x.reshape(B, C, gh * gw).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).view(1, 1, C).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._interp_pos(v.positional_embedding, gh, gw).to(x.dtype)
        x = v.ln_pre(x)
        bf = getattr(v.transformer, "batch_first", False)
        if not bf:
            x = x.permute(1, 0, 2)
        for blk in v.transformer.resblocks[:-1]:
            x = blk(x)
        if not bf:
            x = x.permute(1, 0, 2)
        return x, gh, gw, bf                                  # x: [B, N, C]

    @staticmethod
    def _interp_pos(pe, gh, gw):
        cls_pe, grid = pe[:1], pe[1:]
        g0 = int(math.isqrt(grid.shape[0]))
        if (g0, g0) == (gh, gw):
            return pe
        grid = grid.reshape(1, g0, g0, -1).permute(0, 3, 1, 2).float()
        grid = F.interpolate(grid, size=(gh, gw), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(gh * gw, -1).to(pe.dtype)
        return torch.cat([cls_pe, grid], dim=0)

    def _vit_dense(self, x, mode, return_cls=False):
        v = self.model.visual
        x, gh, gw, bf = self._vit_tokens_before_last(x)
        last = v.transformer.resblocks[-1]
        if mode == "vanilla":
            y = last(x if bf else x.permute(1, 0, 2))
            y = y if bf else y.permute(1, 0, 2)
        else:
            attn = last.attn
            if not hasattr(attn, "in_proj_weight") or attn.in_proj_weight is None:
                raise NotImplementedError("Expected nn.MultiheadAttention with packed in_proj in the last block")
            B, N, C = x.shape
            nh = attn.num_heads
            d = C // nh
            qkv = F.linear(last.ln_1(x), attn.in_proj_weight, attn.in_proj_bias)
            q, k, val = qkv.reshape(B, N, 3, nh, d).permute(2, 0, 3, 1, 4)   # each [B, nh, N, d]
            scale = d ** -0.5
            if mode == "maskclip":
                o = val
            elif mode == "clearclip":
                o = torch.softmax((q @ q.transpose(-2, -1)) * scale, dim=-1) @ val
            elif mode == "sclip":
                a = torch.softmax((q @ q.transpose(-2, -1)) * scale, -1) + torch.softmax((k @ k.transpose(-2, -1)) * scale, -1)
                o = a @ val
            else:
                raise ValueError(f"Unknown dense mode {mode!r}")
            o = o.transpose(1, 2).reshape(B, N, C)
            y = attn.out_proj(o)
            if hasattr(last, "ls_1"):
                y = last.ls_1(y)
        y = v.ln_post(y)
        if v.proj is not None:
            y = y @ v.proj
        if return_cls:
            return y[:, 0]
        return y[:, 1:].reshape(y.shape[0], gh, gw, -1)

    def _timm_parts(self):
        v = self.model.visual
        trunk = v.trunk
        head_norm = getattr(getattr(trunk, "head", None), "norm", None)
        if head_norm is None:
            head_norm = getattr(trunk, "norm_pre", None) or nn.Identity()
        return trunk, head_norm, v.head

    def _timm_dense(self, x, pooled=False):
        trunk, head_norm, proj = self._timm_parts()
        f = trunk.forward_features(x)                         # [B, C, h, w]
        if pooled:
            f = f.mean((2, 3), keepdim=True)
        f = head_norm(f)                                      # LayerNorm2d (channels-first)
        f = f.permute(0, 2, 3, 1)                             # [B, h, w, C]
        B, h, w, C = f.shape
        out = proj(f.reshape(B * h * w, C)).reshape(B, h, w, -1)
        return out[:, 0, 0] if pooled else out

    @torch.no_grad()
    def sanity_check(self, tol: float = 0.99):
        """Our re-implemented forward vs encode_image on a random image."""
        g = torch.Generator().manual_seed(0)
        img = torch.rand(1, 3, self.image_size, self.image_size, generator=g)
        x = self._norm(img)
        ref = F.normalize(self.model.encode_image(x).float(), dim=-1)
        if self.is_vit:
            ours = self._vit_dense(x, "vanilla", return_cls=True)
        elif self.is_timm:
            ours = self._timm_dense(x, pooled=True)
        else:
            return None
        cos = float((F.normalize(ours.float(), dim=-1) * ref).sum())
        if cos < tol:
            raise RuntimeError(
                f"Dense-path sanity check failed (cos={cos:.4f} < {tol}). The OpenCLIP model internals "
                "differ from what mask_cls/teacher/clip.py expects; dense targets would be wrong."
            )
        log.info("CLIP dense-path sanity check passed (cos=%.5f)", cos)
        return cos
