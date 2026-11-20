"""Frozen PixelDiT feature extractor.

Feature groups returned by `PixelDiTFeatureExtractor.forward` (B = batch):
  "patch":    [B, len(layers), h, w, 1536]  image tokens after the chosen MMDiT patch blocks
              (residual stream), h = H/16, w = W/16  -> the "semantic" pathway
  "pit_attn": [B, pixel_depth, h, w, 1152]  per-patch attention token inside each PiT
              (pixel) block (only with pixel_pathway=True)
  "pixel":    [B, pixel_depth, H, W, 16]    per-pixel tokens after each PiT block
              (only with pixel_pathway=True)  -> the "detail" pathway

Timestep convention (flow matching, as in t2i/dpm_solver_wrapper.py): x_t = (1 - t) x_0 + t * eps,
t in [0, 1], t = 0 is the clean image; the network gets t * 1000. With noise=False the clean
image is fed and t only conditions the network (the repo default, t = 0.002).

Conditioning text is embedded once with the model's own text encoder (Gemma-2-2b-it) and
cached to disk, so Gemma is not loaded again on later runs.

Stage 2 reuse: the extractor is an nn.Module with no trainable parameters; a side adapter
can call it and consume the same feature groups.
"""
import logging
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("mask_cls")

MAX_TEXT_TOKENS = 300


def _load_pixeldit_config(config_path: str):
    import pyrallis
    from t2i.diffusion.utils.config import PixDiTConfig

    return pyrallis.parse(config_class=PixDiTConfig, config_path=config_path, args=[])


def encode_prompts(prompts: List[str], text_encoder_name: str, device: str):
    """Embed prompts with PixelDiT's text encoder. Returns (emb [n, 300, 2304] fp32, mask [n, 300])."""
    from t2i.diffusion.model.builder import get_tokenizer_and_text_encoder

    tokenizer, enc = get_tokenizer_and_text_encoder(name=text_encoder_name, device=device)
    embs, masks = encode_with(tokenizer, enc, prompts)
    del tokenizer, enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embs, masks


@torch.no_grad()
def encode_with(tokenizer, enc, prompts: List[str]):
    dev = next(enc.parameters()).device
    tok = tokenizer(prompts, max_length=MAX_TEXT_TOKENS, padding="max_length", truncation=True, return_tensors="pt").to(dev)
    embs = torch.cat([enc(tok.input_ids[[i]], tok.attention_mask[[i]])[0] for i in range(len(prompts))])
    return embs.float().cpu(), tok.attention_mask.cpu()


def cached_prompt_embedding(prompt: str, text_encoder_name: str, cache_dir: str, device: str):
    from ..config import stable_hash

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"prompt_{stable_hash([prompt, text_encoder_name])}.pt")
    if os.path.isfile(path):
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d["emb"], d["mask"]
    emb, mask = encode_prompts([prompt], text_encoder_name, device)
    torch.save({"emb": emb, "mask": mask, "prompt": prompt}, path)
    log.info("Cached prompt embedding (%d tokens) to %s", int(mask.sum()), path)
    return emb, mask


def resize_for_pixeldit(image_hw, target_area: int, patch: int = 16, max_side: int = 2048):
    """Output (H', W'): area ~ target_area, aspect kept, both multiples of `patch`
    (no padding, so every feature cell holds image content)."""
    H, W = image_hw
    s = (target_area / float(H * W)) ** 0.5
    s = min(s, max_side / float(max(H, W)))
    Hn = max(patch, int(round(H * s / patch)) * patch)
    Wn = max(patch, int(round(W * s / patch)) * patch)
    return Hn, Wn


class PixelDiTFeatureExtractor(nn.Module):
    def __init__(
        self,
        checkpoint: str = "t2i/pixeldit_t2i_v1.pth",
        model_config: str = "diffusion_configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml",
        num_blocks: int = 4,
        layers: Sequence[int] = (0, 1, 2, 3),
        timestep: float = 0.002,
        noise: bool = False,
        noise_seed: int = 0,
        prompt: str = "",
        captions: Optional[Dict[str, str]] = None,
        mask_text_padding: bool = True,
        pixel_pathway: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        cache_dir: str = "cache/mask_cls/prompts",
    ):
        super().__init__()
        from t2i.diffusion.model.builder import build_model
        from t2i.diffusion.utils.config import model_init_config
        from t2i.tools.download import resolve_checkpoint

        self.layers = [int(l) for l in layers]
        if max(self.layers) >= num_blocks or min(self.layers) < 0:
            raise ValueError(f"layers {self.layers} must be in [0, num_blocks={num_blocks})")
        self.timestep = float(timestep)
        self.noise = bool(noise)
        self.noise_seed = int(noise_seed)
        self.pixel_pathway = bool(pixel_pathway)
        self.dtype = dtype
        self.device_str = device

        config = _load_pixeldit_config(model_config)
        extra = dict(config.model.extra or {})
        extra["num_patch_blocks"] = int(num_blocks)
        extra["build_pixel_blocks"] = self.pixel_pathway
        config.model.extra = extra
        kwargs = model_init_config(config, latent_size=1024)
        model = build_model(config.model.model, use_fp32_attention=False, **kwargs)
        ckpt_path = resolve_checkpoint(checkpoint)
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd.pop("pos_embed", None)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        del sd
        important = [k for k in missing if not k.startswith("_repa_projector")]
        if important:
            raise RuntimeError(f"PixelDiT weights missing from {ckpt_path}: {important[:10]} ...")
        log.info(
            "PixelDiT: %d patch blocks built%s; %d checkpoint tensors unused (blocks not built).",
            num_blocks, " + pixel pathway" if self.pixel_pathway else "", len(unexpected),
        )
        model.mask_text_padding = bool(mask_text_padding)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.eval().to(device=device, dtype=dtype)
        self.patch_size = self.model.core.patch_size
        self.hidden_size = self.model.core.hidden_size

        # Text conditioning.
        te_name = config.text_encoder.text_encoder_name
        self.captions = captions
        self._te = None
        if captions is None:
            emb, mask = cached_prompt_embedding(prompt, te_name, cache_dir, device)
            self.register_buffer("prompt_emb", emb.to(device), persistent=False)
            self.register_buffer("prompt_mask", mask.to(device), persistent=False)
        else:
            # Per-image captions: keep Gemma loaded (bf16, ~5 GB).
            from t2i.diffusion.model.builder import get_tokenizer_and_text_encoder

            self._te = get_tokenizer_and_text_encoder(name=te_name, device=device)

        self._pit_tokens: List[torch.Tensor] = []
        if self.pixel_pathway:
            for blk in self.model.core.pixel_blocks:
                blk.attn.register_forward_hook(lambda m, i, o: self._pit_tokens.append(o))

    def conditioning(self, keys: Optional[List[str]] = None):
        if self.captions is None:
            return self.prompt_emb, self.prompt_mask
        caps = [self.captions.get(os.path.basename(k), "") for k in keys]
        emb, mask = encode_with(*self._te, caps)
        return emb.to(self.device_str), mask.to(self.device_str)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, keys: Optional[List[str]] = None, seeds: Optional[List[int]] = None):
        """images: [B, 3, H, W] in [-1, 1], H and W multiples of 16.
        keys: image paths (for captions); seeds: per-image noise seeds."""
        B, _, H, W = images.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (H, W)
        x = images.to(self.device_str, torch.float32)
        if self.noise and self.timestep > 0:
            eps = torch.stack([
                torch.randn(x.shape[1:], generator=torch.Generator().manual_seed(self.noise_seed + int(s)))
                for s in (seeds if seeds is not None else range(B))
            ]).to(x.device)
            x = (1.0 - self.timestep) * x + self.timestep * eps
        t = torch.full((B,), self.timestep * 1000.0, device=x.device, dtype=torch.float32)
        emb, mask = self.conditioning(keys)
        seg = {"want_pixel_features": self.pixel_pathway}
        self._pit_tokens = []
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
            self.model(x.to(self.dtype), t, emb, mask=mask, segment_data=seg)
        h, w = seg["hw"]
        pf = seg["patch_features"][:, self.layers]                       # [B, L, hw, C]
        out = {"patch": pf.reshape(B, len(self.layers), h, w, -1), "hw": (h, w)}
        if self.pixel_pathway:
            out["pit_attn"] = torch.stack(self._pit_tokens, 1).reshape(B, len(self._pit_tokens), h, w, -1)
            out["pixel"] = seg["pixel_features"]
        return out


def preprocess_image(img_uint8: torch.Tensor, target_area: int, max_side: int, patch: int = 16) -> torch.Tensor:
    """uint8 [3, H, W] -> float [1, 3, H', W'] in [-1, 1]."""
    Hn, Wn = resize_for_pixeldit(img_uint8.shape[-2:], target_area, patch, max_side)
    x = img_uint8[None].float() / 255.0
    x = F.interpolate(x, size=(Hn, Wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return x * 2 - 1
