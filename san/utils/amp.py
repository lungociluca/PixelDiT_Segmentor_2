import logging

import torch

_PRECISIONS = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_amp_dtype(cfg):
    """Autocast dtype from SOLVER.AMP, or None when AMP is disabled.

    Falls back to float16 when bfloat16 is requested on a GPU without bf16 support
    (pre-Ampere), so the same config runs everywhere.
    """
    if not cfg.SOLVER.AMP.ENABLED:
        return None
    name = cfg.SOLVER.AMP.PRECISION.lower()
    if name not in _PRECISIONS:
        raise ValueError(f"SOLVER.AMP.PRECISION must be one of {list(_PRECISIONS)}, got {name!r}")
    dtype = _PRECISIONS[name]
    # Not torch.cuda.is_bf16_supported(): since torch 2.3 it also returns True on
    # pre-Ampere GPUs that only *emulate* bf16. There, the flash and memory-efficient
    # SDPA kernels don't support bf16, so attention falls back to the math kernel,
    # which materialises the full N x N attention matrix and uses far more memory.
    if (
        dtype == torch.bfloat16
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] < 8
    ):
        logging.getLogger("san").warning(
            "SOLVER.AMP.PRECISION is bfloat16 but this GPU has no native bf16 "
            "(compute capability < 8.0); using float16."
        )
        dtype = torch.float16
    return dtype


def resolve_backbone_dtype(cfg):
    """Dtype of the frozen PixelDiT weights and of its forward pass.

    MODEL.PIXELDIT.PRECISION:
      "auto"    - the autocast dtype when that is bfloat16 (what PixelDiT was trained
                  in), float32 otherwise. float16 has bf16's precision but not its
                  range (max 65504 vs 3.4e38), so a network trained in bf16 can
                  overflow to inf/NaN in it; float32 is the safe substitute.
      "float32" - always float32.
      "amp"     - always the autocast dtype, float16 included (least memory). The
                  model raises if the features come out non-finite.
    """
    name = cfg.MODEL.PIXELDIT.PRECISION.lower()
    if name not in ("auto", "amp", "float32", "fp32"):
        raise ValueError(
            f"MODEL.PIXELDIT.PRECISION must be 'auto', 'float32' or 'amp', got {name!r}"
        )
    amp_dtype = resolve_amp_dtype(cfg)
    if amp_dtype is None or name in ("float32", "fp32"):
        return torch.float32
    if name == "auto" and amp_dtype == torch.float16:
        return torch.float32
    return amp_dtype
