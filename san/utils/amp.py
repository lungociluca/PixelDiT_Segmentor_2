import logging

import torch

_PRECISIONS = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def has_native_bf16(device=None) -> bool:
    """True only on GPUs with hardware bf16 (Ampere / compute capability >= 8.0).

    Not torch.cuda.is_bf16_supported(): since torch 2.3 it also returns True on
    pre-Ampere GPUs (e.g. RTX 20xx / Turing, sm_75) that only *emulate* bf16. There,
    bf16 matmuls are slow, and the flash and memory-efficient SDPA kernels don't
    support bf16, so attention falls back to the math kernel, which materialises
    the full N x N attention matrix and uses far more memory.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device)[0] >= 8


def resolve_amp_dtype(cfg):
    """Autocast dtype from SOLVER.AMP, or None when AMP is disabled.

    When bfloat16 is requested on a GPU without native bf16 (pre-Ampere, e.g. RTX
    2060), uses SOLVER.AMP.BF16_FALLBACK instead: "float32" (default; AMP off) or
    "float16" (faster, uses a GradScaler, but may overflow).
    """
    if not cfg.SOLVER.AMP.ENABLED:
        return None
    name = cfg.SOLVER.AMP.PRECISION.lower()
    if name not in _PRECISIONS:
        raise ValueError(f"SOLVER.AMP.PRECISION must be one of {list(_PRECISIONS)}, got {name!r}")
    dtype = _PRECISIONS[name]
    if dtype == torch.bfloat16 and torch.cuda.is_available() and not has_native_bf16():
        fallback = str(cfg.SOLVER.AMP.get("BF16_FALLBACK", "float32")).lower()
        if fallback not in ("float16", "fp16", "float32", "fp32"):
            raise ValueError(
                f"SOLVER.AMP.BF16_FALLBACK must be 'float16' or 'float32', got {fallback!r}"
            )
        dtype = _PRECISIONS[fallback]
        logging.getLogger("san").warning(
            "SOLVER.AMP.PRECISION is bfloat16 but this GPU has no native bf16 "
            "(compute capability < 8.0); using %s (SOLVER.AMP.BF16_FALLBACK).",
            dtype,
        )
    if dtype == torch.float32:
        return None  # plain fp32, no autocast
    return dtype
