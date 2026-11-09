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
