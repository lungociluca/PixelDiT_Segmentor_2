"""Mask pooling with area-weighted downsampling of masks to the feature grid."""
import torch
import torch.nn.functional as F


def mask_area_weights(masks: torch.Tensor, out_hw) -> torch.Tensor:
    """masks [n, H, W] (bool/float, any resolution) -> [n, h, w] fraction of each
    feature cell covered by the mask ('area' interpolation = adaptive average pooling)."""
    m = masks.float()[:, None]
    if tuple(m.shape[-2:]) == tuple(out_hw):
        return m[:, 0]
    return F.interpolate(m, size=tuple(out_hw), mode="area")[:, 0]


def mask_pool(feats: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted average of grid features inside each mask.

    feats:   [..., h, w, C]  (leading dims, e.g. layers, are kept)
    weights: [n, h, w]
    returns: [n, ..., C]
    """
    lead = feats.shape[:-3]
    h, w, C = feats.shape[-3:]
    f = feats.reshape(-1, h * w, C).float()                      # [L, hw, C]
    wt = weights.reshape(weights.shape[0], h * w).float()       # [n, hw]
    denom = wt.sum(1, keepdim=True).clamp_min(eps)
    pooled = torch.einsum("np,lpc->nlc", wt / denom, f)         # [n, L, C]
    return pooled.reshape(weights.shape[0], *lead, C)
