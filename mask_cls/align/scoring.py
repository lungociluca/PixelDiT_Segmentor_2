"""How a map's predictions are turned into embeddings for scoring, plus collapse diagnostics.

CLIP region embeddings share a large common component ȳ (‖ȳ‖ is typically 0.7-0.9 for
unit vectors), and the class-discriminative part y - ȳ is comparatively small. A regressor
that only partly explains y - ȳ (ridge shrinkage, or a cosine loss that is already high
when the mean alone is predicted) outputs ≈ ȳ + small residual. When scored against text,
ȳ·t is a fixed per-class bias, so the bias dominates and predictions collapse to a few
classes. Modes:

  raw      normalize(p)                                  (what the first runs reported)
  rescale  normalize(ȳ + k (p - p̄)),  k = sqrt(E‖y - ȳ‖² / E‖p - p̄‖²)
           -> residual magnitude matched to the teacher's; uses no labels
  center   normalize(p - p̄)  (drops the shared component, and with it the ȳ·t bias)

p = normalize(map(x)); ȳ, p̄, k are computed on the COCO fit set.
"""
from typing import Dict

import torch
import torch.nn.functional as F

from .fit import predict

MODES = ("raw", "rescale", "center")


@torch.no_grad()
def scoring_stats(map_, X, Y, device="cuda") -> Dict[str, torch.Tensor]:
    P = predict(map_, X, device).double()
    Yn = F.normalize(Y.double(), dim=-1)
    y_mean, p_mean = Yn.mean(0), P.mean(0)
    var_y = ((Yn - y_mean) ** 2).sum(-1).mean()
    var_p = ((P - p_mean) ** 2).sum(-1).mean().clamp_min(1e-12)
    return {"y_mean": y_mean.float(), "p_mean": p_mean.float(), "k": float((var_y / var_p).sqrt()),
            "y_mean_norm": float(y_mean.norm()), "resid_norm_y": float(var_y.sqrt()),
            "resid_norm_p": float(var_p.sqrt())}


def apply_scoring(P: torch.Tensor, mode: str, stats) -> torch.Tensor:
    """P: normalised predictions [N, D] (CPU)."""
    if mode == "raw":
        return P
    if stats is None:
        raise ValueError(f"scoring mode {mode!r} needs fit-set statistics")
    if mode == "rescale":
        return F.normalize(stats["y_mean"] + stats["k"] * (P.float() - stats["p_mean"]), dim=-1)
    if mode == "center":
        return F.normalize(P.float() - stats["p_mean"], dim=-1)
    raise ValueError(mode)


@torch.no_grad()
def diagnostics(P: torch.Tensor, Y, labels, text, stats=None) -> Dict[str, float]:
    """Holdout diagnostics: how much beyond the mean the map explains, and prediction collapse."""
    out = {}
    if Y is not None:
        Yn = F.normalize(Y.float(), dim=-1)
        out["cos_map"] = float((P * Yn).sum(-1).mean())
        mean_only = F.normalize(Yn.mean(0), dim=-1)
        out["cos_mean_only"] = float((Yn @ mean_only).mean())        # predictor that ignores the input
        if stats is not None:
            r_t = Yn - stats["y_mean"]
            r_p = P.float() - stats["p_mean"]
            out["cos_residual"] = float(F.cosine_similarity(r_p, r_t, dim=-1).mean())
            out["teacher_y_mean_norm"] = stats["y_mean_norm"]
            out["k_rescale"] = stats["k"]
    T = F.normalize(text.float(), dim=-1)
    pred = (P.float() @ T.T).argmax(1)
    counts = torch.bincount(pred, minlength=T.shape[0]).float()
    pr = counts / counts.sum()
    out["top_pred_class_frac"] = float(pr.max())
    out["pred_classes_used"] = int((counts > 0).sum())
    out["pred_entropy_bits"] = float(-(pr[pr > 0] * pr[pr > 0].log2()).sum())
    if labels is not None:
        out["coco_top1"] = float((pred == labels).float().mean()) * 100
    return out
