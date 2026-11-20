"""Region-level top-k accuracy and oracle-mask mIoU, split into seen / unseen classes.

Oracle-mask mIoU: every GT mask is painted with its predicted label and semantic
mIoU is computed. GT masks of one image are disjoint (one per class, ignore pixels
excluded), so the painted map's confusion matrix is exactly
    conf[gt, pred] += area(mask)
summed over regions; no image needs to be re-painted. IoU follows detectron2's
SemSegEvaluator: classes with no GT and no prediction pixels are left out of the mean.

Temperature does not change any of these metrics (they are argmax / top-k), so
`eval.temperature` only matters for calibrated scores, not for the reported numbers.
"""
from typing import Dict, List, Optional, Sequence

import torch


def _hmean(a, b):
    if a is None or b is None or a + b == 0:
        return None
    return 2 * a * b / (a + b)


@torch.no_grad()
def region_metrics(
    scores: torch.Tensor,     # [N, C]
    labels: torch.Tensor,     # [N]
    areas: torch.Tensor,      # [N] pixels
    seen: Sequence[bool],     # [C]
    topk: Sequence[int] = (1, 5),
) -> Dict[str, float]:
    C = scores.shape[1]
    seen_t = torch.tensor(list(seen), dtype=torch.bool)
    labels = labels.long().cpu()
    areas = areas.double().cpu()
    kmax = min(max(topk), C)
    top = scores.float().topk(kmax, dim=1).indices.cpu()
    pred = top[:, 0]
    region_seen = seen_t[labels]
    out: Dict[str, Optional[float]] = {"num_regions": int(len(labels)),
                                       "num_regions_seen": int(region_seen.sum()),
                                       "num_regions_unseen": int((~region_seen).sum())}
    for k in topk:
        hit = (top[:, :min(k, C)] == labels[:, None]).any(1).float()
        out[f"top{k}"] = float(hit.mean()) * 100
        out[f"top{k}_seen"] = float(hit[region_seen].mean()) * 100 if region_seen.any() else None
        out[f"top{k}_unseen"] = float(hit[~region_seen].mean()) * 100 if (~region_seen).any() else None
        out[f"top{k}_h"] = _hmean(out[f"top{k}_seen"], out[f"top{k}_unseen"])

    # Class-mean top-1 (each class weighted equally; regions are dominated by frequent classes).
    hit1 = (pred == labels).float()
    per_cls = torch.zeros(C).index_add_(0, labels, hit1) / torch.zeros(C).index_add_(0, labels, torch.ones_like(hit1)).clamp_min(1)
    present = torch.zeros(C, dtype=torch.bool)
    present[labels.unique()] = True
    for name, sel in (("", present), ("_seen", present & seen_t), ("_unseen", present & ~seen_t)):
        out[f"macc{name}"] = float(per_cls[sel].mean()) * 100 if sel.any() else None

    # Oracle-mask mIoU from area-weighted confusion.
    conf = torch.zeros(C * C, dtype=torch.float64).index_add_(0, labels * C + pred, areas).view(C, C)
    tp = conf.diag()
    gt = conf.sum(1)
    pr = conf.sum(0)
    union = gt + pr - tp
    iou = tp / union.clamp_min(1)
    valid = union > 0
    gt_valid = gt > 0
    for name, sel in (("", valid), ("_seen", valid & seen_t), ("_unseen", valid & ~seen_t)):
        out[f"miou{name}"] = float(iou[sel].mean()) * 100 if sel.any() else None
    out["miou_h"] = _hmean(out["miou_seen"], out["miou_unseen"])
    out["pixel_acc"] = float(tp.sum() / gt.sum().clamp_min(1)) * 100
    out["num_classes_gt"] = int(gt_valid.sum())
    out["num_classes_gt_unseen"] = int((gt_valid & ~seen_t).sum())
    return out


@torch.no_grad()
def classify(embeddings: torch.Tensor, text: torch.Tensor, device: str = "cuda", chunk: int = 65536) -> torch.Tensor:
    """Cosine scores [N, C] between L2-normalised region and text embeddings."""
    T = torch.nn.functional.normalize(text.float(), dim=-1).to(device)
    out = []
    for i in range(0, len(embeddings), chunk):
        e = torch.nn.functional.normalize(embeddings[i:i + chunk].float().to(device), dim=-1)
        out.append((e @ T.T).cpu())
    return torch.cat(out)
