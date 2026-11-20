"""Fitting alignment maps on cached region features.

X: [N, D_in] student features (fp16 on CPU), Y: [N, D_out] L2-normalised teacher targets.
Model selection uses a HOLDOUT split of COCO training images (mean cosine to the
teacher, or CE accuracy on COCO classes for the labels-only baseline), never the
evaluation datasets.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .maps import AlignmentMap, LinearMap, ResidualMLPMap

log = logging.getLogger("mask_cls")


@torch.no_grad()
def moments(X: torch.Tensor, Y: Optional[torch.Tensor], device="cuda", chunk: int = 32768):
    """Two-pass centred moments (fp32 per chunk, fp64 accumulation).

    Returns mean_x, std_x, Cxx = Σ(x-μ)(x-μ)ᵀ, Cxy = Σ(x-μ)(y-ȳ)ᵀ, mean_y. Centring first
    avoids cancellation from the few very large channels DiT residual streams tend to have."""
    N, D = X.shape
    sx = torch.zeros(D, dtype=torch.float64, device=device)
    sy = torch.zeros(Y.shape[1], dtype=torch.float64, device=device) if Y is not None else None
    for i in range(0, N, chunk):
        sx += X[i:i + chunk].to(device, torch.float64).sum(0)
        if Y is not None:
            sy += Y[i:i + chunk].to(device, torch.float64).sum(0)
    mx = sx / N
    my = sy / N if Y is not None else None
    cxx = torch.zeros(D, D, dtype=torch.float64, device=device)
    cxy = torch.zeros(D, Y.shape[1], dtype=torch.float64, device=device) if Y is not None else None
    mx32 = mx.float()
    for i in range(0, N, chunk):
        x = X[i:i + chunk].to(device, torch.float32) - mx32
        cxx += (x.T @ x).double()
        if Y is not None:
            y = Y[i:i + chunk].to(device, torch.float32) - my.float()
            cxy += (x.T @ y).double()
    std = (torch.diagonal(cxx) / N).clamp_min(0).sqrt().clamp_min(1e-6)
    return {"N": N, "mean_x": mx, "std_x": std, "cxx": cxx, "cxy": cxy, "mean_y": my}


def _standardized_stats(m):
    """A = ZᵀZ, B = ZᵀYc for Z = (X - μ)/σ, Yc = Y - ȳ."""
    sd = m["std_x"]
    A = m["cxx"] / torch.outer(sd, sd)
    B = None if m["cxy"] is None else m["cxy"] / sd[:, None]
    return A, B


@torch.no_grad()
def predict(map_: AlignmentMap, X: torch.Tensor, device="cuda", chunk: int = 65536) -> torch.Tensor:
    map_ = map_.to(device).eval()
    out = [map_.embed(X[i:i + chunk].to(device)).cpu() for i in range(0, len(X), chunk)]
    return torch.cat(out)


@torch.no_grad()
def mean_cosine(map_: AlignmentMap, X, Y, device="cuda") -> float:
    P = predict(map_, X, device)
    return float((P * F.normalize(Y.float(), dim=-1)).sum(-1).mean())


@torch.no_grad()
def fit_ridge(X, Y, Xh, Yh, feature_spec: List[str], alphas: List[float], device="cuda"):
    """Closed-form ridge on standardised X -> normalised Y, lambda = alpha * N.

    One eigendecomposition of ZᵀZ gives the whole lambda path. Returns
    (best map, best alpha, [{alpha, holdout_cos}], moments)."""
    m = moments(X, Y, device)
    A, B = _standardized_stats(m)
    evals, evecs = torch.linalg.eigh(A)
    evals = evals.clamp_min(0)
    VtB = evecs.T @ B
    best, table = None, []
    for a in alphas:
        lam = a * m["N"]
        W = evecs @ (VtB / (evals + lam)[:, None])
        mp = LinearMap(X.shape[1], Y.shape[1], feature_spec)
        mp.set_standardization(m["mean_x"].float(), m["std_x"].float())
        mp.weight.data.copy_(W.float())
        mp.bias.data.copy_(m["mean_y"].float())
        cos = mean_cosine(mp, Xh, Yh, device)
        table.append({"alpha": a, "holdout_cos": cos})
        log.info("ridge alpha=%g  holdout cos=%.4f", a, cos)
        if best is None or cos > best[1]:
            best = (mp.cpu(), cos, a)
    return best[0], best[2], table, m


@torch.no_grad()
def fit_procrustes(X, Y, Xh, Yh, feature_spec, device="cuda", m=None):
    """Orthogonal (semi-orthogonal if D_in != D_out) Procrustes on standardised X.

    W = U Vᵀ from SVD(ZᵀYc), with the optimal global scale s = tr(Σ)/‖Z‖²."""
    m = m or moments(X, Y, device)
    A, B = _standardized_stats(m)
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    W = U @ Vh
    scale = S.sum() / torch.trace(A).clamp_min(1e-12)
    mp = LinearMap(X.shape[1], Y.shape[1], feature_spec)
    mp.set_standardization(m["mean_x"].float(), m["std_x"].float())
    mp.weight.data.copy_((scale * W).float())
    mp.bias.data.copy_(m["mean_y"].float())
    cos = mean_cosine(mp, Xh, Yh, device)
    log.info("procrustes holdout cos=%.4f", cos)
    return mp.cpu(), cos


def fit_sgd(
    X, Y, labels, Xh, Yh, labels_h, feature_spec, method: str, text_emb: Optional[torch.Tensor],
    epochs: int = 20, batch_size: int = 4096, lr: float = 1e-3, weight_decay: float = 1e-4,
    distill_weight: float = 1.0, ce_weight: float = 0.0, ce_temperature: float = 0.01,
    mlp_hidden: int = 2048, mlp_dropout: float = 0.1, init_map: Optional[LinearMap] = None,
    seed: int = 0, device: str = "cuda", d_out: Optional[int] = None,
) -> Tuple[AlignmentMap, List[dict]]:
    """Linear layer (method="linear") or residual MLP (method="mlp") trained with
    distill_weight * (1 - cos(map(x), teacher)) + ce_weight * CE(cos(map(x), text) / T, label).

    distill_weight=0, ce_weight=1 is the labels-only baseline."""
    if distill_weight == 0 and ce_weight == 0:
        raise ValueError("Both loss weights are 0")
    if ce_weight > 0 and text_emb is None:
        raise ValueError("CE loss needs the COCO text embeddings")
    d_out = d_out or (Y.shape[1] if Y is not None else text_emb.shape[1])
    D = X.shape[1]
    m = moments(X, None, device)
    if method == "linear":
        net = LinearMap(D, d_out, feature_spec)
        if init_map is not None:
            net.load_state_dict(init_map.state_dict())
        else:
            torch.nn.init.normal_(net.weight, std=1.0 / math.sqrt(D))
    elif method == "mlp":
        net = ResidualMLPMap(D, d_out, feature_spec, hidden=mlp_hidden, dropout=mlp_dropout)
        if init_map is not None:
            net.linear.weight.data.copy_(init_map.weight.data.T)
            net.linear.bias.data.copy_(init_map.bias.data)
    else:
        raise ValueError(method)
    net.set_standardization(m["mean_x"].float(), m["std_x"].float())
    net = net.to(device)
    T = F.normalize(text_emb.float(), dim=-1).to(device) if text_emb is not None else None

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(X) // batch_size)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    g = torch.Generator().manual_seed(seed)

    def holdout_metrics():
        net.eval()
        with torch.no_grad():
            P = predict(net, Xh, device)
        out = {}
        if Yh is not None:
            out["holdout_cos"] = float((P * F.normalize(Yh.float(), dim=-1)).sum(-1).mean())
        if T is not None and labels_h is not None:
            out["holdout_coco_acc"] = float(((P.to(device) @ T.T).argmax(1).cpu() == labels_h).float().mean())
        net.train()
        return out

    key = "holdout_cos" if distill_weight > 0 else "holdout_coco_acc"
    best_state, best_val, history = None, -1e9, []
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            x = X[idx].to(device, non_blocking=True)
            e = F.normalize(net(x), dim=-1)
            loss = 0.0
            if distill_weight > 0:
                y = F.normalize(Y[idx].to(device).float(), dim=-1)
                loss = loss + distill_weight * (1 - (e * y).sum(-1)).mean()
            if ce_weight > 0:
                loss = loss + ce_weight * F.cross_entropy((e @ T.T) / ce_temperature, labels[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        hm = holdout_metrics()
        history.append({"epoch": ep, "train_loss": tot / steps_per_epoch, **hm})
        log.info("epoch %d  loss %.4f  %s", ep, tot / steps_per_epoch, hm)
        if hm[key] > best_val:
            best_val = hm[key]
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.cpu().eval(), history
