"""CPU unit tests for the Stage 1 maths (no GPU, data or checkpoints needed).

    python -m pytest tests/mask_cls -q
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mask_cls.align.fit import fit_procrustes, fit_ridge, fit_sgd, moments  # noqa: E402
from mask_cls.config import apply_overrides  # noqa: E402
from mask_cls.data.splits import compute_split  # noqa: E402
from mask_cls.eval.metrics import region_metrics  # noqa: E402
from mask_cls.features.pooling import mask_area_weights, mask_pool  # noqa: E402
from mask_cls.features.pixeldit import resize_for_pixeldit  # noqa: E402

DEV = "cpu"


def test_mask_pool_area_weighting():
    # 4x4 image, 2x2 grid; mask covers the top-left cell fully and 1/4 of the top-right one.
    m = torch.zeros(1, 4, 4, dtype=torch.bool)
    m[0, :2, :2] = True
    m[0, 0, 2] = True
    w = mask_area_weights(m, (2, 2))
    assert torch.allclose(w[0], torch.tensor([[1.0, 0.25], [0.0, 0.0]]))
    feats = torch.tensor([[[1.0], [5.0]], [[100.0], [100.0]]])        # [h, w, C]
    p = mask_pool(feats, w)
    assert torch.allclose(p, torch.tensor([[(1 * 1 + 0.25 * 5) / 1.25]]))
    # leading layer dim kept
    p2 = mask_pool(torch.stack([feats, 2 * feats]), w)
    assert p2.shape == (1, 2, 1) and torch.allclose(p2[0, 1], 2 * p2[0, 0])


def test_resize_multiple_of_16():
    for hw in [(480, 640), (333, 500), (2000, 1500)]:
        H, W = resize_for_pixeldit(hw, 1024 * 1024)
        assert H % 16 == 0 and W % 16 == 0
        assert abs(H * W / (1024 * 1024) - 1) < 0.1
        assert abs(H / W - hw[0] / hw[1]) < 0.05


def test_oracle_miou_equals_painting():
    torch.manual_seed(0)
    C = 5
    # Two images with disjoint per-class masks; paint predictions and compare.
    gts = [torch.randint(0, C, (6, 7)), torch.randint(0, C, (5, 5))]
    gts[0][0, :3] = 255
    labels, areas, painted_gt, painted_pred = [], [], [], []
    preds_per_region = []
    for gt in gts:
        for c in gt.unique().tolist():
            if c == 255:
                continue
            m = gt == c
            p = int(torch.randint(0, C, (1,)))
            labels.append(c); areas.append(int(m.sum())); preds_per_region.append(p)
            painted_gt.append(torch.full((int(m.sum()),), c)); painted_pred.append(torch.full((int(m.sum()),), p))
    labels = torch.tensor(labels)
    scores = F.one_hot(torch.tensor(preds_per_region), C).float()
    seen = [True, True, False, False, True]
    m = region_metrics(scores, labels, torch.tensor(areas), seen, topk=(1,))
    g, p = torch.cat(painted_gt), torch.cat(painted_pred)
    ious = []
    for c in range(C):
        inter = ((g == c) & (p == c)).sum().item()
        union = ((g == c) | (p == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    assert abs(m["miou"] - 100 * sum(ious) / len(ious)) < 1e-6
    assert abs(m["top1"] - 100 * float((labels == torch.tensor(preds_per_region)).float().mean())) < 1e-6


def test_moments_centred_with_large_offsets():
    torch.manual_seed(0)
    X = torch.randn(5000, 8) + torch.tensor([1e3, 0, 0, 0, 0, 0, 0, -5e2])
    m = moments(X, None, DEV, chunk=777)
    Xd = X.double()
    assert torch.allclose(m["mean_x"], Xd.mean(0), atol=1e-6)
    assert torch.allclose(m["std_x"], Xd.std(0, unbiased=False), rtol=1e-4)


def _toy(n=4000, d=16, k=8, noise=0.05):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, d, generator=g) * torch.linspace(0.5, 3, d) + 2.0
    W = torch.randn(d, k, generator=g)
    Y = F.normalize((X - 2.0) @ W + noise * torch.randn(n, k, generator=g), dim=-1)
    return X, Y


def test_ridge_matches_lstsq_and_recovers_map():
    X, Y = _toy()
    mp, alpha, table, _ = fit_ridge(X[:3500], Y[:3500], X[3500:], Y[3500:], ["patch.3"], [1e-8], device=DEV)
    Z = (X[:3500].double() - X[:3500].double().mean(0)) / X[:3500].double().std(0, unbiased=False)
    Yc = Y[:3500].double() - Y[:3500].double().mean(0)
    W = torch.linalg.lstsq(Z, Yc).solution
    assert torch.allclose(mp.weight.double(), W, atol=1e-4)
    assert table[0]["holdout_cos"] > 0.95


def test_procrustes_semi_orthogonal():
    X, Y = _toy(d=16, k=8)
    mp, cos = fit_procrustes(X[:3500], Y[:3500], X[3500:], Y[3500:], ["patch.3"], device=DEV)
    W = mp.weight.double()
    Wn = W / W.norm(dim=0, keepdim=True)
    assert torch.allclose(Wn.T @ Wn, torch.eye(8, dtype=torch.float64), atol=1e-5)


def test_sgd_linear_and_ce_only():
    X, Y = _toy(n=2000)
    labels = (Y @ torch.eye(8)).argmax(1)
    text = torch.eye(8)
    mp, hist = fit_sgd(X[:1800], Y[:1800], labels[:1800], X[1800:], Y[1800:], labels[1800:], ["patch.3"],
                       "linear", text, epochs=5, batch_size=256, lr=1e-2, device=DEV)
    assert hist[-1]["holdout_cos"] > 0.8
    mp2, hist2 = fit_sgd(X[:1800], None, labels[:1800], X[1800:], None, labels[1800:], ["patch.3"],
                         "mlp", text, epochs=3, batch_size=256, lr=1e-2, distill_weight=0.0, ce_weight=1.0,
                         device=DEV)
    assert "holdout_coco_acc" in hist2[-1]


def test_overrides():
    d = apply_overrides({"a": {"b": 1}}, ["a.b=[0,3]", "a.c=null", "x.y=1e-3"])
    assert d["a"]["b"] == [0, 3] and d["a"]["c"] is None and abs(d["x"]["y"] - 1e-3) < 1e-12


def test_split_matching():
    coco = ["person", "couch", "wall-other", "airplane", "tv"]
    voc = ["aeroplane", "sofa", "person", "tv", "sheep"]
    syn = {"aeroplane": "airplane", "sofa": "couch"}
    s = compute_split("voc_sem_seg_val", voc, coco, syn)
    assert s["seen"] == ["aeroplane", "sofa", "person", "tv"] and s["unseen"] == ["sheep"]
    s = compute_split("ade20k_full_sem_seg_val", ["wall", "building, edifice"], coco, {})
    assert s["seen"] == ["wall"] and s["unseen"] == ["building, edifice"]


def test_rescale_undoes_shrinkage_collapse():
    """Shrunk residual + large shared mean -> raw scoring collapses; rescale/center recover."""
    from mask_cls.align.maps import LinearMap
    from mask_cls.align.scoring import apply_scoring, diagnostics, scoring_stats

    g = torch.Generator().manual_seed(0)
    C, D, n = 10, 32, 3000
    text = F.normalize(torch.randn(C, D, generator=g), dim=-1)
    common = F.normalize(torch.randn(D, generator=g), dim=-1)
    labels = torch.randint(0, C, (n,), generator=g)
    Y = F.normalize(2.0 * common + 0.6 * text[labels] + 0.1 * torch.randn(n, D, generator=g), dim=-1)
    # A "map" that predicts mean + 10% of the residual, plus a per-class text bias that favours class 0.
    X = Y.clone()
    mp = LinearMap(D, D, ["patch.3"])
    ym = Y.mean(0)
    mp.weight.data = 0.1 * torch.eye(D)
    mp.bias.data = 0.9 * ym + 0.08 * text[0]
    stats = scoring_stats(mp, X, Y, device="cpu")
    from mask_cls.align.fit import predict
    P = predict(mp, X, device="cpu")
    raw = diagnostics(apply_scoring(P, "raw", stats), Y, labels, text, stats)
    res = diagnostics(apply_scoring(P, "rescale", stats), Y, labels, text, stats)
    cen = diagnostics(apply_scoring(P, "center", stats), Y, labels, text, stats)
    assert raw["top_pred_class_frac"] > 0.3
    assert res["coco_top1"] > raw["coco_top1"] + 30 and cen["coco_top1"] > raw["coco_top1"] + 30


def test_corrclip_scope_and_interaction_mask():
    from mask_cls.teacher.corrclip import interaction_mask, patch_scope

    # 4x4 image, 2x2 grid. Region 0: left column of cells, region 1: top-right cell,
    # bottom-right cell is ignore (no region).
    m = torch.zeros(2, 4, 4, dtype=torch.bool)
    m[0, :, :2] = True
    m[1, :2, 2:] = True
    assign, w = patch_scope(m, (2, 2), min_cover=0.5)
    assert assign.tolist() == [0, 1, 0, -1]
    S = torch.rand(4, 4)
    E = interaction_mask(assign, S, "self")
    assert E[0, 2] and E[2, 0] and not E[0, 1] and not E[1, 0]       # no inter-class attention
    assert not E[3, 0] and E[3, 3]                                   # unassigned: only itself
    assert torch.equal(E.diagonal(), torch.ones(4, dtype=torch.bool))


def test_corrclip_dense_smoke():
    """Tiny random open_clip ViT: every CorrCLIP option runs and gives finite tokens."""
    import contextlib
    import types

    import pytest

    open_clip = pytest.importorskip("open_clip")
    from mask_cls.teacher.clip import ClipTeacher
    from mask_cls.teacher.corrclip import corrclip_dense

    torch.manual_seed(0)
    vis = open_clip.transformer.VisionTransformer(
        image_size=32, patch_size=8, width=64, layers=3, heads=4, mlp_ratio=2.0, output_dim=16)
    vis.eval()
    teacher = types.SimpleNamespace(
        is_vit=True, patch=8, model=types.SimpleNamespace(visual=vis),
        _norm=lambda x: x, _autocast=contextlib.nullcontext, _interp_pos=ClipTeacher._interp_pos)
    x = torch.rand(1, 3, 32, 48)
    masks = torch.zeros(2, 32, 48, dtype=torch.bool)
    masks[0, :, :24] = True
    masks[1, :16, 24:] = True
    base = dict(similarity="clip_qq", tau=0.25, min_cover=0.5, unassigned="above_mean",
                spatial_alpha=0.0, spatial_layers=[-2], mct_beta=0.0)
    cc = types.SimpleNamespace(**base)
    out = corrclip_dense(teacher, x, masks, cc)
    assert out.shape == (4, 6, 16) and torch.isfinite(out).all()
    for k, val in [("similarity", "uniform"), ("spatial_alpha", 1.0), ("mct_beta", 0.5)]:
        o = corrclip_dense(teacher, x, masks, types.SimpleNamespace(**{**base, k: val}))
        assert o.shape == out.shape and torch.isfinite(o).all()
    # Mask class tokens only change patches that belong to a region.
    o = corrclip_dense(teacher, x, masks, types.SimpleNamespace(**{**base, "mct_beta": 0.5}))
    assert torch.allclose(o[2:, 3:], out[2:, 3:], atol=1e-5)          # bottom-right: no region
    assert not torch.allclose(o[:, :3], out[:, :3])
