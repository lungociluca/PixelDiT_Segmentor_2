"""Writes metrics.csv / metrics.json / summary.md and plots from a RawStore."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List

import numpy as np

from .metrics import (
    HEADLINE_METRICS,
    METRIC_DESCRIPTIONS,
    RawStore,
    compute_all_metrics,
    confusion_matrix,
    parse_key,
    per_class_metrics,
)

SPACE_ORDER = ["q", "k", "v", "qk", "kq", "attn_out", "block"]
RANKING_METRIC = "maskcls_top1"


def _fmt(x) -> str:
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def _space_sort(space: str):
    base = space.split("@")[0]
    return (SPACE_ORDER.index(base) if base in SPACE_ORDER else 99, space)


def _rank(metrics: Dict[str, Dict[str, float]], metric: str, keys: List[str], top: int):
    vals = [(k, metrics[k].get(metric, float("nan"))) for k in keys]
    vals = [(k, v) for k, v in vals if not math.isnan(v)]
    reverse = metric != "maskcls_mean_rank"
    return sorted(vals, key=lambda kv: kv[1], reverse=reverse)[:top]


def write_report(store: RawStore, out_dir: str, top: int = 10, make_plots: bool = True) -> Dict[str, Dict[str, float]]:
    os.makedirs(out_dir, exist_ok=True)
    metrics = compute_all_metrics(store)
    keys = list(metrics)
    full_keys = [k for k in keys if "@h" not in parse_key(k)[1]]
    head_keys = [k for k in keys if "@h" in parse_key(k)[1]]
    metric_names = sorted({m for v in metrics.values() for m in v},
                          key=lambda m: (m not in HEADLINE_METRICS, HEADLINE_METRICS.index(m) if m in HEADLINE_METRICS else 0, m))

    # ---------------------------------------------------------------- csv
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "space"] + metric_names)
        for k in sorted(keys, key=lambda k: (_space_sort(parse_key(k)[1]), parse_key(k)[0])):
            layer, space = parse_key(k)
            w.writerow([layer, space] + [metrics[k].get(m, float("nan")) for m in metric_names])

    # --------------------------------------------------------------- best keys
    best = {m: _rank(metrics, m, full_keys, top) for m in HEADLINE_METRICS}
    best_key = best[RANKING_METRIC][0][0] if best[RANKING_METRIC] else (full_keys[0] if full_keys else None)

    js = {
        "num_images": store.num_images,
        "num_segments": len(store.seg_label),
        "class_names": store.class_names,
        "meta": store.meta,
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "ranking_metric": RANKING_METRIC,
        "best": {m: [{"layer": parse_key(k)[0], "space": parse_key(k)[1], "value": v} for k, v in lst]
                 for m, lst in best.items()},
        "metrics": {k: metrics[k] for k in keys},
    }
    if best_key is not None:
        js["best_key_per_class"] = {"key": best_key, "per_class": per_class_metrics(store, best_key)}
        js["best_key_confusion"] = {"key": best_key, "rows_true_cols_pred": confusion_matrix(store, best_key).tolist()}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(js, f, indent=1, default=float)

    # ------------------------------------------------------------ summary.md
    lines = [
        "# Vision-language alignment statistics",
        "",
        f"Images: {store.num_images} · ground-truth segments: {len(store.seg_label)} · classes: {store.num_classes}",
        "",
        "Settings: " + ", ".join(f"{k}={v}" for k, v in store.meta.items() if not isinstance(v, (list, dict))),
        "",
        "## Best layer per space (ranked by oracle-mask classification accuracy)",
        "",
        "| space | best layer | maskcls_top1 | debiased | among present | presence AUROC | presence AUROC (macro) | image mAP | loc AUROC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    spaces = sorted({parse_key(k)[1] for k in full_keys}, key=_space_sort)
    for sp in spaces:
        cands = [k for k in full_keys if parse_key(k)[1] == sp]
        r = _rank(metrics, RANKING_METRIC, cands, 1)
        if not r:
            continue
        k = r[0][0]
        m = metrics[k]
        lines.append(f"| {sp} | {parse_key(k)[0]} | {_fmt(m['maskcls_top1'])} | {_fmt(m['maskcls_top1_debiased'])} | "
                     f"{_fmt(m['maskcls_top1_among_present'])} | {_fmt(m['presence_auroc'])} | "
                     f"{_fmt(m['presence_auroc_macro'])} | {_fmt(m['image_map'])} | {_fmt(m['loc_auroc'])} |")
    for metric in HEADLINE_METRICS:
        lines += ["", f"## Top {top} by `{metric}`", "", METRIC_DESCRIPTIONS.get(metric, ""), "",
                  "| rank | layer | space | value |", "|---|---|---|---|"]
        for i, (k, v) in enumerate(best[metric], 1):
            lines.append(f"| {i} | {parse_key(k)[0]} | {parse_key(k)[1]} | {_fmt(v)} |")
    if head_keys:
        lines += ["", f"## Top {top} single attention heads by `{RANKING_METRIC}`", "",
                  "| rank | layer | space@head | value |", "|---|---|---|---|"]
        for i, (k, v) in enumerate(_rank(metrics, RANKING_METRIC, head_keys, top), 1):
            lines.append(f"| {i} | {parse_key(k)[0]} | {parse_key(k)[1]} | {_fmt(v)} |")
    lines += ["", "## Metric definitions", ""] + [f"- **{k}**: {v}" for k, v in METRIC_DESCRIPTIONS.items()]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    if make_plots and full_keys:
        try:
            _plots(store, metrics, full_keys, head_keys, best, out_dir)
        except Exception as e:  # plotting must never lose the numbers
            print(f"[vl_stats] plotting failed: {e}")
    return metrics


def _plots(store, metrics, full_keys, head_keys, best, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({parse_key(k)[0] for k in full_keys})
    spaces = sorted({parse_key(k)[1] for k in full_keys}, key=_space_sort)

    def grid(metric, ks, rows, row_of):
        g = np.full((len(rows), len(layers)), np.nan)
        for k in ks:
            l, s = parse_key(k)
            g[rows.index(row_of(s)), layers.index(l)] = metrics[k].get(metric, np.nan)
        return g

    # heatmaps space x layer
    plot_metrics = HEADLINE_METRICS
    ncol = 2
    nrow = math.ceil(len(plot_metrics) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.0 + 0.55 * len(layers) * ncol, 0.5 * len(spaces) * nrow + 1.5 * nrow),
                             squeeze=False)
    for ax, metric in zip(axes.flat, plot_metrics):
        g = grid(metric, full_keys, spaces, lambda s: s)
        im = ax.imshow(g, aspect="auto", cmap="viridis")
        ax.set_title(metric, fontsize=10)
        ax.set_xticks(range(len(layers)), layers, fontsize=8)
        ax.set_yticks(range(len(spaces)), spaces, fontsize=8)
        ax.set_xlabel("layer", fontsize=8)
        finite = np.isfinite(g)
        if finite.any():
            i, j = np.unravel_index(np.nanargmax(g), g.shape)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=1.5))
            if g.size <= 160:
                for (ii, jj), v in np.ndenumerate(g):
                    if np.isfinite(v):
                        ax.text(jj, ii, f"{v:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if v < np.nanmean(g) else "black")
        fig.colorbar(im, ax=ax, fraction=0.03)
    for ax in list(axes.flat)[len(plot_metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmaps.png"), dpi=150)
    plt.close(fig)

    # metric vs layer curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["maskcls_top1", "presence_auroc", "loc_auroc"]):
        for sp in spaces:
            ys = [metrics.get(f"{l}|{sp}", {}).get(metric, np.nan) for l in layers]
            ax.plot(layers, ys, marker="o", ms=3, label=sp)
        ax.set_title(metric)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=150)
    plt.close(fig)

    # present vs absent histograms for the best keys
    P = store.presence_array()
    hist_keys = [k for k, _ in best["presence_auroc"][:3]]
    for k, _ in best[RANKING_METRIC][:1]:
        if k not in hist_keys:
            hist_keys.append(k)
    if hist_keys:
        fig, axes = plt.subplots(2, len(hist_keys), figsize=(4.5 * len(hist_keys), 7), squeeze=False)
        for j, k in enumerate(hist_keys):
            a = store.arrays(k)
            S = a["img_scores"]
            ax = axes[0, j]
            ax.hist(S[P == 0], bins=60, alpha=0.6, density=True, label="absent")
            ax.hist(S[P == 1], bins=60, alpha=0.6, density=True, label="present")
            ax.set_title(f"layer {parse_key(k)[0]} · {parse_key(k)[1]}\nimage-level score "
                         f"(AUROC {metrics[k]['presence_auroc']:.3f})", fontsize=9)
            ax.legend(fontsize=8)
            seg = a["seg_scores"]
            y = np.asarray(store.seg_label)
            ok = np.isfinite(seg).all(1)
            seg, y_ok, si = seg[ok], y[ok], np.asarray(store.seg_img)[ok]
            true = seg[np.arange(len(y_ok)), y_ok]
            absent = seg[P[si] == 0]
            other = seg.copy()
            other[np.arange(len(y_ok)), y_ok] = np.nan
            other = other[(P[si] == 1)]
            ax = axes[1, j]
            ax.hist(absent, bins=60, alpha=0.5, density=True, label="absent class")
            ax.hist(other[np.isfinite(other)], bins=60, alpha=0.5, density=True, label="other present class")
            ax.hist(true, bins=60, alpha=0.6, density=True, label="true class")
            ax.set_title(f"segment score (oracle masks, top1 {metrics[k]['maskcls_top1']:.3f})", fontsize=9)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "present_vs_absent.png"), dpi=150)
        plt.close(fig)

    # per-head heatmaps
    if head_keys:
        head_spaces = sorted({parse_key(k)[1].split("@")[0] for k in head_keys}, key=_space_sort)
        fig, axes = plt.subplots(1, len(head_spaces), figsize=(0.5 * len(layers) * len(head_spaces) + 2, 7), squeeze=False)
        for ax, hs in zip(axes[0], head_spaces):
            ks = [k for k in head_keys if parse_key(k)[1].startswith(hs + "@")]
            heads = sorted({parse_key(k)[1].split("@h")[1] for k in ks})
            g = grid(RANKING_METRIC, ks, heads, lambda s: s.split("@h")[1])
            im = ax.imshow(g, aspect="auto", cmap="viridis")
            ax.set_title(f"{hs}: per-head {RANKING_METRIC}", fontsize=10)
            ax.set_xticks(range(len(layers)), layers, fontsize=7)
            ax.set_yticks(range(len(heads)), heads, fontsize=7)
            ax.set_xlabel("layer")
            ax.set_ylabel("head")
            fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "per_head.png"), dpi=150)
        plt.close(fig)

    # confusion matrix of the best key
    k = best[RANKING_METRIC][0][0] if best[RANKING_METRIC] else None
    if k is not None:
        cm = confusion_matrix(store, k).astype(float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(store.num_classes), store.class_names, rotation=90, fontsize=7)
        ax.set_yticks(range(store.num_classes), store.class_names, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"oracle-mask confusion · layer {parse_key(k)[0]} · {parse_key(k)[1]}")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_best.png"), dpi=150)
        plt.close(fig)
