"""Dataset-level vision-language alignment metrics (numpy only).

Everything here works on the small per-image reductions produced by
`t2i.vl_stats.probe.VLAlignmentProbe`, stored in a `RawStore`. Keeping this
module free of torch makes it possible to merge shards and recompute metrics on
any machine.

Conventions
-----------
* A *key* is one feature space at one layer, encoded as "<layer>|<space>",
  e.g. "7|qk" or "7|qk@h03" for a single attention head.
* `presence[i, c]` is 1 if class c is present in image i, 0 if it is absent and
  -1 if it is present only in a negligible area (excluded from presence metrics).
* `img_scores[i, c]` is the image-level score of class c in image i
  (mean of the top-k patch cosine similarities with the class text vector).
* Each present class of an image is one ground-truth *segment*.
  `seg_scores[s, c]` is the cosine similarity between the mean-pooled image
  features inside segment s and the text vector of class c (oracle-mask
  classification, as in the SAN / MaskFormer mask-classification head).
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import numpy as np


def key_name(layer: int, space: str) -> str:
    return f"{int(layer)}|{space}"


def parse_key(key: str):
    layer, space = key.split("|", 1)
    return int(layer), space


# --------------------------------------------------------------------------- #
# Basic statistics
# --------------------------------------------------------------------------- #
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC = P(pos > neg) + 0.5 P(pos == neg). NaN if a side is empty."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    neg = np.sort(neg)
    less = np.searchsorted(neg, pos, side="left")
    less_eq = np.searchsorted(neg, pos, side="right")
    return float((less + 0.5 * (less_eq - less)).sum() / (pos.size * neg.size))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Non-interpolated AP (same as sklearn's average_precision_score without ties handling)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / n_pos)


def cohen_d(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    pooled = np.sqrt(0.5 * (pos.var(ddof=1) + neg.var(ddof=1)))
    return float((pos.mean() - neg.mean()) / (pooled + 1e-12))


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


# --------------------------------------------------------------------------- #
# Raw per-image storage
# --------------------------------------------------------------------------- #
class RawStore:
    """Accumulates per-image probe outputs; can be saved, loaded and merged."""

    FIELDS = ("img_scores", "seg_scores", "loc_auroc", "loc_gap")

    def __init__(self, class_names: List[str], meta: Optional[dict] = None):
        self.class_names = list(class_names)
        self.meta = dict(meta or {})
        self.img_ids: List[str] = []
        self.presence: List[np.ndarray] = []      # (C,) int8 per image
        self.seg_label: List[int] = []
        self.seg_img: List[int] = []
        self.seg_area: List[float] = []
        self.data: Dict[str, Dict[str, list]] = {}  # key -> field -> list of per-image arrays

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_images(self) -> int:
        return len(self.img_ids)

    @property
    def keys(self) -> List[str]:
        return sorted(self.data.keys(), key=lambda k: (parse_key(k)[1], parse_key(k)[0]))

    def add_image(self, img_id: str, presence: np.ndarray, seg_labels: np.ndarray,
                  seg_areas: np.ndarray, results: Dict[str, Dict[str, np.ndarray]]):
        """`results[key]` holds img_scores (C,), seg_scores (S, C), loc_auroc (S,), loc_gap (S,)."""
        i = self.num_images
        C = self.num_classes
        S = int(len(seg_labels))
        self.img_ids.append(str(img_id))
        self.presence.append(np.asarray(presence, dtype=np.int8).reshape(C))
        self.seg_label.extend(int(x) for x in seg_labels)
        self.seg_img.extend([i] * S)
        self.seg_area.extend(float(a) for a in seg_areas)

        empty = {
            "img_scores": np.full((C,), np.nan, np.float32),
            "seg_scores": np.full((S, C), np.nan, np.float32),
            "loc_auroc": np.full((S,), np.nan, np.float32),
            "loc_gap": np.full((S,), np.nan, np.float32),
        }
        for key in set(self.data) | set(results):
            if key not in self.data:
                # a key seen for the first time: back-fill previous images with NaN
                self.data[key] = {f: [] for f in self.FIELDS}
                seg_counts = np.bincount(np.asarray(self.seg_img, np.int64), minlength=i + 1)
                for j in range(i):
                    s_j = int(seg_counts[j])
                    self.data[key]["img_scores"].append(np.full((C,), np.nan, np.float32))
                    self.data[key]["seg_scores"].append(np.full((s_j, C), np.nan, np.float32))
                    self.data[key]["loc_auroc"].append(np.full((s_j,), np.nan, np.float32))
                    self.data[key]["loc_gap"].append(np.full((s_j,), np.nan, np.float32))
            r = results.get(key, empty)
            for f in self.FIELDS:
                arr = np.asarray(r.get(f, empty[f]), dtype=np.float32)
                if arr.shape != empty[f].shape:
                    raise ValueError(f"{key}/{f}: expected shape {empty[f].shape}, got {arr.shape}")
                self.data[key][f].append(arr)

    # ------------------------------------------------------------------ arrays
    def arrays(self, key: str) -> Dict[str, np.ndarray]:
        C = self.num_classes
        d = self.data[key]
        cat = lambda lst, shape: np.concatenate(lst, 0) if lst else np.zeros(shape, np.float32)
        return {
            "img_scores": np.stack(d["img_scores"], 0) if d["img_scores"] else np.zeros((0, C), np.float32),
            "seg_scores": cat(d["seg_scores"], (0, C)),
            "loc_auroc": cat(d["loc_auroc"], (0,)),
            "loc_gap": cat(d["loc_gap"], (0,)),
        }

    def presence_array(self) -> np.ndarray:
        if not self.presence:
            return np.zeros((0, self.num_classes), np.int8)
        return np.stack(self.presence, 0)

    # -------------------------------------------------------------- save/load
    def save(self, path: str):
        out = {
            "class_names": np.array(self.class_names),
            "img_ids": np.array(self.img_ids),
            "presence": self.presence_array(),
            "seg_label": np.asarray(self.seg_label, np.int64),
            "seg_img": np.asarray(self.seg_img, np.int64),
            "seg_area": np.asarray(self.seg_area, np.float32),
            "meta": np.array(json.dumps(self.meta)),
            "keys": np.array(self.keys),
        }
        for key in self.keys:
            for f, arr in self.arrays(key).items():
                out[f"{f}::{key}"] = arr
        np.savez_compressed(path, **out)

    @classmethod
    def load(cls, path: str) -> "RawStore":
        z = np.load(path, allow_pickle=False)
        store = cls([str(c) for c in z["class_names"]], json.loads(str(z["meta"])))
        store.img_ids = [str(x) for x in z["img_ids"]]
        presence = z["presence"]
        store.presence = [presence[i] for i in range(presence.shape[0])]
        store.seg_label = z["seg_label"].tolist()
        store.seg_img = z["seg_img"].tolist()
        store.seg_area = z["seg_area"].tolist()
        seg_img = z["seg_img"]
        n = len(store.img_ids)
        bounds = np.searchsorted(seg_img, np.arange(n + 1), side="left")
        for key in [str(k) for k in z["keys"]]:
            d = {f: [] for f in cls.FIELDS}
            arrs = {f: z[f"{f}::{key}"] for f in cls.FIELDS}   # decompress once
            for i in range(n):
                a, b = bounds[i], bounds[i + 1]
                d["img_scores"].append(arrs["img_scores"][i])
                d["seg_scores"].append(arrs["seg_scores"][a:b])
                d["loc_auroc"].append(arrs["loc_auroc"][a:b])
                d["loc_gap"].append(arrs["loc_gap"][a:b])
            store.data[key] = d
        return store

    @classmethod
    def merge(cls, stores: Iterable["RawStore"]) -> "RawStore":
        stores = list(stores)
        if not stores:
            raise ValueError("nothing to merge")
        base = stores[0]
        merged = cls(base.class_names, dict(base.meta, merged_from=len(stores)))
        for st in stores:
            if st.class_names != base.class_names:
                raise ValueError("cannot merge stores with different class lists")
            bounds = np.searchsorted(np.asarray(st.seg_img, np.int64), np.arange(st.num_images + 1), side="left")
            for i in range(st.num_images):
                a, b = bounds[i], bounds[i + 1]
                results = {k: {f: st.data[k][f][i] for f in cls.FIELDS} for k in st.data}
                merged.add_image(st.img_ids[i], st.presence[i], st.seg_label[a:b], st.seg_area[a:b], results)
        return merged


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
HEADLINE_METRICS = [
    "maskcls_top1",
    "maskcls_top1_debiased",
    "maskcls_top1_among_present",
    "maskcls_margin",
    "presence_auroc",
    "presence_auroc_macro",
    "image_map",
    "loc_auroc",
]

METRIC_DESCRIPTIONS = {
    "maskcls_top1": "Oracle-mask classification accuracy: argmax over all classes of cos(mean-pooled segment features, class text vector).",
    "maskcls_top1_debiased": "Same, after z-scoring every class column over all segments (removes per-class offsets a learned head could absorb; transductive diagnostic).",
    "maskcls_top1_among_present": "Accuracy when only the classes present in the image are candidates (images with >= 2 classes).",
    "maskcls_top5": "Oracle-mask top-5 accuracy.",
    "maskcls_top1_macro": "Oracle-mask accuracy averaged over classes.",
    "maskcls_mean_rank": "Mean rank of the true class (1 = best).",
    "maskcls_margin": "Mean of (score of true class - best other class).",
    "maskcls_true_mean": "Mean segment score of the true class.",
    "maskcls_absent_mean": "Mean segment score of classes absent from the image.",
    "maskcls_other_present_mean": "Mean segment score of other classes present in the image.",
    "presence_auroc": "AUROC separating (image, class) pairs where the class is present vs absent, from image-level top-k scores (all classes pooled).",
    "presence_auroc_macro": "Presence AUROC computed per class, then averaged.",
    "presence_dprime": "Cohen's d between present and absent image-level scores.",
    "present_mean": "Mean image-level score of present classes.",
    "absent_mean": "Mean image-level score of absent classes.",
    "image_map": "Mean over images of the average precision of ranking the classes by image-level score.",
    "image_top1": "Fraction of images whose best-scoring class is present.",
    "loc_auroc": "Per segment, AUROC of the patch similarity map for its class: patches inside vs outside the object (background included in VOC format).",
    "loc_gap": "Mean (inside - outside) patch similarity per segment.",
}


def compute_key_metrics(store: RawStore, key: str) -> Dict[str, float]:
    a = store.arrays(key)
    P = store.presence_array()
    S = a["img_scores"]
    m: Dict[str, float] = {}

    # ---------------- image-level presence (present vs absent classes) --------
    finite = np.isfinite(S)
    pos = S[(P == 1) & finite]
    neg = S[(P == 0) & finite]
    m["presence_auroc"] = auroc(pos, neg)
    m["presence_auroc_macro"] = _nanmean([
        auroc(S[(P[:, c] == 1) & finite[:, c], c], S[(P[:, c] == 0) & finite[:, c], c])
        for c in range(store.num_classes)
    ])
    m["presence_dprime"] = cohen_d(pos, neg)
    m["present_mean"] = _nanmean(pos)
    m["absent_mean"] = _nanmean(neg)
    aps, top1 = [], []
    for i in range(S.shape[0]):
        valid = (P[i] >= 0) & finite[i]
        lab = P[i][valid] == 1
        if lab.sum() == 0 or lab.sum() == lab.size:
            continue
        aps.append(average_precision(S[i][valid], lab))
        top1.append(float(lab[np.argmax(S[i][valid])]))
    m["image_map"] = _nanmean(aps)
    m["image_top1"] = _nanmean(top1)

    # ---------------- oracle-mask classification ----------------------------
    seg = a["seg_scores"]
    y = np.asarray(store.seg_label, np.int64)
    seg_img = np.asarray(store.seg_img, np.int64)
    ok = np.isfinite(seg).all(axis=1)
    seg, y, seg_img = seg[ok], y[ok], seg_img[ok]
    if seg.shape[0]:
        n = seg.shape[0]
        true = seg[np.arange(n), y]
        rank = (seg > true[:, None]).sum(1) + 1
        others = seg.copy()
        others[np.arange(n), y] = -np.inf
        m["maskcls_top1"] = float((rank == 1).mean())
        m["maskcls_top5"] = float((rank <= 5).mean())
        m["maskcls_mean_rank"] = float(rank.mean())
        m["maskcls_margin"] = float((true - others.max(1)).mean())
        m["maskcls_top1_macro"] = _nanmean([(rank[y == c] == 1).mean() if (y == c).any() else np.nan
                                            for c in range(store.num_classes)])
        z = (seg - seg.mean(0, keepdims=True)) / (seg.std(0, keepdims=True) + 1e-8)
        m["maskcls_top1_debiased"] = float((z.argmax(1) == y).mean())

        present_of_seg = P[seg_img] == 1                     # (n, C)
        m["maskcls_true_mean"] = float(true.mean())
        absent_mask = P[seg_img] == 0
        m["maskcls_absent_mean"] = _nanmean(seg[absent_mask])
        other_present = present_of_seg.copy()
        other_present[np.arange(n), y] = False
        m["maskcls_other_present_mean"] = _nanmean(seg[other_present])
        multi = present_of_seg.sum(1) >= 2
        if multi.any():
            restricted = np.where(present_of_seg, seg, -np.inf)[multi]
            m["maskcls_top1_among_present"] = float((restricted.argmax(1) == y[multi]).mean())
        else:
            m["maskcls_top1_among_present"] = float("nan")
    else:
        for k in ("maskcls_top1", "maskcls_top5", "maskcls_mean_rank", "maskcls_margin", "maskcls_top1_macro",
                  "maskcls_top1_debiased", "maskcls_true_mean", "maskcls_absent_mean",
                  "maskcls_other_present_mean", "maskcls_top1_among_present"):
            m[k] = float("nan")

    # ---------------- localization -------------------------------------------
    m["loc_auroc"] = _nanmean(a["loc_auroc"])
    m["loc_gap"] = _nanmean(a["loc_gap"])
    m["num_images"] = float(store.num_images)
    m["num_segments"] = float(seg.shape[0])
    return m


def per_class_metrics(store: RawStore, key: str) -> Dict[str, Dict[str, float]]:
    a = store.arrays(key)
    P = store.presence_array()
    S = a["img_scores"]
    seg = a["seg_scores"]
    y = np.asarray(store.seg_label, np.int64)
    ok = np.isfinite(seg).all(axis=1)
    seg, y = seg[ok], y[ok]
    out = {}
    for c, name in enumerate(store.class_names):
        sel = y == c
        out[name] = {
            "presence_auroc": auroc(S[P[:, c] == 1, c], S[P[:, c] == 0, c]),
            "maskcls_top1": float((seg[sel].argmax(1) == c).mean()) if sel.any() else float("nan"),
            "num_images_present": int((P[:, c] == 1).sum()),
            "num_segments": int(sel.sum()),
        }
    return out


def confusion_matrix(store: RawStore, key: str) -> np.ndarray:
    seg = store.arrays(key)["seg_scores"]
    y = np.asarray(store.seg_label, np.int64)
    ok = np.isfinite(seg).all(axis=1)
    C = store.num_classes
    cm = np.zeros((C, C), np.int64)
    np.add.at(cm, (y[ok], seg[ok].argmax(1)), 1)
    return cm


def compute_all_metrics(store: RawStore) -> Dict[str, Dict[str, float]]:
    return {key: compute_key_metrics(store, key) for key in store.keys}
