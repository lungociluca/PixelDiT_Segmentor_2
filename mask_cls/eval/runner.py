"""Evaluate region embeddings (teacher or student->map) on the eval datasets."""
import logging
from typing import Callable, Dict, Optional

import torch

from ..config import Cfg, stable_hash
from ..utils import append_result
from .metrics import classify, region_metrics

log = logging.getLogger("mask_cls")


def evaluate_dataset(cfg: Cfg, dataset: str, embed_fn: Callable[[str], torch.Tensor], device="cuda") -> Dict:
    """embed_fn(index_key) -> [N, D] region embeddings in index row order."""
    from .. import pipeline as P

    index = P.load_index(cfg, dataset)
    split = P.class_split(cfg, dataset, index.class_names)
    text = P.text_embeddings(cfg, dataset, index.class_names)
    emb = embed_fn(dataset)
    if len(emb) != len(index):
        raise RuntimeError(f"{dataset}: {len(emb)} embeddings for {len(index)} regions")
    scores = classify(emb, text, device)
    m = region_metrics(scores, index.class_id, index.area, [c["seen"] for c in split["classes"]],
                       topk=cfg.eval.topk)
    m["num_classes"] = len(index.class_names)
    m["num_unseen_classes"] = split["num_unseen"]
    return m


def evaluate_all(cfg: Cfg, embed_fn, row_info: dict, datasets=None, device="cuda") -> Dict[str, Dict]:
    from .. import pipeline as P
    from ..data.regions import index_path
    import os

    out = {}
    for d in datasets or cfg.data.eval:
        if not os.path.isfile(index_path(cfg.paths.cache_root, d)):
            log.warning("Skipping %s: no region index (dataset not prepared?)", d)
            continue
        try:
            m = evaluate_dataset(cfg, d, embed_fn, device)
        except FileNotFoundError as e:
            log.warning("Skipping %s: %s", d, e)
            continue
        out[d] = m
        row = {**row_info, "dataset": d, **m}
        append_result(cfg.paths.results_dir, row)
        fmt = lambda v: "-" if v is None else f"{v:.1f}"
        log.info("%-28s %-30s top1 %s (seen %s / unseen %s)  mIoU %s (seen %s / unseen %s)",
                 row_info.get("run_name", ""), d, fmt(m["top1"]), fmt(m["top1_seen"]), fmt(m["top1_unseen"]),
                 fmt(m["miou"]), fmt(m["miou_seen"]), fmt(m["miou_unseen"]))
    return out
