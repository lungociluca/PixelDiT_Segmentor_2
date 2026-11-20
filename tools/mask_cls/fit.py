"""Fit an alignment map on cached COCO region features, save it, and evaluate it.

    python tools/mask_cls/fit.py align.method=ridge align.target=avg align.features=[patch.3]
    python tools/mask_cls/fit.py align.method=linear align.sgd.ce_weight=0.1           # + aux CE
    python tools/mask_cls/fit.py align.method=linear align.sgd.distill_weight=0 \
        align.sgd.ce_weight=1 align.sgd.init=random                                     # labels-only baseline
"""
import logging
import os

import torch

import _common

log = logging.getLogger("mask_cls")


def run_name(cfg, P):
    from mask_cls.config import stable_hash

    a = cfg.align
    if a.name:
        return a.name
    sg = a.sgd
    loss = ""
    if a.method in ("linear", "mlp"):
        loss = f"_d{sg.distill_weight:g}_ce{sg.ce_weight:g}"
    tgt = a.target if (a.method in ("ridge", "procrustes") or sg.distill_weight > 0) else "labels"
    n = "all" if a.train_images is None else str(a.train_images)
    return (f"{a.method}{loss}_{tgt}_{'+'.join(a.features)}_n{n}_{P.clip_key(cfg)}"
            f"_{P.student_key(cfg)}_{stable_hash([a, cfg.teacher, cfg.text])}")


def main():
    args, cfg = _common.parse(__doc__, lambda ap: ap.add_argument("--no-eval", action="store_true"))
    torch.backends.cuda.matmul.allow_tf32 = False
    from mask_cls import pipeline as P
    from mask_cls.align import fit_procrustes, fit_ridge, fit_sgd, predict
    from mask_cls.align.scoring import apply_scoring, diagnostics, scoring_stats
    from mask_cls.eval.runner import evaluate_all

    a, sg = cfg.align, cfg.align.sgd
    fit_key, hold_key = P.train_index_keys(cfg)
    fit_index, hold_index = P.load_index(cfg, fit_key), P.load_index(cfg, hold_key)
    rows = fit_index.rows_of_first_images(a.train_images)
    X = P.load_student(cfg, fit_key, a.features, rows)
    Xh = P.load_student(cfg, hold_key, a.features)
    labels, labels_h = fit_index.class_id[rows], hold_index.class_id
    log.info("fit: %d regions from %s images, D=%d; holdout: %d regions",
             len(X), a.train_images or fit_index.num_images, X.shape[1], len(Xh))

    needs_teacher = a.method in ("ridge", "procrustes") or sg.distill_weight > 0
    Y = Yh = None
    if needs_teacher:
        Y = P.load_teacher(cfg, fit_key, a.target, rows)
        Yh = P.load_teacher(cfg, hold_key, a.target)
    text = P.text_embeddings(cfg, cfg.data.train.name, fit_index.class_names)

    history = {}
    if a.method == "ridge":
        mp, alpha, table, _ = fit_ridge(X, Y, Xh, Yh, a.features, a.ridge.alphas)
        history = {"best_alpha": alpha, "path": table}
    elif a.method == "procrustes":
        mp, cos = fit_procrustes(X, Y, Xh, Yh, a.features)
        history = {"holdout_cos": cos}
    elif a.method in ("linear", "mlp"):
        init = None
        if sg.init == "ridge" and needs_teacher:
            init, alpha, table, _ = fit_ridge(X, Y, Xh, Yh, a.features, a.ridge.alphas)
            history["ridge_init_alpha"] = alpha
        mp, hist = fit_sgd(
            X, Y, labels, Xh, Yh, labels_h, a.features, a.method, text,
            epochs=sg.epochs, batch_size=sg.batch_size, lr=sg.lr, weight_decay=sg.weight_decay,
            distill_weight=sg.distill_weight, ce_weight=sg.ce_weight, ce_temperature=sg.ce_temperature,
            mlp_hidden=sg.mlp_hidden, mlp_dropout=sg.mlp_dropout, init_map=init, seed=cfg.seed,
            d_out=text.shape[1],
        )
        history["epochs"] = hist
    else:
        raise ValueError(a.method)

    # Scoring statistics (teacher mean, residual scale) on the fit set. The labels-only
    # baseline does not train on teacher targets, but uses them here for diagnostics.
    if Y is None:
        try:
            Y, Yh = P.load_teacher(cfg, fit_key, a.target, rows), P.load_teacher(cfg, hold_key, a.target)
        except FileNotFoundError:
            pass
    stats = scoring_stats(mp, X, Y) if Y is not None else None

    name = run_name(cfg, P)
    os.makedirs(P.cache(cfg, "maps"), exist_ok=True)
    path = P.cache(cfg, "maps", f"{name}.pt")
    mp.save(path, meta={"config": cfg.to_dict(), "history": history, "student_key": P.student_key(cfg),
                        "scoring_stats": stats})
    log.info("saved %s", path)

    # Holdout diagnostics per scoring mode (COCO 171 classes; seen recognition and collapse).
    Ph = predict(mp, Xh)
    modes = [m for m in cfg.eval.student_scoring if m == "raw" or stats is not None]
    diags = {}
    for mode in modes:
        diags[mode] = diagnostics(apply_scoring(Ph, mode, stats), Yh, labels_h, text, stats)
        log.info("holdout [%s] %s", mode, {k: round(v, 4) for k, v in diags[mode].items()})
    coco_acc = diags["raw"]["coco_top1"]

    if args.no_eval:
        return
    info = {
        "run_name": name, "kind": "student", "method": a.method, "target": a.target if needs_teacher else "labels",
        "features": "+".join(a.features), "train_images": a.train_images, "ce_weight": sg.ce_weight
        if a.method in ("linear", "mlp") else 0.0, "distill_weight": sg.distill_weight if a.method in ("linear", "mlp") else 1.0,
        "student_key": P.student_key(cfg), "clip": P.clip_key(cfg), "dense_mode": cfg.teacher.dense.mode,
        "timestep": cfg.student.timestep, "noise": cfg.student.noise, "prompt": cfg.student.prompt,
        "templates": cfg.text.templates, "holdout_coco_top1": coco_acc, "map_path": path,
        "ridge_alpha": history.get("best_alpha", history.get("ridge_init_alpha")),
    }
    cache_pred = {}

    def embed(key, mode):
        if key not in cache_pred:
            cache_pred[key] = predict(mp, P.load_student(cfg, key, a.features))
        return apply_scoring(cache_pred[key], mode, stats)

    for mode in modes:
        row = {**info, "run_name": f"{name}__{mode}", "scoring": mode,
               **{f"holdout_{k}": v for k, v in diags[mode].items()}}
        evaluate_all(cfg, lambda key, m=mode: embed(key, m), row)


if __name__ == "__main__":
    main()
