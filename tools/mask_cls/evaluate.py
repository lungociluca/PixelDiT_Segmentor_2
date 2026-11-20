"""Evaluate CLIP-teacher baselines or saved alignment maps on the eval datasets.

    python tools/mask_cls/evaluate.py --teacher crop dense avg            # teacher upper bounds
    python tools/mask_cls/evaluate.py --maps cache/mask_cls/maps/<run>.pt  # re-evaluate saved maps
"""
import logging

import _common

log = logging.getLogger("mask_cls")


def main():
    def extra(ap):
        ap.add_argument("--teacher", nargs="*", default=[], choices=["crop", "dense", "avg"])
        ap.add_argument("--maps", nargs="*", default=[])
        ap.add_argument("--datasets", nargs="*", default=None)

    args, cfg = _common.parse(__doc__, extra)
    from mask_cls import pipeline as P
    from mask_cls.align import load_map, predict
    from mask_cls.align.scoring import apply_scoring, scoring_stats
    from mask_cls.eval.runner import evaluate_all

    for target in args.teacher:
        name = f"teacher_{target}_{P.clip_key(cfg)}"
        if target != "crop":
            name += f"_{cfg.teacher.dense.mode}"
            if cfg.teacher.dense.mode == "corrclip":
                name += f"_{P.dense_key(cfg).rsplit('-', 1)[-1]}"   # one name per CorrCLIP variant
        # "center" for the teacher too, so a centred student is compared with a centred teacher.
        y_mean = P.load_teacher(cfg, P.train_index_keys(cfg)[0], target).float().mean(0)
        tstats = {"y_mean": y_mean, "p_mean": y_mean, "k": 1.0}
        for mode in [m for m in cfg.eval.student_scoring if m in ("raw", "center")]:
            info = {"run_name": name if mode == "raw" else f"{name}__{mode}", "kind": "teacher", "scoring": mode,
                    "method": "clip", "target": target, "features": "-", "clip": P.clip_key(cfg),
                    "dense_mode": cfg.teacher.dense.mode if target != "crop" else None,
                    "templates": cfg.text.templates, "crop_key": P.crop_key(cfg), "dense_key": P.dense_key(cfg)}
            evaluate_all(cfg, lambda key, t=target, md=mode: apply_scoring(
                P.load_teacher(cfg, key, t), md, tstats), info, args.datasets)

    for path in args.maps:
        mp = load_map(path)
        mcfg = mp.meta.get("config", {})
        if mp.meta.get("student_key") != P.student_key(cfg):
            log.warning("%s was fitted on student features %s, but the current config points to %s. "
                        "Pass the same student.* overrides as at fit time.", path, mp.meta.get("student_key"),
                        P.student_key(cfg))
        stats = mp.meta.get("scoring_stats")
        if stats is None:  # maps saved before scoring modes existed
            fit_key, _ = P.train_index_keys(cfg)
            target = mcfg.get("align", {}).get("target", cfg.align.target)
            stats = scoring_stats(mp, P.load_student(cfg, fit_key, mp.feature_spec), P.load_teacher(cfg, fit_key, target))
        base = path.split("/")[-1][:-3]
        for mode in cfg.eval.student_scoring:
            info = {"run_name": f"{base}__{mode}", "kind": "student", "scoring": mode,
                    "method": mcfg.get("align", {}).get("method"), "features": "+".join(mp.feature_spec),
                    "student_key": P.student_key(cfg), "clip": P.clip_key(cfg), "map_path": path,
                    "templates": cfg.text.templates}
            evaluate_all(cfg, lambda key, m=mp, md=mode: apply_scoring(
                predict(m, P.load_student(cfg, key, m.feature_spec)), md, stats), info, args.datasets)


if __name__ == "__main__":
    main()
