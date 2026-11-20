"""Build region indexes (one row per GT region) for the COCO fit/holdout pools and the eval sets.

    python tools/mask_cls/build_index.py [--force] [opts ...]
"""
import logging
import os

import _common

log = logging.getLogger("mask_cls")


def main():
    args, cfg = _common.parse(__doc__, lambda ap: ap.add_argument("--force", action="store_true"))
    from mask_cls import pipeline as P
    from mask_cls.data import build_index, dataset_available, get_dataset, index_path, train_image_split

    for role, (key, name) in P.index_roles(cfg).items():
        path = index_path(cfg.paths.cache_root, key)
        if os.path.isfile(path) and not args.force:
            log.info("%s: index exists (%s)", role, path)
            continue
        if not dataset_available(name):
            log.warning("%s: dataset %s not found under the datasets root, skipping", role, name)
            continue
        info = get_dataset(name)
        if role in ("fit", "holdout"):
            t = cfg.data.train
            fit_ids, hold_ids = train_image_split(len(info.images), t.holdout_images, t.max_images, cfg.seed)
            ids = fit_ids if role == "fit" else hold_ids
            index = build_index(info, min_area=t.min_area, image_subset=ids, num_workers=cfg.data.num_workers,
                                meta={"role": role, "seed": cfg.seed, "min_area": t.min_area})
        else:
            index = build_index(info, min_area=0, num_workers=cfg.data.num_workers, meta={"role": "eval"})
        index.save(path)
        log.info("%s: %d images, %d regions, %d classes -> %s",
                 role, index.num_images, len(index), len(index.class_names), path)


if __name__ == "__main__":
    main()
