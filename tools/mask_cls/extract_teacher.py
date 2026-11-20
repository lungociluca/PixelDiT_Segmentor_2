"""Cache CLIP teacher region targets (crop and/or dense) and class text embeddings. Resumable.

    python tools/mask_cls/extract_teacher.py [--kinds crop dense] [--roles ...] [opts ...]
    e.g. teacher.model=ViT-L-14 teacher.pretrained=openai teacher.dense.mode=sclip
"""
import logging
import os
import time

import torch

import _common

log = logging.getLogger("mask_cls")


def main():
    def extra(ap):
        ap.add_argument("--kinds", nargs="*", default=["crop", "dense"], choices=["crop", "dense"])
        ap.add_argument("--roles", nargs="*", default=None)

    args, cfg = _common.parse(__doc__, extra)
    from mask_cls import pipeline as P
    from mask_cls.data import image_loader, index_path
    from mask_cls.teacher import ClipTeacher, crop_targets, dense_targets
    from mask_cls.utils import ShardWriter, is_complete, resolve_dtype

    t = cfg.teacher
    teacher = ClipTeacher(t.model, t.pretrained, dtype=resolve_dtype(t.dtype))
    if "dense" in args.kinds:
        teacher.sanity_check()
    dino = None
    if "dense" in args.kinds and t.dense.mode == "corrclip" and t.dense.corrclip.similarity == "dino":
        from mask_cls.teacher.corrclip import DinoSimilarity

        dino = DinoSimilarity(t.dense.corrclip.dino_model)

    for role, (key, name) in P.index_roles(cfg).items():
        if args.roles and role not in args.roles:
            continue
        if not os.path.isfile(index_path(cfg.paths.cache_root, key)):
            log.warning("%s: no index, skipping", role)
            continue
        index = P.load_index(cfg, key)
        P.text_embeddings(cfg, name, index.class_names, teacher)   # cache text embeddings too
        writers = {}
        for kind in args.kinds:
            d = P.teacher_dir(cfg, kind, key)
            if not is_complete(d):
                writers[kind] = ShardWriter(d, meta={"teacher": t.to_dict(), "kind": kind, "index": key})
        if not writers:
            log.info("%s: teacher targets already cached", role)
            continue
        dones = {k: w.done_rows() for k, w in writers.items()}
        skip = set.intersection(*dones.values()) if dones else set()
        loader = image_loader(index, cfg.data.num_workers, skip_rows=skip)
        log.info("%s: %d images, kinds %s", role, len(loader), list(writers))
        t0 = time.time()
        for it, item in enumerate(loader):
            img = item["image"].cuda(non_blocking=True).float() / 255.0
            masks = item["label"].cuda(non_blocking=True)[None] == item["class_id"].cuda()[:, None, None]
            first = int(item["rows"][0])
            if "crop" in writers and first not in dones["crop"]:
                c = t.crop
                e = crop_targets(teacher, img, masks, item["bbox"], margin=c.margin, background=c.background,
                                 blur_sigma=c.blur_sigma, square=c.square, batch_size=c.batch_size)
                writers["crop"].add(item["rows"], crop=e)
            if "dense" in writers and first not in dones["dense"]:
                dd = t.dense
                e = dense_targets(teacher, img, masks, mode=dd.mode, short_side=dd.short_side, token_norm=dd.token_norm,
                                  corrclip=dd.get("corrclip"), dino=dino)
                writers["dense"].add(item["rows"], dense=e)
            if (it + 1) % 500 == 0:
                rate = (it + 1) / (time.time() - t0)
                log.info("%s: %d/%d images (%.1f img/s, ETA %.0f min)", role, it + 1, len(loader), rate,
                         (len(loader) - it - 1) / rate / 60)
        for w in writers.values():
            w.close()


if __name__ == "__main__":
    main()
