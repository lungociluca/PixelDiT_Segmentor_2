"""Cache mask-pooled PixelDiT region features (fp16 shards). Resumable.

    python tools/mask_cls/extract_student.py [--roles fit holdout ade20k_sem_seg_val ...] [opts ...]
"""
import json
import logging
import os
import time

import torch

import _common

log = logging.getLogger("mask_cls")


def main():
    args, cfg = _common.parse(__doc__, lambda ap: ap.add_argument("--roles", nargs="*", default=None))
    from mask_cls import pipeline as P
    from mask_cls.data import image_loader, index_path
    from mask_cls.features import PixelDiTFeatureExtractor, mask_area_weights, mask_pool, preprocess_image
    from mask_cls.utils import ShardWriter, is_complete, resolve_dtype

    s = cfg.student
    roles = P.index_roles(cfg)
    todo = []
    for role, (key, _) in roles.items():
        if args.roles and role not in args.roles:
            continue
        if not os.path.isfile(index_path(cfg.paths.cache_root, key)):
            log.warning("%s: no index, skipping (run build_index.py)", role)
            continue
        if is_complete(P.student_dir(cfg, key)):
            log.info("%s: student features already cached", role)
            continue
        todo.append((role, key))
    if not todo:
        return

    captions = None
    if s.get("captions_file"):
        with open(s.captions_file) as f:
            captions = json.load(f)
    ext = PixelDiTFeatureExtractor(
        checkpoint=s.checkpoint, model_config=s.model_config, num_blocks=s.num_blocks, layers=s.layers,
        timestep=s.timestep, noise=s.noise, noise_seed=cfg.seed, prompt=s.prompt, captions=captions,
        mask_text_padding=s.mask_text_padding, pixel_pathway=s.pixel_pathway, dtype=resolve_dtype(s.dtype),
        cache_dir=P.cache(cfg, "prompts"),
    )
    for role, key in todo:
        index = P.load_index(cfg, key)
        out_dir = P.student_dir(cfg, key)
        cache_dtype = resolve_dtype(s.get("cache_dtype", "float16"))
        writer = ShardWriter(out_dir, meta={"student": cfg.student.to_dict(), "index": key}, dtype=cache_dtype)
        done = writer.done_rows()
        loader = image_loader(index, cfg.data.num_workers, skip_rows=done)
        log.info("%s: %d images to process -> %s", role, len(loader), out_dir)
        t0 = time.time()
        for it, item in enumerate(loader):
            img = item["image"].cuda(non_blocking=True)
            lab = item["label"].cuda(non_blocking=True)
            x = preprocess_image(img, s.target_area, s.max_side, ext.patch_size)
            feats = ext(x, keys=[index.images[item["image_idx"]][0]], seeds=[item["image_idx"]])
            masks = lab[None] == item["class_id"].cuda()[:, None, None]
            w = mask_area_weights(masks, feats["hw"])
            out = {"patch": mask_pool(feats["patch"][0], w)}                    # [n, L, 1536]
            if s.pixel_pathway:
                out["pit_attn"] = mask_pool(feats["pit_attn"][0], w)            # [n, 2, 1152]
                wf = mask_area_weights(masks, x.shape[-2:])
                out["pixel"] = mask_pool(feats["pixel"][0], wf)                 # [n, 2, 16]
            for k, v in out.items():
                if not torch.isfinite(v).all():
                    raise RuntimeError(f"Non-finite {k} features for image {index.images[item['image_idx']][0]}")
                if cache_dtype == torch.float16 and v.abs().max() > 6.0e4:
                    raise RuntimeError(f"{k} features exceed the fp16 range (max {float(v.abs().max()):.0f}); "
                                       "re-run with student.cache_dtype=bfloat16")
            writer.add(item["rows"], **out)
            if (it + 1) % 500 == 0:
                rate = (it + 1) / (time.time() - t0)
                log.info("%s: %d/%d images (%.1f img/s, ETA %.0f min)", role, it + 1, len(loader), rate,
                         (len(loader) - it - 1) / rate / 60)
        writer.close()
        log.info("%s: done in %.1f min", role, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
