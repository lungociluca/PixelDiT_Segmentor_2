"""Write the documented seen/unseen class splits (configs/mask_cls/class_splits/<dataset>.json).

    python tools/mask_cls/build_splits.py [opts ...]
"""
import logging

import _common

log = logging.getLogger("mask_cls")


def main():
    args, cfg = _common.parse(__doc__)
    from detectron2.data import MetadataCatalog

    from mask_cls import pipeline as P

    for d in cfg.data.eval:
        meta = MetadataCatalog.get(d)
        names = [c.strip() for c in (getattr(meta, "stuff_classes", None) or meta.thing_classes)]
        s = P.class_split(cfg, d, names)
        log.info("%-28s %4d classes: %4d seen, %4d unseen", d, len(names), s["num_seen"], s["num_unseen"])
        syn = [c for c in s["classes"] if c["rule"] and c["rule"].startswith("synonym")]
        if syn:
            log.info("   via synonyms: %s", ", ".join(f"{c['name']}->{c['coco_match']}" for c in syn[:40]))
        log.info("   unseen (first 30): %s", ", ".join(s["unseen"][:30]))


if __name__ == "__main__":
    main()
