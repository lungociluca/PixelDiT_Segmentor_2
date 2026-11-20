"""Shared setup for the tools/mask_cls scripts (run them from the repository root)."""
import argparse
import logging
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)  # the PixelDiT code uses repo-relative paths


def parse(description: str, extra=None):
    from mask_cls.config import add_common_args, load_config
    from mask_cls.utils import set_seed, setup_logging

    ap = argparse.ArgumentParser(description=description)
    if extra:
        extra(ap)
    add_common_args(ap)
    args = ap.parse_args()
    # `--roles a b key=value` makes argparse eat the overrides; move them back.
    for k, v in list(vars(args).items()):
        if k != "opts" and isinstance(v, list) and any(isinstance(x, str) and "=" in x for x in v):
            args.opts = list(args.opts) + [x for x in v if "=" in x]
            setattr(args, k, [x for x in v if "=" not in x])
    setup_logging()
    cfg = load_config(args.config, args.opts)
    root = cfg.paths.datasets_root
    if not root or root.startswith("$"):
        root = None  # fall back to $DETECTRON2_DATASETS / ./datasets
    from mask_cls.data.datasets import register_datasets

    register_datasets(root)
    set_seed(cfg.seed)
    logging.getLogger("mask_cls").info("datasets root: %s", os.environ.get("DETECTRON2_DATASETS", "datasets"))
    return args, cfg
