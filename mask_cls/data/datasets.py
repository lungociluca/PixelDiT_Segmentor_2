"""Access to the detectron2-registered semantic-segmentation datasets.

Uses the registrations already in the repo (san/data/datasets/*) plus detectron2's
built-in ones (ADE20K-150 = `ade20k_sem_seg_val`). Expected layout under the
datasets root (DETECTRON2_DATASETS), as produced by the repo's prepare_* scripts:

  coco/{train2017,val2017}, coco/stuffthingmaps_detectron2/{train2017,val2017}
  ADEChallengeData2016/images/validation, ADEChallengeData2016/annotations_detectron2/validation
  ADE20K_2021_17_01/{images_detectron2,annotations_detectron2}/validation   (ADE-847, 16-bit tif)
  VOC2012/JPEGImages, VOC2012/annotations_detectron2/val
"""
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger("mask_cls")

_REGISTERED = False


def register_datasets(datasets_root: Optional[str] = None):
    """Register all datasets. Must run before anything imports detectron2.data,
    because detectron2 reads DETECTRON2_DATASETS once, at import time."""
    global _REGISTERED
    if _REGISTERED:
        return
    if datasets_root:
        if "detectron2.data" in sys.modules and os.environ.get("DETECTRON2_DATASETS") != datasets_root:
            log.warning(
                "detectron2.data was imported before DETECTRON2_DATASETS was set to %s; "
                "built-in datasets (ADE-150) may point to the wrong root.", datasets_root,
            )
        os.environ["DETECTRON2_DATASETS"] = datasets_root
    import detectron2.data.datasets  # noqa: F401  (built-in registrations, e.g. ADE-150)
    import san.data.datasets  # noqa: F401  (COCO-Stuff, ADE-847, VOC, Pascal Context)

    _REGISTERED = True


@dataclass
class DatasetInfo:
    name: str
    images: List[Tuple[str, str]]  # (image path, GT label-map path), sorted by image path
    class_names: List[str]         # raw names, index = contiguous GT id
    ignore_label: int


def get_dataset(name: str) -> DatasetInfo:
    from detectron2.data import DatasetCatalog, MetadataCatalog

    meta = MetadataCatalog.get(name)
    classes = getattr(meta, "stuff_classes", None) or meta.thing_classes
    dicts = DatasetCatalog.get(name)
    images = sorted((d["file_name"], d["sem_seg_file_name"]) for d in dicts)
    if not images:
        raise FileNotFoundError(f"Dataset {name} has no images (check {getattr(meta, 'image_root', '?')})")
    return DatasetInfo(name, images, [c.strip() for c in classes], int(getattr(meta, "ignore_label", 255)))


def dataset_available(name: str) -> bool:
    from detectron2.data import MetadataCatalog

    try:
        meta = MetadataCatalog.get(name)
        root = getattr(meta, "sem_seg_root", None)
        return root is not None and os.path.isdir(root) and len(os.listdir(root)) > 0
    except Exception:
        return False


def read_label_map(path: str) -> np.ndarray:
    """GT label map as int64 (handles 8-bit PNG and ADE-847's 16-bit TIFF)."""
    with Image.open(path) as im:
        return np.asarray(im).astype(np.int64)


def read_image(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
