"""Class names used for CLIP text prompts.

Each class gets a list of synonyms; the text embedding is the (template- and
synonym-) averaged CLIP embedding. The raw detectron2 names are kept for
bookkeeping; only prompts use these cleaned names.

Override for a dataset with `data.class_names.<dataset>: path/to/names.json`
(a CAT-Seg-style JSON list, one entry per class in GT-id order; entries may be
comma-separated synonym lists).
"""
import json
import logging
import os
from typing import Dict, List, Optional

log = logging.getLogger("mask_cls")

# COCO-Stuff raw name -> prompt name. Names not listed are used as they are.
COCO_STUFF_PROMPT_NAMES = {
    "tv": "television",
    "hair drier": "hair dryer",
    "building-other": "building",
    "ceiling-other": "ceiling",
    "ceiling-tile": "tiled ceiling",
    "desk-stuff": "desk",
    "door-stuff": "door",
    "floor-marble": "marble floor",
    "floor-other": "floor",
    "floor-stone": "stone floor",
    "floor-tile": "tiled floor",
    "floor-wood": "wooden floor",
    "food-other": "food",
    "furniture-other": "furniture",
    "ground-other": "ground",
    "mirror-stuff": "mirror",
    "plant-other": "plant",
    "playingfield": "playing field",
    "sky-other": "sky",
    "solid-other": "solid",
    "structural-other": "structure",
    "textile-other": "textile",
    "wall-brick": "brick wall",
    "wall-concrete": "concrete wall",
    "wall-other": "wall",
    "wall-panel": "wall panel",
    "wall-stone": "stone wall",
    "wall-tile": "tiled wall",
    "wall-wood": "wooden wall",
    "water-other": "water",
    "waterdrops": "water drops",
    "window-blind": "window blind",
    "window-other": "window",
}

VOC_PROMPT_NAMES = {
    "aeroplane": "aeroplane",
    "diningtable": "dining table",
    "motorbike": "motorbike",
    "pottedplant": "potted plant",
    "tvmonitor": "tv monitor",
    "tv": "tv monitor",
}


def _split_synonyms(name: str) -> List[str]:
    return [s.strip() for s in name.split(",") if s.strip()]


def prompt_names(dataset: str, raw_names: List[str], override_file: Optional[str] = None) -> List[List[str]]:
    """List (per class) of synonym lists used for prompting."""
    if override_file:
        if not os.path.isfile(override_file):
            raise FileNotFoundError(override_file)
        with open(override_file) as f:
            names = [n for n in json.load(f) if n != "unknown"]
        if len(names) != len(raw_names):
            raise ValueError(f"{override_file}: {len(names)} names, dataset {dataset} has {len(raw_names)}")
        return [_split_synonyms(n) for n in names]
    if dataset.startswith("coco"):
        return [[COCO_STUFF_PROMPT_NAMES.get(n, n)] for n in raw_names]
    if dataset.startswith("voc"):
        return [[VOC_PROMPT_NAMES.get(n, n)] for n in raw_names]
    # ADE-150 names are single words; ADE-847 and Pascal Context-459 are synonym lists.
    return [_split_synonyms(n) for n in raw_names]
