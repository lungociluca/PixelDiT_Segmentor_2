"""Seen / unseen class splits relative to the COCO-Stuff training vocabulary.

A target class is SEEN if any of its synonyms, after normalisation (lower case,
'-'/'_' -> space, whitespace collapsed), equals
  * a COCO-Stuff raw name or its prompt name (e.g. 'wall-other' / 'wall'), or
  * a key of the manual synonym table (configs/mask_cls/class_splits/synonyms.yaml)
    that maps to a COCO-Stuff class.
Everything else is UNSEEN. `tools/mask_cls/build_splits.py` writes the result,
with the reason for every class, to configs/mask_cls/class_splits/<dataset>.json
so the matching is documented and reviewable.
"""
import json
import os
import re
from typing import Dict, List, Optional

import yaml

from .class_names import COCO_STUFF_PROMPT_NAMES, prompt_names


def normalize(name: str) -> str:
    n = name.lower().strip().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", n)


def load_synonyms(path: str) -> Dict[str, str]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    return {normalize(k): v for k, v in (d.get("synonyms") or {}).items()}


def compute_split(
    dataset: str,
    raw_names: List[str],
    reference_raw_names: List[str],
    synonyms: Dict[str, str],
    override_file: Optional[str] = None,
) -> dict:
    ref = {}
    for n in reference_raw_names:
        ref[normalize(n)] = n
        ref[normalize(COCO_STUFF_PROMPT_NAMES.get(n, n))] = n
    for k, v in synonyms.items():
        if v not in reference_raw_names:
            raise ValueError(f"synonyms.yaml maps {k!r} to {v!r}, which is not a COCO-Stuff class")

    names = prompt_names(dataset, raw_names, override_file)
    # Raw names may carry synonyms that the prompt names dropped (or vice versa).
    classes = []
    for i, (raw, syns) in enumerate(zip(raw_names, names)):
        cands = list(dict.fromkeys([normalize(s) for s in syns] + [normalize(s) for s in raw.split(",")]))
        match, how = None, None
        for c in cands:
            if c in ref:
                match, how = ref[c], f"exact:{c}"
                break
        if match is None:
            for c in cands:
                if c in synonyms:
                    match, how = synonyms[c], f"synonym:{c}"
                    break
        classes.append({"id": i, "name": raw, "seen": match is not None, "coco_match": match, "rule": how})
    return {
        "dataset": dataset,
        "reference": "coco_stuff_171",
        "num_seen": sum(c["seen"] for c in classes),
        "num_unseen": sum(not c["seen"] for c in classes),
        "seen": [c["name"] for c in classes if c["seen"]],
        "unseen": [c["name"] for c in classes if not c["seen"]],
        "classes": classes,
    }


def seen_mask(split: dict) -> List[bool]:
    return [c["seen"] for c in split["classes"]]


def save_split(split: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split['dataset']}.json")
    with open(path, "w") as f:
        json.dump(split, f, indent=1)
    return path
