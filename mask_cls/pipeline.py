"""Shared glue for the tools/mask_cls scripts: cache keys and loaders.

Cache layout (under paths.cache_root):
  index/<index_key>.pt                               region index (row order shared by all caches)
  student/<student_key>/<index_key>/shard_*.pt        {"rows", "patch": [n, L, 1536], ("pit_attn", "pixel")}
  teacher/<clip_key>/crop-<hash>/<index_key>/...      {"rows", "crop": [n, D]}
  teacher/<clip_key>/dense-<mode>-<hash>/<index_key>/ {"rows", "dense": [n, D]}
  text/<clip_key>/<dataset>_<hash>.pt                 [C, D] class text embeddings
  prompts/prompt_<hash>.pt                            PixelDiT conditioning embedding
  maps/<run_name>.pt                                  fitted alignment maps
"""
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch

from .config import Cfg, stable_hash
from .utils import load_shards

log = logging.getLogger("mask_cls")


# --------------------------------------------------------------------------- keys
def train_index_keys(cfg: Cfg) -> Tuple[str, str]:
    t = cfg.data.train
    n = "all" if t.max_images is None else str(t.max_images)
    base = f"{t.name}__s{cfg.seed}_a{t.min_area}"
    return f"{base}_fit{n}_h{t.holdout_images}", f"{base}_holdout{t.holdout_images}"


def index_roles(cfg: Cfg) -> Dict[str, Tuple[str, str]]:
    """role -> (index_key, dataset name). Roles: fit, holdout, and each eval dataset."""
    fit, hold = train_index_keys(cfg)
    roles = {"fit": (fit, cfg.data.train.name), "holdout": (hold, cfg.data.train.name)}
    for d in cfg.data.eval:
        roles[d] = (d, d)
    return roles


def student_key(cfg: Cfg) -> str:
    s = cfg.student
    prompt_tag = "caption" if s.get("captions_file") else ("empty" if s.prompt == "" else "fixed")
    noise_tag = "noisy" if s.noise else "clean"
    return (f"pixeldit_b{s.num_blocks}_t{s.timestep}_{noise_tag}_{prompt_tag}_a{s.target_area}"
            f"{'_pix' if s.pixel_pathway else ''}_{stable_hash(s)}")


def clip_key(cfg: Cfg) -> str:
    t = cfg.teacher
    return f"{t.model}__{t.pretrained}".replace("/", "-")


def crop_key(cfg: Cfg) -> str:
    return f"crop-{stable_hash(cfg.teacher.crop)}"


def dense_key(cfg: Cfg) -> str:
    d = dict(cfg.teacher.dense)
    if d.get("mode") != "corrclip":
        d.pop("corrclip", None)   # keeps the keys of caches made before corrclip existed
    return f"dense-{d['mode']}-{stable_hash(d)}"


def cache(cfg: Cfg, *parts) -> str:
    return os.path.join(cfg.paths.cache_root, *parts)


def student_dir(cfg, index_key):
    return cache(cfg, "student", student_key(cfg), index_key)


def teacher_dir(cfg, kind, index_key):
    k = crop_key(cfg) if kind == "crop" else dense_key(cfg)
    return cache(cfg, "teacher", clip_key(cfg), k, index_key)


# --------------------------------------------------------------------------- loaders
def load_index(cfg, index_key):
    from .data.regions import RegionIndex, index_path

    p = index_path(cfg.paths.cache_root, index_key)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} missing. Run tools/mask_cls/build_index.py first.")
    return RegionIndex.load(p)


def parse_feature_spec(cfg: Cfg, spec: List[str]) -> Dict[str, List[int]]:
    """["patch.1", "patch.3", "pit_attn.0"] -> {"patch": [pos, ...], ...} (positions in the cache)."""
    out: Dict[str, List[int]] = {}
    for item in spec:
        group, _, idx = item.partition(".")
        idx = int(idx) if idx else 0
        if group == "patch":
            layers = list(cfg.student.layers)
            if idx not in layers:
                raise ValueError(f"{item}: block {idx} not extracted (student.layers={layers})")
            pos = layers.index(idx)
        elif group in ("pit_attn", "pixel"):
            if not cfg.student.pixel_pathway:
                raise ValueError(f"{item} needs student.pixel_pathway=true")
            pos = idx
        else:
            raise ValueError(f"Unknown feature group {group!r}")
        out.setdefault(group, []).append(pos)
    return out


def load_student(cfg: Cfg, index_key: str, spec: List[str], rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Concatenated student features [N, D] (fp16, CPU) in index row order."""
    groups = parse_feature_spec(cfg, spec)
    d = load_shards(student_dir(cfg, index_key), list(groups),
                    select={g: (lambda v, p=pos: v[:, p]) for g, pos in groups.items()})
    # Concatenated group by group, in order of first appearance in `spec`.
    parts = [d[g].reshape(len(d[g]), -1) for g in groups]   # [N, len(pos) * C]
    X = torch.cat(parts, 1)
    if rows is not None:
        X = X[rows]
    if not torch.isfinite(X).all():
        raise RuntimeError(f"Non-finite student features in {student_dir(cfg, index_key)}")
    return X


def load_teacher(cfg: Cfg, index_key: str, target: str, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    from .teacher.targets import combine_targets

    crop = dense = None
    if target in ("crop", "avg"):
        crop = load_shards(teacher_dir(cfg, "crop", index_key), ["crop"])["crop"]
    if target in ("dense", "avg"):
        dense = load_shards(teacher_dir(cfg, "dense", index_key), ["dense"])["dense"]
    Y = combine_targets(crop, dense, target)
    return Y if rows is None else Y[rows]


def class_prompt_names(cfg: Cfg, dataset: str, raw_names: List[str]):
    from .data.class_names import prompt_names

    override = (cfg.data.get("class_names") or {}).get(dataset)
    return prompt_names(dataset, raw_names, override)


def text_embeddings(cfg: Cfg, dataset: str, raw_names: List[str], teacher=None) -> torch.Tensor:
    """[C, D] cached; computed with `teacher` (a ClipTeacher) when missing."""
    from .teacher.text import class_text_embeddings, get_templates, text_cache_path

    names = class_prompt_names(cfg, dataset, raw_names)
    tcfg = cfg.text
    key = stable_hash([names, tcfg.templates, tcfg.synonyms])
    path = text_cache_path(cfg.paths.cache_root, clip_key(cfg), dataset, key)
    if os.path.isfile(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    if teacher is None:
        from .teacher.clip import ClipTeacher
        from .utils import resolve_dtype

        teacher = ClipTeacher(cfg.teacher.model, cfg.teacher.pretrained, dtype=resolve_dtype(cfg.teacher.dtype))
    emb = class_text_embeddings(teacher, names, get_templates(tcfg.templates), tcfg.synonyms)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(emb, path)
    return emb


def class_split(cfg: Cfg, dataset: str, raw_names: List[str]) -> dict:
    """Seen/unseen split vs the COCO-Stuff vocabulary (writes the documented JSON)."""
    from detectron2.data import MetadataCatalog

    from .data.splits import compute_split, load_synonyms, save_split

    ref = [c.strip() for c in MetadataCatalog.get(cfg.data.seen_reference).stuff_classes]
    split = compute_split(dataset, raw_names, ref, load_synonyms(cfg.data.synonyms_file),
                          (cfg.data.get("class_names") or {}).get(dataset))
    save_split(split, cfg.data.splits_dir)
    return split
