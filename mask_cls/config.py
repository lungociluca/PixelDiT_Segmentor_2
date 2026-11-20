"""YAML configs with `_base_` inheritance and dotted command-line overrides.

    cfg = load_config("configs/mask_cls/base.yaml", ["align.method=ridge", "student.timestep=0.1"])
    cfg.align.method  # attribute access

Override values are parsed as YAML, so `a.b=[0,3]`, `a.b=null`, `a.b=1e-3` work.
"""
import copy
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import yaml


class Cfg(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self) -> Dict[str, Any]:
        return _plain(self)


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not v.get("_replace_", False):
            out[k] = _merge(out[k], v)
        else:
            if isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk != "_replace_"}
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    bases = d.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = _merge(merged, _load_yaml(os.path.join(os.path.dirname(path), b)))
    return _merge(merged, d)


def apply_overrides(d: dict, overrides: Iterable[str]) -> dict:
    d = copy.deepcopy(d)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    return d


def _expand_env(x):
    if isinstance(x, str):
        return os.path.expandvars(os.path.expanduser(x))
    if isinstance(x, dict):
        return {k: _expand_env(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand_env(v) for v in x]
    return x


def load_config(path: str, overrides: Optional[List[str]] = None) -> Cfg:
    d = _load_yaml(path)
    d = apply_overrides(d, overrides)
    d = _expand_env(d)
    d.setdefault("_config_path", path)
    return _wrap(d)


def stable_hash(obj: Any, n: int = 10) -> str:
    """Short deterministic hash of a JSON-serialisable object (cache keys)."""
    s = json.dumps(_plain(obj), sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def add_common_args(parser):
    parser.add_argument("--config", default="configs/mask_cls/base.yaml")
    parser.add_argument("opts", nargs="*", help="Config overrides, e.g. align.method=ridge")
    return parser
