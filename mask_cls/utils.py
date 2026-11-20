"""Seeding, sharded fp16 feature caches and result logging."""
import csv
import glob
import json
import logging
import os
import random
import time
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

log = logging.getLogger("mask_cls")


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
    }[name.lower()]


# --------------------------------------------------------------------------- shards
class ShardWriter:
    """Writes per-region features to `dir/shard_XXXXX.pt`.

    Each shard holds {"rows": int64 [n] (row ids in the region index), <name>: fp16 [n, ...]}.
    Shards are written atomically, so an interrupted extraction can resume: images
    whose regions are all present in finished shards are skipped (see `done_rows`).
    """

    def __init__(self, out_dir: str, regions_per_shard: int = 20000, meta: Optional[dict] = None,
                 dtype: torch.dtype = torch.float16):
        self.dir = out_dir
        self.dtype = dtype
        os.makedirs(out_dir, exist_ok=True)
        self.per_shard = regions_per_shard
        self.buf: Dict[str, List[torch.Tensor]] = {}
        self.rows: List[torch.Tensor] = []
        self.n_buf = 0
        existing = sorted(glob.glob(os.path.join(out_dir, "shard_*.pt")))
        self.next_id = (int(os.path.basename(existing[-1])[6:11]) + 1) if existing else 0
        if meta is not None:
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=1, default=str)

    def done_rows(self) -> set:
        done = set()
        for p in sorted(glob.glob(os.path.join(self.dir, "shard_*.pt"))):
            # mmap: only the small "rows" tensor is read, not the features.
            done.update(torch.load(p, map_location="cpu", weights_only=True, mmap=True)["rows"].tolist())
        return done

    def add(self, rows: torch.Tensor, **feats: torch.Tensor):
        self.rows.append(rows.cpu().long())
        for k, v in feats.items():
            self.buf.setdefault(k, []).append(v.detach().to("cpu", self.dtype))
        self.n_buf += len(rows)
        if self.n_buf >= self.per_shard:
            self.flush()

    def flush(self):
        if self.n_buf == 0:
            return
        out = {"rows": torch.cat(self.rows)}
        for k, v in self.buf.items():
            out[k] = torch.cat(v)
        path = os.path.join(self.dir, f"shard_{self.next_id:05d}.pt")
        torch.save(out, path + ".tmp")
        os.replace(path + ".tmp", path)
        self.next_id += 1
        self.buf, self.rows, self.n_buf = {}, [], 0

    def close(self):
        self.flush()
        with open(os.path.join(self.dir, "DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))


def is_complete(cache_dir: str) -> bool:
    return os.path.isfile(os.path.join(cache_dir, "DONE"))


def load_shards(
    cache_dir: str,
    keys: Sequence[str],
    n_rows: Optional[int] = None,
    select=None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Load cached features in region-index row order.

    `select`: optional dict key -> callable(tensor) applied per shard before
    concatenation (e.g. to keep only some layers, saving memory).
    Returns {key: tensor [N, ...]} plus "rows" (sorted row ids). If `n_rows` is
    given, checks that every row 0..n_rows-1 is present.
    """
    paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shards in {cache_dir}. Run the extraction first.")
    rows, out = [], {k: [] for k in keys}
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=True)
        rows.append(d["rows"])
        for k in keys:
            if k not in d:
                raise KeyError(f"{p} has no '{k}' (has {list(d)})")
            v = d[k]
            if select and k in select:
                v = select[k](v)
            out[k].append(v if dtype is None else v.to(dtype))
    rows_t = torch.cat(rows)
    order = torch.argsort(rows_t)
    res = {k: torch.cat(v)[order] for k, v in out.items()}
    res["rows"] = rows_t[order]
    if n_rows is not None:
        if len(res["rows"]) != n_rows or not torch.equal(res["rows"], torch.arange(n_rows)):
            raise RuntimeError(
                f"{cache_dir}: expected rows 0..{n_rows - 1}, got {len(res['rows'])} rows "
                "(duplicates or missing). Delete the folder and re-extract."
            )
    return res


# --------------------------------------------------------------------------- results
def append_result(results_dir: str, row: dict):
    """Append one result row to results.jsonl and results.csv."""
    os.makedirs(results_dir, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    with open(os.path.join(results_dir, "results.jsonl"), "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    csv_path = os.path.join(results_dir, "results.csv")
    flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
    header = None
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            header = next(csv.reader(f), None)
    if header is not None and set(flat) <= set(header):
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(flat)
    else:
        # New columns: rewrite the CSV from the JSONL so every row has all columns.
        rows = [json.loads(l) for l in open(os.path.join(results_dir, "results.jsonl"))]
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
