"""Region extraction and the region index.

A *region* is one binary mask with a class id. The index fixes a global row order
for all regions of a dataset, so student and teacher caches line up row by row.

Region sources are pluggable: `SemanticGTRegions` gives one mask per class present
in an image (ignore label excluded). A class-agnostic mask source for Stage 2 only
needs to implement `describe` and `masks` the same way (class_id = -1 when unknown).
"""
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .datasets import DatasetInfo, read_image, read_label_map

log = logging.getLogger("mask_cls")


class RegionSource:
    def describe(self, label_map: np.ndarray) -> List[dict]:
        """[{class_id, area, bbox=(x0, y0, x1, y1) exclusive}] for one image."""
        raise NotImplementedError

    def masks(self, label_map: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        """Bool masks [n, H, W] for the given regions of one image."""
        raise NotImplementedError


class SemanticGTRegions(RegionSource):
    def __init__(self, ignore_label: int, num_classes: int, min_area: int = 0):
        self.ignore = ignore_label
        self.num_classes = num_classes
        self.min_area = min_area

    def describe(self, label_map):
        ids, counts = np.unique(label_map, return_counts=True)
        out = []
        for c, a in zip(ids.tolist(), counts.tolist()):
            if c == self.ignore or c < 0 or c >= self.num_classes or a < max(self.min_area, 1):
                continue
            m = label_map == c
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            bbox = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
            out.append({"class_id": int(c), "area": int(a), "bbox": bbox})
        return out

    def masks(self, label_map, class_ids):
        return label_map[None] == class_ids.to(label_map.device)[:, None, None]


@dataclass
class RegionIndex:
    dataset: str
    images: List[List[str]]          # [image path, GT path]
    hw: torch.Tensor                 # [n_img, 2]
    img_ptr: torch.Tensor            # [n_img + 1]; regions of image i are rows img_ptr[i]:img_ptr[i+1]
    class_id: torch.Tensor           # [N]
    area: torch.Tensor               # [N] pixels at GT resolution
    bbox: torch.Tensor               # [N, 4] x0, y0, x1, y1 (exclusive)
    image_id: torch.Tensor           # [N]
    class_names: List[str] = field(default_factory=list)
    ignore_label: int = 255
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.class_id)

    @property
    def num_images(self):
        return len(self.images)

    def rows_of(self, i: int) -> torch.Tensor:
        return torch.arange(int(self.img_ptr[i]), int(self.img_ptr[i + 1]))

    def rows_of_first_images(self, k: Optional[int]) -> torch.Tensor:
        """Rows of the first k images (data-scaling ablations); all rows if k is None."""
        if k is None or k >= self.num_images:
            return torch.arange(len(self))
        return torch.arange(int(self.img_ptr[k]))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.__dict__, path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


class _DescribeDS(Dataset):
    def __init__(self, images, source):
        self.images, self.source = images, source

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        lab = read_label_map(self.images[i][1])
        return lab.shape, self.source.describe(lab)


def build_index(
    info: DatasetInfo,
    min_area: int = 0,
    image_subset: Optional[Sequence[int]] = None,
    num_workers: int = 8,
    meta: Optional[dict] = None,
) -> RegionIndex:
    images = info.images if image_subset is None else [info.images[i] for i in image_subset]
    source = SemanticGTRegions(info.ignore_label, len(info.class_names), min_area)
    dl = DataLoader(_DescribeDS(images, source), batch_size=None, num_workers=num_workers)
    hw, ptr, cls, area, bbox, img_id = [], [0], [], [], [], []
    for i, (shape, regs) in enumerate(dl):
        hw.append(tuple(shape))
        for r in regs:
            cls.append(r["class_id"]); area.append(r["area"]); bbox.append(r["bbox"]); img_id.append(i)
        ptr.append(len(cls))
        if (i + 1) % 5000 == 0:
            log.info("%s: indexed %d/%d images, %d regions", info.name, i + 1, len(images), len(cls))
    return RegionIndex(
        dataset=info.name,
        images=[list(x) for x in images],
        hw=torch.tensor(hw, dtype=torch.long),
        img_ptr=torch.tensor(ptr, dtype=torch.long),
        class_id=torch.tensor(cls, dtype=torch.long),
        area=torch.tensor(area, dtype=torch.long),
        bbox=torch.tensor(bbox, dtype=torch.long).reshape(-1, 4),
        image_id=torch.tensor(img_id, dtype=torch.long),
        class_names=list(info.class_names),
        ignore_label=info.ignore_label,
        meta=meta or {},
    )


def train_image_split(n_images: int, holdout: int, max_images: Optional[int], seed: int):
    """Seeded permutation of the training images -> (fit ids, holdout ids).

    The fit list keeps the permutation order, so "the first K images" is a
    random K-image subset (used for data-scaling ablations)."""
    perm = list(range(n_images))
    random.Random(seed).shuffle(perm)
    hold = perm[:holdout]
    fit = perm[holdout:]
    if max_images is not None:
        fit = fit[:max_images]
    return fit, hold


class ImageRegionsDataset(Dataset):
    """Per image: uint8 image [3, H, W], label map [H, W] and the image's region rows.

    Images with no regions, or whose rows are all in `skip_rows`, are dropped."""

    def __init__(self, index: RegionIndex, skip_rows: Optional[set] = None):
        self.index = index
        self.ids = []
        for i in range(index.num_images):
            rows = index.rows_of(i)
            if len(rows) == 0:
                continue
            if skip_rows and all(int(r) in skip_rows for r in rows):
                continue
            self.ids.append(i)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, j):
        i = self.ids[j]
        img_path, gt_path = self.index.images[i]
        img = torch.from_numpy(read_image(img_path)).permute(2, 0, 1).contiguous()
        lab = torch.from_numpy(read_label_map(gt_path))
        if tuple(img.shape[-2:]) != tuple(lab.shape):
            raise ValueError(f"Image {img_path} {tuple(img.shape[-2:])} and GT {tuple(lab.shape)} differ in size")
        rows = self.index.rows_of(i)
        return {
            "image_idx": i,
            "image": img,
            "label": lab,
            "rows": rows,
            "class_id": self.index.class_id[rows],
            "bbox": self.index.bbox[rows],
        }


def image_loader(index: RegionIndex, num_workers: int = 6, skip_rows: Optional[set] = None):
    ds = ImageRegionsDataset(index, skip_rows)
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=True)


def index_path(cache_root: str, key: str) -> str:
    return os.path.join(cache_root, "index", f"{key}.pt")
