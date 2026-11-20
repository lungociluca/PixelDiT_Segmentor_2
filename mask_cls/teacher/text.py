"""CLIP text embeddings of class names with a prompt-template ensemble."""
import os
from typing import List

import torch
import torch.nn.functional as F

from .clip import ClipTeacher


def get_templates(name: str) -> List[str]:
    if name == "single":
        return ["a photo of a {}."]
    from san.model.clip_utils.utils import get_predefined_templates

    return get_predefined_templates(name)   # "vild" (14) or "imagenet" (80)


@torch.no_grad()
def class_text_embeddings(teacher: ClipTeacher, names: List[List[str]], templates: List[str],
                          synonyms: str = "ensemble") -> torch.Tensor:
    """names: per class, a list of synonyms. Returns L2-normalised [C, D].

    Per synonym: mean of normalised template embeddings, renormalised.
    Per class: mean over synonyms ("ensemble") or first synonym only ("first")."""
    out = []
    for syns in names:
        syns = syns if synonyms == "ensemble" else syns[:1]
        per_syn = []
        for s in syns:
            e = teacher.encode_text([t.format(s) for t in templates])
            per_syn.append(F.normalize(e.mean(0), dim=-1))
        out.append(F.normalize(torch.stack(per_syn).mean(0), dim=-1))
    return torch.stack(out).cpu()


def text_cache_path(cache_root: str, teacher_key: str, dataset: str, names_key: str) -> str:
    return os.path.join(cache_root, "text", teacher_key, f"{dataset}_{names_key}.pt")
