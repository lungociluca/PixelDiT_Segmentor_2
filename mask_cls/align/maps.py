"""Alignment maps: PixelDiT region features -> CLIP embedding space.

Every map is an nn.Module taking RAW pooled features [..., D_in] (the
concatenation described by `feature_spec`, e.g. ["patch.3"] or ["patch.1", "patch.3"])
and returning UNNORMALISED embeddings [..., D_out]. Standardisation statistics
are buffers inside the module, so a Stage 2 head can call it on pooled features
directly:  emb = F.normalize(align_map(pooled), dim=-1).
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentMap(nn.Module):
    kind = "base"

    def __init__(self, d_in: int, d_out: int, feature_spec: List[str]):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.feature_spec = list(feature_spec)
        self.register_buffer("mean", torch.zeros(d_in))
        self.register_buffer("std", torch.ones(d_in))

    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean.copy_(mean)
        self.std.copy_(std)

    def standardize(self, x):
        return (x.float() - self.mean) / self.std

    def embed(self, x):
        """Normalised embedding."""
        return F.normalize(self(x), dim=-1)

    # ---- persistence
    def extra_state(self) -> dict:
        return {}

    def save(self, path: str, meta: Optional[dict] = None):
        torch.save({"kind": self.kind, "d_in": self.d_in, "d_out": self.d_out, "feature_spec": self.feature_spec,
                    "extra": self.extra_state(), "state_dict": self.state_dict(), "meta": meta or {}}, path)


class LinearMap(AlignmentMap):
    """y = s * (standardize(x) @ W) + b. Used for ridge, Procrustes and linear-cosine."""
    kind = "linear"

    def __init__(self, d_in, d_out, feature_spec):
        super().__init__(d_in, d_out, feature_spec)
        self.weight = nn.Parameter(torch.zeros(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x):
        return self.standardize(x) @ self.weight + self.bias


class ResidualMLPMap(AlignmentMap):
    """y = linear(z) + mlp(z), z = standardize(x); the MLP branch starts at zero."""
    kind = "mlp"

    def __init__(self, d_in, d_out, feature_spec, hidden: int = 2048, dropout: float = 0.1):
        super().__init__(d_in, d_out, feature_spec)
        self.hidden, self.dropout_p = hidden, dropout
        self.linear = nn.Linear(d_in, d_out)
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def extra_state(self):
        return {"hidden": self.hidden, "dropout": self.dropout_p}

    def forward(self, x):
        z = self.standardize(x)
        return self.linear(z) + self.mlp(z)


def load_map(path: str, map_location="cpu") -> AlignmentMap:
    d = torch.load(path, map_location=map_location, weights_only=False)
    if d["kind"] == "linear":
        m = LinearMap(d["d_in"], d["d_out"], d["feature_spec"])
    elif d["kind"] == "mlp":
        m = ResidualMLPMap(d["d_in"], d["d_out"], d["feature_spec"], **d["extra"])
    else:
        raise ValueError(d["kind"])
    m.load_state_dict(d["state_dict"])
    m.meta = d.get("meta", {})
    return m.eval()
