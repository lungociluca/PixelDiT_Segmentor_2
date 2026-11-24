"""SAN-style open-vocabulary mask classification on PixelDiT, with a frozen CLIP
text encoder (MODEL.REGION_CLS).

Like SAN's [SOS] tokens: every mask query gets one region token that runs
through PixelDiT block L (default 5) of the frozen backbone, attending to the
block's input image tokens with an additive attention bias predicted by the
side adapter (one bias map per query and per head). The token is one-way, so
the backbone's own image and text streams are untouched, and the block can be
re-run on the region tokens after the backbone pass, once the side adapter has
produced the biases from PixelDiT features (block L included).

The region token's block-L output is mapped by a small MLP into the CLIP text
space and compared (cosine) with frozen CLIP text embeddings of the class
names, plus a learned "no object" embedding. No parameter is tied to a class,
so new vocabularies only need their CLIP text embeddings. The CLIP text encoder
is used once, at construction, and then dropped: there is no CLIP at inference.
"""
import logging
import math
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .clip_utils.utils import get_predefined_templates
from .concept_classifier import class_name_variants, reduce_synonyms

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_clip_vocabularies(
    vocabularies: Dict[str, List[str]],
    model_name: str = "ViT-B-16",
    pretrained: str = "openai",
    template_set: str = "vild",
    clean_names: bool = True,
    split_synonyms: bool = True,
    device: str = "cuda",
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Frozen CLIP text embeddings of every vocabulary.

    Every class name is cleaned / split into synonyms (as for the concept
    classifier), every variant is encoded in every template of `template_set`,
    and the unit-norm embeddings are averaged over the templates.
    Returns {dataset_name: ((N, D) unit-norm embeddings, (N,) class index of
    every variant)}. The CLIP model is freed before returning.
    """
    import open_clip

    model = open_clip.create_model(model_name, pretrained=pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    templates = get_predefined_templates(template_set)
    out = {}
    cache: Dict[str, torch.Tensor] = {}
    for name, labels in vocabularies.items():
        embs, classes = [], []
        for k, label in enumerate(labels):
            for variant in class_name_variants(label, clean=clean_names, split=split_synonyms):
                if variant not in cache:
                    tokens = tokenizer([t.format(variant) for t in templates]).to(device)
                    e = F.normalize(model.encode_text(tokens).float(), dim=-1).mean(0)
                    cache[variant] = F.normalize(e, dim=-1)
                embs.append(cache[variant])
                classes.append(k)
        out[name] = (torch.stack(embs), torch.tensor(classes, dtype=torch.long, device=device))
    logger.info(
        "Region classifier: CLIP %s/%s text embeddings, %s variants",
        model_name, pretrained, {k: v[0].shape[0] for k, v in out.items()},
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


class RegionClassifier(nn.Module):
    """Trainable part of the SAN-style classifier (the PixelDiT block is frozen).

    region_embed: added to the global mean of the block's input image tokens to
                  form the initial region token (all queries start equal, like
                  SAN's copies of [CLS]; the attention bias tells them apart).
    proj:         LayerNorm + MLP, region feature (hidden) -> CLIP text space.
    void_embed:   learned "no object" embedding in the CLIP text space.
    logit_scale:  learned, initialised at `init_logit_scale`, capped at 100.
    """

    def __init__(
        self,
        hidden_size: int,
        text_dim: int,
        proj_hidden_dim: int = 1024,
        init_logit_scale: float = 50.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.region_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, text_dim),
        )
        self.void_embed = nn.Parameter(torch.randn(1, text_dim) * text_dim ** -0.5)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale)))

    def init_tokens(self, x: torch.Tensor, num_queries: int) -> torch.Tensor:
        """x: (B, P, hidden) input image tokens of the block -> (B, Q, hidden)."""
        r = x.float().mean(1, keepdim=True) + self.region_embed
        return r.expand(-1, num_queries, -1).to(x.dtype)

    def logits(
        self,
        region_feats: torch.Tensor,
        text_embs: torch.Tensor,
        variant_class: torch.Tensor,
        num_classes: int,
        synonym_reduce: str = "max",
    ) -> torch.Tensor:
        """region_feats: (B, Q, hidden) block outputs of the region tokens,
        text_embs: (N, D) unit-norm CLIP embeddings -> (B, Q, K + 1) logits, last
        column = no object."""
        with torch.autocast(region_feats.device.type, enabled=False):
            img = F.normalize(self.proj(region_feats.float()), dim=-1)
            s = img @ text_embs.float().t()                                   # (B, Q, N)
            s = reduce_synonyms(s, variant_class, num_classes, synonym_reduce)  # (B, Q, K)
            void = img @ F.normalize(self.void_embed.float(), dim=-1).t()     # (B, Q, 1)
            return torch.cat([s, void], dim=-1) * self.logit_scale.exp().clamp(max=100.0)
