"""Side adapter network for a frozen PixelDiT backbone.

Same side ViT, learnable queries and feature fusion as SAN's
RegionwiseSideAdapterNetwork, but the CLIP-specific attention-bias branch is
removed: there is no CLIP recognition head to feed. Instead each query predicts
a mask and a class-agnostic objectness score (object / no object).
"""
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.layers import ShapeSpec
from timm import create_model
from torch import nn

from ..layers import MLP, build_fusion_layer
from .side_adapter import SIDE_ADAPTER_REGISTRY, RegionwiseSideAdapterNetwork
from .timm_wrapper import PatchEmbed


class MaskObjectnessDecoder(nn.Module):
    """Query/pixel MLPs -> mask logits, plus either a linear class head on the
    queries (objectness / closed-set classes) or, with `num_classes=None`, SAN's
    attention-bias branch for the region classifier (san.model.region_classifier).

    attn_bias_heads > 0: per query and per PixelDiT attention head, a bias map
    over the patch grid, bias[b, n, q] = <bias_query[b, q], bias_pix[b, n]>,
    as in SAN's MLPMaskDecoder (with its own, smaller embedding width).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: Optional[int] = 1,
        embed_channels: int = 256,
        mlp_channels: int = 256,
        mlp_num_layers: int = 3,
        attn_bias_heads: int = 0,
        attn_bias_channels: int = 64,
    ):
        super().__init__()
        # Query branch (same as SAN's MLPMaskDecoder).
        self.query_mlp = MLP(in_channels, mlp_channels, embed_channels, mlp_num_layers)
        # Pixel branch: 1x1 convs on the dense side features.
        dense = partial(nn.Conv2d, kernel_size=1)
        self.pix_mlp = MLP(in_channels, mlp_channels, embed_channels, mlp_num_layers, affine_func=dense)
        # num_classes + 1 logits, last one = no object (Mask2Former convention).
        self.class_embed = nn.Linear(in_channels, num_classes + 1) if num_classes is not None else None
        self.attn_bias_heads = int(attn_bias_heads)
        self.attn_bias_channels = int(attn_bias_channels)
        if self.attn_bias_heads > 0:
            self.bias_query = nn.Linear(embed_channels, self.attn_bias_channels)
            self.attn_mlp = MLP(
                in_channels, mlp_channels, self.attn_bias_channels * self.attn_bias_heads,
                mlp_num_layers, affine_func=dense,
            )

    def forward(self, query: torch.Tensor, x: torch.Tensor):
        # query: [B,N,C]  x: [B,C,H,W]
        # Returns (mask_preds [B,N,H,W], class_logits [B,N,K+1] or None,
        #          attn_bias [B,heads,N,H,W] or None).
        class_logits = self.class_embed(query) if self.class_embed is not None else None
        query_embed = self.query_mlp(query)
        pix_embed = self.pix_mlp(x)
        # Mask logits in fp32: an unnormalised 256-term dot product, which is where
        # float16 autocast first overflows / loses resolution.
        with torch.autocast(query.device.type, enabled=False):
            mask_preds = torch.einsum(
                "bqc,bchw->bqhw", query_embed.float(), pix_embed.float()
            )  # [B,N,H,W]
        attn_bias = None
        if self.attn_bias_heads > 0:
            b, _, h, w = x.shape
            bq = self.bias_query(query_embed)
            attn = self.attn_mlp(x).reshape(b, self.attn_bias_heads, self.attn_bias_channels, h, w)
            with torch.autocast(query.device.type, enabled=False):
                attn_bias = torch.einsum("bqc,bnchw->bnqhw", bq.float(), attn.float())
        return mask_preds, class_logits, attn_bias


@SIDE_ADAPTER_REGISTRY.register()
class PixelDiTSideAdapterNetwork(RegionwiseSideAdapterNetwork):
    """SAN side adapter that fuses PixelDiT patch features and decodes masks only.

    `input_shape` maps a PixelDiT layer index (0 = patch embedding, k = output of
    patch block k) to its ShapeSpec. MODEL.SIDE_ADAPTER.FUSION_MAP entries
    "k->j" add PixelDiT layer k to the side ViT tokens before side block j.

    forward(image, features) returns two lists, one entry per deep-supervision
    block during training (only the last one at inference):
      mask_preds:   [B, N, h, w] mask logits on the side ViT's patch grid
      class_logits: [B, N, K + 1] (K = 1 for class-agnostic training), or with
                    MODEL.REGION_CLS.ENABLED the attention biases
                    [B, heads, N, h, w] for the region classifier
    """

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[int, ShapeSpec]):
        sa = cfg.MODEL.SIDE_ADAPTER
        vit = create_model(
            sa.VIT_NAME,
            sa.PRETRAINED,
            img_size=sa.IMAGE_SIZE,
            drop_path_rate=sa.DROP_PATH_RATE,
            fc_norm=False,
            num_classes=0,
            embed_layer=PatchEmbed,
        )
        # "src->tgt": backbone layer src is fused before side block tgt.
        x2side_map = {
            int(tgt): int(src) for src, tgt in [x.split("->") for x in sa.FUSION_MAP]
        }
        missing = sorted(set(x2side_map.values()) - set(input_shape))
        if missing:
            raise ValueError(
                f"SIDE_ADAPTER.FUSION_MAP uses PixelDiT layers {missing}, but only "
                f"{sorted(input_shape)} are extracted (MODEL.PIXELDIT.FEATURE_LAYERS)."
            )
        too_deep = sorted(t for t in x2side_map if t >= len(vit.blocks))
        if too_deep:
            raise ValueError(
                f"SIDE_ADAPTER.FUSION_MAP targets side blocks {too_deep}, but the "
                f"side ViT has {len(vit.blocks)} blocks (valid targets 0..{len(vit.blocks) - 1})."
            )
        bad_ds = [i for i in sa.DEEP_SUPERVISION_IDXS if not 1 <= i <= len(vit.blocks)]
        if bad_ds:
            raise ValueError(
                f"SIDE_ADAPTER.DEEP_SUPERVISION_IDXS {bad_ds} out of range 1..{len(vit.blocks)}."
            )
        fusion_layers = nn.ModuleDict(
            {
                f"layer_{tgt}": build_fusion_layer(
                    sa.FUSION_TYPE, input_shape[src].channels, vit.num_features
                )
                for tgt, src in x2side_map.items()
            }
        )
        region = cfg.MODEL.REGION_CLS.ENABLED
        if region:
            # Classes come from the region classifier; no closed-set class head.
            num_classes = None
        else:
            num_classes = 1 if cfg.MODEL.SAN.CLASS_AGNOSTIC else cfg.MODEL.SAN.NUM_CLASSES
        return {
            "vit_model": vit,
            "num_queries": sa.NUM_QUERIES,
            "fusion_layers": fusion_layers,
            "fusion_map": x2side_map,
            "mask_decoder": MaskObjectnessDecoder(
                in_channels=vit.num_features,
                num_classes=num_classes,
                embed_channels=sa.ATTN_BIAS.EMBED_CHANNELS,
                mlp_channels=sa.ATTN_BIAS.MLP_CHANNELS,
                mlp_num_layers=sa.ATTN_BIAS.MLP_NUM_LAYERS,
                attn_bias_heads=cfg.MODEL.REGION_CLS.BIAS_HEADS if region else 0,
                attn_bias_channels=cfg.MODEL.REGION_CLS.BIAS_CHANNELS,
            ),
            "deep_supervision_idxs": sa.DEEP_SUPERVISION_IDXS,
        }

    def decode_masks(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Returns (mask_preds, second), one entry per decoded block. `second` is
        the class logits, or the attention biases [B, heads, N, h, w] when the
        decoder has no class head (MODEL.REGION_CLS.ENABLED)."""
        if not self.training:
            features = [features[-1]]
        mask_preds, second = [], []
        for feature in features:
            mask_pred, class_logit, attn_bias = self.mask_decoder(**feature)
            mask_preds.append(mask_pred)
            second.append(class_logit if class_logit is not None else attn_bias)
        return mask_preds, second
