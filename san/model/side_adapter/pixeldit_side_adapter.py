"""Side adapter network for a frozen PixelDiT backbone.

Same side ViT, learnable queries and feature fusion as SAN's
RegionwiseSideAdapterNetwork. Each query predicts a mask and a class-agnostic
objectness score (object / no object). With MODEL.REC.ENABLED SAN's
attention-bias branch is back: each query also predicts, for every recognition
block and every PixelDiT attention head, a bias map over the patch grid, which
steers that query's mask token in the frozen PixelDiT (san.model.rec_head).
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
    """Query/pixel MLPs -> mask logits, plus a linear class head on the queries.

    bias_heads > 0 adds SAN's attention-bias branch (as in MLPMaskDecoder):
    attn_bias[b, l, n, q] = <query_embed[b, q], attn_mlp(x)[b, l, n]> for
    `bias_layers` recognition blocks l and `bias_heads` heads n, optionally
    rescaled by a learned affine map (RESCALE_ATTN_BIAS)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int = 1,
        embed_channels: int = 256,
        mlp_channels: int = 256,
        mlp_num_layers: int = 3,
        bias_heads: int = 0,
        bias_layers: int = 1,
        rescale_attn_bias: bool = False,
    ):
        super().__init__()
        # Query branch (same as SAN's MLPMaskDecoder).
        self.query_mlp = MLP(in_channels, mlp_channels, embed_channels, mlp_num_layers)
        # Pixel branch: 1x1 convs on the dense side features.
        dense = partial(nn.Conv2d, kernel_size=1)
        self.pix_mlp = MLP(in_channels, mlp_channels, embed_channels, mlp_num_layers, affine_func=dense)
        # num_classes + 1 logits, last one = no object (Mask2Former convention).
        self.class_embed = nn.Linear(in_channels, num_classes + 1)
        # Attention-bias branch (SAN).
        self.bias_heads = int(bias_heads)
        self.bias_layers = int(bias_layers)
        if self.bias_heads > 0:
            self.attn_mlp = MLP(
                in_channels,
                mlp_channels,
                embed_channels * self.bias_heads * self.bias_layers,
                mlp_num_layers,
                affine_func=dense,
            )
            self.bias_scaling = nn.Linear(1, 1) if rescale_attn_bias else nn.Identity()

    def forward(
        self, query: torch.Tensor, x: torch.Tensor, with_bias: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        # query: [B,N,C]  x: [B,C,H,W]
        # Returns mask_preds [B,N,H,W], class_logits [B,N,K+1] and, with the bias
        # branch (and with_bias), one [B,heads,N,H,W] fp32 bias per recognition block.
        class_logits = self.class_embed(query)  # [B,N,K+1]
        query_embed = self.query_mlp(query)
        pix_embed = self.pix_mlp(x)
        # Mask logits in fp32: an unnormalised 256-term dot product, which is where
        # float16 autocast first overflows / loses resolution.
        with torch.autocast(query.device.type, enabled=False):
            mask_preds = torch.einsum(
                "bqc,bchw->bqhw", query_embed.float(), pix_embed.float()
            )  # [B,N,H,W]
        attn_bias = None
        if self.bias_heads > 0 and with_bias:
            b, _, h, w = x.shape
            attn = self.attn_mlp(x).reshape(b, self.bias_layers, self.bias_heads, -1, h, w)
            with torch.autocast(query.device.type, enabled=False):
                bias = torch.einsum("bqc,blnchw->blnqhw", query_embed.float(), attn.float())
                bias = self.bias_scaling(bias[..., None]).squeeze(-1)
            attn_bias = [bias[:, j] for j in range(self.bias_layers)]
        return mask_preds, class_logits, attn_bias


@SIDE_ADAPTER_REGISTRY.register()
class PixelDiTSideAdapterNetwork(RegionwiseSideAdapterNetwork):
    """SAN side adapter that fuses PixelDiT patch features and decodes masks only.

    `input_shape` maps a PixelDiT layer index (0 = patch embedding, k = output of
    patch block k) to its ShapeSpec. MODEL.SIDE_ADAPTER.FUSION_MAP entries
    "k->j" add PixelDiT layer k to the side ViT tokens before side block j.

    forward(image, features) returns three lists, one entry per deep-supervision
    block during training (only the last one at inference):
      mask_preds:   [B, N, h, w] mask logits on the side ViT's patch grid
      class_logits: [B, N, K + 1] (K = 1 for class-agnostic training)
      attn_biases:  with MODEL.REC.ENABLED, for the last block only, a list with
                    one [B, heads, N, h, w] bias per MODEL.REC.LAYERS entry;
                    None for every other entry and without MODEL.REC
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
                bias_heads=sa.ATTN_BIAS.NUM_HEADS if cfg.MODEL.REC.ENABLED else 0,
                bias_layers=len(cfg.MODEL.REC.LAYERS),
                rescale_attn_bias=sa.ATTN_BIAS.RESCALE_ATTN_BIAS,
            ),
            "deep_supervision_idxs": sa.DEEP_SUPERVISION_IDXS,
        }

    def decode_masks(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Optional[List[torch.Tensor]]]]:
        if not self.training:
            features = [features[-1]]
        mask_preds, class_logits, attn_biases = [], [], []
        for j, feature in enumerate(features):
            # Only the final output is classified: no bias for deep supervision.
            mask_pred, class_logit, attn_bias = self.mask_decoder(
                **feature, with_bias=j == len(features) - 1
            )
            mask_preds.append(mask_pred)
            class_logits.append(class_logit)
            attn_biases.append(attn_bias)
        return mask_preds, class_logits, attn_biases
