"""SAN-style open-vocabulary mask classification inside the frozen PixelDiT
(MODEL.REC).

SAN (Xu et al., CVPR 2023) classifies a mask by giving CLIP one extra [SOS]
token per mask query in its last layers; the token attends to the image tokens
with an attention bias predicted by the side adapter, and its output is matched
against CLIP text embeddings. Here the frozen model is PixelDiT:

  * one mask token per side-adapter query enters PixelDiT block L (default 5)
    and is processed with that block's frozen *image-stream* weights (adaLN,
    qkv_x, q/k norms, proj_x); it attends to the block's input image tokens,
    and to itself, with the side adapter's per-query, per-head attention bias
    added to the attention logits;
  * attention is one-way: the image, prompt and concept tokens never attend to
    the mask tokens, so the backbone pass is unchanged and block L can be
    re-run on the mask tokens alone after the side adapter has produced the
    biases (from features of the same backbone pass);
  * the mask token's attention output (block L's image attention-output space)
    is compared by cosine similarity with the attention output of the class
    concept tokens in the same block, as ConceptAttention (arXiv:2502.04320)
    compares image tokens with concepts. The classifier therefore uses
    PixelDiT's own text-image alignment: no CLIP, no projection head and no
    parameter tied to a class. The only trained parts are the attention-bias
    branch of the side adapter (through which the classification loss also
    reaches the rest of the side adapter, as in SAN) and a logit scale.

The mask tokens use the image keys without rotary position embedding (as if
the token sat at every key's position): where a mask token looks is set by
the bias, not by a position it does not have.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _linear(mod: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """A frozen nn.Linear in fp32, whatever dtype its weights are stored in."""
    return F.linear(x, mod.weight.float(), mod.bias.float() if mod.bias is not None else None)


def _seq(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """A frozen Linear or Sequential of Linear / activation modules, in fp32."""
    if isinstance(mod, nn.Linear):
        return _linear(mod, x)
    for m in mod:
        x = _linear(m, x) if isinstance(m, nn.Linear) else m(x)
    return x


def _norm(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """RMSNorm in fp32 (its weight may be bf16: the product is promoted to fp32)."""
    return mod(x).float()


def _feed_forward(ff: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return _linear(ff.w2, F.silu(_linear(ff.w1, x)) * _linear(ff.w3, x))


class AttnBiasMaskRecognizer(nn.Module):
    """Classifies the side adapter's masks by running mask tokens through
    PixelDiT blocks `layers` (1-based, consecutive) with an attention bias.

    query_init: "global" - every mask token starts as the mean image token
                           entering the first block (SAN copies [CLS] the same
                           way; the bias tells the queries apart);
                "masked" - the image tokens averaged over each predicted mask
                           (sigmoid, detached), a region-specific start.
    self_attn:  the mask token also attends to itself (SAN's [SOS] tokens do).
    """

    def __init__(
        self,
        layers: Sequence[int],
        query_init: str = "global",
        self_attn: bool = True,
        init_logit_scale: float = 20.0,
    ):
        super().__init__()
        layers = [int(k) for k in layers]
        if not layers or layers != list(range(layers[0], layers[0] + len(layers))) or layers[0] < 1:
            raise ValueError(f"MODEL.REC.LAYERS must be consecutive PixelDiT blocks >= 1, got {layers}")
        if query_init not in ("global", "masked"):
            raise ValueError(f"MODEL.REC.QUERY_INIT must be 'global' or 'masked', got {query_init!r}")
        self.layers = layers
        self.query_init = query_init
        self.self_attn = bool(self_attn)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale)))

    # ------------------------------------------------------------------ tokens
    def init_tokens(
        self, x: torch.Tensor, mask_logits: torch.Tensor, hw: Tuple[int, int]
    ) -> torch.Tensor:
        """x: (B, P, C) image tokens entering the first block, mask_logits:
        (B, Q, hm, wm) -> (B, Q, C) fp32 initial mask tokens."""
        x = x.float()
        B, Q = mask_logits.shape[:2]
        if self.query_init == "global":
            return x.mean(1, keepdim=True).expand(B, Q, -1)
        m = mask_logits.detach().float()
        if tuple(m.shape[-2:]) != tuple(hw):
            m = F.interpolate(m, size=hw, mode="bilinear", align_corners=False)
        w = m.sigmoid().flatten(2)  # (B, Q, P)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
        return torch.einsum("bqp,bpc->bqc", w, x)

    @staticmethod
    def _bias_to_grid(bias: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """(B, n, Q, hb, wb) -> (B, n, Q, h*w) fp32 on the PixelDiT patch grid."""
        bias = bias.float()
        B, n, Q = bias.shape[:3]
        if tuple(bias.shape[-2:]) != tuple(hw):
            bias = F.interpolate(
                bias.flatten(0, 2)[:, None], size=hw, mode="bilinear", align_corners=False
            )[:, 0].reshape(B, n, Q, *hw)
        return bias.flatten(3)

    # ------------------------------------------------------------------- block
    def _block(
        self,
        block: nn.Module,
        condition: torch.Tensor,
        x: torch.Tensor,
        m: torch.Tensor,
        bias: torch.Tensor,
        last: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One PixelDiT block (MMDiTBlockT2I) on the mask tokens.

        x: (B, P, C) image tokens entering the block, m: (B, Q, C) mask tokens,
        bias: (B, n, Q, P) with n = heads or 1.
        Returns (attention output (B, Q, C), mask tokens after the block, or
        None when `last`: the block's MLP is not needed for the score)."""
        attn = block.attn
        H, hd = attn.num_heads, attn.head_dim
        B, P, C = x.shape
        Q = m.shape[1]
        cond = condition.float()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _seq(
            block.adaLN_modulation_img, cond
        ).chunk(6, dim=-1)

        with torch.no_grad():
            # Image keys / values: frozen, so no graph is kept for them.
            x_norm = _norm(block.norm_x1, x.float()) * (1 + scale_msa) + shift_msa
            kv = _linear(attn.qkv_x, x_norm).reshape(B, P, 3, H, hd)
            k_img = _norm(attn.k_norm_x, kv[:, :, 1]).transpose(1, 2)  # (B, H, P, hd)
            v_img = kv[:, :, 2].transpose(1, 2)                         # (B, H, P, hd)
            del kv, x_norm

        m_norm = _norm(block.norm_x1, m) * (1 + scale_msa) + shift_msa
        qkv = _linear(attn.qkv_x, m_norm).reshape(B, Q, 3, H, hd)
        q = _norm(attn.q_norm_x, qkv[:, :, 0]).transpose(1, 2)       # (B, H, Q, hd)
        scale = hd ** -0.5
        logits = torch.einsum("bhqd,bhpd->bhqp", q, k_img) * scale + bias  # (B, H, Q, P)
        if self.self_attn:
            k_self = _norm(attn.k_norm_x, qkv[:, :, 1]).transpose(1, 2)
            v_self = qkv[:, :, 2].transpose(1, 2)
            self_logit = (q * k_self).sum(-1, keepdim=True) * scale    # (B, H, Q, 1)
            probs = torch.cat([logits, self_logit], dim=-1).softmax(-1)
            out = torch.einsum("bhqp,bhpd->bhqd", probs[..., :-1], v_img) + probs[..., -1:] * v_self
        else:
            out = torch.einsum("bhqp,bhpd->bhqd", logits.softmax(-1), v_img)
        out = _linear(attn.proj_x, out.transpose(1, 2).reshape(B, Q, C))  # (B, Q, C)
        if last:
            return out, None
        m = m + gate_msa * out
        m = m + gate_mlp * _feed_forward(
            block.mlp_x, _norm(block.norm_x2, m) * (1 + scale_mlp) + shift_mlp
        )
        return out, m

    def forward(
        self,
        blocks: nn.ModuleList,
        rec_state: Dict,
        attn_biases: List[torch.Tensor],
        mask_logits: torch.Tensor,
    ) -> torch.Tensor:
        """rec_state: segment_data of a backbone pass run with rec_layers (see
        PixDiT_T2I.forward): "rec_inputs", "rec_concepts", "rec_condition", "hw".
        attn_biases: one (B, n, Q, hb, wb) bias per block of `layers`.
        mask_logits: (B, Q, hm, wm), only used by QUERY_INIT "masked".
        Returns (B, Q, N) logits = scale * cos(mask token, concept) at the last
        block, one column per concept (synonyms not reduced yet)."""
        if len(attn_biases) != len(self.layers):
            raise ValueError(f"{len(attn_biases)} attention biases for MODEL.REC.LAYERS {self.layers}")
        hw = tuple(rec_state["hw"])
        with torch.autocast(mask_logits.device.type, enabled=False):
            m = self.init_tokens(rec_state["rec_inputs"][self.layers[0]], mask_logits, hw)
            out = None
            for j, k in enumerate(self.layers):
                bias = self._bias_to_grid(attn_biases[j], hw)
                n_heads = blocks[k - 1].attn.num_heads
                if bias.shape[1] not in (1, n_heads):
                    raise ValueError(
                        f"Attention bias has {bias.shape[1]} heads; PixelDiT block {k} has {n_heads} "
                        "(set MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_HEADS to that or 1)."
                    )
                out, m = self._block(
                    blocks[k - 1],
                    rec_state["rec_condition"],
                    rec_state["rec_inputs"][k],
                    m,
                    bias,
                    last=j == len(self.layers) - 1,
                )
            concepts = rec_state["rec_concepts"][self.layers[-1]].float()  # (B, N, C)
            s = torch.einsum(
                "bqc,bnc->bqn", F.normalize(out, dim=-1), F.normalize(concepts, dim=-1)
            )
            return s * self.logit_scale.exp().clamp(max=100.0)
