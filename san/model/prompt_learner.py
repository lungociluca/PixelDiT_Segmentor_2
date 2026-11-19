"""Learnable language embeddings for the last concept layer (MODEL.PROMPT_LEARNER).

PixelDiT's text stream is computed from the default prompt (t2i.config.
default_prompt) by every block, as before. Only at the input of one block, by
default the last of MODEL.CONCEPT.LAYERS (the one block used for mask
classification), learned embeddings are *added* to the text states the
previous blocks produced:

    y_k  <-  y_k + delta

so the image tokens of that block attend to a learned prompt, and the
concept scores of that block (image-token / class-concept similarities in its
attention output) change with it. Blocks before it, and the side adapter's
pass, are untouched: no gradient flows through them, so the cost is one block's
backward in the concept-scoring pass. No pretrained model is added:

  * static embeddings (always): one learned delta shared by every image;
  * implicit captioner (IMPLICIT.ENABLED, ODISE-style): a small transformer,
    trained from scratch, reads the image tokens entering that block (the
    frozen PixelDiT is its own image encoder, as CLIP is in ODISE) and adds a
    per-image delta.

The delta lives in PixelDiT's hidden text space and covers either every text
position (SPAN "all") or NUM_TOKENS padding positions right after the default
prompt's last real token (SPAN "append"). Both parts start at zero (zero
embeddings, zero-initialised captioner output), so training starts exactly
from the fixed-prompt model.
"""
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ImplicitCaptioner(nn.Module):
    """Image tokens (B, h*w, in_dim) -> (B, num_tokens, out_dim) prompt deltas.

    The image tokens are average-pooled to a pool_size x pool_size grid, and
    num_tokens learned queries read them with `depth` transformer decoder layers
    (self-attention among the queries, cross-attention to the image). The output
    projection is zero-initialised, so the delta is 0 at the start.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_tokens: int,
        dim: int = 512,
        depth: int = 2,
        num_heads: int = 8,
        pool_size: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pool_size = int(pool_size)
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, dim)
        # Names match SOLVER.WEIGHT_DECAY_EMBED_GROUP ("pos_embed", "query_embed").
        self.pos_embed = nn.Parameter(torch.zeros(1, self.pool_size * self.pool_size, dim))
        self.query_embed = nn.Parameter(torch.zeros(num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    dim,
                    num_heads,
                    dim_feedforward=int(dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, feats: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        B, P, C = feats.shape
        h, w = hw
        assert P == h * w, (P, hw)
        x = feats.float().transpose(1, 2).reshape(B, C, h, w)
        x = F.adaptive_avg_pool2d(x, self.pool_size).flatten(2).transpose(1, 2)  # (B, S, C)
        x = self.in_proj(self.in_norm(x)) + self.pos_embed
        q = self.query_embed[None].expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, x)
        return self.out_proj(self.out_norm(q)).float()


class PromptLearner(nn.Module):
    """Learned delta of PixelDiT's text states at the input of one block.

    hidden_size: PixelDiT hidden size (text stream width).
    text_length: number of text positions (300).
    start, num_tokens: the positions [start, start + num_tokens) that get a delta.
    captioner: optional ImplicitCaptioner (per-image delta).
    """

    def __init__(
        self,
        hidden_size: int,
        text_length: int,
        start: int,
        num_tokens: int,
        captioner: Optional[ImplicitCaptioner] = None,
    ):
        super().__init__()
        if num_tokens < 1 or start < 0 or start + num_tokens > text_length:
            raise ValueError(
                f"Learned span [{start}, {start + num_tokens}) does not fit in the {text_length} "
                "text positions. Use fewer MODEL.PROMPT_LEARNER.NUM_TOKENS or SPAN 'all'."
            )
        self.text_length = int(text_length)
        self.start = int(start)
        self.num_tokens = int(num_tokens)
        self.prompt_tokens = nn.Parameter(torch.zeros(num_tokens, hidden_size))
        self.captioner = captioner
        self._last_residual = None

    @property
    def implicit(self) -> bool:
        return self.captioner is not None

    def forward(self, image_tokens: Optional[torch.Tensor] = None, hw=None) -> torch.Tensor:
        """(1, L, hidden) delta, or (B, L, hidden) with the captioner (image_tokens:
        (B, h*w, hidden) entering the block). fp32, zero outside the span."""
        with torch.autocast(self.prompt_tokens.device.type, enabled=False):
            delta = self.prompt_tokens[None].float()  # (1, N, hidden)
            self._last_residual = None
            if self.captioner is not None:
                residual = self.captioner(image_tokens.detach(), hw)  # (B, N, hidden)
                self._last_residual = residual.detach()
                delta = delta + residual
            pad = (0, 0, self.start, self.text_length - self.start - self.num_tokens)
            return F.pad(delta, pad)  # (b, L, hidden)

    def text_residual(self, layer: int) -> dict:
        """segment_data["text_residual"] entry for PixDiT_T2I.forward."""
        return {"layer": int(layer), "fn": self.forward}

    @torch.no_grad()
    def stats(self) -> dict:
        """RMS of the static delta and of the last implicit delta (for logging)."""
        out = {"prompt_delta_rms": self.prompt_tokens.pow(2).mean().sqrt().item()}
        if self._last_residual is not None:
            out["prompt_residual_rms"] = self._last_residual.pow(2).mean().sqrt().item()
        return out
