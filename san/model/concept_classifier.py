"""Open-vocabulary mask classification from PixelDiT's own attention (ConceptAttention).

The class names of a vocabulary are encoded once (Gemma -> PixelDiT text embedding)
into "concept" tokens. During the PixelDiT pass they form an extra stream that runs
through the patch blocks next to the image and prompt tokens: concepts attend to the
image tokens and to each other with the text-stream weights, while image and prompt
never attend to them, so the pass itself is unchanged (Helbling et al., ICML 2025).

(The pooling described next is POOLING "probs". With "scores", the default, each
mask averages the standardised scores and the softmax over classes comes after.)

At each readout block, every image token is scored against every class by the dot
product of the two attention outputs (before the output projection), averaged over
the class's tokens, and turned into class probabilities by a softmax over classes.
The per-block probabilities are averaged with (learnable) block weights, and each
predicted mask averages them over its tokens: p(class | mask). Together with the
side adapter's objectness this gives Mask2Former-style C + 1 logits:

    logits = [log(p_obj * p(c | mask)) for each class c,  log(1 - p_obj)]

Before the per-token softmax, each token's class scores are standardised across
classes (score_norm "zscore"): the raw dot products differ by tens to hundreds
between classes, which makes the softmax one-hot per token, gives correct classes
no token votes for a probability of ~0 and stops the gradients. Standardising
keeps every token's class ranking (so the zero-shot readout is unchanged in the
argmax sense) and leaves the sharpness to the learnable temperature.

Trainable here: block weights, the temperature, and optionally a linear adapter on
the image attention outputs (identity at init). Everything on the
text side stays frozen, so new vocabularies can be swapped in at test time. With the
initial parameters this is exactly the zero-shot readout (uniform block average,
temperature 1).
"""
import logging
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger("san")


class Concepts(NamedTuple):
    """Concept tokens of one vocabulary (C classes, Nt tokens in total)."""
    tokens: torch.Tensor     # (Nt, hidden) embedded tokens, fp32
    positions: torch.Tensor  # (Nt,) token positions in their prompt (for the text RoPE)
    pool: torch.Tensor       # (Nt, C) averages each class's tokens: scores @ pool


class ConceptClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        readout_blocks: List[int],
        prompt_template: str = "A picture of a {target}",
        prompt_prefix: str = "",
        learn_block_weights: bool = True,
        learn_temperature: bool = True,
        image_adapter: bool = False,
        detach_masks: bool = False,
        mask_pool_threshold: float = 0.5,
        fp32_residual: bool = True,
        score_norm: str = "zscore",
        init_temperature: float = 1.0,
        pooling: str = "probs",
    ):
        super().__init__()
        self.readout_blocks = sorted(set(int(b) for b in readout_blocks))
        if not self.readout_blocks or self.readout_blocks[0] < 1:
            raise ValueError(f"READOUT_BLOCKS must be 1-based block numbers, got {readout_blocks}")
        if "{target}" not in prompt_template:
            raise ValueError(f"PROMPT_TEMPLATE must contain '{{target}}', got {prompt_template!r}")
        self.prompt_template = prompt_template
        self.prompt_prefix = prompt_prefix
        self.detach_masks = bool(detach_masks)
        self.mask_pool_threshold = float(mask_pool_threshold)
        self.fp32_residual = bool(fp32_residual)
        if score_norm not in ("zscore", "none"):
            raise ValueError(f"SCORE_NORM must be 'zscore' or 'none', got {score_norm!r}")
        self.score_norm = score_norm
        if pooling not in ("probs", "scores"):
            raise ValueError(f"POOLING must be 'probs' or 'scores', got {pooling!r}")
        self.pooling = pooling
        if init_temperature <= 0:
            raise ValueError(f"INIT_TEMPERATURE must be > 0, got {init_temperature}")
        log_t0 = torch.tensor(float(init_temperature)).log()
        # Training statistics of the last token_outputs call (see stats()).
        self._last_raw_std = None

        n = len(self.readout_blocks)
        if learn_block_weights:
            self.block_logits = nn.Parameter(torch.zeros(n))
        else:
            self.register_buffer("block_logits", torch.zeros(n), persistent=False)
        if learn_temperature:
            self.log_temperature = nn.Parameter(log_t0.clone())
        else:
            self.register_buffer("log_temperature", log_t0.clone(), persistent=False)
        self.image_adapter = None
        if image_adapter:
            self.image_adapter = nn.Linear(hidden_size, hidden_size)
            nn.init.eye_(self.image_adapter.weight)
            nn.init.zeros_(self.image_adapter.bias)

    @property
    def num_blocks(self) -> int:
        """PixelDiT blocks the concept stream has to run through."""
        return self.readout_blocks[-1]

    # --------------------------------------------------------------- vocabulary
    def prompts(self, class_names: List[str]) -> List[str]:
        return [self.prompt_prefix + self.prompt_template.format(target=c) for c in class_names]

    def name_token_mask(self, tokenizer, prompts: List[str], attention_mask: torch.Tensor) -> torch.Tensor:
        """(N, L) bool: the tokens of each prompt after the prefix and the template's
        text before the class name (e.g. "dog" in "A picture of a dog"). The
        tokenizer pads on the right, so a prompt's tokens are at positions 0..n-1."""
        select = torch.zeros_like(attention_mask, dtype=torch.bool)
        prefix = (self.prompt_prefix + self.prompt_template.split("{target}")[0]).rstrip()
        prefix_ids = tokenizer(prefix)["input_ids"] if prefix else []
        bos = getattr(tokenizer, "bos_token_id", None)
        for i, prompt in enumerate(prompts):
            ids = tokenizer(prompt)["input_ids"]
            n_real = int(attention_mask[i].sum())
            if n_real != len(ids) or not bool(attention_mask[i, :n_real].all()):
                raise RuntimeError(
                    f"Unexpected tokenization for {prompt!r}: {len(ids)} ids vs {n_real} real "
                    "positions (expected right padding, no truncation)."
                )
            start = 1 if (ids and ids[0] == bos) else 0
            if prefix_ids and ids[: len(prefix_ids)] == prefix_ids and len(ids) > len(prefix_ids):
                start = len(prefix_ids)
            elif prefix_ids:
                logger.warning("Could not locate the class name in %r; using all its tokens.", prompt)
            select[i, start : len(ids)] = True
        return select

    @torch.no_grad()
    def build_concepts(self, core, caption_embs: torch.Tensor, token_select: torch.Tensor) -> Concepts:
        """caption_embs: (C, L, D) text-encoder outputs of the class prompts;
        token_select: (C, L) bool. The selected tokens are embedded as PixelDiT embeds
        prompt tokens (y_embedder + y_pos_embedding at the token's position)."""
        weight = core.y_embedder.proj.weight
        device = weight.device
        tokens, positions, owner = [], [], []
        for c in range(caption_embs.shape[0]):
            pos = token_select[c].nonzero().flatten().to(device)
            emb = core.y_embedder(caption_embs[c, pos].to(device, weight.dtype)[None])[0]
            emb = emb + core.y_pos_embedding[0, pos].to(emb.dtype)
            tokens.append(emb.float())
            positions.append(pos)
            owner.append(torch.full((len(pos),), c, device=device))
        owner = torch.cat(owner)
        pool = F.one_hot(owner, caption_embs.shape[0]).float()
        pool = pool / pool.sum(0, keepdim=True)
        return Concepts(torch.cat(tokens), torch.cat(positions), pool)

    # ------------------------------------------------------------ PixelDiT pass
    @torch.no_grad()
    def run_backbone(
        self,
        core,
        s0: torch.Tensor,
        y0: torch.Tensor,
        condition: torch.Tensor,
        pos: torch.Tensor,
        concepts: Optional[Concepts],
        feature_layers: List[int],
    ) -> Tuple[Dict[int, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run PixelDiT's patch blocks from the embeddings (layer 0), with the concept
        stream up to the last readout block.

        s0, y0: (B, L, hidden) image / (B, Ltxt, hidden) prompt tokens at layer 0.
        Returns the image tokens at feature_layers ({k: (B, L, hidden)}) and, for each
        readout block, the pre-projection attention outputs (o_x (B, L, D),
        o_c (B, Nt, D)); the latter is empty when concepts is None.
        """
        Ny = y0.shape[1]
        dtype = torch.float32 if self.fp32_residual else s0.dtype
        x, y = s0.to(dtype), y0.to(dtype)
        pos_y = core.fetch_pos_text(Ny, s0.device) if core.use_text_rope else None
        cpt = pos_c = None
        last = max(feature_layers) if feature_layers else 0
        if concepts is not None:
            cpt = concepts.tokens[None].to(dtype)
            if core.use_text_rope:
                n = max(Ny, int(concepts.positions.max()) + 1)
                pos_c = core.fetch_pos_text(n, s0.device)[concepts.positions]
            last = max(last, self.num_blocks)
        if last > len(core.patch_blocks):
            raise ValueError(f"Need PixelDiT block {last}, but only {len(core.patch_blocks)} are built.")

        features = {0: s0} if 0 in feature_layers else {}
        readout = []
        for i in range(last):
            b = i + 1
            block = core.patch_blocks[i]
            if cpt is not None and b <= self.num_blocks:
                x, y, cpt, o_x, o_c = block.forward_with_concepts(x, y, cpt, condition, pos, pos_y, pos_c)
                if b in self.readout_blocks:
                    readout.append((o_x, o_c))
            else:
                x, y = block(x, y, condition, pos, pos_y)
            if b in feature_layers:
                features[b] = x
        return features, readout

    # --------------------------------------------------------------------- head
    def token_outputs(self, readout, concepts: Concepts) -> torch.Tensor:
        """(B, L, C) per-token class outputs, block-weighted over the readout blocks.
        Per block, the output-space dot products are averaged over each class's
        tokens and (score_norm "zscore") standardised across classes. Then
          pooling "probs":  softmax(temperature * scores) per token (probabilities);
          pooling "scores": the standardised scores themselves; the softmax comes
                            after averaging them over a mask (mask_class_logits)."""
        weights = F.softmax(self.block_logits.float(), dim=0)
        scale = self.log_temperature.float().exp()
        out = 0.0
        raw_std = []
        for w, (o_x, o_c) in zip(weights, readout):
            o_x = o_x.float()
            if self.image_adapter is not None:
                o_x = self.image_adapter(o_x)
            scores = torch.einsum("bld,bnd->bln", o_x, o_c.float()) @ concepts.pool  # (B, L, C)
            std = scores.std(dim=-1, keepdim=True)
            raw_std.append(std.detach().mean())
            if self.score_norm == "zscore":
                scores = (scores - scores.mean(dim=-1, keepdim=True)) / std.clamp(min=1e-6)
            if self.pooling == "probs":
                scores = F.softmax(scale * scores, dim=-1)
            out = out + w * scores
        self._last_raw_std = torch.stack(raw_std)
        return out

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        """Head parameters and the raw score spread of the last token_outputs call."""
        out = {"temperature": self.log_temperature.float().exp().item()}
        for b, w in zip(self.readout_blocks, F.softmax(self.block_logits.float(), dim=0)):
            out[f"block_weight_{b}"] = w.item()
        if self._last_raw_std is not None:
            for b, s in zip(self.readout_blocks, self._last_raw_std):
                out[f"raw_score_std_{b}"] = s.item()
        return out

    def mask_class_logits(
        self, token_outputs: torch.Tensor, mask_preds: torch.Tensor, objectness_logits: torch.Tensor
    ) -> torch.Tensor:
        """C + 1 logits (log-probabilities) for each mask.

        token_outputs:     (B, L, C) from token_outputs
        mask_preds:        (B, Q, h, w) mask logits on PixelDiT's token grid (h * w = L)
        objectness_logits: (B, Q, 2) side adapter [object, no object] logits
        Each mask averages the token outputs over its tokens, weighted by its sigmoid;
        with mask_pool_threshold > 0 only tokens above it count (all tokens, if none
        is), so the many near-zero background tokens cannot outweigh a small object.
        pooling "probs": p(c | mask) = the averaged probabilities. pooling "scores":
        p(c | mask) = softmax(temperature * averaged scores): token noise cancels in
        the average while the class signal adds up, giving sharper distributions.
        """
        B, Q = mask_preds.shape[:2]
        if mask_preds.shape[-2] * mask_preds.shape[-1] != token_outputs.shape[1]:
            raise ValueError(
                f"Mask grid {tuple(mask_preds.shape[-2:])} does not match the "
                f"{token_outputs.shape[1]} PixelDiT tokens."
            )
        m = mask_preds.float().sigmoid().flatten(2)  # (B, Q, L)
        if self.detach_masks:
            m = m.detach()
        if self.mask_pool_threshold > 0:
            keep = m > self.mask_pool_threshold
            m = torch.where(keep.any(-1, keepdim=True), m * keep, m)
        w = m / m.sum(-1, keepdim=True).clamp(min=1e-6)
        pooled = w @ token_outputs.float()  # (B, Q, C)
        if self.pooling == "scores":
            log_p_cls = F.log_softmax(self.log_temperature.float().exp() * pooled, dim=-1)
        else:
            log_p_cls = torch.log(pooled.clamp(min=1e-8))
        lo = F.log_softmax(objectness_logits.float(), dim=-1)
        return torch.cat([lo[..., :1] + log_p_cls, lo[..., 1:]], dim=-1)
