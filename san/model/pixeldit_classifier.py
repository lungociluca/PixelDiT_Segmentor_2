"""Zero-shot mask classification with PixelDiT's own image-text attention.

Stage 2 of the pipeline, after the side adapter (or GT masks) has produced regions:

  1. Image tokens from PixelDiT layer `input_layer` go through `num_masked_blocks`
     further (frozen) patch blocks run on image tokens only. Their attention is
     restricted so that each token attends to tokens of its own region, which
     pulls the features of different regions apart.
  2. Class prompts ("A picture of a {c}") are encoded once per vocabulary through
     the text stream of blocks 1..readout_block-1 with text-to-text attention only,
     and turned into the readout block's attention keys.
  3. The readout block's image queries (computed on the region-separated tokens)
     are matched against those class keys with the block's own scaled dot product,
     pooled over each class's tokens and the heads -> per-token class scores, then
     averaged over each mask -> (num_masks, num_classes) scores.

Nothing here is trained. PixelDiT is passed in at call time so its weights are not
registered twice in the state dict.
"""
import logging
import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger("san")

READOUT_MODES = ("qk", "concept")
MASK_ATTENTION_MODES = ("soft", "hard", "none")
TOKEN_POOL_MODES = ("lme", "max", "mean")
TEXT_TOKEN_MODES = ("name", "all")


class PixelDiTRegionClassifier(nn.Module):
    def __init__(
        self,
        *,
        readout_block: int,
        num_masked_blocks: int = 1,
        mask_attention: str = "soft",
        soft_eps: float = 1e-4,
        token_pool: str = "lme",
        text_tokens: str = "name",
        prompt_template: str = "A picture of a {target}",
        prompt_prefix: str = "",
        readout: str = "qk",
        concept_num_blocks: int = 14,
        concept_readout_blocks: Optional[List[int]] = None,
        concept_masked_blocks: Optional[List[int]] = None,
        concept_token_softmax: bool = True,
        fp32_residual: bool = True,
    ):
        super().__init__()
        if readout not in READOUT_MODES:
            raise ValueError(f"READOUT must be one of {READOUT_MODES}, got {readout!r}")
        self.readout = readout
        self.concept_num_blocks = int(concept_num_blocks)
        self.concept_readout_blocks = sorted(set(concept_readout_blocks or range(1, self.concept_num_blocks + 1)))
        self.concept_masked_blocks = set(concept_masked_blocks or [])
        self.concept_token_softmax = bool(concept_token_softmax)
        self.fp32_residual = bool(fp32_residual)
        if readout == "concept":
            bad = [b for b in self.concept_readout_blocks + sorted(self.concept_masked_blocks)
                   if not 1 <= b <= self.concept_num_blocks]
            if bad:
                raise ValueError(
                    f"CONCEPT.READOUT_BLOCKS / MASKED_BLOCKS {bad} outside 1..NUM_BLOCKS={self.concept_num_blocks}"
                )
        if mask_attention not in MASK_ATTENTION_MODES:
            raise ValueError(f"MASK_ATTENTION must be one of {MASK_ATTENTION_MODES}, got {mask_attention!r}")
        if token_pool not in TOKEN_POOL_MODES:
            raise ValueError(f"TOKEN_POOL must be one of {TOKEN_POOL_MODES}, got {token_pool!r}")
        if text_tokens not in TEXT_TOKEN_MODES:
            raise ValueError(f"TEXT_TOKENS must be one of {TEXT_TOKEN_MODES}, got {text_tokens!r}")
        if "{target}" not in prompt_template:
            raise ValueError(f"PROMPT_TEMPLATE must contain '{{target}}', got {prompt_template!r}")
        if num_masked_blocks < 0 or readout_block - num_masked_blocks - 1 < 0:
            raise ValueError(
                f"READOUT_BLOCK={readout_block} with NUM_MASKED_BLOCKS={num_masked_blocks} needs "
                f"input layer READOUT_BLOCK - NUM_MASKED_BLOCKS - 1 >= 0."
            )
        self.readout_block = int(readout_block)  # 1-based PixelDiT block number
        self.num_masked_blocks = int(num_masked_blocks)
        self.mask_attention = mask_attention
        self.soft_eps = float(soft_eps)
        self.token_pool = token_pool
        self.text_tokens = text_tokens
        self.prompt_template = prompt_template
        self.prompt_prefix = prompt_prefix  # e.g. the CHI instruction, prepended verbatim

    # ------------------------------------------------------------------ layout
    @property
    def input_layer(self) -> int:
        """PixelDiT layer whose image tokens feed the first masked block
        (0 = patch embedding, k = output of block k). The concept readout runs every
        block itself, starting from the embeddings."""
        if self.readout == "concept":
            return 0
        return self.readout_block - self.num_masked_blocks - 1

    @property
    def deepest_block(self) -> int:
        return self.concept_num_blocks if self.readout == "concept" else self.readout_block

    @property
    def masked_blocks(self) -> List[int]:
        """1-based numbers of the blocks run with region-restricted attention."""
        return list(range(self.input_layer + 1, self.readout_block))

    # --------------------------------------------------------------- vocabulary
    def prompts(self, class_names: List[str]) -> List[str]:
        return [self.prompt_prefix + self.prompt_template.format(target=c) for c in class_names]

    def class_token_mask(self, tokenizer, prompts: List[str], attention_mask: torch.Tensor) -> torch.Tensor:
        """(N, L) bool: which caption tokens represent each class.

        "name": the tokens after the prompt prefix and the template's text before
        the name (e.g. "dog" in "A picture of a dog", plus anything the template puts
        after the name). "all": every real token except BOS (including the prefix). The tokenizer pads on the right, so a prompt's real tokens
        are at positions 0..n-1 of the padded sequence.
        """
        select = torch.zeros_like(attention_mask, dtype=torch.bool)
        prefix = (self.prompt_prefix + self.prompt_template.split("{target}")[0]).rstrip()
        prefix_ids = tokenizer(prefix)["input_ids"] if prefix else []
        bos = getattr(tokenizer, "bos_token_id", None)
        for i, prompt in enumerate(prompts):
            ids = tokenizer(prompt)["input_ids"]
            real = attention_mask[i].nonzero().flatten()
            if len(real) != len(ids) or (len(real) and real[-1].item() != len(real) - 1):
                raise RuntimeError(
                    f"Unexpected tokenization for {prompt!r}: {len(ids)} ids vs {len(real)} real "
                    "positions (expected right padding)."
                )
            start = 1 if (len(ids) and ids[0] == bos) else 0
            if self.text_tokens == "name":
                if prefix_ids and ids[: len(prefix_ids)] == prefix_ids and len(ids) > len(prefix_ids):
                    start = len(prefix_ids)
                elif prefix_ids:
                    logger.warning(
                        "Could not locate the class name in %r; using all its tokens.", prompt
                    )
            select[i, start : len(ids)] = True
        return select

    @torch.no_grad()
    def build_text_keys(
        self,
        core,
        caption_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        token_select: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Readout-block keys of the selected class tokens.

        caption_embs: (N, L, D) text-encoder outputs; attention_mask / token_select: (N, L).
        Returns keys (N, T, heads, head_dim) in fp32 and valid (N, T) bool, T = max
        number of selected tokens.
        """
        y, pos_txt = core.encode_text_only(
            caption_embs, attention_mask, condition, num_blocks=self.readout_block - 1
        )
        keys = core.patch_blocks[self.readout_block - 1].readout_keys(y, condition, pos_txt)
        counts = token_select.sum(1)
        T = int(counts.max().item())
        out = keys.new_zeros(keys.shape[0], T, *keys.shape[2:], dtype=torch.float32)
        valid = torch.zeros(keys.shape[0], T, dtype=torch.bool, device=keys.device)
        for i in range(keys.shape[0]):
            k = keys[i][token_select[i].to(keys.device)]
            out[i, : k.shape[0]] = k.float()
            valid[i, : k.shape[0]] = True
        return out, valid

    # --------------------------------------------------------- region attention
    def attention_bias(self, region_weights: torch.Tensor) -> Optional[torch.Tensor]:
        """Attention mask / bias from soft region memberships.

        region_weights: (L, K) non-negative weight of token i in region k (rows may
        be all zero: e.g. unlabeled pixels for GT masks).
          soft: log(P Pᵀ + eps) with P = row-normalized weights, i.e. the log
                probability that two tokens belong to the same region. Tokens with
                no region get a constant row (= unrestricted attention).
          hard: token i attends to j iff argmax regions match; tokens with no region
                attend everywhere.
          none: no restriction.
        Returns (1, 1, L, L) float bias, bool mask, or None.
        """
        if self.mask_attention == "none":
            return None
        w = region_weights.float()
        has_region = w.sum(1) > 0
        if self.mask_attention == "hard":
            label = torch.where(has_region, w.argmax(1), torch.full_like(has_region, -1, dtype=torch.long))
            allowed = (label[:, None] == label[None, :]) | (label[:, None] < 0)
            return allowed[None, None]
        p = w / w.sum(1, keepdim=True).clamp(min=1e-12)
        affinity = p @ p.T  # (L, L), in [0, 1]
        return torch.log(affinity + self.soft_eps)[None, None]

    # ------------------------------------------------------------------ scoring
    def token_class_scores(self, queries: torch.Tensor, keys: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """queries (L, H, d); keys (C, T, H, d); valid (C, T) -> scores (L, C).

        Per head: q·k/sqrt(d) (the block's attention logits), pooled over each class's
        tokens ("lme" = log-mean-exp, i.e. logsumexp without favouring classes with
        more tokens; "max"; "mean"), then averaged over heads.
        """
        L, H, d = queries.shape
        C, T = valid.shape
        q = queries.float() * (1.0 / math.sqrt(d))
        k = keys.float()
        n_valid = valid.sum(1).clamp(min=1).float()  # (C,)
        chunk = max(1, int(2 ** 24 // max(1, H * C * T)))
        out = []
        for s in range(0, L, chunk):
            logits = torch.einsum("lhd,cthd->lhct", q[s : s + chunk], k)  # (l, H, C, T)
            if self.token_pool == "mean":
                pooled = (logits * valid).sum(-1) / n_valid
            else:
                logits = logits.masked_fill(~valid, float("-inf"))
                if self.token_pool == "max":
                    pooled = logits.amax(-1)
                else:
                    pooled = torch.logsumexp(logits, dim=-1) - torch.log(n_valid)
            out.append(pooled.mean(1))  # (l, C)
        return torch.cat(out, 0)

    def forward(
        self,
        core,
        image_tokens: torch.Tensor,
        condition: torch.Tensor,
        pos: torch.Tensor,
        region_weights: torch.Tensor,
        pool_weights: torch.Tensor,
        text_keys: torch.Tensor,
        text_valid: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Class scores for one image.

        image_tokens:   (1, L, hidden) PixelDiT tokens at `input_layer`
        condition, pos: from the stage-1 PixelDiT run (segment_data)
        region_weights: (L, K) region membership used to restrict attention
        pool_weights:   (K, L) weights for averaging token scores within each mask
        text_keys, text_valid: from build_text_keys for the current vocabulary
        token_valid:    (L,) bool, tokens inside the image (not padding); all if None
        Returns (K, C) mask scores (not normalized) and (C,) the mean token score over
        the image's valid tokens. Since pooling is a weighted average, subtracting a
        per-class vector from the mask scores equals subtracting it from every token
        score: scores - image_mean is the region-vs-image contrast.
        """
        x = image_tokens
        bias = self.attention_bias(region_weights) if self.masked_blocks else None
        for b in self.masked_blocks:
            x = core.patch_blocks[b - 1].forward_image_only(x, condition, pos, bias)
        queries = core.patch_blocks[self.readout_block - 1].readout_queries(x, condition, pos)[0]
        token_scores = self.token_class_scores(queries, text_keys, text_valid)  # (L, C)
        w = pool_weights.float()
        scores = (w @ token_scores) / w.sum(1, keepdim=True).clamp(min=1e-6)
        image_mean = (token_scores if token_valid is None else token_scores[token_valid]).mean(0)
        return scores, image_mean

    # ---------------------------------------------------------- concept readout
    def _pool_class_tokens(self, logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """logits (..., C, T), valid (C, T) -> (..., C), pooled over each class's tokens."""
        if self.token_pool == "mean":
            return (logits * valid).sum(-1) / valid.sum(1).clamp(min=1)
        logits = logits.masked_fill(~valid, float("-inf"))
        if self.token_pool == "max":
            return logits.amax(-1)
        return torch.logsumexp(logits, dim=-1) - torch.log(valid.sum(1).clamp(min=1).float())

    @torch.no_grad()
    def build_concepts(self, core, caption_embs, token_select):
        """Initial concept tokens for a vocabulary: the selected (class-name) tokens of
        each class prompt, embedded as PixelDiT embeds prompt tokens (y_embedder +
        y_pos_embedding at the token's position in its prompt).

        caption_embs: (N, L, D) text-encoder outputs; token_select: (N, L) bool.
        Returns tokens (Nt, hidden) fp32, positions (Nt,) long, and the per-class
        token index (N, T) long with validity (N, T) bool, T = max tokens per class.
        """
        weight = core.y_embedder.proj.weight
        tokens, positions, index = [], [], []
        start = 0
        for i in range(caption_embs.shape[0]):
            pos = token_select[i].nonzero().flatten().to(weight.device)
            emb = core.y_embedder(caption_embs[i, pos].to(weight.device, weight.dtype)[None])[0]
            emb = emb + core.y_pos_embedding[0, pos].to(emb.dtype)
            tokens.append(emb.float())
            positions.append(pos)
            index.append(torch.arange(start, start + len(pos), device=weight.device))
            start += len(pos)
        T = max(len(ix) for ix in index)
        idx = torch.zeros(len(index), T, dtype=torch.long, device=weight.device)
        valid = torch.zeros(len(index), T, dtype=torch.bool, device=weight.device)
        for i, ix in enumerate(index):
            idx[i, : len(ix)] = ix
            valid[i, : len(ix)] = True
        return torch.cat(tokens), torch.cat(positions), idx, valid

    def _concept_token_scores(self, o_x, o_c, idx, valid):
        """o_x (L, D), o_c (Nt, D) pre-projection attention outputs -> (L, C):
        output-space dot products pooled over each class's tokens, then (optionally)
        a softmax over classes per image token, as in ConceptAttention."""
        dots = o_x.float() @ o_c.float().T  # (L, Nt)
        scores = self._pool_class_tokens(dots[:, idx], valid)  # (L, C)
        return F.softmax(scores, dim=-1) if self.concept_token_softmax else scores

    def concept_scores(
        self,
        core,
        s0: torch.Tensor,
        y0: torch.Tensor,
        condition: torch.Tensor,
        pos: torch.Tensor,
        concepts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        region_weights: torch.Tensor,
        pool_weights: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
        debug: bool = False,
    ):
        """ConceptAttention readout for one image.

        s0, y0: (1, L, hidden) / (1, Ltxt, hidden) image and prompt tokens after the
        embeddings (PixelDiT layer 0); concepts: from build_concepts. Runs blocks
        1..concept_num_blocks; in concept_masked_blocks, image-to-image attention is
        biased by the regions (attention_bias), image-to-prompt attention is not.
        Returns [(block, (K, C) mask scores, (C,) image-mean token score)] for every
        readout block, plus ("avg", ...) for the token scores averaged over them.
        With debug, self.last_debug holds per-block activation statistics.
        """
        self.last_debug = [] if debug else None
        cpt0, cpt_pos, idx, valid = concepts
        Ny, L = y0.shape[1], s0.shape[1]
        pos_y = pos_c = None
        if core.use_text_rope:
            table = core.fetch_pos_text(max(Ny, int(cpt_pos.max()) + 1), s0.device)
            pos_y, pos_c = table[:Ny], table[cpt_pos.to(s0.device)]
        joint_bias = None
        if self.concept_masked_blocks and self.mask_attention != "none":
            region_bias = self.attention_bias(region_weights)[0, 0]  # (L, L)
            if region_bias.dtype == torch.bool:
                region_bias = torch.zeros_like(region_bias, dtype=torch.float32).masked_fill(
                    ~region_bias, float("-inf")
                )
            joint_bias = torch.zeros(1, 1, Ny + L, Ny + L, device=s0.device)
            joint_bias[0, 0, Ny:, Ny:] = region_bias

        w = pool_weights.float()
        w = w / w.sum(1, keepdim=True).clamp(min=1e-6)

        def pool(token_scores):
            image_mean = (token_scores if token_valid is None else token_scores[token_valid]).mean(0)
            return w @ token_scores, image_mean

        dtype = torch.float32 if self.fp32_residual else s0.dtype
        x, y, cpt = s0.to(dtype), y0.to(dtype), cpt0[None].to(dtype)
        results, acc = [], 0.0
        for i in range(self.concept_num_blocks):
            b = i + 1
            bias = joint_bias if b in self.concept_masked_blocks else None
            x, y, cpt, o_x, o_c = core.patch_blocks[i].forward_with_concepts(
                x, y, cpt, condition, pos, pos_y, pos_c, bias
            )
            if debug:
                self.last_debug.append(
                    (b, x.dtype, *[(t.float().abs().max().item(), bool(torch.isfinite(t).all()))
                                   for t in (x, y, cpt, o_x, o_c)])
                )
            if b in self.concept_readout_blocks:
                ts = self._concept_token_scores(o_x[0], o_c[0], idx, valid)
                results.append((b, *pool(ts)))
                acc = acc + ts
        if len(self.concept_readout_blocks) > 1:
            results.append(("avg", *pool(acc / len(self.concept_readout_blocks))))
        return results
