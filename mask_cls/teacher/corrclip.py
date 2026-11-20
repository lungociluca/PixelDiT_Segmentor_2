"""CorrCLIP-style dense CLIP features (Zhang et al., arXiv:2411.10086), adapted to GT masks.

CorrCLIP's finding: inter-class correlations (patches attending to patches of OTHER
classes) are the main reason CLIP's dense features are poor; restricting attention
to intra-class / intra-object patches fixes most of it. Its components and how they
map here:

  scope reconstruction  E = Σ m_i ⊗ m_i + (m_0 ⊗ m_0) ⊙ (S > mean S)  (Eq. 7)
      CorrCLIP: SAM2 masks merged with DBSCAN. Here: the GT region masks (Stage 1
      uses GT masks everywhere). A patch belongs to the region covering most of it,
      if regions cover >= `min_cover` of the patch; otherwise it is "unassigned"
      (m_0, mostly ignore pixels) and attends to unassigned patches with
      above-mean similarity (or only itself / all of them).
  value reconstruction  Attn = MaskedSoftmax(S / tau, E), S from DINO ViT-B/8
      q+k features, tau = 0.25 (Eqs. 9-10). Options: "dino", "clip_qq"
      (ClearCLIP's q-q similarity), "uniform" (plain average inside the scope).
  last block            no residual, no FFN (as ClearCLIP)
  spatial branch        + alpha * Proj(Attn · V_C'), V_C' = final-layer value projection
      of lower-layer tokens (Eq. 11, alpha = 1). The paper does not name the layer(s);
      set `spatial_layers` (block indices, negative = from the end).
  semantic branch       + beta * MCT: one mask class token per region, attending only
      to its region's patches in every block, added back to the region's patches
      (Eq. 11, beta = 0.5).
  map correction        (majority vote inside each mask) is implicit: every GT mask gets
      one label by pooling.

The scope comes from GT masks, so teacher *dense* numbers on the eval sets use the
GT partition inside CLIP's attention. That is an oracle setting (as is the GT-mask
pooling); at Stage 2 inference CLIP is not used at all.
"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from ..features.pooling import mask_area_weights

log = logging.getLogger("mask_cls")


class DinoSimilarity:
    """DINO q+k features of the last block -> cosine similarity on the CLIP patch grid."""

    def __init__(self, name: str = "dino_vitb8", device: str = "cuda"):
        self.model = torch.hub.load("facebookresearch/dino:main", name).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        ps = self.model.patch_embed.patch_size
        self.patch = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(self, image01: torch.Tensor, out_hw) -> torch.Tensor:
        """image01 [1, 3, H, W] -> L2-normalised q+k features [h*w, C] on the `out_hw` grid."""
        H, W = image01.shape[-2:]
        p = self.patch
        Hn, Wn = max(p, H // p * p), max(p, W // p * p)
        x = image01 if (Hn, Wn) == (H, W) else F.interpolate(image01, size=(Hn, Wn), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        tok = self.model.prepare_tokens(x)
        for blk in self.model.blocks[:-1]:
            tok = blk(tok)
        last = self.model.blocks[-1]
        y = last.norm1(tok)
        B, N, C = y.shape
        qkv = last.attn.qkv(y).reshape(B, N, 3, C)            # heads concatenated
        f = (qkv[:, 1:, 0] + qkv[:, 1:, 1]).float()             # q + k, drop CLS
        f = f.reshape(1, Hn // p, Wn // p, C).permute(0, 3, 1, 2)
        f = F.interpolate(f, size=tuple(out_hw), mode="area" if (Hn // p) >= out_hw[0] else "bilinear")
        return F.normalize(f.permute(0, 2, 3, 1).reshape(-1, C), dim=-1)


def patch_scope(masks: torch.Tensor, grid_hw, min_cover: float = 0.5):
    """masks [n, H, W] -> (assign [P] region id or -1, weights [n, P])."""
    w = mask_area_weights(masks, grid_hw).flatten(1)            # [n, P]
    total = w.sum(0)
    best = w.argmax(0)
    assign = torch.where(total >= min_cover, best, torch.full_like(best, -1))
    return assign, w


def interaction_mask(assign: torch.Tensor, sim: torch.Tensor, unassigned: str = "above_mean") -> torch.Tensor:
    """E [P, P] bool: True = patch i may attend to patch j (Eq. 7)."""
    a = assign
    E = (a[:, None] == a[None, :]) & (a[:, None] >= 0)
    u = a < 0
    if u.any():
        if unassigned == "all":
            E |= u[:, None] & u[None, :]
        elif unassigned == "above_mean":
            uu = u[:, None] & u[None, :]
            thr = sim[uu].mean()
            E |= uu & (sim > thr)
        elif unassigned != "self":
            raise ValueError(unassigned)
    E |= torch.eye(len(a), dtype=torch.bool, device=a.device)
    return E


def _split_heads(t, nh):
    N, C = t.shape
    return t.reshape(N, nh, C // nh).transpose(0, 1)         # [nh, N, d]


@torch.no_grad()
def corrclip_dense(teacher, x01: torch.Tensor, masks: torch.Tensor, cc, dino: Optional[DinoSimilarity] = None):
    """x01 [1, 3, Hd, Wd] in [0, 1]; masks [n, H, W] (any resolution) -> tokens [gh, gw, D]."""
    if not teacher.is_vit:
        raise NotImplementedError("corrclip mode needs a ViT CLIP (ViT-B-16, ViT-L-14, ...)")
    v = teacher.model.visual
    x = teacher._norm(x01)
    gh, gw = x.shape[-2] // teacher.patch, x.shape[-1] // teacher.patch
    P = gh * gw
    assign, w = patch_scope(masks, (gh, gw), cc.min_cover)
    n = masks.shape[0]
    use_mct = cc.mct_beta > 0 and n > 0
    nblocks = len(v.transformer.resblocks)
    spatial = [l % nblocks for l in (cc.spatial_layers or [])] if cc.spatial_alpha > 0 else []
    if any(l == nblocks - 1 for l in spatial):
        raise ValueError("spatial_layers must be below the last block")

    with teacher._autocast():
        # ---- tokens through all but the last block (+ mask class tokens) -------------
        t = v.conv1(x)
        C = t.shape[1]
        t = t.reshape(1, C, P).permute(0, 2, 1)
        pe = teacher._interp_pos(v.positional_embedding, gh, gw).to(t.dtype)
        cls = v.class_embedding.to(t.dtype).view(1, 1, C)
        t = torch.cat([cls, t], 1) + pe
        N0 = 1 + P
        attn_mask = None
        if use_mct:
            mct = (cls + pe[:1]).expand(1, n, C)
            t = torch.cat([t, mct], 1)
            region_patches = (assign[None, :] == torch.arange(n, device=assign.device)[:, None])   # [n, P]
            empty = ~region_patches.any(1)
            region_patches[empty] = w[empty] > 0                                                  # tiny regions
            N = N0 + n
            blocked = torch.ones(N, N, dtype=torch.bool, device=x.device)                       # True = blocked
            blocked[:N0, :N0] = False                     # image tokens ignore mask tokens (features unchanged)
            blocked[N0:, 1:N0] = ~region_patches           # mask token j -> patches of region j
            idx = torch.arange(N0, N, device=x.device)
            blocked[idx, idx] = False                      # and itself
            # Additive float mask: open_clip casts attn_mask to the activation dtype, which
            # would turn a bool mask into +1/0 instead of blocking.
            attn_mask = torch.zeros(N, N, device=x.device).masked_fill(blocked, float("-inf"))
        t = v.ln_pre(t)
        bf = getattr(v.transformer, "batch_first", False)
        t = t if bf else t.permute(1, 0, 2)
        kept = {}
        for i, blk in enumerate(v.transformer.resblocks[:-1]):
            t = blk(t, attn_mask=attn_mask) if attn_mask is not None else blk(t)
            if i in spatial:
                kept[i] = t if bf else t.permute(1, 0, 2)
        t = t if bf else t.permute(1, 0, 2)                # [1, N, C]

        # ---- last block: scoped attention, no residual / FFN ------------------------
        last = v.transformer.resblocks[-1]
        attn = last.attn
        nh = attn.num_heads
        d = C // nh
        Wq, Wk, Wv = attn.in_proj_weight.chunk(3)
        bq, bk, bv = attn.in_proj_bias.chunk(3) if attn.in_proj_bias is not None else (None, None, None)
        y = last.ln_1(t[0])                                 # [N, C]
        yp = y[1:N0]
        q = _split_heads(F.linear(yp, Wq, bq), nh)          # [nh, P, d]
        val = _split_heads(F.linear(yp, Wv, bv), nh)

        qq = (q @ q.transpose(-2, -1)).float() * d ** -0.5  # [nh, P, P]
        if cc.similarity == "dino":
            if dino is None:
                raise ValueError("similarity=dino needs a DinoSimilarity")
            f = dino.features(x01, (gh, gw))
            S = (f @ f.T)[None]                              # cosine, [1, P, P]
            logits = S / cc.tau
        elif cc.similarity == "clip_qq":
            S = qq.mean(0, keepdim=True)
            logits = qq
        elif cc.similarity == "uniform":
            S = qq.mean(0, keepdim=True)
            logits = torch.zeros_like(S)
        else:
            raise ValueError(cc.similarity)
        E = interaction_mask(assign, S[0], cc.unassigned)
        A = torch.softmax(logits.masked_fill(~E[None], float("-inf")), dim=-1).to(val.dtype)   # [h|1, P, P]

        def project(values):                                # [nh, P, d] -> [P, D]
            o = (A @ values).transpose(0, 1).reshape(P, C)
            o = attn.out_proj(o)
            if hasattr(last, "ls_1"):
                o = last.ls_1(o)
            o = v.ln_post(o)
            return o @ v.proj if v.proj is not None else o

        out = project(val)
        for i in spatial:                                   # spatial branch (lower-layer values)
            yl = last.ln_1(kept[i][0][1:N0])
            out = out + cc.spatial_alpha / len(spatial) * project(_split_heads(F.linear(yl, Wv, bv), nh))

        if use_mct:                                         # semantic branch (mask class tokens)
            qm = _split_heads(F.linear(y[N0:], Wq, bq), nh)                        # [nh, n, d]
            lm = (qm @ q.transpose(-2, -1)).float() * d ** -0.5                    # [nh, n, P]
            Am = torch.softmax(lm.masked_fill(~region_patches[None], float("-inf")), -1).to(val.dtype)
            om = attn.out_proj((Am @ val).transpose(0, 1).reshape(n, C))
            if hasattr(last, "ls_1"):
                om = last.ls_1(om)
            om = v.ln_post(om)
            om = om @ v.proj if v.proj is not None else om                        # [n, D]
            has = assign >= 0
            add = torch.zeros_like(out)
            add[has] = om[assign[has]]
            out = out + cc.mct_beta * add
    return out.float().reshape(gh, gw, -1)
