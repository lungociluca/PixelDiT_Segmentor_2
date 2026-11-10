#!/usr/bin/env python
"""
Vision-language correlation analysis of the frozen PixelDiT backbone on PASCAL VOC 2012.

For every image we run ONE forward pass with a batch of 21 prompts, one per VOC class
(background included), built from `--prompt-format` (default: t2i.config.prompt_format,
"A picture of a {target}"). The image and the timestep (t2i.config.timestep) are the
same for every item of the batch; only the text differs. The image is NOT noised, as in
the current segmentation pipeline.

Hooks on every joint-attention block record, for each class prompt, one map over the
patch grid (32x32 at 512 px) saying how strongly each patch relates to the class-name
tokens of that prompt:

  img2txt_attn         image queries -> class-name keys, softmax over the 300 text keys (per head)
  txt2img_attn         class-name queries -> image keys, softmax over the image keys (per head)
  attn_out_cos         cosine(image attention output, class-token attention output) (per head)
  proj_out_cos         same, after the attention output projection (all heads)
  block_out_cos        cosine(image tokens, class tokens) in the residual stream after the block
  block_out_cos_text0  cosine(image tokens after the block, class tokens entering block 0):
                       a text-only baseline, the class tokens have not seen the image yet
  *_ctrl               img2txt / txt2img computed on a control word of the template
                       ("picture") instead of the class name: what any token gives you

Per-head map types have 25 "heads": the 24 heads and their mean (index 24).

Metrics, each for every (map type, layer, head):
  loc_auroc     AUROC of a class map against the GT patch mask, for each class present
                in the image (class-balanced mean over foreground classes)
  miou_oracle   per-patch argmax over the classes present in the image (min-max
                normalised maps), mIoU over the 21 classes and over the 20 foreground ones
  miou_open     per-patch argmax over all 21 classes
  recog_auroc   image-level present/absent AUROC per class, scoring each map by the
                mean of its top `--top-frac` patches

The sampled images are split in two halves: the best (layer, head) of every metric is
chosen on the first half and reported on the second, so the 14 x 25 search does not
inflate the numbers.

Run from anywhere (the script switches to the repo root, where the checkpoint lives):
  python analysis/vl_correlation.py --voc-root ../datasets/VOC2012 --num-images 300

The VOC root must contain JPEGImages/, SegmentationClassAug/ and val.txt (or
ImageSets/Segmentation/val.txt).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import t2i.config as local_config  # noqa: E402
from t2i.pixdit_core.modules import apply_rotary_emb  # noqa: E402

# Index = VOC label id in SegmentationClassAug. Natural names, not the VOC ids
# ("diningtable", "pottedplant", "tvmonitor"), since they are fed to the text encoder.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant",
    "sheep", "sofa", "train", "tv monitor",
]
IGNORE_LABEL = 255
TEXT_LEN = 300  # PixelDiT text length (txt_max_length / model_max_length)

PER_HEAD_TYPES = ["img2txt_attn", "txt2img_attn", "attn_out_cos", "img2txt_attn_ctrl", "txt2img_attn_ctrl"]
FULL_TYPES = ["proj_out_cos", "block_out_cos", "block_out_cos_text0"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voc-root", required=True, help="VOC2012 folder (JPEGImages, SegmentationClassAug, val.txt)")
    p.add_argument("--split-file", default=None, help="image id list; default: val.txt in the VOC root")
    p.add_argument("--num-images", type=int, default=300)
    p.add_argument("--seed", type=int, default=0, help="seed for sampling the images")
    p.add_argument("--image-size", type=int, default=local_config.image_size)
    p.add_argument("--num-layers", type=int, default=14, help="patch blocks to build and analyse (PixelDiT has 14)")
    p.add_argument("--prompt-format", default=local_config.prompt_format)
    p.add_argument("--control-word", default="picture", help="template word used as the control token span")
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--purity", type=float, default=0.5, help="min class fraction for a patch to get a hard label")
    p.add_argument("--top-frac", type=float, default=0.05, help="fraction of patches pooled for recognition scores")
    p.add_argument("--out", default="output/vl_correlation")
    return p.parse_args()


# ----------------------------------------------------------------------------- data
def load_image_ids(args):
    root = Path(args.voc_root)
    candidates = [Path(args.split_file)] if args.split_file else [
        root / "val.txt", root / "ImageSets" / "Segmentation" / "val.txt"]
    for c in candidates:
        if c.is_file():
            ids = [l.split()[0] for l in c.read_text().splitlines() if l.strip()]
            break
    else:
        raise FileNotFoundError(f"No split file among {candidates}")
    ids = [i for i in ids if (root / "SegmentationClassAug" / f"{i}.png").is_file()]
    n = min(args.num_images, len(ids))
    return random.Random(args.seed).sample(ids, n)


def load_sample(root, image_id, size):
    """Image in [-1, 1] (as in SAN.forward) and GT label map, both resized to size x size."""
    img = Image.open(root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
    gt = Image.open(root / "SegmentationClassAug" / f"{image_id}.png")
    img = np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.float32)
    gt = np.asarray(gt.resize((size, size), Image.NEAREST), dtype=np.int64)
    if gt.ndim != 2:
        raise ValueError(f"{image_id}: GT must be a label-index PNG, got shape {gt.shape}")
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0 * 2 - 1
    return img, torch.from_numpy(gt)


def patch_labels(gt, num_classes, patch, purity):
    """Hard label per patch (-1 = ignore) and the set of classes present at pixel level."""
    g = gt.clone()
    g[(g < 0) | (g >= num_classes)] = num_classes  # 255 -> extra "ignore" channel
    onehot = F.one_hot(g, num_classes + 1).permute(2, 0, 1).float()[None]
    frac = F.avg_pool2d(onehot, patch)[0].flatten(1)[:num_classes]  # (C, N)
    max_frac, lab = frac.max(0)
    lab[max_frac < purity] = -1
    present = torch.zeros(num_classes, dtype=torch.bool)
    present[g[g < num_classes].unique()] = True
    return lab, present


# ----------------------------------------------------------------------------- text
def encode_prompts(tokenizer, text_encoder, prompt_format, class_names, control_word, device):
    """Text embeddings (C, 300, D) and the token positions of each class name / control word.

    Embedding follows SAN.embed_condition (no CHI window): first 300 positions, right padding.
    """
    if "{target}" not in prompt_format:
        raise ValueError("--prompt-format must contain {target}")
    prefix, suffix = prompt_format.split("{target}", 1)
    prompts = [prefix + name + suffix for name in class_names]
    tok = tokenizer(prompts, max_length=TEXT_LEN, padding="max_length", truncation=True,
                    return_tensors="pt", return_offsets_mapping=True)
    offsets = tok.pop("offset_mapping")
    ids, attn = tok.input_ids.to(device), tok.attention_mask.to(device)
    with torch.no_grad():
        embs = torch.cat([text_encoder(ids[[i]], attn[[i]])[0] for i in range(len(prompts))], 0)

    def span(i, cs, ce):
        return [j for j, (s, e) in enumerate(offsets[i].tolist())
                if attn[i, j] and e > s and s < ce and e > cs]

    spans, ctrl_spans, info = [], [], []
    for i, name in enumerate(class_names):
        cs = len(prefix)
        sp = span(i, cs, cs + len(name))
        if not sp:
            raise RuntimeError(f"Could not locate the tokens of '{name}' in '{prompts[i]}'")
        spans.append(sp)
        cw = prompts[i].find(control_word) if control_word else -1
        if cw >= 0 and not (cs <= cw < cs + len(name)):
            ctrl_spans.append(span(i, cw, cw + len(control_word)))
        else:
            ctrl_spans.append(None)
        info.append({
            "class": name, "prompt": prompts[i], "positions": sp,
            "tokens": tokenizer.convert_ids_to_tokens(tok.input_ids[i, sp].tolist()),
            "control_positions": ctrl_spans[-1],
            "num_tokens": int(attn[i].sum()),
        })
    if any(c is None or len(c) == 0 for c in ctrl_spans):
        ctrl_spans = None
    return embs, spans, ctrl_spans, info


# ----------------------------------------------------------------------------- hooks
class MapCollector:
    """Turns the activations of every joint-attention block into per-class patch maps."""

    def __init__(self, num_layers, num_heads, num_classes, num_patches, spans, ctrl_spans, device):
        self.L, self.H = num_layers, num_heads
        self.spans = [torch.tensor(s, device=device) for s in spans]
        self.ctrl = [torch.tensor(s, device=device) for s in ctrl_spans] if ctrl_spans else None
        self.types = [t for t in PER_HEAD_TYPES if self.ctrl is not None or not t.endswith("_ctrl")] + FULL_TYPES
        self.maps = {
            t: torch.zeros(num_layers, num_heads + 1 if t in PER_HEAD_TYPES else 1, num_classes, num_patches,
                           device=device)
            for t in self.types
        }
        self.text0 = None

    def register(self, core):
        handles = []
        for blk in core.patch_blocks[: self.L]:
            handles.append(blk.register_forward_pre_hook(self._block_pre_hook))
            handles.append(blk.register_forward_hook(self._block_hook))
            handles.append(blk.attn.register_forward_hook(self._attn_hook))
        return handles

    def finalize(self):
        for t in PER_HEAD_TYPES:
            if t in self.maps:
                self.maps[t][:, self.H] = self.maps[t][:, : self.H].mean(1)
        return self.maps

    @torch.no_grad()
    def _block_pre_hook(self, module, args):
        if module.block_id == 0:
            y = args[1]  # text tokens entering the first block: projected Gemma features only
            self.text0 = torch.stack([y[b, s].float().mean(0) for b, s in enumerate(self.spans)])

    @torch.no_grad()
    def _block_hook(self, module, args, output):
        l = module.block_id
        if l >= self.L:
            return
        x, y = output[0].float(), output[1].float()
        for b, s in enumerate(self.spans):  # cosine_similarity runs in fp32 under autocast
            self.maps["block_out_cos"][l, 0, b] = F.cosine_similarity(x[b], y[b, s].mean(0, keepdim=True), dim=-1)
            self.maps["block_out_cos_text0"][l, 0, b] = F.cosine_similarity(x[b], self.text0[b][None], dim=-1)

    @torch.no_grad()
    def _attn_hook(self, module, args, output):
        """Recomputes q/k/v exactly as MMDiTJointAttention.forward does (it is called positionally)."""
        l = module.block_id
        if l >= self.L:
            return
        x, y, pos_img = args[0], args[1], args[2]
        pos_txt = args[3] if len(args) > 3 else None
        B, Nx, D = x.shape
        Ny = y.shape[1]
        H = module.num_heads
        d = D // H

        qx, kx, vx = module.qkv_x(x).reshape(B, Nx, 3, H, d).unbind(2)
        qy, ky, vy = module.qkv_y(y).reshape(B, Ny, 3, H, d).unbind(2)
        qx, kx = module.q_norm_x(qx), module.k_norm_x(kx)
        qy, ky = module.q_norm_y(qy), module.k_norm_y(ky)
        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)
        qx, kx, vx, qy, ky, vy = (t.transpose(1, 2) for t in (qx, kx, vx, qy, ky, vy))  # (B, H, N, d)

        # Attention outputs per head, before the output projection (no mask, as in the model).
        out = F.scaled_dot_product_attention(torch.cat([qy, qx], 2), torch.cat([ky, kx], 2), torch.cat([vy, vx], 2))
        oy, ox = out[:, :, :Ny].float(), out[:, :, Ny:].float()
        proj_x, proj_y = output[0].float(), output[1].float()
        scale = d ** -0.5
        with torch.autocast("cuda", enabled=False):  # maps in fp32, the forward itself stays in AMP
            self._attn_maps(l, H, scale, qx, kx, qy, ky, ox, oy, proj_x, proj_y)

    def _attn_maps(self, l, H, scale, qx, kx, qy, ky, ox, oy, proj_x, proj_y):
        for b, s in enumerate(self.spans):
            # Image -> text: share of each patch's text attention landing on the class tokens.
            p_it = torch.softmax(qx[b].float() @ ky[b].float().transpose(1, 2) * scale, dim=-1)  # (H, Nx, Ny)
            self.maps["img2txt_attn"][l, :H, b] = p_it[:, :, s].sum(-1)
            # Text -> image: where the class tokens look, averaged over the class tokens.
            p_ti = torch.softmax(qy[b][:, s].float() @ kx[b].float().transpose(1, 2) * scale, dim=-1)
            self.maps["txt2img_attn"][l, :H, b] = p_ti.mean(1)
            # Output space, per head and after projection.
            self.maps["attn_out_cos"][l, :H, b] = F.cosine_similarity(ox[b], oy[b][:, s].mean(1)[:, None], dim=-1)
            self.maps["proj_out_cos"][l, 0, b] = F.cosine_similarity(
                proj_x[b], proj_y[b, s].mean(0, keepdim=True), dim=-1)
            if self.ctrl is not None:
                c = self.ctrl[b]
                self.maps["img2txt_attn_ctrl"][l, :H, b] = p_it[:, :, c].sum(-1)
                p_ci = torch.softmax(qy[b][:, c].float() @ kx[b].float().transpose(1, 2) * scale, dim=-1)
                self.maps["txt2img_attn_ctrl"][l, :H, b] = p_ci.mean(1)


# ----------------------------------------------------------------------------- metrics
@torch.no_grad()
def image_metrics(maps, lab, present_px, top_frac):
    """maps: (L, Hh, C, N); lab: (N,) hard patch labels with -1 = ignore."""
    L, Hh, C, N = maps.shape
    valid = lab >= 0
    labv = lab[valid]
    res = {"loc": []}

    # Localisation AUROC (Mann-Whitney from ranks; ties are negligible for continuous maps).
    mv = maps[..., valid]
    nv = labv.numel()
    ranks = mv.argsort(-1).argsort(-1).float() + 1
    for c in labv.unique().tolist():
        pos = labv == c
        P = int(pos.sum())
        Nn = nv - P
        if P == 0 or Nn == 0:
            continue
        auc = (ranks[:, :, c][..., pos].sum(-1) - P * (P + 1) / 2) / (P * Nn)
        res["loc"].append((c, P, auc.cpu().numpy().astype(np.float32)))

    # Segmentation by per-patch argmax over min-max normalised class maps.
    mn, mx = mv.amin(-1, keepdim=True), mv.amax(-1, keepdim=True)
    norm = (mv - mn) / (mx - mn).clamp_min(1e-8)
    present_patch = torch.zeros(C, dtype=torch.bool, device=maps.device)
    present_patch[labv.unique()] = True
    gt1 = F.one_hot(labv, C).bool()[None, None]  # (1, 1, Nv, C)
    for mode, scores in (("open", norm), ("oracle", norm.masked_fill(~present_patch[None, None, :, None], -1.0))):
        pred1 = F.one_hot(scores.argmax(2), C).bool()  # (L, Hh, Nv, C)
        res[f"inter_{mode}"] = (pred1 & gt1).sum(2).int().cpu().numpy()
        res[f"union_{mode}"] = (pred1 | gt1).sum(2).int().cpu().numpy()

    # Image-level recognition score: mean of the top patches of the raw map.
    k = max(1, int(round(top_frac * N)))
    res["recog"] = maps.topk(k, dim=-1).values.mean(-1).cpu().numpy().astype(np.float16)
    res["present"] = present_px.numpy()
    return res


def summarize_type(arrays, t, img_subset, num_classes):
    """Metric grids (L, Hh) for one map type on a subset of image indices."""
    sub = np.zeros(arrays[f"{t}__recog"].shape[0], dtype=bool)
    sub[img_subset] = True
    out = {}

    meta, auc = arrays[f"{t}__loc_meta"], arrays[f"{t}__loc_auroc"]
    rows = sub[meta[:, 0]] & (meta[:, 1] > 0)  # foreground classes only
    per_class = [auc[rows & (meta[:, 1] == c)].mean(0) for c in range(1, num_classes)
                 if (rows & (meta[:, 1] == c)).any()]
    out["loc_auroc"] = np.mean(per_class, 0)

    for mode in ("oracle", "open"):
        inter = arrays[f"{t}__inter_{mode}"][sub].sum(0).astype(np.float64)
        union = arrays[f"{t}__union_{mode}"][sub].sum(0).astype(np.float64)
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        out[f"miou_{mode}"] = np.nanmean(iou, -1)
        out[f"miou_{mode}_fg"] = np.nanmean(iou[..., 1:], -1)

    score = arrays[f"{t}__recog"][sub].astype(np.float32)  # (n, L, Hh, C)
    present = arrays["present"][sub]
    aucs = []
    for c in range(1, num_classes):
        pos = present[:, c]
        P, Nn = pos.sum(), (~pos).sum()
        if P == 0 or Nn == 0:
            continue
        r = score[..., c].argsort(0).argsort(0) + 1
        aucs.append((r[pos].sum(0) - P * (P + 1) / 2) / (P * Nn))
    out["recog_auroc"] = np.mean(aucs, 0) if aucs else np.full(score.shape[1:3], np.nan)
    return out


def save_heatmaps(full, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping heatmaps")
        return
    per_head = [t for t in PER_HEAD_TYPES if t in full]
    for metric in ("loc_auroc", "miou_oracle", "miou_open", "recog_auroc"):
        fig, axes = plt.subplots(1, len(per_head), figsize=(4.2 * len(per_head), 4.8), squeeze=False)
        for ax, t in zip(axes[0], per_head):
            grid = full[t][metric]
            im = ax.imshow(grid, aspect="auto", cmap="viridis")
            ax.set_title(t, fontsize=9)
            ax.set_xlabel("head (24 = mean)")
            ax.set_ylabel("layer")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(metric)
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for t in full:
            ax.plot(full[t][metric][:, -1], marker="o", ms=3, label=t)
        ax.set_xlabel("layer")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}, head mean / full-dim maps")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"layers_{metric}.png", dpi=120)
        plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    voc_root = Path(args.voc_root).resolve()
    args.voc_root = str(voc_root)
    if args.split_file:
        args.split_file = str(Path(args.split_file).resolve())
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)  # get_model() loads diffusion_configs/ and t2i/pixeldit_t2i_v1.pth relatively

    # Must be set before the model is built: PixDiT_T2I only builds `layer_count` patch blocks.
    local_config.layer_count = args.num_layers
    local_config.image_size = args.image_size
    device = local_config.device
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]

    from san.model.san import get_model  # heavy import (detectron2), after the config edits

    image_ids = load_image_ids(args)
    print(f"{len(image_ids)} images from {voc_root}")

    model, tokenizer, text_encoder = get_model()
    core = model.core
    if len(core.patch_blocks) != args.num_layers:
        raise RuntimeError(f"Model built {len(core.patch_blocks)} patch blocks, expected {args.num_layers}")
    num_heads, patch = core.num_groups, core.patch_size
    grid = args.image_size // patch
    C = len(VOC_CLASSES)

    text_embs, spans, ctrl_spans, span_info = encode_prompts(
        tokenizer, text_encoder, args.prompt_format, VOC_CLASSES, args.control_word, device)
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    for s in span_info:
        print(f"  {s['class']:>13}: positions {s['positions']} tokens {s['tokens']}")
    if ctrl_spans is None:
        print(f"  control word '{args.control_word}' not found in every prompt: control maps disabled")

    collector = MapCollector(args.num_layers, num_heads, C, grid * grid, spans, ctrl_spans, device)
    handles = collector.register(core)

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"args": vars(args), "timestep": local_config.timestep, "grid": grid,
                   "classes": VOC_CLASSES, "image_ids": image_ids, "prompts": span_info,
                   "map_types": collector.types}, f, indent=2)

    acc = {t: {"loc_meta": [], "loc_auroc": [], "inter_oracle": [], "union_oracle": [],
               "inter_open": [], "union_open": [], "recog": []} for t in collector.types}
    present_all = []
    t_cond = torch.full((C,), float(local_config.timestep), device=device)
    start = time.time()

    for i, image_id in enumerate(image_ids):
        img, gt = load_sample(voc_root, image_id, args.image_size)
        lab, present_px = patch_labels(gt, C, patch, args.purity)
        lab = lab.to(device)

        x = img[None].to(device).expand(C, -1, -1, -1).contiguous()
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            model(x, t_cond, text_embs)
        maps = collector.finalize()

        present_all.append(present_px.numpy())
        for t, m in maps.items():
            r = image_metrics(m, lab, present_px, args.top_frac)
            for c, P, auc in r["loc"]:
                acc[t]["loc_meta"].append((i, c, P))
                acc[t]["loc_auroc"].append(auc)
            for k in ("inter_oracle", "union_oracle", "inter_open", "union_open", "recog"):
                acc[t][k].append(r[k])

        if (i + 1) % 25 == 0 or i + 1 == len(image_ids):
            el = time.time() - start
            print(f"[{i + 1}/{len(image_ids)}] {el / (i + 1):.2f}s/img, eta {el / (i + 1) * (len(image_ids) - i - 1) / 60:.1f} min")

    for h in handles:
        h.remove()

    arrays = {"present": np.stack(present_all)}
    for t, a in acc.items():
        arrays[f"{t}__loc_meta"] = np.array(a["loc_meta"], dtype=np.int64).reshape(-1, 3)
        arrays[f"{t}__loc_auroc"] = np.stack(a["loc_auroc"])
        for k in ("inter_oracle", "union_oracle", "inter_open", "union_open", "recog"):
            arrays[f"{t}__{k}"] = np.stack(a[k])
    np.savez_compressed(out_dir / "per_image_metrics.npz", **arrays)

    # Select (layer, head) on the first half, report on the second half; also full-set grids.
    n = len(image_ids)
    sel, held = np.arange(n // 2), np.arange(n // 2, n)
    full, summary = {}, {}
    for t in collector.types:
        full[t] = summarize_type(arrays, t, np.arange(n), C)
        s_sel = summarize_type(arrays, t, sel, C)
        s_held = summarize_type(arrays, t, held, C)
        summary[t] = {}
        for metric, grid_sel in s_sel.items():
            l, h = np.unravel_index(np.nanargmax(grid_sel), grid_sel.shape)
            summary[t][metric] = {
                "best_layer": int(l), "best_head": int(h) if grid_sel.shape[1] > 1 else None,
                "selection_half": float(grid_sel[l, h]), "held_out_half": float(s_held[metric][l, h]),
                "head_mean_per_layer_full": [float(v) for v in full[t][metric][:, -1]],
            }
    np.savez_compressed(out_dir / "metric_grids_full.npz",
                        **{f"{t}__{m}": g for t, d in full.items() for m, g in d.items()})
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    save_heatmaps(full, out_dir)

    print("\nBest (layer, head) chosen on the first half, value on the held-out half:")
    metrics = ["loc_auroc", "miou_oracle", "miou_open", "recog_auroc"]
    print(f"{'map type':>22} " + " ".join(f"{m:>22}" for m in metrics))
    for t in collector.types:
        cells = []
        for m in metrics:
            d = summary[t][m]
            where = f"L{d['best_layer']}" + (f"H{d['best_head']}" if d["best_head"] is not None else "")
            cells.append(f"{d['held_out_half']:.3f} ({where})")
        print(f"{t:>22} " + " ".join(f"{c:>22}" for c in cells))
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
