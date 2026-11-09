"""Vision-language alignment statistics for the PixelDiT backbone on PASCAL VOC 2012.

For every MMDiT block and every feature space (Q, K, V, Q_img-K_txt, K_img-Q_txt,
attention output, block output) this measures how well image patch features
align with the features of class-name tokens, separating classes that are
present in the image from classes that are not, using the VOC annotations.

Run from the repository root (the checkpoint paths in san/model/san.py are relative):

    export DETECTRON2_DATASETS=../datasets
    python vl_alignment_stats.py --out-dir output/vl_stats --max-images 300

    # all of VOC val split over 4 GPUs, then merge
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python vl_alignment_stats.py \
        --num-shards 4 --shard-id $i --out-dir output/vl_stats & done; wait
    python vl_alignment_stats.py --merge output/vl_stats/raw_shard*of4.npz --out-dir output/vl_stats

See docs/vl_alignment_stats.md for what each number means.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from t2i.vl_stats.metrics import RawStore, compute_key_metrics
from t2i.vl_stats.report import write_report

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv monitor",
]
# names exactly as registered for detectron2 (san/data/datasets/register_voc.py), i.e. what SAN embeds
VOC_CLASSES_DETECTRON2 = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tv",
]
SPACES = ("q", "k", "v", "qk", "kq", "attn_out", "block")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--voc-root", default=None,
                   help="VOC2012 folder (default: $DETECTRON2_DATASETS/VOC2012, or datasets/VOC2012)")
    d.add_argument("--gt-format", choices=["voc", "detectron2"], default="voc",
                   help="voc: SegmentationClass(Aug) PNGs, 0=background 1..20 classes 255=boundary. "
                        "detectron2: annotations_detectron2/<split>, 0..19 classes, 255=background+boundary "
                        "(background then cannot be used as negative region for localization).")
    d.add_argument("--gt-dir", default=None, help="override the annotation folder")
    d.add_argument("--image-dir", default=None, help="override the image folder (default VOC2012/JPEGImages)")
    d.add_argument("--split", default="val", help="ImageSets/Segmentation/<split>.txt (voc format)")
    d.add_argument("--max-images", type=int, default=0, help="0 = all")
    d.add_argument("--num-shards", type=int, default=1)
    d.add_argument("--shard-id", type=int, default=0)
    d.add_argument("--detectron2-class-names", action="store_true",
                   help="use the exact registered names ('diningtable', 'tv', ...) instead of natural ones")

    m = p.add_argument_group("model")
    m.add_argument("--image-size", type=int, default=1024,
                   help="resize so the long side is this many pixels (rounded to the 16-px patch grid)")
    m.add_argument("--timestep", type=float, default=None, help="flow time in [0,1] (default t2i/config.timestep)")
    m.add_argument("--add-noise", action="store_true", help="noise the image to the flow time x_t=(1-t)x+t*eps")
    m.add_argument("--num-layers", type=int, default=None,
                   help="build and analyse only the first N MMDiT blocks (e.g. 5 -> layers 0-4). "
                        "Blocks after the last analysed one are never built, which saves memory and time.")
    m.add_argument("--layers", default="all",
                   help="which of the built blocks to analyse: 'all', '0-4' or '0,2,4' "
                        "(without --num-layers, blocks up to the highest one listed are built)")
    m.add_argument("--no-amp", action="store_true", help="run in fp32 instead of bf16 autocast")
    m.add_argument("--fp32-weights", action="store_true",
                   help="keep the backbone weights in fp32 (default bf16, half the memory)")
    m.add_argument("--seed", type=int, default=0)

    t = p.add_argument_group("text")
    t.add_argument("--text-mode", choices=["per_class", "list"], default="per_class",
                   help="per_class: one prompt per class (SAN's class prompts); "
                        "list: a single prompt naming all classes (1 extra pass per image, order-biased)")
    t.add_argument("--image-source", choices=["default", "same"], default="default",
                   help="default: image features from a pass with the default prompt, as SAN conditions the "
                        "backbone; same: image features from the same pass as the class prompt")
    t.add_argument("--prompt-template", default=None,
                   help="class prompt containing {target} (default: t2i/config.subprompt)")
    t.add_argument("--no-prefix", action="store_true",
                   help="do not prepend t2i/config.default_prompt to class prompts (SAN prepends it)")
    t.add_argument("--list-template", default="There may be any of the following in the picture: {targets}.")
    t.add_argument("--max-length", type=int, default=506, help="tokenizer padding length (SAN uses 506)")
    t.add_argument("--text-batch", type=int, default=4,
                   help="class prompts per forward pass; lower it (down to 1) if CUDA runs out of memory")

    s = p.add_argument_group("statistics")
    s.add_argument("--spaces", default=",".join(SPACES), help=f"comma list from {SPACES}")
    s.add_argument("--per-head", default="", help="spaces to also analyse per attention head, e.g. 'qk,k'")
    s.add_argument("--topk-frac", type=float, default=0.1,
                   help="image-level class score = mean of the top fraction of patch similarities")
    s.add_argument("--purity", type=float, default=0.5, help="a patch belongs to a class if it covers >= this fraction")
    s.add_argument("--min-area", type=float, default=1.0, help="present classes must cover >= this many patches")

    o = p.add_argument_group("output")
    o.add_argument("--out-dir", default="output/vl_stats")
    o.add_argument("--save-every", type=int, default=50, help="checkpoint the raw statistics every N images")
    o.add_argument("--no-plots", action="store_true")
    o.add_argument("--merge", nargs="+", default=None,
                   help="only merge raw_*.npz files from shards and write the report (no GPU needed)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def fill_template(template: str, mapping: Dict[str, str]) -> Tuple[str, Dict[str, List[Tuple[int, int]]]]:
    """Format `{name}` placeholders and return the char spans of every inserted value."""
    out, spans, i = "", {k: [] for k in mapping}, 0
    while i < len(template):
        hit = next((k for k in mapping if template.startswith("{" + k + "}", i)), None)
        if hit is None:
            out += template[i]
            i += 1
            continue
        val = mapping[hit]
        spans[hit].append((len(out), len(out) + len(val)))
        out += val
        i += len(hit) + 2
    return out, spans


def build_prompts(class_names, args, local_config):
    """Returns list of (name, text, [(class_id, [char spans])])."""
    template = args.prompt_template or local_config.subprompt
    prefix = "" if args.no_prefix else local_config.default_prompt + " "
    prompts = []
    if args.text_mode == "per_class":
        for c, name in enumerate(class_names):
            body, spans = fill_template(template, {"target": name})
            if not spans["target"]:
                raise ValueError("--prompt-template must contain {target}")
            prompts.append((f"class:{name}", prefix + body,
                            [(c, [(a + len(prefix), b + len(prefix)) for a, b in spans["target"]])]))
    else:
        head, _, tail = args.list_template.partition("{targets}")
        text = prefix + head
        cls_spans = []
        for c, name in enumerate(class_names):
            if c:
                text += ", "
            cls_spans.append((c, [(len(text), len(text) + len(name))]))
            text += name
        prompts.append(("list", text + tail, cls_spans))
    prompts.append(("default", local_config.default_prompt, []))
    return prompts


def encode_prompts(tokenizer, text_encoder, prompts, max_length, keep, usable, device):
    """Mirror SAN.embed_condition: pad to `max_length`, keep token 0 + the last keep-1 tokens.

    Returns embeddings (P, 1, keep, D) on CPU and, per prompt, [(class_id, LongTensor of kept token indices)].
    """
    import torch

    select = [0] + list(range(max_length - (keep - 1), max_length))
    pos_map = {p: j for j, p in enumerate(select)}
    embs, all_spans = [], []
    for name, text, cls_spans in prompts:
        kwargs = dict(max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        offsets = None
        if getattr(tokenizer, "is_fast", False):
            tok = tokenizer([text], return_offsets_mapping=True, **kwargs)
            offsets = tok.pop("offset_mapping")[0].tolist()
        else:
            tok = tokenizer([text], **kwargs)
        ids = tok["input_ids"].to(device)
        am = tok["attention_mask"].to(device)
        with torch.no_grad():
            e = text_encoder(ids, am)[0][:, None][:, :, select]
        embs.append(e.float().cpu())
        n_real = int(am.sum())
        visible = 1 + sum(1 for p in select[1:usable] if p < n_real)
        if name in ("default", "list") or name.startswith("class:") and not all_spans:
            print(f"[vl_stats] prompt '{name}': {n_real} tokens, {visible} of them visible to the model "
                  f"(token 0 + positions {max_length - keep + 1}..{max_length - keep + usable - 1})")

        resolved = []
        for c, char_spans in cls_spans:
            if offsets is not None:
                orig = [i for i, (s, t) in enumerate(offsets[:n_real])
                        if t > s and any(s < ce and t > cs for cs, ce in char_spans)]
            else:  # slow tokenizer: count tokens of the text before/after each span
                orig = []
                for cs, ce in char_spans:
                    a = len(tokenizer(text[:cs])["input_ids"])
                    b = len(tokenizer(text[:ce])["input_ids"])
                    orig += list(range(a, max(b, a + 1)))
            kept = [pos_map[p] for p in orig if p in pos_map and pos_map[p] < usable]
            if not kept:
                raise RuntimeError(
                    f"prompt '{name}': the class tokens (positions {orig}) are not among the tokens the model sees "
                    f"(token 0 and positions {max_length - keep + 1}..{max_length - 1} with right padding to "
                    f"{max_length}; the prompt has {n_real} tokens). The prompt must be at least "
                    f"{max_length - keep + 1} tokens long for its end to be kept - keep the default-prompt prefix "
                    f"or lower --max-length to about prompt_length + {keep - 1}.")
            if len(kept) < len(orig):
                print(f"[vl_stats] warning: prompt '{name}': {len(orig) - len(kept)} of {len(orig)} class tokens dropped")
            resolved.append((c, torch.tensor(kept, dtype=torch.long)))
        all_spans.append(resolved)
    return torch.cat(embs, 0), all_spans


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def list_samples(args) -> List[Tuple[str, str, str]]:
    root = args.voc_root or os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "VOC2012")
    image_dir = args.image_dir or os.path.join(root, "JPEGImages")
    if args.gt_dir:
        gt_dir = args.gt_dir
    elif args.gt_format == "voc":
        gt_dir = os.path.join(root, "SegmentationClass")
    else:
        gt_dir = os.path.join(root, "annotations_detectron2", args.split)
    split_file = os.path.join(root, "ImageSets", "Segmentation", f"{args.split}.txt")
    if args.gt_format == "voc" and os.path.isfile(split_file):
        with open(split_file) as f:
            ids = [l.strip() for l in f if l.strip()]
    else:
        ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(gt_dir, "*.png")))
    samples = [(i, os.path.join(image_dir, i + ".jpg"), os.path.join(gt_dir, i + ".png")) for i in ids]
    missing = [s for s in samples if not (os.path.isfile(s[1]) and os.path.isfile(s[2]))]
    if missing:
        print(f"[vl_stats] skipping {len(missing)} ids without image or annotation (e.g. {missing[0][1:]})")
    samples = [s for s in samples if s not in missing]
    if not samples:
        raise FileNotFoundError(f"no VOC samples found (images: {image_dir}, annotations: {gt_dir})")
    samples = samples[args.shard_id::args.num_shards]
    if args.max_images:
        samples = samples[: args.max_images]
    return samples


def load_sample(img_path, gt_path, image_size, patch, gt_format, num_classes):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    img = np.asarray(Image.open(img_path).convert("RGB"))
    gt = np.asarray(Image.open(gt_path))
    if gt.ndim == 3:
        gt = gt[..., 0]
    H0, W0 = gt.shape
    s = image_size / max(H0, W0)
    H = max(patch, int(round(H0 * s / patch)) * patch)
    W = max(patch, int(round(W0 * s / patch)) * patch)
    x = torch.from_numpy(img.copy()).permute(2, 0, 1).float()[None] / 127.5 - 1.0
    x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False, antialias=True).clamp(-1, 1)
    g = torch.from_numpy(gt.astype(np.int64))
    g = F.interpolate(g[None, None].float(), (H, W), mode="nearest")[0, 0].long()
    if gt_format == "voc":
        out = torch.full_like(g, 255)
        cls = (g >= 1) & (g <= num_classes)
        out[cls] = g[cls] - 1
        out[g == 0] = num_classes          # background
        g, background_id = out, num_classes
    else:
        background_id = None
    return x, g, background_id


# --------------------------------------------------------------------------- #
def parse_layers(spec: str, depth: int) -> List[int]:
    if spec == "all":
        return list(range(depth))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


def plan_passes(args, prompt_index, spans, C):
    """Forward passes per image: list of (prompt rows, text_roles, image_roles)."""
    classes = list(range(C))
    default_row = prompt_index["default"]
    passes = []
    if args.text_mode == "per_class":
        rows = [prompt_index[f"class:{i}"] for i in classes]
        for a in range(0, C, args.text_batch):
            chunk = rows[a: a + args.text_batch]
            text_roles = [spans[r] for r in chunk]
            image_roles = [[c for c, _ in spans[r]] if args.image_source == "same" else [] for r in chunk]
            passes.append((chunk, text_roles, image_roles))
    else:
        r = prompt_index["list"]
        passes.append(([r], [spans[r]], [classes if args.image_source == "same" else []]))
    if args.image_source == "default":
        passes.append(([default_row], [[]], [classes]))
    return passes


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge:
        files = sorted({f for pat in args.merge for f in glob.glob(pat)})
        print(f"[vl_stats] merging {len(files)} files")
        store = RawStore.merge(RawStore.load(f) for f in files)
        store.save(os.path.join(args.out_dir, "raw_merged.npz"))
        write_report(store, args.out_dir, make_plots=not args.no_plots)
        print(f"[vl_stats] report written to {args.out_dir}/summary.md")
        return

    import torch
    import t2i.config as local_config

    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        sys.exit("CUDA is required (t2i/config.device is 'cuda').")
    torch.manual_seed(args.seed)

    class_names = VOC_CLASSES_DETECTRON2 if args.detectron2_class_names else VOC_CLASSES
    C = len(class_names)
    spaces = [s.strip() for s in args.spaces.split(",") if s.strip()]
    per_head = [s.strip() for s in args.per_head.split(",") if s.strip()]

    # Only build the blocks that are analysed: layers beyond the last requested one are never needed.
    requested = None if args.layers == "all" else parse_layers(args.layers, 10 ** 6)
    if args.num_layers is not None:
        if args.num_layers < 1:
            sys.exit("--num-layers must be >= 1")
        local_config.layer_count = args.num_layers
        if requested is not None and max(requested) >= args.num_layers:
            sys.exit(f"--layers {args.layers} asks for blocks beyond --num-layers {args.num_layers}")
    else:
        local_config.layer_count = (max(requested) + 1) if requested else 10 ** 6

    from san.model.san import get_model
    from t2i.vl_stats.probe import VLAlignmentProbe, build_patch_gt

    device = local_config.device
    model, tokenizer, text_encoder = get_model(dtype=torch.float32 if args.fp32_weights else torch.bfloat16)
    assert not any(p.requires_grad for p in model.parameters()), "backbone must be frozen"
    core = model.core
    depth = len(core.patch_blocks)
    layers = [l for l in (requested or range(depth)) if l < depth]
    patch = core.patch_size
    keep = core.txt_max_length                      # 300 tokens fed to the model ...
    usable = min(keep, max(1, core.txt_max_length - 1))  # ... of which the first 299 are used
    torch.cuda.empty_cache()
    print(f"[vl_stats] PixelDiT blocks built: {depth}, probing layers {layers}, spaces {spaces}, per-head {per_head}")

    prompts = build_prompts(class_names, args, local_config)
    if args.text_mode == "per_class":
        names = [f"class:{i}" for i in range(C)] + ["default"]
    else:
        names = ["list", "default"]
    prompt_index = {n: i for i, n in enumerate(names)}
    embs, spans = encode_prompts(tokenizer, text_encoder, prompts, args.max_length, keep, usable, device)
    first = next(s for s in spans if s)
    print(f"[vl_stats] example class token positions: {class_names[first[0][0]]} -> {first[0][1].tolist()}")
    del tokenizer, text_encoder
    torch.cuda.empty_cache()

    probe = VLAlignmentProbe(C, spaces=spaces, layers=layers, per_head_spaces=per_head, topk_frac=args.topk_frac)
    passes = plan_passes(args, prompt_index, spans, C)
    timestep = args.timestep if args.timestep is not None else local_config.timestep
    # identical to DPM_Solver_Wrapper.segment + model_wrapper(discrete_flow): t_input = t * 1000
    t_cont = torch.tensor(timestep, device=device, dtype=torch.bfloat16)
    t_input = t_cont * 1000

    samples = list_samples(args)
    meta = {
        "dataset": "VOC2012", "split": args.split, "gt_format": args.gt_format, "image_size": args.image_size,
        "timestep": timestep, "add_noise": args.add_noise, "text_mode": args.text_mode,
        "image_source": args.image_source, "prompt_prefix": not args.no_prefix,
        "prompt_template": args.prompt_template or local_config.subprompt, "topk_frac": args.topk_frac,
        "purity": args.purity, "min_area": args.min_area, "amp": not args.no_amp,
        "shard": f"{args.shard_id}/{args.num_shards}",
    }
    store = RawStore(class_names, meta)
    suffix = f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    raw_path = os.path.join(args.out_dir, f"raw{suffix}.npz")
    print(f"[vl_stats] {len(samples)} images, {len(passes)} forward passes per image")

    gen = torch.Generator(device=device)
    t0 = time.time()
    for n, (img_id, img_path, gt_path) in enumerate(samples, 1):
        x, g, background_id = load_sample(img_path, gt_path, args.image_size, patch, args.gt_format, C)
        x, g = x.to(device), g.to(device)
        if args.add_noise:
            gen.manual_seed(args.seed + n)
            eps = torch.randn(x.shape, generator=gen, device=device)
            x = (1 - timestep) * x + timestep * eps
        gt = build_patch_gt(g, C, patch, background_id=background_id, purity=args.purity, min_area=args.min_area)
        probe.start_image(gt)
        for rows, text_roles, image_roles in passes:
            y = embs[rows].to(device)                                     # (B, 1, 300, D)
            xb = x.expand(len(rows), -1, -1, -1)
            probe.set_roles(text_roles, image_roles)
            segment_data = {"vl_probe": probe}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp):
                model.forward_with_dpmsolver(
                    xb, t_input, y, mask=None,
                    data_info={"img_hw": x.shape[-2], "aspect_ratio": 1}, segment_data=segment_data,
                )
            del segment_data
        store.add_image(img_id, gt.presence, np.asarray(gt.seg_classes), gt.seg_area, probe.end_image())

        if n % args.save_every == 0 or n == len(samples):
            store.save(raw_path)
            rate = n / (time.time() - t0)
            best = max(((k, compute_key_metrics(store, k)["maskcls_top1"]) for k in store.keys if "@h" not in k),
                       key=lambda kv: -1 if np.isnan(kv[1]) else kv[1])
            print(f"[vl_stats] {n}/{len(samples)} images ({rate:.2f} img/s) - "
                  f"best so far: layer {best[0].split('|')[0]} {best[0].split('|')[1]} maskcls_top1={best[1]:.3f}")

    store.save(raw_path)
    if args.num_shards == 1:
        write_report(store, args.out_dir, make_plots=not args.no_plots)
        print(f"[vl_stats] report written to {args.out_dir}/summary.md")
    else:
        print(f"[vl_stats] shard saved to {raw_path}; merge with --merge {args.out_dir}/raw_shard*of{args.num_shards}.npz")


if __name__ == "__main__":
    main()
