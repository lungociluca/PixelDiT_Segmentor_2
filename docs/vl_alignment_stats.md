# Vision-language alignment statistics (VOC 2012)

`vl_alignment_stats.py` measures, for every PixelDiT MMDiT block and feature space, how well
image-patch features align with the features of the class-name tokens, using the VOC 2012
annotations to separate classes that are **present** in an image from classes that are **absent**.
The goal is to pick the layer/space to use for mask classification.

## How it works

1. Each class prompt (by default exactly SAN's: `default_prompt + " " + subprompt.format(target=...)`)
   is encoded with Gemma; the token positions of the class name are located with the tokenizer
   offsets and mapped through SAN's token selection (token 0 + last 299 of 506).
2. Each image is run through the frozen backbone at `t2i/config.timestep`:
   * class prompts are batched with the image (`--text-batch` per pass) to read the class-token
     features at every layer (text features in an MMDiT depend on the image through joint attention);
   * image features come from a pass with the default prompt (`--image-source default`, how SAN
     conditions the backbone) or from the class prompt's own pass (`--image-source same`).
3. Hooks in `t2i/pixdit_core/pixeldit_t2i.py` hand the features to `t2i/vl_stats/probe.py`, which
   reduces them on the fly (nothing large is stored). Similarities are cosine.

| space | image side | text side |
|---|---|---|
| `q`, `k`, `v` | q_x / k_x / v_x | q_y / k_y / v_y |
| `qk` | q_x | k_y (the image->text attention logits) |
| `kq` | k_x | q_y (the text->image attention logits) |
| `attn_out` | proj_x(attention) | proj_y(attention) |
| `block` | residual stream x after the block | residual stream y after the block |

Q/K/V are taken after the per-head RMSNorm and **before RoPE** (RoPE only encodes position).
After per-head RMSNorm the cosine of the concatenated heads is close to the mean per-head cosine;
`--per-head qk` additionally reports every head.

## Metrics (per layer x space)

* **Oracle-mask classification** (`maskcls_*`): mean-pool the features inside each ground-truth
  segment, compare with all 20 class vectors. `maskcls_top1` is the headline metric for choosing
  the classification layer; `_debiased` z-scores each class column (removes constant class offsets
  a learned head can absorb), `_among_present` only discriminates between the classes in the image.
* **Presence** (`presence_*`, `image_map`, `image_top1`): image-level class score = mean of the top
  10% patch similarities; AUROC of present vs absent (image, class) pairs, pooled and per class.
* **Localization** (`loc_*`): AUROC of each class similarity map inside vs outside its mask.

Outputs in `--out-dir`: `summary.md` (best layer per space, top-10 rankings), `metrics.csv`,
`metrics.json` (plus per-class results and confusion matrix of the best key), `heatmaps.png`,
`curves.png`, `present_vs_absent.png`, `confusion_best.png`, `per_head.png`, and `raw*.npz`.

## Usage

Run from the repository root (checkpoint paths are relative). The backbone is fully frozen
(no learnable tokens, no parameter requires grad) and runs under `torch.inference_mode`.

By default all 14 blocks are built and analysed. `--num-layers N` builds and analyses only the
first N blocks (blocks after them are never built or loaded); `--layers` picks a subset of the
built blocks, e.g. `--num-layers 5 --layers 0,2,4`.

```bash
export DETECTRON2_DATASETS=../datasets
# quick look
python vl_alignment_stats.py --max-images 200 --out-dir output/vl_stats
# only the first 5 blocks
python vl_alignment_stats.py --num-layers 5 --out-dir output/vl_stats_5
# full val set on 4 GPUs, with per-head analysis of the attention logits
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python vl_alignment_stats.py --per-head qk \
    --num-shards 4 --shard-id $i --out-dir output/vl_stats & done; wait
python vl_alignment_stats.py --merge "output/vl_stats/raw_shard*of4.npz" --out-dir output/vl_stats
```

If CUDA runs out of memory, in order of impact: lower `--text-batch` (default 4; 1 is the
minimum), use `--num-layers`, lower `--image-size` (e.g. 768 or 512), or use `--text-mode list`.
Backbone weights are kept in bf16 (`--fp32-weights` to change that).

Useful options: `--gt-format detectron2` (uses `annotations_detectron2/val`; background is then
ignored), `--text-mode list` (one prompt naming all classes: 2 passes per image instead of 4),
`--layers 0-6`, `--image-size 512`, `--add-noise`, `--timestep`, `--detectron2-class-names`.
