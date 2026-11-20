# Stage 1 — mask classification on GT masks

Frozen PixelDiT region features → a light alignment map → CLIP embedding space,
scored against CLIP **text** embeddings. CLIP's image encoder is only a teacher
that produces cached region targets; nothing from it is needed at inference.

```
image ─► PixelDiT (4 patch blocks, frozen) ─► block outputs [h, w, 1536]
GT masks ─► area-weighted downsampling ─► mask pooling ─► f_region
f_region ─► W (ridge / Procrustes / linear-cos / residual MLP) ─► normalize ─► · CLIP text(class names)
                     ▲ fit on COCO-Stuff regions to CLIP image embeddings of the same regions
```

## Run

From the repository root (an RTX 4090 runs every step; all steps are cached and resumable):

```bash
export DETECTRON2_DATASETS=/path/to/datasets
bash tools/mask_cls/run_stage1.sh main        # indexes, caches, baselines, ours, summary table
bash tools/mask_cls/run_stage1.sh ablations   # sweeps (see the script)
python -m pytest tests/mask_cls -q            # CPU unit tests for pooling / ridge / metrics / splits
```

Single steps take `--config configs/mask_cls/base.yaml` plus dotted overrides, e.g.

```bash
python tools/mask_cls/fit.py align.method=procrustes "align.features=[patch.1,patch.3]" align.target=crop
python tools/mask_cls/extract_student.py student.timestep=0.1 student.noise=true
```

Results go to `output/mask_cls/results.{jsonl,csv}`. `summarize.py` writes
`summary.{csv,md}` (teacher vs student, seen / unseen, per dataset).

## What each step does

| step | script | output |
|---|---|---|
| region index | `build_index.py` | one row per GT region: image, class, area, bbox. COCO train is split (seeded) into a **fit** pool (`data.train.max_images`) and a **holdout** set used only for model selection |
| class splits | `build_splits.py` | `configs/mask_cls/class_splits/<dataset>.json`: seen/unseen per class, with the matching rule used |
| student cache | `extract_student.py` | mask-pooled PixelDiT features, fp16 shards, `[n, 4, 1536]` per region (+ PiT features with `student.pixel_pathway=true`) |
| teacher cache | `extract_teacher.py` | (a) masked-crop and (b) dense CLIP region embeddings; (c) = their normalised mean, built on load. Also class text embeddings |
| fit + eval | `fit.py` | the map in `cache/mask_cls/maps/`, plus result rows for every eval dataset |
| baselines | `evaluate.py --teacher crop dense avg` | CLIP teacher reference numbers on the same masks |

## Design notes

* **Student features.** PixelDiT is built with 4 of its 14 MMDiT patch blocks
  (`student.num_blocks`; weights of the other blocks are not loaded). Features are
  the image tokens (residual stream) after each block, on the 1/16 grid. Images are
  resized so that H·W ≈ 1024² (PixelDiT's training area) with both sides multiples of
  16, so no padding enters the grid. Masks are downsampled with `area` interpolation
  (fraction of each cell covered) and used as pooling weights.
* **Timestep.** Flow matching: `x_t = (1-t)·x + t·ε`, t = 0 is clean, the network
  gets `t·1000`. Default is the repo's t = 0.002 on the clean image;
  `student.noise=true` actually adds the noise (seeded per image).
* **Prompt.** `student.prompt: ""` (empty, the model's own null prompt) by default;
  a fixed string, or per-image captions via `student.captions_file`. Embedded once
  with Gemma-2-2b-it and cached.
* **Pixel pathway** (`student.pixel_pathway=true`): runs the 2 PiT blocks,
  conditioned on the last *built* patch block as `silu(t_emb + s)`. With 4 of 14
  blocks this conditioning differs from pretraining, so treat it as an ablation.
  Exposes `pit_attn.{0,1}` (1152-d per patch) and `pixel.{0,1}` (16-d per pixel).
* **Teacher.** `crop`: square bbox crop with margin, background gray (CLIP mean) or
  blurred. `dense`: MaskCLIP / ClearCLIP / SCLIP-style last block for ViTs (positional
  embeddings interpolated), per-location head for ConvNeXt. `sanity_check()` checks
  the re-implemented forward pass against `encode_image` before extraction.
* **CorrCLIP teacher** (`teacher.dense.mode=corrclip`, `mask_cls/teacher/corrclip.py`).
  CorrCLIP (arXiv:2411.10086) finds that inter-class correlations, i.e. patches
  attending to patches of other classes, are the main cause of poor dense CLIP
  features. In the last ViT block each patch attends only to patches of its own GT
  region (GT masks replace SAM2 + DBSCAN). Patches outside every region attend to
  outside patches with above-mean similarity. The similarity is DINO ViT-B/8 q+k with
  τ = 0.25 (or CLIP q-q / uniform), and there is no residual or FFN. Optional spatial
  branch (`spatial_alpha`, lower-layer values) and mask-class-token branch
  (`mct_beta`). The scope uses GT masks, so its teacher numbers are an oracle
  setting, like the GT-mask pooling itself.
  `bash tools/mask_cls/run_stage1.sh corrclip` runs the variants.
* **Maps.** Ridge is closed form on standardised features with a λ path from one
  eigendecomposition (λ = α·N); Procrustes is the semi-orthogonal solution with the
  optimal global scale; linear / MLP are trained with cosine distillation, optionally
  + CE against frozen COCO text embeddings. Selection uses the COCO **holdout**,
  never an eval set. Maps are `nn.Module`s that take raw pooled features and hold
  their own standardisation, so the Stage 2 head can call them unchanged.
* **Metrics.** Region top-1 / top-5, class-mean accuracy, and oracle-mask mIoU. GT
  masks are disjoint, so the painted map's confusion matrix is the area-weighted
  (gt, pred) histogram over regions. All split seen / unseen, plus the harmonic mean.
  The softmax temperature cannot change argmax / top-k metrics; it only matters for
  the CE loss.
* **Seen / unseen.** Seen = matches a COCO-Stuff name (raw or cleaned, e.g.
  `wall-other` = `wall`) or an entry of `class_splits/synonyms.yaml`. With the
  shipped table: VOC 20/0 (every VOC class is a COCO class, so VOC only has seen
  numbers), ADE-150 74/76, ADE-847 123/724, COCO 171/0.
