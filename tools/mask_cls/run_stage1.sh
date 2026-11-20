#!/usr/bin/env bash
# Stage 1 end to end: GT-mask classification with PixelDiT features aligned to CLIP.
# Run from the repository root:   bash tools/mask_cls/run_stage1.sh [main|ablations|all]
# Every step is cached and resumable; re-running skips finished work.
set -euo pipefail
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-../datasets}"
CFG=configs/mask_cls/base.yaml
PY="python tools/mask_cls"
STAGE="${1:-main}"

main() {
  # 1. Region indexes and the documented seen/unseen splits
  $PY/build_index.py --config $CFG
  $PY/build_splits.py --config $CFG

  # 2. Feature caches (student: PixelDiT 4 blocks; teacher: OpenCLIP ViT-B/16 crop + ClearCLIP dense)
  $PY/extract_student.py --config $CFG
  $PY/extract_teacher.py --config $CFG

  # 3. Baselines
  $PY/evaluate.py --config $CFG --teacher crop dense avg                      # CLIP teacher (reference)
  $PY/fit.py --config $CFG align.method=linear align.sgd.distill_weight=0 \
      align.sgd.ce_weight=1.0 align.sgd.init=random                          # labels-only CE (overfitting baseline)

  # 4. Distilled maps (ours)
  $PY/fit.py --config $CFG align.method=ridge
  $PY/fit.py --config $CFG align.method=procrustes
  $PY/fit.py --config $CFG align.method=linear
  $PY/fit.py --config $CFG align.method=linear align.sgd.ce_weight=0.1       # + auxiliary CE
  $PY/fit.py --config $CFG align.method=mlp

  $PY/summarize.py --config $CFG
}

ablations() {
  # Cheap (cached features only): layers, teacher target, lambda path is logged by every ridge run
  for F in "[patch.0]" "[patch.1]" "[patch.2]" "[patch.3]" "[patch.1,patch.3]" "[patch.0,patch.1,patch.2,patch.3]"; do
    $PY/fit.py --config $CFG align.method=ridge "align.features=$F"
  done
  for T in crop dense avg; do
    $PY/fit.py --config $CFG align.method=ridge align.target=$T
  done
  for W in 0.03 0.1 0.3 1.0; do
    $PY/fit.py --config $CFG align.method=linear align.sgd.ce_weight=$W
  done
  for N in 1000 3000 10000 30000; do                                         # data scaling (COCO images)
    $PY/fit.py --config $CFG align.method=ridge align.train_images=$N
  done
  $PY/evaluate.py --config $CFG --teacher avg text.templates=imagenet        # prompt templates
  $PY/fit.py --config $CFG align.method=ridge text.templates=imagenet

  # Need new teacher caches
  for M in maskclip sclip; do
    $PY/extract_teacher.py --config $CFG --kinds dense teacher.dense.mode=$M
    $PY/evaluate.py --config $CFG --teacher dense avg teacher.dense.mode=$M
    $PY/fit.py --config $CFG align.method=ridge teacher.dense.mode=$M
  done
  $PY/extract_teacher.py --config $CFG --kinds crop teacher.crop.background=blur
  $PY/evaluate.py --config $CFG --teacher crop teacher.crop.background=blur
  for TM in "ViT-L-14 openai" "convnext_large_d_320 laion2b_s29b_b131k_ft_soup"; do
    set -- $TM
    $PY/extract_teacher.py --config $CFG teacher.model=$1 teacher.pretrained=$2
    $PY/evaluate.py --config $CFG --teacher crop dense avg teacher.model=$1 teacher.pretrained=$2
    $PY/fit.py --config $CFG align.method=ridge teacher.model=$1 teacher.pretrained=$2
  done

  # Need new student caches (one PixelDiT pass over fit + holdout + eval each)
  for S in "student.timestep=0.1" "student.timestep=0.3" "student.timestep=0.1 student.noise=true" \
           "student.prompt='A photo of a scene.'" "student.pixel_pathway=true"; do
    $PY/extract_student.py --config $CFG $S
    $PY/fit.py --config $CFG align.method=ridge $S
  done
  $PY/fit.py --config $CFG align.method=ridge "align.features=[pit_attn.1]" student.pixel_pathway=true
  $PY/fit.py --config $CFG align.method=ridge "align.features=[pixel.1]" student.pixel_pathway=true

  $PY/summarize.py --config $CFG
}

rescore() {
  # Re-fit / re-evaluate with the raw, rescale and center scoring modes and holdout
  # diagnostics (cached features only, a few minutes).
  $PY/evaluate.py --config $CFG --teacher crop dense avg
  $PY/fit.py --config $CFG align.method=ridge
  $PY/fit.py --config $CFG align.method=linear
  $PY/fit.py --config $CFG align.method=mlp
  $PY/fit.py --config $CFG align.method=linear align.sgd.ce_weight=0.1
  $PY/fit.py --config $CFG align.method=linear align.sgd.distill_weight=0 align.sgd.ce_weight=1.0 align.sgd.init=random
  for F in "[patch.0]" "[patch.1]" "[patch.2]" "[patch.0,patch.1,patch.2,patch.3]"; do
    $PY/fit.py --config $CFG align.method=ridge "align.features=$F"
  done
  $PY/summarize.py --config $CFG
}

depth() {
  # Deeper PixelDiT blocks (needs new student caches: one PixelDiT pass each).
  for B in 8 14; do
    L=$(python -c "print([$B//4-1, $B//2-1, 3*$B//4-1, $B-1])" | tr -d ' ')
    $PY/extract_student.py --config $CFG student.num_blocks=$B "student.layers=$L"
    for F in $(python -c "print(' '.join(f'[patch.{i}]' for i in $L))"); do
      $PY/fit.py --config $CFG align.method=ridge student.num_blocks=$B "student.layers=$L" "align.features=$F"
    done
  done
  $PY/summarize.py --config $CFG
}

corrclip() {
  # CorrCLIP teacher: last-block attention restricted to the patch's own GT region
  # (no inter-class attention). New dense caches only; crop targets are reused.
  for V in "similarity=dino" "similarity=clip_qq" "similarity=uniform" \
           "similarity=dino spatial_alpha=1.0" "similarity=dino mct_beta=0.5" \
           "similarity=dino spatial_alpha=1.0 mct_beta=0.5"; do
    OV="teacher.dense.mode=corrclip"
    for kv in $V; do OV="$OV teacher.dense.corrclip.$kv"; done
    $PY/extract_teacher.py --config $CFG --kinds dense $OV
    $PY/evaluate.py --config $CFG --teacher dense avg $OV
    $PY/fit.py --config $CFG align.method=ridge $OV
    $PY/fit.py --config $CFG align.method=ridge align.target=dense $OV
  done
  $PY/summarize.py --config $CFG
}

case "$STAGE" in
  corrclip) corrclip ;;
  rescore) rescore ;;
  depth) depth ;;
  main) main ;;
  ablations) ablations ;;
  all) main; ablations ;;
  *) echo "usage: $0 [main|rescore|corrclip|depth|ablations|all]"; exit 1 ;;
esac
