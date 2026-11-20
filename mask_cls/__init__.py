"""Stage 1: open-vocabulary mask classification with frozen PixelDiT features
aligned to CLIP, evaluated on ground-truth masks.

Subpackages
-----------
data/      datasets, class names, seen/unseen splits, region extraction
features/  PixelDiT feature extractor (hooks) and mask pooling
teacher/   CLIP teacher: masked-crop and dense (MaskCLIP/ClearCLIP/SCLIP) region targets, text embeddings
align/     alignment maps (ridge, Procrustes, linear-cosine, residual MLP) and their fitting
eval/      zero-shot region classification, top-k accuracy, oracle-mask mIoU, summaries
"""
