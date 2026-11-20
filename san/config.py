# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_san_config(cfg):
    # copied from maskformer2
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # optimizer
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP = [
        "absolute_pos_embed",
        "positional_embedding",
        "pos_embed",
        "query_embed",
        "relative_position_bias_table",
    ]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.CLIP_MULTIPLIER = 1.0
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1
    # Gradient accumulation: micro-batches per optimizer step. Each step sees
    # IMS_PER_BATCH * GRAD_ACCUM_STEPS images; MAX_ITER counts optimizer steps.
    cfg.SOLVER.GRAD_ACCUM_STEPS = 1
    # Autocast dtype when SOLVER.AMP.ENABLED: "float16" (uses a GradScaler) or
    # "bfloat16" (no scaler needed; falls back to float16 if the GPU lacks bf16).
    # This is the dtype of the trainable side adapter's forward pass; the frozen
    # PixelDiT follows MODEL.PIXELDIT.PRECISION.
    cfg.SOLVER.AMP.PRECISION = "float16"

    # san
    cfg.MODEL.SAN = CN()
    cfg.MODEL.SAN.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.SAN.CLASS_WEIGHT = 2.0
    cfg.MODEL.SAN.DICE_WEIGHT = 5.0
    cfg.MODEL.SAN.MASK_WEIGHT = 5.0
    cfg.MODEL.SAN.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.SAN.NUM_CLASSES = 171
    # Train the query class head as object / no-object only (all GT segments get
    # label 0, NUM_CLASSES is ignored). Classes then come from the oracle at
    # inference, so this requires ORACLE_CLASSES.ENABLED.
    cfg.MODEL.SAN.CLASS_AGNOSTIC = True
    cfg.MODEL.SAN.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.SAN.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.SAN.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.SAN.CLIP_PRETRAINED_NAME = "openai"
    cfg.MODEL.SAN.CLIP_TEMPLATE_SET = "vild"
    cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX = 9
    cfg.MODEL.SAN.CLIP_FROZEN_EXCLUDE = ["positional_embedding"]
    cfg.MODEL.SAN.CLIP_DEEPER_FROZEN_EXCLUDE = []
    cfg.MODEL.SAN.REC_CROSS_ATTN = False
    cfg.MODEL.SAN.REC_DOWNSAMPLE_METHOD = "max"
    cfg.MODEL.SAN.SOS_TOKEN_FORMAT = "cls_token"
    cfg.MODEL.SAN.SIZE_DIVISIBILITY = 32
    cfg.MODEL.SAN.ASYMETRIC_INPUT = True
    cfg.MODEL.SAN.CLIP_RESOLUTION = 0.5

    cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE = True
    # Oracle mask classification (inference only): replace the predicted class
    # scores of every mask with the ground-truth class it overlaps best, so that
    # evaluation measures only the quality of the class-agnostic masks.
    cfg.MODEL.SAN.ORACLE_CLASSES = CN()
    cfg.MODEL.SAN.ORACLE_CLASSES.ENABLED = True
    # "max_iou": GT segment with the highest IoU with the binarized mask.
    # "majority": GT class covering most of the binarized mask's pixels.
    cfg.MODEL.SAN.ORACLE_CLASSES.ASSIGNMENT = "max_iou"
    # Threshold on sigmoid(mask logits) used to binarize masks for the matching.
    cfg.MODEL.SAN.ORACLE_CLASSES.MASK_THRESHOLD = 0.5
    # Masks whose best match has IoU (or covered fraction, for "majority") below
    # this value are treated as "no object" and contribute nothing.
    cfg.MODEL.SAN.ORACLE_CLASSES.MIN_SCORE = 0.0
    # Scale each mask's one-hot oracle class by its predicted objectness
    # (softmax prob. of not being "no object"), as Mask2Former does with class
    # scores. Without it every query votes with weight 1, including the ones the
    # model has learned are empty.
    cfg.MODEL.SAN.ORACLE_CLASSES.WEIGHT_BY_OBJECTNESS = True
    # Also measure the class-agnostic masks against every GT class region: mean
    # best IoU, recall@0.5 / 0.75 and kept objects per image (logged after eval
    # and added to the results as "mask_stats").
    cfg.MODEL.SAN.ORACLE_CLASSES.MASK_STATS = True

    # Open-vocabulary mask classification from PixelDiT concept attention
    # (ConceptAttention, arXiv:2502.04320). Inference only and training-free: every
    # class name is a concept token run through the frozen PixelDiT's text stream;
    # each mask gets the classes of the image tokens it covers. Mutually exclusive
    # with ORACLE_CLASSES.ENABLED.
    cfg.MODEL.CONCEPT = CN()
    cfg.MODEL.CONCEPT.ENABLED = False
    # PixelDiT blocks (1-based) whose attention outputs are compared; their scores
    # are averaged. Blocks up to the deepest one are built and, at inference, run.
    cfg.MODEL.CONCEPT.LAYERS = [1, 2, 3, 4]
    # Prompt each class name is encoded in; only the name's tokens are pooled.
    cfg.MODEL.CONCEPT.TEMPLATE = "A picture of a {target}"
    # Template ensemble: when non-empty, replaces TEMPLATE. Every class variant is
    # encoded in every template and its pooled embeddings are averaged.
    cfg.MODEL.CONCEPT.TEMPLATES = []
    # Prepend the instruction ("complex human instruction") of the default prompt
    # (t2i.config.default_prompt, up to its last "User Prompt:") to every template,
    # so the text encoder sees class names in the context PixelDiT was trained on.
    # The concept tokens then sit at the text position of the name in that prompt.
    cfg.MODEL.CONCEPT.INSTRUCTION_CONTEXT = False
    # Clean class names before encoding (fixes like "diningtable" -> "dining table",
    # COCO-Stuff suffixes like "-merged" / "-other" removed, "-" / "_" -> " ").
    cfg.MODEL.CONCEPT.CLEAN_NAMES = True
    # Split comma-separated synonyms ("chest of drawers, chest, bureau, dresser")
    # into one concept each; their per-patch scores are combined into one class
    # score with SYNONYM_REDUCE ("max" or "mean").
    cfg.MODEL.CONCEPT.SPLIT_SYNONYMS = True
    cfg.MODEL.CONCEPT.SYNONYM_REDUCE = "max"
    # "mean": mean of the name's tokens. "last": its last token.
    cfg.MODEL.CONCEPT.POOL = "mean"
    # Divides the similarities before the softmax over classes.
    cfg.MODEL.CONCEPT.TEMPERATURE = 1.0
    # True: softmax over classes per patch and layer, then average over the mask
    # (as in the paper). False: average raw similarities over the mask, then softmax.
    cfg.MODEL.CONCEPT.PIXEL_SOFTMAX = True
    # Pool over the thresholded (0.5) mask instead of the soft sigmoid mask.
    cfg.MODEL.CONCEPT.BINARIZE_MASKS = False
    # Scale each mask's class probabilities by its predicted objectness.
    cfg.MODEL.CONCEPT.WEIGHT_BY_OBJECTNESS = True
    # Flow time of the PixelDiT pass that scores the concepts, = fraction of noise:
    # the input is (1 - t) * image + t * gaussian noise. With learned masks this is
    # a second backbone pass (the side adapter keeps its clean-image pass). <= 0:
    # no noise, reuse the side adapter's pass (t2i.config.timestep).
    cfg.MODEL.CONCEPT.TIMESTEP = 0.4
    # Seed of that noise; the same noise is used for every image, so evaluation
    # is reproducible.
    cfg.MODEL.CONCEPT.NOISE_SEED = 0
    # Classify the ground-truth segments instead of the side adapter's masks (one
    # mask per GT class present in the image). The side adapter is not run, so no
    # trained checkpoint is needed: this isolates mask classification. pACC is then
    # the area-weighted accuracy of the classifier; the per-segment top-1 accuracy
    # is logged during evaluation.
    cfg.MODEL.CONCEPT.GT_MASKS = False
    # Trained variant: low-rank adapters on the PixelDiT text-stream attention
    # weights, applied to the concept tokens only, plus learned layer weights and
    # a learned logit scale. No per-class parameters, so new class names work
    # without retraining. When enabled, *training* trains only this classifier:
    # ground-truth segments are classified with a cross-entropy over the class
    # names, the side adapter is frozen and not run. Evaluation needs the same
    # setting to load the trained weights.
    cfg.MODEL.CONCEPT.LORA = CN()
    cfg.MODEL.CONCEPT.LORA.ENABLED = False
    cfg.MODEL.CONCEPT.LORA.RANK = 16
    cfg.MODEL.CONCEPT.LORA.ALPHA = 16.0
    # Cosine instead of dot-product similarity between image and concept outputs
    # (bounded logits; the learned scale below sets their range).
    cfg.MODEL.CONCEPT.LORA.COSINE = True
    cfg.MODEL.CONCEPT.LORA.INIT_LOGIT_SCALE = 20.0
    cfg.MODEL.CONCEPT.LORA.LABEL_SMOOTHING = 0.1
    # Class names left out of training (their segments are skipped and their
    # concepts are not in the training vocabulary), to measure generalisation to
    # unseen classes: evaluate on the same dataset and compare their per-class IoU.
    cfg.MODEL.CONCEPT.LORA.HOLDOUT_CLASSES = []
    # "gt_masks": train only the concept classifier, on ground-truth segments (the
    #             side adapter is frozen and not run).
    # "joint":    train the side adapter (masks + objectness, as without concepts)
    #             and the concept classifier together: every predicted mask that
    #             the Hungarian matcher assigns to a ground-truth segment is
    #             classified with the concept scores pooled over that predicted
    #             mask; cross-entropy against the segment's class.
    cfg.MODEL.CONCEPT.LORA.TRAIN_MODE = "gt_masks"
    # Weight of that cross-entropy ("joint" only; "gt_masks" uses 1).
    cfg.MODEL.CONCEPT.LORA.LOSS_WEIGHT = 2.0
    # "joint" only: stop the classification loss from flowing into the masks.
    # False lets the classifier reshape the masks (end-to-end, as in SAN), which
    # can also shrink them to their most discriminative parts.
    cfg.MODEL.CONCEPT.LORA.DETACH_MASKS = True
    # Trained mask-embedding head (ODISE-style): image tokens of the concept pass
    # from LAYERS are pooled over every mask (LayerNorm per block, learned block
    # weights, MLP) and compared by cosine similarity with a class-agnostic
    # projection of the concept text embeddings. Trained next to the concept
    # classifier (needs LORA.ENABLED; same segments, same targets, in "gt_masks"
    # and "joint" mode); at inference the two are ensembled geometrically:
    # p ∝ p_concept^(1 - ENSEMBLE_ALPHA) * p_head^ENSEMBLE_ALPHA.
    cfg.MODEL.CONCEPT.MASK_HEAD = CN()
    cfg.MODEL.CONCEPT.MASK_HEAD.ENABLED = False
    cfg.MODEL.CONCEPT.MASK_HEAD.LAYERS = [2, 3, 4]
    cfg.MODEL.CONCEPT.MASK_HEAD.EMBED_DIM = 768
    cfg.MODEL.CONCEPT.MASK_HEAD.HIDDEN_DIM = 1024
    # "linear" (less prone to overfitting the training vocabulary) or "mlp".
    cfg.MODEL.CONCEPT.MASK_HEAD.TEXT_PROJ = "linear"
    cfg.MODEL.CONCEPT.MASK_HEAD.INIT_LOGIT_SCALE = 20.0
    cfg.MODEL.CONCEPT.MASK_HEAD.DROPOUT = 0.0
    cfg.MODEL.CONCEPT.MASK_HEAD.LOSS_WEIGHT = 1.0
    # 0 = concept classifier only, 1 = mask head only.
    cfg.MODEL.CONCEPT.MASK_HEAD.ENSEMBLE_ALPHA = 0.5
    # Low-rank adapters on the image stream (qkv_x / proj_x of every block the
    # concept pass runs), in the concept-scoring pass only: the side adapter's
    # clean-image pass keeps the frozen features. Gradients then flow through
    # the image tokens of every block, so memory grows with depth and batch size;
    # GRAD_CHECKPOINT recomputes the blocks in backward. Needs LORA.ENABLED.
    cfg.MODEL.CONCEPT.IMAGE_LORA = CN()
    cfg.MODEL.CONCEPT.IMAGE_LORA.ENABLED = False
    cfg.MODEL.CONCEPT.IMAGE_LORA.RANK = 16
    cfg.MODEL.CONCEPT.IMAGE_LORA.ALPHA = 16.0
    cfg.MODEL.CONCEPT.IMAGE_LORA.GRAD_CHECKPOINT = True
    # PixelDiT -> CLIP modality adapter (san/model/clip_align.py). Training: a frozen
    # CLIP image tower with attention restricted to the GT segments (CorrCLIP-style
    # scope; one segment per class) gives a target embedding per segment, and a
    # dense MLP on the clean-image tokens of PixelDiT blocks LAYERS, pooled over the
    # same segment (GT mask in "gt_masks" mode, the matched predicted mask in
    # "joint" mode), is pulled to it with a cosine loss (AGO-style). Inference: no
    # CLIP image tower and no GT; the adapter is pooled over the predicted masks,
    # compared with CLIP text embeddings of the class names and ensembled with the
    # concept classifier (+ mask head). Needs LORA.ENABLED. pip: open_clip_torch.
    cfg.MODEL.CONCEPT.ALIGN = CN()
    cfg.MODEL.CONCEPT.ALIGN.ENABLED = False
    # PixelDiT blocks (1-based) of the clean-image pass fed to the adapter.
    cfg.MODEL.CONCEPT.ALIGN.LAYERS = [1, 2, 3, 4, 5]
    # open_clip model / weights of the teacher and of the text embeddings.
    cfg.MODEL.CONCEPT.ALIGN.CLIP_MODEL = "ViT-B-16"
    cfg.MODEL.CONCEPT.ALIGN.CLIP_PRETRAINED = "openai"
    # Side of the square CLIP input (672 with a /16 model = a 42x42 grid).
    cfg.MODEL.CONCEPT.ALIGN.INPUT_SIZE = 672
    # Final CLIP blocks whose attention is restricted to the GT segments (1 = the
    # last one only, as in CorrCLIP). The last block is always ClearCLIP-style.
    cfg.MODEL.CONCEPT.ALIGN.SCOPE_LAYERS = 1
    # Last block: query-query similarities (CorrCLIP baseline) instead of query-key.
    cfg.MODEL.CONCEPT.ALIGN.QUERY_QUERY = True
    # A CLIP patch belongs to a segment covering at least this fraction of it.
    cfg.MODEL.CONCEPT.ALIGN.MIN_COVERAGE = 0.5
    # Images per teacher forward (bounds the masked-attention memory).
    cfg.MODEL.CONCEPT.ALIGN.TEACHER_CHUNK = 4
    # CLIP text templates ({target}); empty = the concept TEMPLATES / TEMPLATE.
    cfg.MODEL.CONCEPT.ALIGN.TEMPLATES = []
    cfg.MODEL.CONCEPT.ALIGN.HIDDEN_DIM = 1024
    cfg.MODEL.CONCEPT.ALIGN.DROPOUT = 0.0
    # Cosine distillation loss weight.
    cfg.MODEL.CONCEPT.ALIGN.LOSS_WEIGHT = 1.0
    # Optional cross-entropy of the aligned embeddings against the CLIP text
    # embeddings of the training classes. 0 = pure distillation (class-agnostic;
    # AGO found grounding + alignment on one embedding hurts).
    cfg.MODEL.CONCEPT.ALIGN.CLS_LOSS_WEIGHT = 0.0
    # Scale of the aligned cosine logits; <= 0 = CLIP's own logit scale (100).
    cfg.MODEL.CONCEPT.ALIGN.LOGIT_SCALE = -1.0
    # "geometric": p ∝ p_concept^(1 - ENSEMBLE_ALPHA) * p_align^ENSEMBLE_ALPHA.
    # "entropy":   per mask, the branch with the lower entropy (AGO).
    cfg.MODEL.CONCEPT.ALIGN.ENSEMBLE = "geometric"
    # 0 = concept classifier (+ mask head) only, 1 = aligned CLIP branch only.
    cfg.MODEL.CONCEPT.ALIGN.ENSEMBLE_ALPHA = 0.5
    # Diagnostic: ignore the masks and output the per-patch class probabilities as
    # the segmentation, to measure the concept scores on their own.
    cfg.MODEL.CONCEPT.PIXEL_BASELINE = False

    # PixelDiT backbone
    cfg.MODEL.PIXELDIT = CN()
    # PixelDiT layers whose image tokens are passed to the side adapter:
    # 0 = patch embedding, k = output of patch block k (1-based). Only blocks up to
    # the deepest listed layer are run; FUSION_MAP refers to these indices.
    cfg.MODEL.PIXELDIT.FEATURE_LAYERS = [1, 2, 3]
    # Dtype of the frozen PixelDiT (weights and forward pass):
    # "auto": bfloat16 when autocast runs in bfloat16, float32 otherwise (PixelDiT was
    #         trained in bf16; float16 lacks its range and can overflow to inf/NaN).
    # "float32": always float32.  "amp": always the autocast dtype, even float16.
    cfg.MODEL.PIXELDIT.PRECISION = "auto"
    # side adapter
    cfg.MODEL.SIDE_ADAPTER = CN()
    # "PixelDiTSideAdapterNetwork": masks + objectness, fused with PixelDiT features.
    # "RegionwiseSideAdapterNetwork": original SAN (needs the CLIP recognition head).
    cfg.MODEL.SIDE_ADAPTER.NAME = "PixelDiTSideAdapterNetwork"
    cfg.MODEL.SIDE_ADAPTER.VIT_NAME = "vit_w240n6d8_patch16"
    cfg.MODEL.SIDE_ADAPTER.PRETRAINED = False
    # Size the side ViT's positional embedding is built for (resized otherwise).
    cfg.MODEL.SIDE_ADAPTER.IMAGE_SIZE = 1024
    cfg.MODEL.SIDE_ADAPTER.DROP_PATH_RATE = 0.0
    cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES = 100
    cfg.MODEL.SIDE_ADAPTER.FUSION_TYPE = "add"
    # "k->j": backbone layer k is added to the side ViT tokens before side block j.
    # For PixelDiT, k indexes MODEL.PIXELDIT.FEATURE_LAYERS.
    cfg.MODEL.SIDE_ADAPTER.FUSION_MAP = ["1->0", "2->1", "3->2"]
    cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS = [7, 8]

    # Mask decoder MLPs. PixelDiTSideAdapterNetwork uses EMBED_CHANNELS,
    # MLP_CHANNELS and MLP_NUM_LAYERS; NUM_HEADS, NUM_LAYERS and RESCALE_ATTN_BIAS
    # only apply to SAN's CLIP attention-bias branch.
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS = CN()
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_HEADS = 12
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_LAYERS = 1
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.EMBED_CHANNELS = 256
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_CHANNELS = 256
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_NUM_LAYERS = 3
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.RESCALE_ATTN_BIAS = True

    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "san"
    cfg.WANDB.NAME = None
    # use flash attention
    cfg.MODEL.FLASH = False
