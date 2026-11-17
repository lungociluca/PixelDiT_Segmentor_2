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
