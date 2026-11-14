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
        "block_logits",      # concept classifier block weights
        "log_temperature",   # concept classifier temperature
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
    # The frozen PixelDiT weights are stored in this dtype too.
    cfg.SOLVER.AMP.PRECISION = "float16"

    # san
    cfg.MODEL.SAN = CN()
    cfg.MODEL.SAN.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.SAN.CLASS_WEIGHT = 2.0
    cfg.MODEL.SAN.DICE_WEIGHT = 5.0
    cfg.MODEL.SAN.MASK_WEIGHT = 5.0
    cfg.MODEL.SAN.TRAIN_NUM_POINTS = 112 * 112
    # Number of classes of the TRAINING vocabulary (must match DATASETS.TRAIN).
    cfg.MODEL.SAN.NUM_CLASSES = 171
    # Mask-only mode, used when MODEL.PIXELDIT.CLASSIFIER.ENABLED is False: train the
    # query head as object / no-object only (all GT segments get label 0) and take
    # the classes from the oracle at inference (requires ORACLE_CLASSES.ENABLED).
    # Ignored when the classifier is enabled (real labels are used).
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
    # With the classifier enabled, the oracle replaces its class predictions at
    # inference (to measure mask quality alone).
    cfg.MODEL.SAN.ORACLE_CLASSES.ENABLED = False
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

    # PixelDiT backbone
    cfg.MODEL.PIXELDIT = CN()
    # PixelDiT layers whose image tokens are passed to the side adapter:
    # 0 = patch embedding, k = output of patch block k (1-based). Only blocks up to
    # the deepest listed layer are run; FUSION_MAP refers to these indices.
    cfg.MODEL.PIXELDIT.FEATURE_LAYERS = [1, 2, 3]
    # One PixelDiT pass per image, at continuous flow time TIMESTEP on the noised
    # image x_t = (1 - t) * x + t * eps (the model receives 1000 * t). It feeds both
    # the side adapter (FEATURE_LAYERS) and the classifier (CLASSIFIER.READOUT_BLOCKS).
    # Overrides t2i/config.py timestep.
    cfg.MODEL.PIXELDIT.TIMESTEP = 0.5
    # Evaluation noise is drawn per image from NOISE_SEED + crc32(file_name), so
    # results are reproducible; training draws fresh noise every iteration.
    cfg.MODEL.PIXELDIT.NOISE_SEED = 0

    # Open-vocabulary mask classifier read from PixelDiT's attention
    # (san/model/concept_classifier.py).
    cfg.MODEL.PIXELDIT.CLASSIFIER = CN()
    cfg.MODEL.PIXELDIT.CLASSIFIER.ENABLED = True
    # Blocks (1-based) whose attention outputs score image tokens against classes.
    # Zero-shot on VOC GT segments (t = 0.5): blocks 4-8 averaged gave 44% top-1;
    # blocks after 8 are weaker.
    cfg.MODEL.PIXELDIT.CLASSIFIER.READOUT_BLOCKS = [4, 5, 6, 7, 8]
    # Every class is encoded with each template (scores averaged over them). Only the
    # name's own tokens become concept tokens, not the template words around it.
    cfg.MODEL.PIXELDIT.CLASSIFIER.PROMPT_TEMPLATES = [
        "A picture of a {target}",
        "A photo of the {target}.",
        "There is {target} in the scene.",
    ]
    # Class names that are synonym lists ("road, route", "sidewalk, pavement") get one
    # prompt per synonym instead of one prompt with the whole list.
    cfg.MODEL.PIXELDIT.CLASSIFIER.SPLIT_SYNONYMS = True
    # Each prompt's tokens attend to the image and to themselves only, not to other
    # classes, so scores do not depend on the vocabulary (171 classes in training,
    # 150 / 847 at test).
    cfg.MODEL.PIXELDIT.CLASSIFIER.ISOLATE_CONCEPTS = True
    # Prepend the "Enhanced prompt" instruction (CHI) that t2i default_prompt
    # starts with: "<CHI> User Prompt: A picture of a dog", as in generation.
    cfg.MODEL.PIXELDIT.CLASSIFIER.USE_CHI_PREFIX = False
    # Trainable parts (the text side always stays frozen). At their initial values
    # the classifier is the zero-shot readout.
    cfg.MODEL.PIXELDIT.CLASSIFIER.LEARN_BLOCK_WEIGHTS = True
    cfg.MODEL.PIXELDIT.CLASSIFIER.LEARN_TEMPERATURE = True
    # Linear map on the image attention outputs (identity at init). More capacity,
    # more risk of overfitting to the training vocabulary.
    cfg.MODEL.PIXELDIT.CLASSIFIER.IMAGE_ADAPTER = False
    # Stop the classification loss from shaping the masks.
    cfg.MODEL.PIXELDIT.CLASSIFIER.DETACH_MASKS = False
    # A mask averages token class probabilities over tokens with sigmoid above this
    # (all tokens if none is); 0 = weight every token by its sigmoid.
    cfg.MODEL.PIXELDIT.CLASSIFIER.MASK_POOL_THRESHOLD = 0.5
    # Keep the residual streams in fp32 during the pass (the prompt stream exceeds
    # the float16 range at block 7; matmuls stay in the AMP dtype).
    cfg.MODEL.PIXELDIT.CLASSIFIER.FP32_RESIDUAL = True
    # Standardise each token's class scores across classes before the softmax
    # ("zscore"), or use the raw dot products ("none": the softmax saturates).
    cfg.MODEL.PIXELDIT.CLASSIFIER.SCORE_NORM = "zscore"
    cfg.MODEL.PIXELDIT.CLASSIFIER.INIT_TEMPERATURE = 1.0
    # How a mask turns token scores into class probabilities.
    #   "probs":  softmax per token, then average the probabilities over the mask
    #             (ConceptAttention; distributions come out flat).
    #   "scores": average the standardised scores over the mask, then softmax;
    #             token noise cancels, the class signal adds up.
    cfg.MODEL.PIXELDIT.CLASSIFIER.POOLING = "scores"
    # LR multiplier for the head's scalar parameters (temperature, block weights).
    # Adam moves a parameter by about lr per step, so at BASE_LR 1e-4 a log-temperature
    # could change by only ~0.2 in 2000 iterations.
    cfg.MODEL.PIXELDIT.CLASSIFIER.SCALAR_LR_MULTIPLIER = 100.0
    # Write head statistics (temperature, block weights, raw score spread, and the
    # probability / top-1 of the GT class on matched queries) to the training
    # metrics every LOG_PERIOD iterations (0 = never). Costs one extra matching.
    cfg.MODEL.PIXELDIT.CLASSIFIER.LOG_PERIOD = 20
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
