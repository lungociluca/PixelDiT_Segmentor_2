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
    # The frozen PixelDiT weights are stored in this dtype too.
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

    # PixelDiT backbone
    cfg.MODEL.PIXELDIT = CN()
    # PixelDiT layers whose image tokens are passed to the side adapter:
    # 0 = patch embedding, k = output of patch block k (1-based). Only blocks up to
    # the deepest listed layer are run; FUSION_MAP refers to these indices.
    cfg.MODEL.PIXELDIT.FEATURE_LAYERS = [1, 2, 3]

    # Zero-shot mask classification with PixelDiT's own image-text attention
    # (inference only, for now). See san/model/pixeldit_classifier.py.
    cfg.MODEL.PIXELDIT.CLASSIFIER = CN()
    cfg.MODEL.PIXELDIT.CLASSIFIER.ENABLED = False
    # Regions to classify: "gt" = GT segments (one per class present; measures
    # classification alone, side adapter not used), "pred" = side adapter masks
    # (needs trained weights; masks weighted by objectness).
    cfg.MODEL.PIXELDIT.CLASSIFIER.MASK_SOURCE = "gt"
    # Block (1-based) whose attention q/k are used to score image tokens against
    # class tokens. The NUM_MASKED_BLOCKS blocks before it run on image tokens only,
    # with attention restricted to each token's region; their input is PixelDiT
    # layer READOUT_BLOCK - NUM_MASKED_BLOCKS - 1.
    cfg.MODEL.PIXELDIT.CLASSIFIER.READOUT_BLOCK = 5
    cfg.MODEL.PIXELDIT.CLASSIFIER.NUM_MASKED_BLOCKS = 1
    # "soft": bias log(P(same region) + SOFT_EPS); "hard": attend only within the
    # argmax region; "none": unrestricted image-only attention (ablation).
    cfg.MODEL.PIXELDIT.CLASSIFIER.MASK_ATTENTION = "soft"
    cfg.MODEL.PIXELDIT.CLASSIFIER.SOFT_EPS = 1e-4
    cfg.MODEL.PIXELDIT.CLASSIFIER.PROMPT_TEMPLATE = "A picture of a {target}"
    # Class tokens used as keys: "name" = tokens of the class name only,
    # "all" = all prompt tokens except BOS.
    cfg.MODEL.PIXELDIT.CLASSIFIER.TEXT_TOKENS = "name"
    # Pooling of a token's logits over a class's tokens: "lme" (log-mean-exp),
    # "max" or "mean". Heads are averaged.
    cfg.MODEL.PIXELDIT.CLASSIFIER.TOKEN_POOL = "lme"
    # Mask scores are divided by this before the softmax over classes.
    cfg.MODEL.PIXELDIT.CLASSIFIER.TEMPERATURE = 1.0
    # Noise levels for the classifier branch: continuous flow times t in (0, 1]
    # (same units as t2i/config.py timestep; the model receives 1000 * t). For each
    # t, PixelDiT runs (up to the classifier's input layer) on
    # x_t = (1 - t) * x + t * eps. Empty = reuse the stage-1 pass (clean image at
    # t2i timestep). With several values, one eval logs segment accuracy for each
    # (plus their average) and the sem_seg output averages their class probabilities.
    # The side adapter always uses the stage-1 pass.
    cfg.MODEL.PIXELDIT.CLASSIFIER.TIMESTEPS = []
    # eps is drawn per image from seed NOISE_SEED + crc32(file_name), identical for
    # every t, so runs are reproducible and timesteps are compared on the same noise.
    cfg.MODEL.PIXELDIT.CLASSIFIER.NOISE_SEED = 0
    # Removing the per-class offset of the scores (one class otherwise wins every
    # region). All listed variants are logged; the FIRST one is used for the sem_seg
    # output. "none"; "image": subtract each class's mean token score over the image
    # (region vs rest of image); "dataset": subtract each class's mean token score
    # over all eval images so far (label-free, transductive; noisy for the first
    # images).
    cfg.MODEL.PIXELDIT.CLASSIFIER.CALIBRATION = ["none"]
    # How image tokens are scored against classes.
    #   "qk": the READOUT_BLOCK's attention logits between image queries and class
    #         keys from a text-only pass (READOUT_BLOCK / NUM_MASKED_BLOCKS apply).
    #   "concept": ConceptAttention (Helbling et al., ICML 2025). The class-name
    #         tokens of each prompt form a concept stream that runs through every
    #         block next to the normal image + prompt pass: concepts attend to the
    #         image tokens and to each other, image and prompt never attend to them.
    #         Score = dot product of the image and concept attention outputs (before
    #         the output projection). Per-block results and their average are logged.
    cfg.MODEL.PIXELDIT.CLASSIFIER.READOUT = "qk"
    cfg.MODEL.PIXELDIT.CLASSIFIER.CONCEPT = CN()
    # Blocks 1..NUM_BLOCKS are run (PixelDiT has 14).
    cfg.MODEL.PIXELDIT.CLASSIFIER.CONCEPT.NUM_BLOCKS = 8
    # Blocks whose outputs are scored; empty = all. With more than one, "avg" (the
    # token scores averaged over them) gives the sem_seg output.
    cfg.MODEL.PIXELDIT.CLASSIFIER.CONCEPT.READOUT_BLOCKS = []
    # Blocks where image-to-image attention is restricted to each token's region
    # (MASK_ATTENTION soft/hard); image-to-prompt attention stays unrestricted.
    # Empty = plain ConceptAttention.
    cfg.MODEL.PIXELDIT.CLASSIFIER.CONCEPT.MASKED_BLOCKS = []
    # Softmax over classes of each image token's scores (as in ConceptAttention)
    # before averaging over masks.
    cfg.MODEL.PIXELDIT.CLASSIFIER.CONCEPT.TOKEN_SOFTMAX = True
    # Keep the image / prompt / concept residual streams in fp32 (matmuls stay in
    # the AMP dtype). Needed under float16: the prompt stream exceeds 65504 by block 7.
    cfg.MODEL.PIXELDIT.CLASSIFIER.CONCEPT.FP32_RESIDUAL = True
    # Prepend the same "Enhanced prompt" instruction (CHI) the image prompt
    # (t2i/config.py default_prompt) starts with, so class prompts are
    # "<CHI> User Prompt: A picture of a dog" as in generation.
    cfg.MODEL.PIXELDIT.CLASSIFIER.USE_CHI_PREFIX = False
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
