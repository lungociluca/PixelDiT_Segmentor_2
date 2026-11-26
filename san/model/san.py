from typing import List

import open_clip
import torch
from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Optional
import pyrallis
from argparse import Namespace
import json

from .concept_classifier import (
    ConceptLoRA,
    ConceptMaskClassifier,
    MaskEmbedHead,
    encode_concepts,
    ensemble_probs,
    instruction_prefix,
    reduce_synonyms,
)
from .criterion import SetCriterion
from .matcher import HungarianMatcher
from .side_adapter import build_side_adapter_network
from ..utils.amp import resolve_amp_dtype, resolve_backbone_dtype


from t2i.diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from t2i.diffusion.model.utils import get_weight_dtype
from t2i.diffusion.utils.config import PixDiTConfig, model_init_config
from t2i.diffusion.utils.logger import get_root_logger

from t2i.diffusion.model.flow_dpm import DPMS
from t2i.diffusion.model.trainer import PixDiTTrainer

import t2i.config as local_config
from detectron2.data import MetadataCatalog


def get_dataset_labels(dataset_name: str) -> List[str]:
    """Class names used to build text embeddings for `dataset_name`.
    
    Priority:
      1. local_config.label_json_overrides[dataset_name] (a catseg-style JSON list)
      2. MetadataCatalog stuff_classes (falls back to thing_classes)
    The order must match the contiguous class ids of that dataset's GT masks.
    """
    overrides = getattr(local_config, "label_json_overrides", {}) or {}
    if dataset_name in overrides:
        with open(overrides[dataset_name]) as f:
            return [x for x in json.load(f) if x != "unknown"]
    meta = MetadataCatalog.get(dataset_name)
    classes = meta.stuff_classes if hasattr(meta, "stuff_classes") else meta.thing_classes
    return [c.strip() for c in classes]

@dataclass
class PixelDiTInference(PixDiTConfig):
    config: Optional[str] = "diffusion_configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml"
    model_path: Optional[str] = ".."
    work_dir: Optional[str] = None
    version: str = "sigma"
    txt_file: str = "asset/samples/samples_mini.txt"
    json_file: Optional[str] = None
    sample_nums: int = 1
    bs: int = 1
    cfg_scale: float = 3.5
    sampling_algo: str = "flow_dpm-solver"
    seed: int = 0
    dataset: str = "custom"
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_image_size: Optional[int] = None
    custom_height: Optional[int] = None
    custom_width: Optional[int] = None
    start_index: int = 0
    end_index: int = 30_000
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    if_save_dirname: bool = False
    negative_prompt: str = ""  # optional negative prompt applied at inference

def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    # torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)

def get_model():
    args = Namespace(config='diffusion_configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml')
    config = args = pyrallis.parse(config_class=PixelDiTInference, config_path=args.config, args=[])
    from t2i.tools.download import resolve_checkpoint
    args.model_path = resolve_checkpoint("t2i/pixeldit_t2i_v1.pth")

    args.image_size = local_config.image_size
    if args.custom_image_size:
        args.image_size = args.custom_image_size
        print(f"custom_image_size: {args.image_size}")

    set_env(args.seed, args.image_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    # only support fixed latent size currently
    latent_size = args.image_size
    max_sequence_length = config.text_encoder.model_max_length
    flow_shift = config.scheduler.flow_shift
    guidance_type = "classifier-free"
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [max(0, args.interval_guidance[0]), min(1, args.interval_guidance[1])]
    default_sample_steps = 50
    sample_steps = args.step if args.step != -1 else default_sample_steps

    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}")

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name, device=device)
    for param in text_encoder.parameters():
        param.requires_grad = False

    null_caption_token = tokenizer(
        args.negative_prompt if hasattr(args, "negative_prompt") and len(args.negative_prompt) > 0 else "",
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask, use_cache=False)[0]

    # model setting
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model, use_fp32_attention=config.model.get("fp32_attention", False), **model_kwargs
    )
    # The PixelDiT backbone is frozen.
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    state_dict = torch.load("t2i/pixeldit_t2i_v1.pth", map_location=lambda storage, loc: storage)
    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    return model.eval().to(torch.float32), tokenizer, text_encoder.eval()


@META_ARCH_REGISTRY.register()
class SAN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        criterion: SetCriterion,
        side_adapter_cfg: CfgNode,
        feature_layers: List[int],
        class_agnostic: bool,
        train_dataset_names: List[str],
        test_dataset_names: List[str],
        sem_seg_postprocess_before_inference: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
        backbone_dtype: torch.dtype = torch.float32,
        oracle_classes: bool = False,
        oracle_assignment: str = "max_iou",
        oracle_mask_threshold: float = 0.5,
        oracle_min_score: float = 0.0,
        oracle_weight_by_objectness: bool = True,
        concept_enabled: bool = False,
        concept_layers: Optional[List[int]] = None,
        concept_template: str = "A picture of a {target}",
        concept_templates: Optional[List[str]] = None,
        concept_instruction_context: bool = False,
        concept_clean_names: bool = True,
        concept_split_synonyms: bool = True,
        concept_synonym_reduce: str = "max",
        concept_pool: str = "mean",
        concept_temperature: float = 1.0,
        concept_pixel_softmax: bool = True,
        concept_binarize_masks: bool = False,
        concept_weight_by_objectness: bool = True,
        concept_pixel_baseline: bool = False,
        concept_timestep: float = 0.4,
        concept_noise_seed: int = 0,
        concept_gt_masks: bool = False,
        concept_lora: bool = False,
        concept_lora_rank: int = 16,
        concept_lora_alpha: float = 16.0,
        concept_cosine: bool = True,
        concept_init_logit_scale: float = 20.0,
        concept_label_smoothing: float = 0.1,
        concept_holdout_classes: Optional[List[str]] = None,
        concept_train_mode: str = "gt_masks",
        concept_loss_weight: float = 2.0,
        concept_detach_masks: bool = True,
        mask_head: bool = False,
        mask_head_layers: Optional[List[int]] = None,
        mask_head_embed_dim: int = 768,
        mask_head_hidden_dim: int = 1024,
        mask_head_text_proj: str = "linear",
        mask_head_init_logit_scale: float = 20.0,
        mask_head_dropout: float = 0.0,
        mask_head_loss_weight: float = 1.0,
        mask_head_ensemble_alpha: float = 0.5,
        mask_head_only: bool = False,
        image_lora: bool = False,
        image_lora_rank: int = 16,
        image_lora_alpha: float = 16.0,
        image_lora_grad_checkpoint: bool = True,
        oracle_mask_stats: bool = True,
    ):
        super().__init__()
        self.size_divisibility = 16
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.criterion = criterion

        assert oracle_assignment in ("max_iou", "majority"), oracle_assignment
        self.oracle_classes = oracle_classes
        self.oracle_assignment = oracle_assignment
        self.oracle_mask_threshold = oracle_mask_threshold
        self.oracle_min_score = oracle_min_score
        self.oracle_weight_by_objectness = oracle_weight_by_objectness

        # Concept-attention mask classification (inference only, training-free).
        self.concept_enabled = concept_enabled
        # MASK_HEAD.ONLY: the mask head is the only classifier, no concept attention.
        self.mask_head_only = mask_head_only
        if mask_head_only:
            if not (concept_enabled and mask_head):
                raise ValueError(
                    "MODEL.CONCEPT.MASK_HEAD.ONLY needs MODEL.CONCEPT.ENABLED and MODEL.CONCEPT.MASK_HEAD.ENABLED."
                )
            if concept_lora:
                raise ValueError(
                    "MODEL.CONCEPT.MASK_HEAD.ONLY trains no concept LoRA: set MODEL.CONCEPT.LORA.ENABLED False."
                )
            if concept_pixel_baseline:
                raise ValueError("MODEL.CONCEPT.PIXEL_BASELINE needs the concept classifier (MASK_HEAD.ONLY False).")
            concept_layers = []
        self.concept_layers = sorted(set(int(k) for k in (concept_layers or [])))
        self.concept_weight_by_objectness = concept_weight_by_objectness
        self.concept_pixel_baseline = concept_pixel_baseline
        if concept_enabled and oracle_classes:
            raise ValueError(
                "MODEL.CONCEPT.ENABLED and MODEL.SAN.ORACLE_CLASSES.ENABLED both label the "
                "masks at inference; enable only one of them."
            )
        if concept_enabled and not mask_head_only and (not self.concept_layers or self.concept_layers[0] < 1):
            raise ValueError(
                f"MODEL.CONCEPT.LAYERS must list PixelDiT blocks >= 1, got {concept_layers}"
            )
        if concept_pixel_baseline and not concept_enabled:
            raise ValueError("MODEL.CONCEPT.PIXEL_BASELINE needs MODEL.CONCEPT.ENABLED.")
        if concept_gt_masks and not concept_enabled:
            raise ValueError("MODEL.CONCEPT.GT_MASKS needs MODEL.CONCEPT.ENABLED.")
        if concept_timestep >= 1:
            raise ValueError(f"MODEL.CONCEPT.TIMESTEP must be < 1, got {concept_timestep}")
        self.concept_timestep = float(concept_timestep)
        self.concept_noise_seed = int(concept_noise_seed)
        self.concept_gt_masks = concept_gt_masks
        # dataset_name -> [correctly classified GT segments, GT segments]
        self.concept_gt_stats = {}
        if concept_lora and not concept_enabled:
            raise ValueError("MODEL.CONCEPT.LORA.ENABLED needs MODEL.CONCEPT.ENABLED.")
        self.concept_lora = concept_lora
        self.concept_cosine = concept_cosine and concept_lora
        self.concept_label_smoothing = concept_label_smoothing
        self.concept_adapter = None
        if concept_train_mode not in ("gt_masks", "joint"):
            raise ValueError(
                f"MODEL.CONCEPT.LORA.TRAIN_MODE must be 'gt_masks' or 'joint', got {concept_train_mode!r}"
            )
        # Trainable open-vocabulary classifier: concept LoRA (+ mask head), or the
        # mask head alone.
        self.cls_trainable = concept_lora or mask_head_only
        self.concept_joint = self.cls_trainable and concept_train_mode == "joint"
        self.concept_loss_weight = concept_loss_weight
        self.concept_detach_masks = concept_detach_masks

        # Trained mask-embedding head on PixelDiT image features (MASK_HEAD) and
        # low-rank adapters on the image stream of the concept pass (IMAGE_LORA).
        # Both are trained with the concept classifier (they need LORA.ENABLED).
        if (mask_head or image_lora) and not self.cls_trainable:
            raise ValueError(
                "MODEL.CONCEPT.MASK_HEAD.ENABLED / IMAGE_LORA.ENABLED are trained with the concept "
                "classifier and need MODEL.CONCEPT.LORA.ENABLED (or MODEL.CONCEPT.MASK_HEAD.ONLY)."
            )
        self.mask_head_layers = sorted(set(int(k) for k in (mask_head_layers or []))) if mask_head else []
        if mask_head and (not self.mask_head_layers or self.mask_head_layers[0] < 1):
            raise ValueError(f"MODEL.CONCEPT.MASK_HEAD.LAYERS must list PixelDiT blocks >= 1, got {mask_head_layers}")
        if not 0.0 <= mask_head_ensemble_alpha <= 1.0:
            raise ValueError(f"MODEL.CONCEPT.MASK_HEAD.ENSEMBLE_ALPHA must be in [0, 1], got {mask_head_ensemble_alpha}")
        self.mask_head_loss_weight = mask_head_loss_weight
        self.mask_head_alpha = mask_head_ensemble_alpha
        self.mask_head = None
        self.image_adapter = None
        self.image_lora_grad_checkpoint = image_lora_grad_checkpoint
        # Deepest PixelDiT block the concept pass has to run.
        self.concept_depth = max(self.concept_layers + self.mask_head_layers) if concept_enabled else 0
        self.oracle_mask_stats = oracle_mask_stats
        # dataset_name -> [sum of best IoU, GT segments, recalled @0.5, @0.75, masks kept, images]
        self.oracle_stats = {}

        self.class_agnostic = class_agnostic
        if class_agnostic and not (oracle_classes or concept_enabled):
            raise ValueError(
                "MODEL.SAN.CLASS_AGNOSTIC predicts no classes, so it needs "
                "MODEL.CONCEPT.ENABLED or MODEL.SAN.ORACLE_CLASSES.ENABLED to label the "
                "masks at inference."
            )

        # PixelDiT layers fed to the side adapter (0 = patch embedding, k = block k).
        # Only blocks up to the deepest one are built and run.
        self.feature_layers = sorted(set(int(k) for k in feature_layers))
        if not self.feature_layers or self.feature_layers[0] < 0:
            raise ValueError(f"MODEL.PIXELDIT.FEATURE_LAYERS must be non-negative, got {feature_layers}")
        # Blocks built = deepest layer needed by the side adapter or by the concept
        # classifier. Training only runs the side adapter's layers.
        local_config.layer_count = max(self.feature_layers[-1], self.concept_depth)

        self.model, self.tokenizer, self.text_encoder = get_model()
        num_blocks = len(self.model.core.patch_blocks)
        if self.feature_layers[-1] > num_blocks:
            raise ValueError(
                f"MODEL.PIXELDIT.FEATURE_LAYERS {self.feature_layers} goes deeper than the "
                f"{num_blocks} PixelDiT patch blocks."
            )
        if concept_enabled and self.concept_depth > num_blocks:
            raise ValueError(
                f"MODEL.CONCEPT.LAYERS {self.concept_layers} / MASK_HEAD.LAYERS "
                f"{self.mask_head_layers} go deeper than the {num_blocks} PixelDiT patch blocks."
            )

        # The diffusion model is always conditioned on the dataset-agnostic default prompt.
        default_emb, default_mask, default_tokens = self.embed_condition([local_config.default_prompt])
        self.default_emb = default_emb.to(torch.float32)      # (1, 1, 300, D)
        self.default_emb_mask = default_mask                  # (1, 300)
        self.default_tokens_count = default_tokens[0]

        # Class names per dataset (order = contiguous GT class ids).
        self.default_dataset = train_dataset_names[0]
        self.class_labels = {}   # dataset_name -> List[str]
        for name in dict.fromkeys(list(train_dataset_names) + list(test_dataset_names)):
            self.class_labels[name] = get_dataset_labels(name)

        # One concept vector per class name, per dataset (needs the text encoder,
        # so computed before it is dropped). dataset_name -> (K, txt_embed_dim).
        self.concept_embs = {}
        # dataset_name -> (N,) class index of every concept (N > K with synonyms).
        self.concept_variant_class = {}
        if concept_synonym_reduce not in ("max", "mean"):
            raise ValueError(
                f"MODEL.CONCEPT.SYNONYM_REDUCE must be 'max' or 'mean', got {concept_synonym_reduce!r}"
            )
        self.concept_synonym_reduce = concept_synonym_reduce
        self.concept_position = 0
        self.concept_classifier = None
        if concept_enabled and not mask_head_only:
            self.concept_classifier = ConceptMaskClassifier(
                temperature=concept_temperature,
                pixel_softmax=concept_pixel_softmax,
                binarize_masks=concept_binarize_masks,
                learned=concept_lora,
                num_layers=len(self.concept_layers),
                init_logit_scale=concept_init_logit_scale,
            )
        if concept_enabled:
            # Class-name embeddings: concept tokens, and the mask head's text side.
            templates = list(concept_templates or []) or [concept_template]
            context = instruction_prefix(local_config.default_prompt) if concept_instruction_context else ""
            for name, labels in self.class_labels.items():
                embs, variant_class, self.concept_position = encode_concepts(
                    self.tokenizer,
                    self.text_encoder,
                    labels,
                    template=templates,
                    pool=concept_pool,
                    device=local_config.device,
                    context=context,
                    clean_names=concept_clean_names,
                    split_synonyms=concept_split_synonyms,
                )
                self.concept_embs[name] = embs
                self.concept_variant_class[name] = variant_class.to(local_config.device)
            get_root_logger().info(
                ("Mask-head-only classifier, " if mask_head_only else "")
                + f"Concept classifier: PixelDiT layers {self.concept_layers}, "
                f"{ {k: v.shape[0] for k, v in self.concept_embs.items()} } concepts "
                f"({len(templates)} templates, instruction context {concept_instruction_context}), "
                f"text position {self.concept_position}"
            )

        # Training vocabulary without the held-out classes:
        # dataset_name -> (indices of the kept classes, class id -> new id or -1).
        self.concept_train_vocab = {}
        holdout = set(concept_holdout_classes or [])
        if self.cls_trainable and holdout:
            for name in train_dataset_names:
                labels = self.class_labels[name]
                unknown = holdout - set(labels)
                if unknown:
                    raise ValueError(
                        f"MODEL.CONCEPT.LORA.HOLDOUT_CLASSES {sorted(unknown)} are not classes of '{name}'."
                    )
                keep = torch.tensor(
                    [i for i, l in enumerate(labels) if l not in holdout],
                    dtype=torch.long, device=local_config.device,
                )
                remap = torch.full((len(labels),), -1, dtype=torch.long, device=local_config.device)
                remap[keep] = torch.arange(keep.numel(), device=local_config.device)
                self.concept_train_vocab[name] = (keep, remap)

        del self.tokenizer
        del self.text_encoder

        # amp_dtype: autocast dtype of the trainable side adapter (None = fp32).
        # backbone_dtype: dtype the frozen PixelDiT is stored and run in. The two
        # differ when autocast is float16: PixelDiT was trained in bf16, and float16
        # does not have bf16's range, so the backbone then stays in fp32 (see
        # san.utils.amp.resolve_backbone_dtype). Storing the weights directly in a
        # half dtype (rather than fp32 + autocast) avoids keeping two copies of them.
        self.amp_dtype = amp_dtype
        self.backbone_dtype = backbone_dtype
        if backbone_dtype != torch.float32:
            for p in self.model.parameters():
                if not p.requires_grad:
                    p.data = p.data.to(backbone_dtype)
            torch.cuda.empty_cache()
        get_root_logger().info(
            f"Precision: side adapter autocast={amp_dtype or torch.float32}, "
            f"frozen PixelDiT={backbone_dtype}"
        )

        # Trainable side adapter (kept in fp32; autocast handles the compute dtype).
        hidden_size = self.model.core.hidden_size
        self.side_adapter_network = build_side_adapter_network(
            side_adapter_cfg,
            {k: ShapeSpec(channels=hidden_size) for k in self.feature_layers},
        )

        if self.cls_trainable:
            # Trainable concept path. With TRAIN_MODE "gt_masks", training optimises
            # only the concept classifier (see concept_train_forward) and the side
            # adapter is frozen; with "joint" both are trained. MASK_HEAD.ONLY: no
            # concept adapter, the mask head (+ image LoRA) is the classifier.
            if concept_lora:
                self.concept_adapter = ConceptLoRA(
                    hidden_size, self.concept_layers[-1], rank=concept_lora_rank, alpha=concept_lora_alpha
                )
            if not self.concept_joint:
                self.side_adapter_network.requires_grad_(False)
            if mask_head:
                text_dim = next(iter(self.concept_embs.values())).shape[-1]
                self.mask_head = MaskEmbedHead(
                    hidden_size,
                    text_dim,
                    num_layers=len(self.mask_head_layers),
                    embed_dim=mask_head_embed_dim,
                    hidden_dim=mask_head_hidden_dim,
                    text_proj=mask_head_text_proj,
                    init_logit_scale=mask_head_init_logit_scale,
                    dropout=mask_head_dropout,
                )
            if image_lora:
                # Same low-rank form as the concept adapter, on qkv_x / proj_x of
                # every block the concept pass runs. Zero-initialised up projections:
                # training starts from the frozen model.
                self.image_adapter = ConceptLoRA(
                    hidden_size, self.concept_depth, rank=image_lora_rank, alpha=image_lora_alpha
                )

    @classmethod
    def from_config(cls, cfg):
        ## copied from maskformer2
        # Loss parameters
        no_object_weight = cfg.MODEL.SAN.NO_OBJECT_WEIGHT
        # loss weights
        class_weight = cfg.MODEL.SAN.CLASS_WEIGHT
        dice_weight = cfg.MODEL.SAN.DICE_WEIGHT
        mask_weight = cfg.MODEL.SAN.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }
        aux_weight_dict = {}
        for i in range(len(cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS) - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks"]

        class_agnostic = cfg.MODEL.SAN.CLASS_AGNOSTIC
        criterion = SetCriterion(
            num_classes=1 if class_agnostic else cfg.MODEL.SAN.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.SAN.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.SAN.IMPORTANCE_SAMPLE_RATIO,
        )
        ## end of copy
        return {
            "criterion": criterion,
            "side_adapter_cfg": cfg,
            "feature_layers": list(cfg.MODEL.PIXELDIT.FEATURE_LAYERS),
            "class_agnostic": class_agnostic,
            "train_dataset_names": list(cfg.DATASETS.TRAIN),
            "test_dataset_names": list(cfg.DATASETS.TEST),
            "sem_seg_postprocess_before_inference": cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,
            "amp_dtype": resolve_amp_dtype(cfg),
            "backbone_dtype": resolve_backbone_dtype(cfg),
            "oracle_classes": cfg.MODEL.SAN.ORACLE_CLASSES.ENABLED,
            "oracle_assignment": cfg.MODEL.SAN.ORACLE_CLASSES.ASSIGNMENT,
            "oracle_mask_threshold": cfg.MODEL.SAN.ORACLE_CLASSES.MASK_THRESHOLD,
            "oracle_min_score": cfg.MODEL.SAN.ORACLE_CLASSES.MIN_SCORE,
            "oracle_weight_by_objectness": cfg.MODEL.SAN.ORACLE_CLASSES.WEIGHT_BY_OBJECTNESS,
            "concept_enabled": cfg.MODEL.CONCEPT.ENABLED,
            "concept_layers": list(cfg.MODEL.CONCEPT.LAYERS),
            "concept_template": cfg.MODEL.CONCEPT.TEMPLATE,
            "concept_templates": list(cfg.MODEL.CONCEPT.TEMPLATES),
            "concept_instruction_context": cfg.MODEL.CONCEPT.INSTRUCTION_CONTEXT,
            "concept_clean_names": cfg.MODEL.CONCEPT.CLEAN_NAMES,
            "concept_split_synonyms": cfg.MODEL.CONCEPT.SPLIT_SYNONYMS,
            "concept_synonym_reduce": cfg.MODEL.CONCEPT.SYNONYM_REDUCE,
            "concept_pool": cfg.MODEL.CONCEPT.POOL,
            "concept_temperature": cfg.MODEL.CONCEPT.TEMPERATURE,
            "concept_pixel_softmax": cfg.MODEL.CONCEPT.PIXEL_SOFTMAX,
            "concept_binarize_masks": cfg.MODEL.CONCEPT.BINARIZE_MASKS,
            "concept_weight_by_objectness": cfg.MODEL.CONCEPT.WEIGHT_BY_OBJECTNESS,
            "concept_pixel_baseline": cfg.MODEL.CONCEPT.PIXEL_BASELINE,
            "concept_timestep": cfg.MODEL.CONCEPT.TIMESTEP,
            "concept_noise_seed": cfg.MODEL.CONCEPT.NOISE_SEED,
            "concept_gt_masks": cfg.MODEL.CONCEPT.GT_MASKS,
            "concept_lora": cfg.MODEL.CONCEPT.LORA.ENABLED,
            "concept_lora_rank": cfg.MODEL.CONCEPT.LORA.RANK,
            "concept_lora_alpha": cfg.MODEL.CONCEPT.LORA.ALPHA,
            "concept_cosine": cfg.MODEL.CONCEPT.LORA.COSINE,
            "concept_init_logit_scale": cfg.MODEL.CONCEPT.LORA.INIT_LOGIT_SCALE,
            "concept_label_smoothing": cfg.MODEL.CONCEPT.LORA.LABEL_SMOOTHING,
            "concept_holdout_classes": list(cfg.MODEL.CONCEPT.LORA.HOLDOUT_CLASSES),
            "concept_train_mode": cfg.MODEL.CONCEPT.LORA.TRAIN_MODE,
            "concept_loss_weight": cfg.MODEL.CONCEPT.LORA.LOSS_WEIGHT,
            "concept_detach_masks": cfg.MODEL.CONCEPT.LORA.DETACH_MASKS,
            "mask_head": cfg.MODEL.CONCEPT.MASK_HEAD.ENABLED,
            "mask_head_layers": list(cfg.MODEL.CONCEPT.MASK_HEAD.LAYERS),
            "mask_head_embed_dim": cfg.MODEL.CONCEPT.MASK_HEAD.EMBED_DIM,
            "mask_head_hidden_dim": cfg.MODEL.CONCEPT.MASK_HEAD.HIDDEN_DIM,
            "mask_head_text_proj": cfg.MODEL.CONCEPT.MASK_HEAD.TEXT_PROJ,
            "mask_head_init_logit_scale": cfg.MODEL.CONCEPT.MASK_HEAD.INIT_LOGIT_SCALE,
            "mask_head_dropout": cfg.MODEL.CONCEPT.MASK_HEAD.DROPOUT,
            "mask_head_loss_weight": cfg.MODEL.CONCEPT.MASK_HEAD.LOSS_WEIGHT,
            "mask_head_ensemble_alpha": cfg.MODEL.CONCEPT.MASK_HEAD.ENSEMBLE_ALPHA,
            "mask_head_only": cfg.MODEL.CONCEPT.MASK_HEAD.ONLY,
            "image_lora": cfg.MODEL.CONCEPT.IMAGE_LORA.ENABLED,
            "image_lora_rank": cfg.MODEL.CONCEPT.IMAGE_LORA.RANK,
            "image_lora_alpha": cfg.MODEL.CONCEPT.IMAGE_LORA.ALPHA,
            "image_lora_grad_checkpoint": cfg.MODEL.CONCEPT.IMAGE_LORA.GRAD_CHECKPOINT,
            "oracle_mask_stats": cfg.MODEL.SAN.ORACLE_CLASSES.MASK_STATS,
        }
    
    def embed_condition(self, prompts, with_chi_window=False):
        max_length = 506 if with_chi_window else 300
        caption_token = self.tokenizer(
            prompts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).to(local_config.device)
        select_index = [0] + list(range(-300 + 1, 0)) if with_chi_window else list(range(300))
        caption_embs_list = [self.text_encoder(caption_token.input_ids[[i]], caption_token.attention_mask[[i]])[0][:, None][
            :, :, select_index
        ] for i in range(caption_token.input_ids.shape[0])]
        caption_embs = torch.cat(caption_embs_list, dim=0)
        tokens_count = caption_token.attention_mask.sum(dim=-1)
        emb_masks = caption_token.attention_mask[:, select_index]
        return caption_embs, emb_masks, tokens_count
    
    def run_diffusion_model(self, image, segment_data, timestep=None):
        """One forward pass of the frozen PixelDiT on `image` at flow time `timestep`
        (default local_config.timestep, for the clean image), conditioned on the
        default prompt. `image` must already carry the noise of that timestep. Fills
        segment_data["hw"] and segment_data["features"] (see PixDiT_T2I.forward)."""
        emb = self.default_emb
        emb_mask = self.default_emb_mask
        tokens_count = self.default_tokens_count
        model_kwargs = dict(data_info={"img_hw": image.shape[-2], "aspect_ratio": 1}, mask=emb_mask)
        segment_data["tokens_count"] = tokens_count

        dpm_solver = DPMS(
            self.model.forward_with_dpmsolver,
            condition=emb,
            uncondition=None,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
            cfg_scale=3.5
        )

        samples = dpm_solver.segment(
            image,
            segment_data,
            timestep=timestep,
        )

    def _run_backbone(self, image_tensor, segment_data, timestep=None, grad=False):
        # The backbone is frozen and nothing before it is trained, so no autograd
        # graph is needed through it, even during training. The exception is
        # grad=True, for training the concept adapter: the graph then only covers
        # the concept tokens, the image stream has nothing that requires grad.
        # The backbone runs in its own dtype: this overrides the surrounding
        # autocast (and switches it off for an fp32 backbone, whose weights must
        # not be cast down to the side adapter's float16).
        half_backbone = self.backbone_dtype != torch.float32
        with torch.set_grad_enabled(grad), torch.autocast(
            image_tensor.device.type,
            dtype=self.backbone_dtype if half_backbone else None,
            enabled=half_backbone,
        ):
            self.run_diffusion_model(image_tensor, segment_data, timestep)

    def _concept_variants(self, dataset_name, keep=None):
        """Indices of the concepts to run (None = all) and the class index of each
        of them (renumbered to positions in `keep` when given), plus the number
        of classes their scores are reduced to."""
        variant_class = self.concept_variant_class[dataset_name]
        if keep is None:
            return None, variant_class, len(self.class_labels[dataset_name])
        new_id = torch.full(
            (len(self.class_labels[dataset_name]),), -1, dtype=torch.long, device=variant_class.device
        )
        new_id[keep.to(variant_class.device)] = torch.arange(keep.numel(), device=variant_class.device)
        sel = (new_id[variant_class] >= 0).nonzero().squeeze(1)
        return sel, new_id[variant_class[sel]], keep.numel()

    def _concept_inputs(self, dataset_name, keep=None):
        if dataset_name not in self.concept_embs:
            raise KeyError(f"No concept embeddings for '{dataset_name}'.")
        concepts = self.concept_embs[dataset_name]
        sel, _, _ = self._concept_variants(dataset_name, keep)
        return {
            "concepts": concepts if sel is None else concepts[sel],
            "concept_layers": self.concept_layers,
            "concept_position": self.concept_position,
            "concept_adapter": self.concept_adapter,
            "concept_cosine": self.concept_cosine,
        }

    def concept_pass(self, image_tensor, dataset_name, train=False, keep=None):
        """Backbone pass that only scores the class concepts, on the image noised
        to MODEL.CONCEPT.TIMESTEP: x_t = (1 - t) * image + t * noise (the flow the
        model was trained on). Returns {"scores": {layer: (B, h*w, C)}, "hw": (h, w)}.

        train: fresh noise instead of the fixed evaluation noise, and gradients
               flow into the concept adapter.
        keep:  indices of the classes to use (default: the whole vocabulary)."""
        t = self.concept_timestep
        x = image_tensor
        if t > 0:
            g = None
            if not train:
                g = torch.Generator(device=image_tensor.device).manual_seed(self.concept_noise_seed)
            noise = torch.randn(
                image_tensor.shape, generator=g, device=image_tensor.device, dtype=image_tensor.dtype
            )
            x = (1.0 - t) * image_tensor + t * noise
        segment_data = {"feature_layers": list(self.mask_head_layers)}
        if self.mask_head_only:
            if dataset_name not in self.concept_embs:
                raise KeyError(f"No concept embeddings for '{dataset_name}'.")
        else:
            segment_data.update(self._concept_inputs(dataset_name, keep))
        if self.image_adapter is not None:
            segment_data["image_adapter"] = self.image_adapter
            segment_data["grad_checkpoint"] = self.image_lora_grad_checkpoint
        # Gradients through the backbone only reach an adapter; without one (mask
        # head only) the backbone pass needs no graph.
        grad = train and (self.concept_adapter is not None or self.image_adapter is not None)
        self._run_backbone(x, segment_data, timestep=t if t > 0 else None, grad=grad)
        scores = None
        if "concept_scores" in segment_data:
            scores = self._reduce_concept_scores(segment_data["concept_scores"], dataset_name, keep)
        return {
            # {layer: (B, h*w, K)} concept scores (None with MASK_HEAD.ONLY).
            "scores": scores,
            "hw": segment_data["hw"],
            # {layer: (B, h*w, hidden)} image tokens for the mask head ({} without it).
            "features": segment_data.get("features", {}),
            "keep": keep,
        }

    def _head_logits(self, concept_out, i, weights, dataset_name):
        """Mask-head class logits (S, K) of image i for mask weights (S, h*w) on
        the patch grid, or None without MASK_HEAD."""
        if self.mask_head is None:
            return None
        keep = concept_out.get("keep")
        sel, variant_class, num_classes = self._concept_variants(dataset_name, keep)
        concepts = self.concept_embs[dataset_name]
        if sel is not None:
            concepts = concepts[sel]
        feats = {k: v[i] for k, v in concept_out["features"].items()}
        with torch.autocast(weights.device.type, enabled=False):
            return self.mask_head.logits(
                feats, weights, concepts, variant_class, num_classes, self.concept_synonym_reduce
            )

    def _classification_loss(self, concept_logits, head_logits, targets):
        """Cross-entropy of the concept classifier (+ the mask head), with the
        accuracies logged. Returns the loss dict."""
        ls = self.concept_label_smoothing
        losses = {}
        if concept_logits is not None:
            losses["loss_concept_cls"] = F.cross_entropy(concept_logits, targets, label_smoothing=ls)
        if head_logits is not None:
            losses["loss_mask_head_cls"] = self.mask_head_loss_weight * F.cross_entropy(
                head_logits, targets, label_smoothing=ls
            )
        try:
            from detectron2.utils.events import get_event_storage

            storage = get_event_storage()
            if concept_logits is not None:
                storage.put_scalar(
                    "concept_cls_acc", (concept_logits.argmax(-1) == targets).float().mean().item()
                )
                storage.put_scalar("concept_logit_scale", self.concept_classifier.logit_scale.exp().item())
            if head_logits is not None:
                with torch.no_grad():
                    storage.put_scalar(
                        "mask_head_acc", (head_logits.argmax(-1) == targets).float().mean().item()
                    )
                    if concept_logits is not None:
                        ens = ensemble_probs(concept_logits.softmax(-1), head_logits, self.mask_head_alpha)
                        storage.put_scalar("ensemble_acc", (ens.argmax(-1) == targets).float().mean().item())
                    storage.put_scalar("mask_head_logit_scale", self.mask_head.logit_scale.exp().item())
        except AssertionError:
            pass  # no event storage outside a trainer
        return losses

    def _reduce_concept_scores(self, scores, dataset_name, keep=None):
        """{layer: (B, P, N) per-concept scores} -> {layer: (B, P, K) per-class
        scores}: the synonyms of a class are combined (MODEL.CONCEPT.SYNONYM_REDUCE)
        per patch and layer, before any scaling or softmax."""
        _, variant_class, num_classes = self._concept_variants(dataset_name, keep)
        return {
            k: reduce_synonyms(v, variant_class, num_classes, self.concept_synonym_reduce)
            for k, v in scores.items()
        }

    def concept_train_forward(self, batched_inputs, images, dataset_name):
        """Training step of the concept classifier (MODEL.CONCEPT.LORA.ENABLED):
        every ground-truth segment is classified by pooling the concept scores
        over its mask; cross-entropy over the (non held-out) class names."""
        keep, remap = self.concept_train_vocab.get(dataset_name, (None, None))
        with torch.autocast(
            "cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None
        ):
            concept_out = self.concept_pass(images.tensor, dataset_name, train=True, keep=keep)
        h, w = concept_out["hw"]
        H_pad, W_pad = images.tensor.shape[-2:]
        with torch.autocast(images.tensor.device.type, enabled=False):
            pixel_logits = None
            if self.concept_classifier is not None:
                pixel_logits = self.concept_classifier.pixel_scores(concept_out["scores"])  # (B, h*w, C)
            logits, head_logits, targets = [], [], []
            for i, x in enumerate(batched_inputs):
                instances = x["instances"].to(local_config.device)
                masks, classes = instances.gt_masks, instances.gt_classes
                if remap is not None:
                    classes = remap[classes]
                    masks, classes = masks[classes >= 0], classes[classes >= 0]
                if classes.numel() == 0:
                    continue
                padded = F.pad(
                    masks.float(), (0, W_pad - masks.shape[-1], 0, H_pad - masks.shape[-2])
                )
                # Fraction of every PixelDiT patch covered by each segment.
                weights = F.adaptive_avg_pool2d(padded[None], (h, w))[0].flatten(1)  # (S, h*w)
                visible = weights.sum(-1) > 0
                if pixel_logits is not None:
                    logits.append(self.concept_classifier.pool_logits(pixel_logits[i], weights[visible]))
                hl = self._head_logits(concept_out, i, weights[visible], dataset_name)
                if hl is not None:
                    head_logits.append(hl)
                targets.append(classes[visible])
            if not targets or sum(t.numel() for t in targets) == 0:
                return self._zero_classification_loss(pixel_logits)
            targets = torch.cat(targets)
            logits = torch.cat(logits) if logits else None
            head_logits = torch.cat(head_logits) if head_logits else None
            return self._classification_loss(logits, head_logits, targets)

    def _zero_classification_loss(self, ref):
        """Zero losses (with every trainable classifier part in the graph) for a
        batch without usable segments. ref: the concept pixel logits, or None
        with MASK_HEAD.ONLY."""
        losses = {}
        zero = 0.0
        if ref is not None:
            zero = ref.sum() * 0.0
            losses["loss_concept_cls"] = zero
        if self.mask_head is not None:
            params = list(self.mask_head.parameters())
            if ref is None and self.image_adapter is not None:
                params += list(self.image_adapter.parameters())
            losses["loss_mask_head_cls"] = zero + sum(p.sum() * 0.0 for p in params)
        return losses

    def call_model(self, image_tensor, dataset_name):
        # Autocast here too (not only in the training loop) because evaluation runs
        # without it, and the frozen weights may be in half precision. Inside the
        # trainer's autocast with the same dtype this is a no-op.
        with torch.autocast(
            "cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None
        ):
            mask_logits, mask_preds, concept_out = self._call_model(image_tensor, dataset_name)
        return [x.float() for x in mask_logits], [x.float() for x in mask_preds], concept_out

    def _call_model(self, image_tensor, dataset_name):
        """Run the frozen backbone and the side adapter.

        Returns two lists with one entry per deep-supervision block of the side
        adapter during training (the last entry is the final prediction), and a
        single entry at inference:
          mask_logits: (B, Q, K + 1) class logits per mask, last column = no object
                       (K = 1 when MODEL.SAN.CLASS_AGNOSTIC)
          mask_preds:  (B, Q, h, w) mask logits on the patch grid
        and, at inference with MODEL.CONCEPT.ENABLED, a dict (None otherwise):
          "scores": {layer: (B, h*w, C)} image-token / class-concept similarities
          "hw":     (h, w) PixelDiT patch grid
        """
        segment_data = {"feature_layers": self.feature_layers}
        use_concepts = self.concept_enabled and not self.training
        # Concepts share this (clean image) pass only when no noise is requested;
        # otherwise they get their own noised pass below.
        # The mask head taps its own layers and the image adapter changes the image
        # stream, so either one needs the separate concept pass.
        share_pass = (
            use_concepts and self.concept_timestep <= 0
            and self.mask_head is None and self.image_adapter is None
        )
        if share_pass:
            segment_data.update(self._concept_inputs(dataset_name))
        self._run_backbone(image_tensor, segment_data)

        for k, v in segment_data["features"].items():
            if not torch.isfinite(v).all():
                raise FloatingPointError(
                    f"PixelDiT layer {k} features contain inf/NaN with the backbone in "
                    f"{self.backbone_dtype}. Set MODEL.PIXELDIT.PRECISION to 'auto' or "
                    "'float32' (float16 overflows where bfloat16 does not)."
                )

        h, w = segment_data["hw"]
        # {layer: (B, h*w, hidden)} -> {layer: (B, hidden, h, w)}. fp32 so the
        # fusion LayerNorm statistics are not computed in half precision.
        backbone_features = {
            k: v.detach().float().transpose(1, 2).reshape(v.shape[0], v.shape[-1], h, w)
            for k, v in segment_data["features"].items()
        }
        mask_preds, mask_logits = self.side_adapter_network(image_tensor, backbone_features)
        concept_out = None
        if share_pass:
            concept_out = {
                "scores": self._reduce_concept_scores(segment_data["concept_scores"], dataset_name),
                "hw": (h, w),
            }
        elif use_concepts:
            concept_out = self.concept_pass(image_tensor, dataset_name)
        return mask_logits, mask_preds, concept_out

    def forward(self, batched_inputs):
        images = [x["image"].to(local_config.device).float() / 255.0 for x in batched_inputs]
        images = [x * 2 - 1 for x in images]
        images = ImageList.from_tensors(
            images,
            self.size_divisibility,
            pad_value=-1.0,
        )
        images_tensor = images.tensor

        dataset_name = self._dataset_name(batched_inputs)
        if self.concept_gt_masks and not self.training:
            return self.gt_mask_inference(batched_inputs, images, dataset_name)
        if self.cls_trainable and self.training and not self.concept_joint:
            return self.concept_train_forward(batched_inputs, images, dataset_name)
        mask_logits, mask_preds, concept_out = self.call_model(images_tensor, dataset_name)
        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(local_config.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            outputs = {
                "pred_logits": mask_logits[-1],
                "pred_masks": mask_preds[-1],
                "aux_outputs": [
                    {
                        "pred_logits": aux_pred_logits,
                        "pred_masks": aux_pred_masks,
                    }
                    for aux_pred_logits, aux_pred_masks in zip(
                        mask_logits[:-1], mask_preds[:-1]
                    )
                ],
            }

            # Matching and losses in fp32, whatever the trainer's autocast dtype:
            # they sum over thousands of points and take logs of probabilities.
            with torch.autocast(images_tensor.device.type, enabled=False):
                losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if self.concept_joint and targets is not None:
                cls_losses = self.concept_joint_loss(
                    images_tensor, mask_preds[-1], gt_instances, self.criterion.last_indices, dataset_name
                )
                for k, v in cls_losses.items():
                    losses[k] = self.concept_loss_weight * v
            return losses
        else:
            mask_preds = mask_preds[-1]
            mask_logits = mask_logits[-1]
            # Inference
            concept_cls = concept_pixels = None
            if self.concept_enabled:
                if dataset_name not in self.concept_embs:
                    raise KeyError(f"No concept embeddings for '{dataset_name}'.")
                with torch.autocast(images_tensor.device.type, enabled=False):
                    if self.concept_pixel_baseline:
                        # Diagnostic: per-patch class probabilities, no masks.
                        concept_pixels = F.interpolate(
                            self.concept_classifier.pixel_map(
                                concept_out["scores"], concept_out["hw"]
                            ),
                            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                            mode="bilinear",
                            align_corners=False,
                        )
                    elif self.mask_head_only:
                        # (B, Q, C) class probabilities from the mask head alone.
                        concept_cls = self._ensemble_with_head(
                            None, concept_out, mask_preds, dataset_name
                        )
                    else:
                        # (B, Q, C) class probabilities, pooled on the patch grid.
                        concept_cls = self.concept_classifier(
                            concept_out["scores"], mask_preds, concept_out["hw"]
                        )
                        if self.mask_head is not None:
                            concept_cls = self._ensemble_with_head(
                                concept_cls, concept_out, mask_preds, dataset_name
                            )
            mask_preds = F.interpolate(
                mask_preds,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            processed_results = []
            for i, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(
                mask_logits, mask_preds, batched_inputs, images.image_sizes
            )):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if concept_pixels is not None:
                    processed_results[-1]["sem_seg"] = retry_if_cuda_oom(sem_seg_postprocess)(
                        concept_pixels[i], image_size, height, width
                    )
                    continue

                if concept_cls is not None:
                    # Replace the (object / no object) scores with open-vocabulary
                    # class probabilities from concept attention.
                    cls_probs = concept_cls[i]
                    if self.concept_weight_by_objectness:
                        objectness = F.softmax(mask_cls_result.float(), dim=-1)[:, :-1].sum(-1)
                        cls_probs = cls_probs * objectness[:, None]
                    mask_cls_result = cls_probs

                if self.oracle_classes:
                    # Replace the predicted class scores with one-hot GT classes.
                    # Matching is done at the network input resolution (unpadded),
                    # which is the resolution of the GT the test mapper provides.
                    gt_sem_seg = self._gt_sem_seg(input_per_image, image_size)
                    oracle = retry_if_cuda_oom(self.oracle_mask_classification)(
                        mask_pred_result[:, : image_size[0], : image_size[1]],
                        gt_sem_seg,
                        num_classes=len(self.class_labels[dataset_name]),
                        ignore_label=self._ignore_label(dataset_name),
                    )
                    if self.oracle_mask_stats:
                        retry_if_cuda_oom(self.accumulate_mask_stats)(
                            mask_pred_result[:, : image_size[0], : image_size[1]],
                            mask_cls_result,
                            gt_sem_seg,
                            dataset_name,
                        )
                    if self.oracle_weight_by_objectness:
                        # P(mask is an object) = 1 - P(no object).
                        objectness = F.softmax(mask_cls_result.float(), dim=-1)[:, :-1].sum(-1)
                        oracle = oracle * objectness[:, None]
                    mask_cls_result = oracle

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result,
                    mask_pred_result,
                    cls_are_probs=self.oracle_classes or concept_cls is not None,
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r
            return processed_results

    @torch.no_grad()
    def _ensemble_with_head(self, concept_cls, concept_out, mask_preds, dataset_name):
        """(B, Q, K) concept probabilities of the predicted masks, ensembled with
        the mask head (MASK_HEAD.ENSEMBLE_ALPHA). concept_cls None (MASK_HEAD.ONLY):
        the mask head's probabilities alone."""
        h, w = concept_out["hw"]
        m = mask_preds.float()
        if tuple(m.shape[-2:]) != (h, w):
            m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        m = m.sigmoid().flatten(2)  # (B, Q, h*w)
        if self.concept_classifier is not None and self.concept_classifier.binarize_masks:
            m = (m > self.concept_classifier.mask_threshold).float()
        out = []
        for i in range(m.shape[0]):
            hl = self._head_logits(concept_out, i, m[i], dataset_name)
            if concept_cls is None:
                out.append(hl.float().softmax(-1) * (m[i].sum(-1, keepdim=True) > 0))
            else:
                out.append(ensemble_probs(concept_cls[i], hl, self.mask_head_alpha))
        return torch.stack(out)

    def concept_joint_loss(self, images_tensor, mask_preds, gt_instances, indices, dataset_name):
        """Cross-entropy of the concept classifier on the predicted masks that the
        Hungarian matcher assigned to ground-truth segments (TRAIN_MODE "joint").

        mask_preds: (B, Q, hm, wm) mask logits of the last side-adapter output.
        indices:    per image, (query indices, GT segment indices) from the matcher.
        Returns the loss dict of _classification_loss (+ the mask head's loss).
        """
        keep, remap = self.concept_train_vocab.get(dataset_name, (None, None))
        with torch.autocast(
            "cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None
        ):
            concept_out = self.concept_pass(images_tensor, dataset_name, train=True, keep=keep)
        h, w = concept_out["hw"]
        with torch.autocast(images_tensor.device.type, enabled=False):
            pixel_logits = None
            if self.concept_classifier is not None:
                pixel_logits = self.concept_classifier.pixel_scores(concept_out["scores"])  # (B, h*w, C)
            masks = mask_preds.float()
            if self.concept_detach_masks:
                masks = masks.detach()
            if tuple(masks.shape[-2:]) != (h, w):
                masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
            masks = masks.sigmoid().flatten(2)  # (B, Q, h*w)
            logits, head_logits, targets = [], [], []
            for i, (src, tgt) in enumerate(indices):
                classes = gt_instances[i].gt_classes[tgt.to(local_config.device)]
                src = src.to(local_config.device)
                if remap is not None:
                    classes = remap[classes]
                    src, classes = src[classes >= 0], classes[classes >= 0]
                if classes.numel() == 0:
                    continue
                if pixel_logits is not None:
                    logits.append(self.concept_classifier.pool_logits(pixel_logits[i], masks[i, src]))
                hl = self._head_logits(concept_out, i, masks[i, src], dataset_name)
                if hl is not None:
                    head_logits.append(hl)
                targets.append(classes)
            if not targets:
                return self._zero_classification_loss(pixel_logits)
            targets = torch.cat(targets)
            logits = torch.cat(logits) if logits else None
            head_logits = torch.cat(head_logits) if head_logits else None
            return self._classification_loss(logits, head_logits, targets)

    @torch.no_grad()
    def gt_mask_inference(self, batched_inputs, images, dataset_name):
        """Classify the ground-truth segments with concept attention
        (MODEL.CONCEPT.GT_MASKS). The side adapter is not run. Every GT class
        present in an image gives one mask; it is painted with the class
        probabilities the classifier predicts for it."""
        with torch.autocast(
            "cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None
        ):
            concept_out = self.concept_pass(images.tensor, dataset_name)
        h, w = concept_out["hw"]
        H_pad, W_pad = images.tensor.shape[-2:]
        num_classes = len(self.class_labels[dataset_name])
        ignore_label = self._ignore_label(dataset_name)
        pixel_scores = None
        if self.concept_classifier is not None:
            pixel_scores = self.concept_classifier.pixel_scores(concept_out["scores"])  # (B, h*w, C)
        stats = self.concept_gt_stats.setdefault(dataset_name, [0, 0])

        processed_results = []
        for i, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            gt = self._gt_sem_seg(input_per_image, image_size)  # (H, W), unpadded input size
            valid = (gt != ignore_label) & (gt >= 0) & (gt < num_classes)
            gt_classes = torch.unique(gt[valid])
            sem_seg = torch.zeros(num_classes, *gt.shape, device=gt.device, dtype=torch.float32)
            if gt_classes.numel() > 0:
                masks = (gt[None] == gt_classes[:, None, None]).float()  # (S, H, W)
                # Fraction of every PixelDiT patch covered by each segment.
                padded = F.pad(masks, (0, W_pad - gt.shape[1], 0, H_pad - gt.shape[0]))
                weights = F.adaptive_avg_pool2d(padded[None], (h, w))[0].flatten(1)  # (S, h*w)
                if self.mask_head_only:
                    hl = self._head_logits(concept_out, i, weights, dataset_name)
                    with torch.autocast(weights.device.type, enabled=False):
                        probs = hl.float().softmax(-1)                                  # (S, C)
                else:
                    probs = self.concept_classifier.pool(pixel_scores[i], weights)   # (S, C)
                    if self.mask_head is not None:
                        hl = self._head_logits(concept_out, i, weights, dataset_name)
                        probs = ensemble_probs(probs, hl, self.mask_head_alpha)
                sem_seg = torch.einsum("sc,shw->chw", probs, masks)
                stats[0] += int((probs.argmax(dim=-1) == gt_classes).sum())
                stats[1] += int(gt_classes.numel())
                if stats[1] // 500 != (stats[1] - gt_classes.numel()) // 500:
                    get_root_logger().info(
                        f"[{dataset_name}] GT-mask classification top-1: "
                        f"{100.0 * stats[0] / stats[1]:.2f}% of {stats[1]} segments"
                    )
            processed_results.append(
                {"sem_seg": retry_if_cuda_oom(sem_seg_postprocess)(sem_seg, image_size, height, width)}
            )
        return processed_results

    def _dataset_name(self, batched_inputs):
        names = {x.get("meta", {}).get("dataset_name", self.default_dataset) for x in batched_inputs}
        assert len(names) == 1, f"All images in a batch must be from the same dataset, got {names}"
        name = names.pop()
        if name not in self.class_labels:
            raise KeyError(
                f"No class names for '{name}'. Add it to DATASETS.TEST (or TRAIN) so its "
                "vocabulary is loaded at model construction."
            )
        return name

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            labels = targets_per_image.gt_classes
            if self.class_agnostic:
                labels = torch.zeros_like(labels)  # every GT segment is "object"
            new_targets.append(
                {
                    "labels": labels,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred, cls_are_probs=False):
        # cls_are_probs: mask_cls is already (Q, C) class probabilities without the
        # no-object column (oracle mode), so skip the softmax / no-object removal.
        if not cls_are_probs:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    # ------------------------------------------------------------------ oracle
    @staticmethod
    def _ignore_label(dataset_name):
        return getattr(MetadataCatalog.get(dataset_name), "ignore_label", 255)

    def _gt_sem_seg(self, input_per_image, image_size):
        """GT semantic map (H, W) at the unpadded network-input resolution."""
        if "sem_seg" not in input_per_image:
            raise KeyError(
                "MODEL.SAN.ORACLE_CLASSES.ENABLED needs the ground truth at inference, but "
                "the test mapper did not provide 'sem_seg'. Make sure the test dataset "
                "dicts have 'sem_seg_file_name'."
            )
        gt = input_per_image["sem_seg"].to(local_config.device)
        if tuple(gt.shape[-2:]) != tuple(image_size):
            gt = F.interpolate(
                gt[None, None].float(), size=tuple(image_size), mode="nearest"
            )[0, 0].long()
        return gt

    @torch.no_grad()
    def accumulate_mask_stats(self, mask_pred, mask_cls, sem_seg_gt, dataset_name, log_every=200):
        """Class-agnostic quality of the predicted masks (ORACLE_CLASSES.MASK_STATS),
        against every GT class region of the image (semantic GT: one segment per
        class present). Accumulated per dataset and logged every `log_every`
        images, and returned by mask_stats():
          mean best IoU: for every GT segment, the IoU of the best predicted mask
                         (binarised at ORACLE_CLASSES.MASK_THRESHOLD), averaged;
          recall@0.5/0.75: fraction of GT segments with such a mask;
          objects/image: masks with objectness > 0.5 per image.
        Unlabelled pixels are ignored, as in the oracle assignment."""
        num_classes = len(self.class_labels[dataset_name])
        ignore_label = self._ignore_label(dataset_name)
        st = self.oracle_stats.setdefault(dataset_name, [0.0, 0, 0, 0, 0, 0])
        objectness = F.softmax(mask_cls.float(), dim=-1)[:, :-1].sum(-1)
        st[4] += int((objectness > 0.5).sum())
        st[5] += 1
        valid = (sem_seg_gt != ignore_label) & (sem_seg_gt >= 0) & (sem_seg_gt < num_classes)
        gt_classes = torch.unique(sem_seg_gt[valid])
        if gt_classes.numel() > 0:
            pred = ((mask_pred.float().sigmoid() > self.oracle_mask_threshold) & valid[None]).flatten(1).float()
            gt = (sem_seg_gt[None] == gt_classes[:, None, None]).flatten(1).float()
            inter = pred @ gt.T                                                   # (Q, S)
            union = pred.sum(1, keepdim=True) + gt.sum(1)[None] - inter
            best = (inter / union.clamp(min=1)).max(dim=0).values                 # (S,)
            st[0] += float(best.sum())
            st[1] += int(best.numel())
            st[2] += int((best >= 0.5).sum())
            st[3] += int((best >= 0.75).sum())
        if st[5] % log_every == 0:
            get_root_logger().info(f"[{dataset_name}] masks: {self.format_mask_stats(dataset_name)}")

    def mask_stats(self, dataset_name):
        """Accumulated mask statistics of `dataset_name` (see accumulate_mask_stats)."""
        st = self.oracle_stats.get(dataset_name)
        if not st or st[5] == 0:
            return {}
        n = max(st[1], 1)
        return {
            "mean_best_iou": 100.0 * st[0] / n,
            "recall@0.5": 100.0 * st[2] / n,
            "recall@0.75": 100.0 * st[3] / n,
            "objects_per_image": st[4] / st[5],
            "gt_segments": st[1],
            "images": st[5],
        }

    def format_mask_stats(self, dataset_name):
        m = self.mask_stats(dataset_name)
        if not m:
            return "no images"
        return (
            f"mean best IoU {m['mean_best_iou']:.2f}, recall@0.5 {m['recall@0.5']:.2f}, "
            f"recall@0.75 {m['recall@0.75']:.2f}, {m['objects_per_image']:.1f} objects/image "
            f"({m['gt_segments']} GT segments, {m['images']} images)"
        )

    @torch.no_grad()
    def oracle_mask_classification(self, mask_pred, sem_seg_gt, num_classes, ignore_label):
        """Assign each predicted mask the GT class it overlaps best.

        mask_pred:  (Q, H, W) mask logits.
        sem_seg_gt: (H, W) GT class ids, `ignore_label` for unlabeled pixels.
        Returns (Q, C) one-hot class probabilities; masks with no (sufficient)
        overlap with any labeled GT segment get an all-zero row (= no object).
        """
        Q = mask_pred.shape[0]
        oracle = torch.zeros(Q, num_classes, device=mask_pred.device, dtype=torch.float32)

        valid = (sem_seg_gt != ignore_label) & (sem_seg_gt >= 0) & (sem_seg_gt < num_classes)
        gt_classes = torch.unique(sem_seg_gt[valid])
        if gt_classes.numel() == 0:
            return oracle

        # Binarized predicted masks, restricted to labeled pixels so that unlabeled
        # regions neither help nor hurt the matching.
        pred = (mask_pred.float().sigmoid() > self.oracle_mask_threshold) & valid[None]
        pred = pred.flatten(1).float()                                           # (Q, P)
        gt = (sem_seg_gt[None] == gt_classes[:, None, None]).flatten(1).float()  # (K, P)

        # fp32 on purpose: pixel counts overflow fp16.
        inter = pred @ gt.T                                                      # (Q, K)
        pred_area = pred.sum(1, keepdim=True)                                    # (Q, 1)
        if self.oracle_assignment == "max_iou":
            union = pred_area + gt.sum(1)[None] - inter
            score = inter / union.clamp(min=1)
        else:  # "majority": fraction of the mask's pixels lying on each GT class
            score = inter / pred_area.clamp(min=1)

        best_score, best_k = score.max(dim=1)
        best_inter = inter.gather(1, best_k[:, None]).squeeze(1)
        keep = (best_inter > 0) & (best_score >= self.oracle_min_score)
        oracle[keep, gt_classes[best_k[keep]]] = 1.0
        return oracle

    @property
    def device(self):
        return torch.device(local_config.device)