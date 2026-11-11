import contextlib
import logging
import zlib
from collections import defaultdict
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

from .criterion import SetCriterion
from .matcher import HungarianMatcher
from .pixeldit_classifier import PixelDiTRegionClassifier
from .side_adapter import build_side_adapter_network
from ..utils.amp import resolve_amp_dtype


from t2i.diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from t2i.diffusion.model.utils import get_weight_dtype
from t2i.diffusion.utils.config import PixDiTConfig, model_init_config
from t2i.diffusion.utils.logger import get_root_logger

from t2i.diffusion.model.flow_dpm import DPMS
from t2i.diffusion.model.dpm_solver import NoiseScheduleFlow
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


def model_timestep(device, t: Optional[float] = None) -> torch.Tensor:
    """The model-time value PixelDiT receives in run_diffusion_model for continuous
    flow time t (default local_config.timestep): DPM_Solver_Wrapper builds t in bf16,
    and get_model_input_time scales it by total_N for the discrete flow schedule."""
    t = local_config.timestep if t is None else t
    t = torch.tensor(t, device=device, dtype=torch.bfloat16)
    return t * NoiseScheduleFlow(schedule="discrete_flow").total_N


@contextlib.contextmanager
def override_timestep(t: float):
    """Temporarily change the timestep DPM_Solver_Wrapper.segment passes to PixelDiT."""
    old = local_config.timestep
    local_config.timestep = t
    try:
        yield
    finally:
        local_config.timestep = old


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
        oracle_classes: bool = False,
        oracle_assignment: str = "max_iou",
        oracle_mask_threshold: float = 0.5,
        oracle_min_score: float = 0.0,
        oracle_weight_by_objectness: bool = True,
        pixeldit_classifier: Optional[dict] = None,
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

        # Zero-shot PixelDiT classifier (inference only for now).
        self.classifier = None
        if pixeldit_classifier is not None:
            pixeldit_classifier = dict(pixeldit_classifier)
            self.classifier_mask_source = pixeldit_classifier.pop("mask_source")
            self.classifier_temperature = float(pixeldit_classifier.pop("temperature"))
            self.classifier_noise_seed = int(pixeldit_classifier.pop("noise_seed"))
            self.classifier_calibrations = list(pixeldit_classifier.pop("calibration"))
            bad = [c for c in self.classifier_calibrations if c not in ("none", "image", "dataset")]
            if not self.classifier_calibrations or bad or len(set(self.classifier_calibrations)) != len(self.classifier_calibrations):
                raise ValueError(
                    "MODEL.PIXELDIT.CLASSIFIER.CALIBRATION must be a non-empty list of distinct "
                    f"'none' / 'image' / 'dataset', got {self.classifier_calibrations}"
                )
            # Running per-class sums of token scores for "dataset": (dataset, t) -> [sum, count]
            self._calib_stats = {}
            timesteps = [float(t) for t in pixeldit_classifier.pop("timesteps")]
            if any(not 0.0 < t <= 1.0 for t in timesteps):
                raise ValueError(f"MODEL.PIXELDIT.CLASSIFIER.TIMESTEPS must lie in (0, 1], got {timesteps}")
            # None = reuse the stage-1 pass (clean image at local_config.timestep).
            self.classifier_timesteps = timesteps or [None]
            if pixeldit_classifier.pop("use_chi_prefix"):
                marker = "User Prompt: "
                if marker not in local_config.default_prompt:
                    raise ValueError("USE_CHI_PREFIX: no 'User Prompt: ' in t2i default_prompt")
                pixeldit_classifier["prompt_prefix"] = (
                    local_config.default_prompt.rsplit(marker, 1)[0] + marker
                )
            if self.classifier_mask_source not in ("gt", "pred"):
                raise ValueError(
                    f"MODEL.PIXELDIT.CLASSIFIER.MASK_SOURCE must be 'gt' or 'pred', got {self.classifier_mask_source!r}"
                )
            self.classifier = PixelDiTRegionClassifier(**pixeldit_classifier)
            if oracle_classes:
                raise ValueError(
                    "MODEL.PIXELDIT.CLASSIFIER.ENABLED and MODEL.SAN.ORACLE_CLASSES.ENABLED "
                    "both assign the mask classes; enable only one."
                )
        self._diag_stats = defaultdict(lambda: defaultdict(int))
        self._diag_images = defaultdict(int)
        self._concept_debug_logged = set()

        self.class_agnostic = class_agnostic
        if class_agnostic and not (oracle_classes or self.classifier is not None):
            raise ValueError(
                "MODEL.SAN.CLASS_AGNOSTIC predicts no classes, so it needs "
                "MODEL.SAN.ORACLE_CLASSES.ENABLED or MODEL.PIXELDIT.CLASSIFIER.ENABLED "
                "to label the masks at inference."
            )

        # PixelDiT layers fed to the side adapter (0 = patch embedding, k = block k).
        # Only blocks up to the deepest one are built and run.
        self.feature_layers = sorted(set(int(k) for k in feature_layers))
        if not self.feature_layers or self.feature_layers[0] < 0:
            raise ValueError(f"MODEL.PIXELDIT.FEATURE_LAYERS must be non-negative, got {feature_layers}")
        # Layers the stage-1 PixelDiT run must return; blocks needed by the
        # classifier (masked blocks + readout) are built but run separately.
        self._tap_layers = set(self.feature_layers)
        deepest_block = self.feature_layers[-1]
        if self.classifier is not None:
            self._tap_layers.add(self.classifier.input_layer)
            deepest_block = max(deepest_block, self.classifier.deepest_block)
        self._tap_layers = sorted(self._tap_layers)
        local_config.layer_count = deepest_block

        self.model, self.tokenizer, self.text_encoder = get_model()
        num_blocks = len(self.model.core.patch_blocks)
        if deepest_block > num_blocks:
            raise ValueError(
                f"Need PixelDiT block {deepest_block} (FEATURE_LAYERS {self.feature_layers}, "
                f"classifier readout block), but the model has only {num_blocks} patch blocks."
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

        # Class keys for the PixelDiT classifier, per dataset vocabulary. Built now,
        # while the text encoder is loaded and PixelDiT is still in fp32.
        self._vocab_keys = {}
        self._vocab_concepts = {}
        if self.classifier is not None:
            self._build_vocabulary_keys()

        del self.tokenizer
        del self.text_encoder

        # Store the frozen PixelDiT weights in the autocast dtype. Under fp32 weights,
        # autocast keeps a half-precision copy of every weight during the forward pass,
        # so this roughly removes 1.5x the backbone size from GPU memory. Any text
        # embeddings that need fp32 should be computed above, before this cast.
        self.amp_dtype = amp_dtype
        if amp_dtype is not None:
            for p in self.model.parameters():
                if not p.requires_grad:
                    p.data = p.data.to(amp_dtype)
            torch.cuda.empty_cache()

        # Trainable side adapter (kept in fp32; autocast handles the compute dtype).
        hidden_size = self.model.core.hidden_size
        self.side_adapter_network = build_side_adapter_network(
            side_adapter_cfg,
            {k: ShapeSpec(channels=hidden_size) for k in self.feature_layers},
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
            "oracle_classes": cfg.MODEL.SAN.ORACLE_CLASSES.ENABLED,
            "oracle_assignment": cfg.MODEL.SAN.ORACLE_CLASSES.ASSIGNMENT,
            "oracle_mask_threshold": cfg.MODEL.SAN.ORACLE_CLASSES.MASK_THRESHOLD,
            "oracle_min_score": cfg.MODEL.SAN.ORACLE_CLASSES.MIN_SCORE,
            "oracle_weight_by_objectness": cfg.MODEL.SAN.ORACLE_CLASSES.WEIGHT_BY_OBJECTNESS,
            "pixeldit_classifier": cls._classifier_args(cfg.MODEL.PIXELDIT.CLASSIFIER),
        }

    @staticmethod
    def _classifier_args(c):
        if not c.ENABLED:
            return None
        return {
            "mask_source": c.MASK_SOURCE,
            "temperature": c.TEMPERATURE,
            "readout_block": c.READOUT_BLOCK,
            "num_masked_blocks": c.NUM_MASKED_BLOCKS,
            "mask_attention": c.MASK_ATTENTION,
            "soft_eps": c.SOFT_EPS,
            "token_pool": c.TOKEN_POOL,
            "text_tokens": c.TEXT_TOKENS,
            "prompt_template": c.PROMPT_TEMPLATE,
            "readout": c.READOUT,
            "concept_num_blocks": c.CONCEPT.NUM_BLOCKS,
            "concept_readout_blocks": list(c.CONCEPT.READOUT_BLOCKS),
            "concept_masked_blocks": list(c.CONCEPT.MASKED_BLOCKS),
            "concept_token_softmax": c.CONCEPT.TOKEN_SOFTMAX,
            "fp32_residual": c.CONCEPT.FP32_RESIDUAL,
            "timesteps": list(c.TIMESTEPS),
            "noise_seed": c.NOISE_SEED,
            "calibration": list(c.CALIBRATION),
            "use_chi_prefix": c.USE_CHI_PREFIX,
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
    
    @torch.no_grad()
    def _build_vocabulary_keys(self, chunk_size: int = 16):
        """Readout-block keys of every class prompt, for each dataset vocabulary."""
        core = self.model.core
        device = next(core.parameters()).device
        # The text stream's adaLN depends on t, so the keys are built per timestep.
        conditions = {
            t: core.timestep_condition(model_timestep(device, t).float())
            for t in self.classifier_timesteps
        }
        self._vocab_conditions = conditions
        self._checked_conditions = set()
        log = logging.getLogger("san")
        if self.classifier.readout == "concept":
            # Concept tokens do not depend on t (their stream is run per image).
            for name, labels in self.class_labels.items():
                prompts = self.classifier.prompts(labels)
                embs, attn_mask, _ = self.embed_condition(prompts)
                if (attn_mask.sum(1) >= attn_mask.shape[1]).any():
                    log.warning("PixelDiT classifier: some class prompts hit the 300-token limit")
                select = self.classifier.class_token_mask(self.tokenizer, prompts, attn_mask)
                self._vocab_concepts[name] = self.classifier.build_concepts(core, embs[:, 0], select)
                log.info(
                    "PixelDiT classifier (concept readout): %d classes for %s, %d concept tokens, e.g. %r",
                    len(prompts), name, self._vocab_concepts[name][0].shape[0], prompts[0][-80:],
                )
            return
        for name, labels in self.class_labels.items():
            prompts = self.classifier.prompts(labels)
            keys = {t: [] for t in conditions}
            valid = {t: [] for t in conditions}
            for i in range(0, len(prompts), chunk_size):
                chunk = prompts[i : i + chunk_size]
                embs, attn_mask, _ = self.embed_condition(chunk)  # (n, 1, 300, D), (n, 300)
                if (attn_mask.sum(1) >= attn_mask.shape[1]).any():
                    log.warning("PixelDiT classifier: some class prompts hit the 300-token limit")
                select = self.classifier.class_token_mask(self.tokenizer, chunk, attn_mask)
                for t, condition in conditions.items():
                    k, v = self.classifier.build_text_keys(core, embs[:, 0], attn_mask, select, condition)
                    keys[t].append(k)
                    valid[t].append(v)
            for t in conditions:
                T = max(k.shape[1] for k in keys[t])
                k = torch.cat([F.pad(k, (0, 0, 0, 0, 0, T - k.shape[1])) for k in keys[t]])
                v = torch.cat(
                    [torch.cat([v, v.new_zeros(v.shape[0], T - v.shape[1])], dim=1) for v in valid[t]]
                )
                self._vocab_keys[(name, t)] = (k, v)
            log.info(
                "PixelDiT classifier: %d class prompts for %s (up to %d key tokens each), "
                "timesteps %s, e.g. %r",
                len(prompts), name, T, [self._t_label(t) for t in conditions], prompts[0][-80:],
            )

    @staticmethod
    def _t_label(t):
        return f"t={local_config.timestep:g} (stage 1, clean)" if t is None else f"t={t:g}"

    @torch.no_grad()
    def _classifier_passes(self, batched_inputs, images_tensor, segment_data):
        """PixelDiT tokens at the classifier's input layer and their adaLN condition,
        for every classifier timestep: [(t, tokens (B, L, hidden), condition,
        prompt tokens (B, Ltxt, hidden) at the same layer)]. t=None
        reuses the stage-1 pass; otherwise the image is noised to x_t = (1-t) x + t eps
        with per-image deterministic eps (the same for every t)."""
        layer = self.classifier.input_layer
        passes = []
        eps = None
        for t in self.classifier_timesteps:
            if t is None:
                passes.append((
                    t, segment_data["features"][layer], segment_data["condition"],
                    segment_data["text_features"][layer],
                ))
                continue
            if eps is None:
                eps = torch.empty_like(images_tensor, dtype=torch.float32)
                for b, inp in enumerate(batched_inputs):
                    key = str(inp.get("file_name", inp.get("image_id", b)))
                    g = torch.Generator(device=images_tensor.device)
                    g.manual_seed(self.classifier_noise_seed + zlib.crc32(key.encode()))
                    eps[b] = torch.randn(
                        eps.shape[1:], generator=g, device=images_tensor.device, dtype=torch.float32
                    )
            x_t = (1.0 - t) * images_tensor.float() + t * eps
            sd = {"feature_layers": [layer]}
            with override_timestep(t), torch.autocast(
                "cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None
            ):
                self.run_diffusion_model(x_t, sd)
            assert sd["hw"] == segment_data["hw"], (sd["hw"], segment_data["hw"])
            passes.append((t, sd["features"][layer], sd["condition"], sd["text_features"][layer]))
        # Once per timestep: the image pass must have used the same adaLN condition as
        # the class keys, otherwise image and text streams are at different t.
        for t, _, condition, _ in passes:
            if t in self._checked_conditions:
                continue
            ref = self._vocab_conditions[t].to(condition.device, torch.float32)
            err = (condition.float()[:1] - ref).abs().max().item()
            scale = ref.abs().max().item()
            rel = err / max(scale, 1e-6)
            msg = (
                f"PixelDiT classifier {self._t_label(t)}: image-pass timestep condition vs class-key "
                f"condition: max abs diff {err:.3g} (relative {rel:.3g})"
            )
            if rel > 0.2:  # far beyond bf16 rounding: the two streams are at different t
                raise RuntimeError(msg)
            logging.getLogger("san").log(logging.WARNING if rel > 2e-2 else logging.INFO, msg)
            self._checked_conditions.add(t)
        return passes

    def run_diffusion_model(self, image, segment_data):
        """One forward pass of the frozen PixelDiT on the clean image at the fixed
        timestep local_config.timestep, conditioned on the default prompt. Fills
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
            segment_data
        )

    def call_model(self, image_tensor, dataset_name):
        # Autocast here too (not only in the training loop) because evaluation runs
        # without it, and the frozen weights may be in half precision. Inside the
        # trainer's autocast with the same dtype this is a no-op.
        with torch.autocast(
            "cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None
        ):
            mask_logits, mask_preds, segment_data = self._call_model(image_tensor, dataset_name)
        return [x.float() for x in mask_logits], [x.float() for x in mask_preds], segment_data

    def _call_model(self, image_tensor, dataset_name):
        """Run the frozen backbone and the side adapter.

        Returns two lists with one entry per deep-supervision block of the side
        adapter during training (the last entry is the final prediction), and a
        single entry at inference:
          mask_logits: (B, Q, K + 1) class logits per mask, last column = no object
                       (K = 1 when MODEL.SAN.CLASS_AGNOSTIC)
          mask_preds:  (B, Q, h, w) mask logits on the patch grid
        and the PixelDiT segment_data (features of all tapped layers, condition, pos).
        Both lists are empty when the side adapter is not needed (GT-mask classifier
        diagnostic).
        """
        segment_data = {"feature_layers": self._tap_layers}
        # The backbone is frozen and nothing before it is trained, so no autograd
        # graph is needed through it, even during training.
        with torch.no_grad():
            self.run_diffusion_model(image_tensor, segment_data)

        h, w = segment_data["hw"]
        # {layer: (B, h*w, hidden)} -> {layer: (B, hidden, h, w)}. fp32 so the
        # fusion LayerNorm statistics are not computed in half precision.
        if self.classifier is not None and self.classifier_mask_source == "gt" and not self.training:
            return [], [], segment_data
        backbone_features = {
            k: segment_data["features"][k].detach().float().transpose(1, 2).reshape(-1, segment_data["features"][k].shape[-1], h, w)
            for k in self.feature_layers
        }
        mask_preds, mask_logits = self.side_adapter_network(image_tensor, backbone_features)
        return mask_logits, mask_preds, segment_data

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
        if self.training and self.classifier is not None:
            raise NotImplementedError(
                "MODEL.PIXELDIT.CLASSIFIER is inference-only for now (zero-shot diagnostic); "
                "disable it for training."
            )
        mask_logits, mask_preds, segment_data = self.call_model(images_tensor, dataset_name)
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

            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        elif self.classifier is not None:
            passes = self._classifier_passes(batched_inputs, images.tensor, segment_data)
            return self._classifier_inference(
                batched_inputs, images, mask_logits, mask_preds, segment_data, dataset_name, passes
            )
        else:
            mask_preds = mask_preds[-1]
            mask_logits = mask_logits[-1]
            # Inference
            mask_preds = F.interpolate(
                mask_preds,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_logits, mask_preds, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.oracle_classes:
                    # Replace the predicted class scores with one-hot GT classes.
                    # Matching is done at the network input resolution (unpadded),
                    # which is the resolution of the GT the test mapper provides.
                    oracle = retry_if_cuda_oom(self.oracle_mask_classification)(
                        mask_pred_result[:, : image_size[0], : image_size[1]],
                        self._gt_sem_seg(input_per_image, image_size),
                        num_classes=len(self.class_labels[dataset_name]),
                        ignore_label=self._ignore_label(dataset_name),
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
                    mask_cls_result, mask_pred_result, cls_are_probs=self.oracle_classes
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r
            return processed_results

    # -------------------------------------------------------- PixelDiT classifier
    def _classifier_inference(
        self, batched_inputs, images, mask_logits, mask_preds, segment_data, dataset_name, passes
    ):
        """Semantic segmentation with masks classified by PixelDiTRegionClassifier.

        MASK_SOURCE "gt": the regions are the GT segments (one per class present), so
        the result measures classification alone. "pred": side adapter masks and
        objectness, as in normal inference but with PixelDiT class scores.
        `passes` (from _classifier_passes) holds one PixelDiT pass per classifier
        timestep; class probabilities are averaged over them.
        """
        core = self.model.core
        num_classes = len(self.class_labels[dataset_name])
        ignore_label = self._ignore_label(dataset_name)
        h, w = segment_data["hw"]
        pad_h, pad_w = images.tensor.shape[-2:]
        stride = pad_h // h
        assert pad_h == h * stride and pad_w == w * stride, (images.tensor.shape, h, w)
        pos = segment_data["pos"]

        def classify(b, region_weights, pool_weights, token_valid):
            """Scores (K, C) for every (timestep, readout block, calibration) as
            (t, block, calib, scores), block = None for the q·k readout; and for each
            calibration the class probabilities averaged over timesteps (concept
            readout: of the block average, or the single readout block)."""
            all_scores, probs = [], {c: 0.0 for c in self.classifier_calibrations}
            n_valid = int(token_valid.sum())
            for t, tokens, condition, text_tokens in passes:
                cond = condition[b : b + 1] if condition.shape[0] > 1 else condition
                if self.classifier.readout == "concept":
                    debug = t not in self._concept_debug_logged
                    results = self.classifier.concept_scores(
                        core, tokens[b : b + 1], text_tokens[b : b + 1], cond, pos,
                        self._vocab_concepts[dataset_name], region_weights, pool_weights, token_valid,
                        debug=debug,
                    )
                    if debug:
                        self._concept_debug_logged.add(t)
                        rows = [
                            f"  block {blk:>2} ({dt}): " + "  ".join(
                                f"{n} {m:9.3g}{'' if ok else ' NON-FINITE'}"
                                for n, (m, ok) in zip(("|x|", "|y|", "|cpt|", "|o_x|", "|o_c|"), st)
                            )
                            for blk, dt, *st in self.classifier.last_debug
                        ]
                        logging.getLogger("san").info(
                            "PixelDiT concept readout, max activations on the first image (%s; "
                            "float16 overflows above 65504):\n%s",
                            self._t_label(t), "\n".join(rows),
                        )
                    final = "avg" if len(results) > 1 else results[0][0]
                else:
                    keys, valid = self._vocab_keys[(dataset_name, t)]
                    scores, image_mean = self.classifier(
                        core, tokens[b : b + 1], cond, pos, region_weights, pool_weights, keys, valid,
                        token_valid=token_valid,
                    )
                    results, final = [(None, scores, image_mean)], None
                for block, scores, image_mean in results:
                    scores, image_mean = scores.float(), image_mean.float()
                    if "dataset" in probs:
                        acc = self._calib_stats.setdefault((dataset_name, t, block), [0.0, 0])
                        acc[0] = acc[0] + image_mean * n_valid
                        acc[1] += n_valid
                    for calib in self.classifier_calibrations:
                        if calib == "image":
                            s = scores - image_mean
                        elif calib == "dataset":
                            acc = self._calib_stats[(dataset_name, t, block)]
                            s = scores - acc[0] / acc[1]
                        else:
                            s = scores
                        all_scores.append((t, block, calib, s))
                        if block == final:
                            probs[calib] = probs[calib] + F.softmax(s / self.classifier_temperature, dim=-1)
            return all_scores, {c: p / len(passes) for c, p in probs.items()}

        processed_results = []
        for b, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            ih, iw = image_size
            # Tokens whose patch overlaps the image (the rest is ImageList padding).
            token_valid = torch.zeros(h, w, dtype=torch.bool, device=images.tensor.device)
            token_valid[: -(-ih // stride), : -(-iw // stride)] = True
            token_valid = token_valid.flatten()
            with torch.autocast(
                "cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None
            ):
                if self.classifier_mask_source == "gt":
                    gt = self._gt_sem_seg(input_per_image, image_size)
                    labeled = (gt != ignore_label) & (gt >= 0) & (gt < num_classes)
                    seg_classes = torch.unique(gt[labeled])
                    if seg_classes.numel() == 0:
                        r = torch.zeros(num_classes, ih, iw, device=gt.device)
                    else:
                        masks = torch.zeros(len(seg_classes), pad_h, pad_w, device=gt.device)
                        masks[:, :ih, :iw] = (gt[None] == seg_classes[:, None, None]).float()
                        # Fraction of each token's patch covered by each segment.
                        weights = F.avg_pool2d(masks[None], kernel_size=stride)[0].flatten(1)  # (K, L)
                        all_scores, probs = classify(b, weights.T, weights, token_valid)
                        r = torch.einsum(
                            "kc,khw->chw", probs[self.classifier_calibrations[0]], masks[:, :ih, :iw]
                        )
                        for t, block, calib, scores in all_scores:
                            label = f"{self._t_label(t)} | {calib}"
                            if block is not None:
                                label += f" | block {block}"
                            self._update_diag_stats(dataset_name, label, scores, seg_classes)
                        if len(passes) > 1:
                            for calib, p in probs.items():
                                self._update_diag_stats(
                                    dataset_name, f"mean of timesteps | {calib}", p, seg_classes
                                )
                        self._maybe_log_diag_stats(dataset_name)
                else:
                    m = mask_preds[-1][b]  # (Q, h, w) logits on the token grid
                    objectness = F.softmax(mask_logits[-1][b].float(), dim=-1)[:, :-1].sum(-1)  # (Q,)
                    member = m.float().sigmoid().flatten(1)  # (Q, L)
                    _, probs = classify(b, (objectness[:, None] * member).T, member, token_valid)
                    mask_cls = objectness[:, None] * probs[self.classifier_calibrations[0]]
                    masks_up = F.interpolate(
                        m[None].float(), size=(pad_h, pad_w), mode="bilinear", align_corners=False
                    )[0][:, :ih, :iw]
                    r = self.semantic_inference(mask_cls, masks_up, cls_are_probs=True)
            r = retry_if_cuda_oom(sem_seg_postprocess)(r.float(), image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results

    def _update_diag_stats(self, dataset_name, label, scores, seg_classes):
        """Segment-level top-1 / top-5 accuracy on GT segments (each segment counts
        once, regardless of its area), and how often each class is predicted."""
        st = self._diag_stats[(dataset_name, label)]
        finite = torch.isfinite(scores).all(-1)
        st["nonfinite"] += int((~finite).sum())
        # Non-finite rows never count as correct (topk would pick an arbitrary class).
        scores = torch.where(finite[:, None], scores, torch.full_like(scores, float("-inf")))
        topk = scores.topk(min(5, scores.shape[-1]), dim=-1).indices
        topk = torch.where(finite[:, None], topk, torch.full_like(topk, -1))
        st["segments"] += len(seg_classes)
        st["top1"] += int((topk[:, 0] == seg_classes).sum())
        st["top5"] += int((topk == seg_classes[:, None]).any(1).sum())
        for c in topk[:, 0].tolist():
            if c >= 0:
                st[("pred", c)] += 1

    def _maybe_log_diag_stats(self, dataset_name, log_every=50):
        self._diag_images[dataset_name] += 1
        n = self._diag_images[dataset_name]
        if n % log_every:
            return
        names = self.class_labels[dataset_name]
        lines = [f"PixelDiT classifier on {dataset_name} GT segments after {n} images:"]
        for (ds, label), st in self._diag_stats.items():
            if ds != dataset_name or not st["segments"]:
                continue
            seg = st["segments"]
            preds = sorted(
                ((v, k[1]) for k, v in st.items() if isinstance(k, tuple)), reverse=True
            )[:3]
            most = ", ".join(f"{names[c]} {100.0 * v / seg:.0f}%" for v, c in preds)
            if st["nonfinite"]:
                most += f"; NON-FINITE scores in {st['nonfinite']} segments"
            lines.append(
                f"  {label:>48}: top-1 {100.0 * st['top1'] / seg:5.2f}%  "
                f"top-5 {100.0 * st['top5'] / seg:5.2f}%  ({seg} segments; most predicted: {most})"
            )
        logging.getLogger("san").info("\n".join(lines))

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