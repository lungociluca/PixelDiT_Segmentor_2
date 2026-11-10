from typing import List
import math
import einops

import open_clip
import torch
from detectron2.config import configurable
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
from .side_adapter import build_side_adapter_network
from ..utils.amp import resolve_amp_dtype


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
    for name, param in model.named_parameters():
        param.requires_grad = ("learnable_token" in name) and local_config.save_learned_tokens
        
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
        train_dataset_names: List[str],
        test_dataset_names: List[str],
        sem_seg_postprocess_before_inference: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
        oracle_classes: bool = False,
        oracle_assignment: str = "max_iou",
        oracle_mask_threshold: float = 0.5,
        oracle_min_score: float = 0.0,
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

        self.model, self.tokenizer, self.text_encoder = get_model()

        # The diffusion model is always conditioned on the dataset-agnostic default prompt.
        default_emb, default_mask, default_tokens = self.embed_condition([local_config.default_prompt])
        self.default_emb = default_emb.to(torch.float32)      # (1, 1, 300, D)
        self.default_emb_mask = default_mask                  # (1, 300)
        self.default_tokens_count = default_tokens[0]

        # Per-dataset class embeddings, computed once while the text encoder is loaded.
        # Training datasets stay on GPU; validation-only ones are kept on CPU to save memory.
        self.default_dataset = train_dataset_names[0]
        self.class_labels = {}   # dataset_name -> List[str]
        self.class_embs = {}     # dataset_name -> (C, 1, 300, D)
        for name in dict.fromkeys(list(train_dataset_names) + list(test_dataset_names)):
            labels = get_dataset_labels(name)
            if labels in self.class_labels.values():
                # identical vocabulary already embedded (e.g. coco train/test splits)
                src = next(k for k, v in self.class_labels.items() if v == labels)
                self.class_labels[name], self.class_embs[name] = labels, self.class_embs[src]
                continue
            embs = self.precompute_prompts(labels)

            if name not in train_dataset_names:
                embs = embs.cpu()
            self.class_labels[name], self.class_embs[name] = labels, embs
        # Cache of token-weighted class vectors, valid only while weights are frozen (eval).
        self._language_target_cache = {}

        del self.tokenizer
        del self.text_encoder

        # Store the frozen PixelDiT weights in the autocast dtype. Under fp32 weights, 
        # autocast keeps a half-precision copy of every weight during the forward pass,
        # so this roughly removes 1.5x the backbone size from GPU memory. Trainable
        # learnable_token* params stay fp32 for the optimizer. Done after the class
        # embeddings above so those are still computed in fp32.
        self.amp_dtype = amp_dtype
        if amp_dtype is not None:
            for p in self.model.parameters():
                if not p.requires_grad:
                    p.data = p.data.to(amp_dtype)
            torch.cuda.empty_cache()

        if local_config.use_learned_tokens:
            self.learnable_token_language_weights = torch.load("learnable_token_language_weights.pt", weights_only=False)
            self.learnable_token_language_no_object = torch.load("learnable_token_language_no_object.pt", weights_only=False)
        else:
            self.learnable_token_language_weights = torch.nn.Embedding(300, 1).to(local_config.device)
            self.learnable_token_language_no_object = torch.nn.Embedding(1, 1536).to(local_config.device)

        self.learnable_token_logit_scale = nn.Parameter(
            torch.tensor(math.log(1 / 0.07), device=local_config.device)
        )
        self.learnable_token_mask_scale = nn.Parameter(
            torch.tensor(math.log(5.0), device=local_config.device)
        )
        self.learnable_token_mask_bias = nn.Parameter(torch.zeros((), device=local_config.device))

        # Learned weighted sum of the per-layer masks. Weights are softmax(logits) * L,
        # so they stay positive and sum to L; zero-initialized logits reproduce the
        # previous plain sum exactly (also when loading checkpoints without this param).
        self.num_mask_layers = local_config.layer_count
        self.learnable_token_mask_layer_logits = nn.Parameter(
            torch.zeros(self.num_mask_layers, device=local_config.device)
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

        criterion = SetCriterion(
            num_classes=cfg.MODEL.SAN.NUM_CLASSES,
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
            "train_dataset_names": list(cfg.DATASETS.TRAIN),
            "test_dataset_names": list(cfg.DATASETS.TEST),
            "sem_seg_postprocess_before_inference": cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,
            "amp_dtype": resolve_amp_dtype(cfg),
            "oracle_classes": cfg.MODEL.SAN.ORACLE_CLASSES.ENABLED,
            "oracle_assignment": cfg.MODEL.SAN.ORACLE_CLASSES.ASSIGNMENT,
            "oracle_mask_threshold": cfg.MODEL.SAN.ORACLE_CLASSES.MASK_THRESHOLD,
            "oracle_min_score": cfg.MODEL.SAN.ORACLE_CLASSES.MIN_SCORE,
        }
    
    @torch.no_grad()
    def precompute_prompts_legacy(self, labels: List[str]):
        prompts = [local_config.default_prompt + " " + local_config.subprompt.format(target=t) for t in labels]
        embs, _, _ = self.embed_condition(prompts)
        return embs  # (C, 1, 300, D)
    
    @torch.no_grad()
    def precompute_prompts(self, labels: List[str]):
        prompts = [local_config.subprompt.format(target=t) for t in labels]
        embs, emb_masks, tokens_count = self.embed_condition(prompts, with_chi_window=False)
        assert int(tokens_count.max()) <= embs.shape[2], "class prompt longer than 300 tokens"
        language_embs_list = []
        extract_representative_emb = lambda emb, tok_count: emb[:, [int(tok_count) - 1], :]
        for i in range(embs.shape[0]):
            current_emb = embs[[i]]
            current_mask = emb_masks[[i]]
            tokens_count = tokens_count
            model_kwargs = dict(data_info={"img_hw": 1024, "aspect_ratio": 1}, mask=current_mask) # TODO: hardcoded resolution

            dpm_solver = DPMS(
                self.model.forward_with_dpmsolver_prompts,
                condition=current_emb,
                uncondition=None,
                model_type="flow",
                model_kwargs=model_kwargs,
                schedule="FLOW",
                cfg_scale=3.5,
                precompute_embs=True
            )

            language_embs_list.append(
                extract_representative_emb(dpm_solver.segment(torch.zeros(1), {}), tokens_count[i])
            )
        return torch.stack(language_embs_list, dim=0)

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
    
    def run_diffusion_model(self, image, segment_data):
        # TODO
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

    @staticmethod
    def language_targets(language_tokens, language_weights, chunk_size=64):
        """(C, 1, 300, D) class token embeddings -> (C, 1, D) using learned token weights.
        Chunked so large vocabularies stored on CPU never sit on the GPU all at once."""
        device = language_weights.device
        return language_tokens.to(device)

    def get_language_targets(self, dataset_name):
        weights = self.learnable_token_language_weights.weight
        if self.training:
            self._language_target_cache.clear()
            return SAN.language_targets(self.class_embs[dataset_name], weights)
        if dataset_name not in self._language_target_cache:
            with torch.no_grad():
                self._language_target_cache[dataset_name] = SAN.language_targets(
                    self.class_embs[dataset_name], weights
                )
        return self._language_target_cache[dataset_name]

    def train(self, mode: bool = True):
        self._language_target_cache = {}
        return super().train(mode)

    @staticmethod
    def clasify_masks(visual_tokens, language_targets, language_no_object, logit_scale=None):
        # visual_tokens -> (B, layers, Q, D), language_targets -> (C, 1, D)
        B = visual_tokens.shape[0]
        visual_tokens = visual_tokens.sum(dim=1) # sum across layers TODO: try other ways of aggregating visual features for comparison with language
        language_targets = torch.cat([language_targets[:,0], language_no_object[None,:,:]], dim=0).permute(1,2,0).expand(B, -1, -1)
        visual_tokens = visual_tokens.permute(0,2,1)
        my_norm = lambda x,d: torch.nn.functional.normalize(x, dim=d)
        classes = my_norm(visual_tokens, 1).permute(0,2,1) @ my_norm(language_targets, 1)
        if logit_scale is not None:
            # clamp as in CLIP so the temperature can't blow up
            classes = classes * logit_scale.exp().clamp(max=100.0)
        return classes
    
    def mask_layer_weights(self):
        """(L,) positive per-layer weights summing to L."""
        logits = self.learnable_token_mask_layer_logits.float()
        return logits.softmax(dim=0) * logits.numel()

    @staticmethod
    def masks_from_tokens_and_dense_features(visual_features, aggregate_layers=True, layer_weights=None):
        learned_vision = visual_features[:,:,:local_config.no_vision_learnable_tokens,:]
        dense_vision = visual_features[:,:,local_config.no_vision_learnable_tokens:,:]
        my_norm = lambda x: x if not local_config.norm_before_dot else torch.nn.functional.normalize(x, dim=-1)
        mask_pred = my_norm(dense_vision) @ my_norm(learned_vision).permute(0, 1, 3, 2)

        # masks (b, l, p, t)
        if not aggregate_layers:
            return mask_pred
        if layer_weights is None:
            return mask_pred.sum(dim=1)
        assert layer_weights.numel() == mask_pred.shape[1], (
            f"{layer_weights.numel()} layer weights for {mask_pred.shape[1]} layers"
        )
        # weighted sum over layers: (b, l, p, t) x (l,) -> (b, p, t)
        return torch.einsum("blpt,l->bpt", mask_pred, layer_weights.to(mask_pred.dtype))

    def call_model(self, image_tensor, dataset_name):
        # Autocast here too (not only in the training loop) because evaluation runs
        # without it, and the frozen weights may be in half precision. Inside the
        # trainer's autocast with the same dtype this is a no-op.
        with torch.autocast(
            "cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None
        ):
            mask_logits, mask_pred = self._call_model(image_tensor, dataset_name)
        return mask_logits.float(), mask_pred.float()

    def _call_model(self, image_tensor, dataset_name):
        segment_data = {}
        # Grad mode is left to the caller: on during training, off (torch.no_grad)
        # during detectron2 evaluation, so no autograd graph is built at test time.
        self.run_diffusion_model(image_tensor, segment_data)

        h, w = segment_data["hw"]
        visual_features = segment_data["all_vision_features"]
        mask_pred = SAN.masks_from_tokens_and_dense_features(
            visual_features, layer_weights=self.mask_layer_weights()
        )
        mask_pred = (
            mask_pred * self.learnable_token_mask_scale.exp().clamp(max=100.0)
            + self.learnable_token_mask_bias
        )

        vl_features = segment_data["vl_learned"]
        attn_bias = SAN.masks_from_tokens_and_dense_features(vl_features, aggregate_layers=False)
        
        language_features = segment_data["language_targets"]
        my_norm = lambda x: x if not local_config.norm_before_dot else torch.nn.functional.normalize(x, dim=-1)
        masked_vl_featues = my_norm(attn_bias.permute(0,1,3,2)) @ my_norm(einops.rearrange(language_features, "b l p h d -> b l p (h d)"))

        mask_logits = SAN.clasify_masks(
            masked_vl_featues,
            self.get_language_targets(dataset_name),
            self.learnable_token_language_no_object.weight,
            logit_scale=self.learnable_token_logit_scale,
        )
        B, P, L = mask_pred.shape
        return mask_logits, mask_pred.permute(0, 2, 1).view(B, L, h, w)

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
        mask_logits, mask_preds = self.call_model(images_tensor, dataset_name)
        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(local_config.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            outputs = {
                "pred_logits": mask_logits,
                "pred_masks": mask_preds,
                # "aux_outputs": [
                #     {
                #         "pred_logits": aux_pred_logits,
                #         "pred_masks": aux_pred_masks,
                #     }
                #     for aux_pred_logits, aux_pred_masks in zip(
                #         mask_logits[:-1], mask_preds[:-1]
                #     )
                # ],
            }

            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            # mask_preds = mask_preds[-1]
            # mask_logits = mask_logits[-1]
            # torch.cuda.empty_cache()
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
                    mask_cls_result = retry_if_cuda_oom(self.oracle_mask_classification)(
                        mask_pred_result[:, : image_size[0], : image_size[1]],
                        self._gt_sem_seg(input_per_image, image_size),
                        num_classes=len(self.class_labels[dataset_name]),
                        ignore_label=self._ignore_label(dataset_name),
                    )

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

    def _dataset_name(self, batched_inputs):
        names = {x.get("meta", {}).get("dataset_name", self.default_dataset) for x in batched_inputs}
        assert len(names) == 1, f"All images in a batch must be from the same dataset, got {names}"
        name = names.pop()
        if name not in self.class_embs:
            raise KeyError(
                f"No text embeddings for '{name}'. Add it to DATASETS.TEST (or TRAIN) so its "
                "class names are embedded at model construction."
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
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
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