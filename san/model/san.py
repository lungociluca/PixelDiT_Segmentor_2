from typing import List

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


from t2i.diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from t2i.diffusion.model.utils import get_weight_dtype
from t2i.diffusion.utils.config import PixDiTConfig, model_init_config
from t2i.diffusion.utils.logger import get_root_logger

from t2i.diffusion.model.flow_dpm import DPMS
from t2i.diffusion.model.trainer import PixDiTTrainer
import t2i.pixdit_core.compute_masks as compute_masks_utils

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
    ):
        super().__init__()
        self.size_divisibility = 16
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.criterion = criterion

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
            embs = self.precompute_prompts(labels).to(torch.float32)
            if name not in train_dataset_names:
                embs = embs.cpu()
            self.class_labels[name], self.class_embs[name] = labels, embs
        # Cache of token-weighted class vectors, valid only while weights are frozen (eval).
        self._language_target_cache = {}

        del self.tokenizer
        del self.text_encoder

        self.learnable_token_norm_weights = torch.nn.LayerNorm(local_config.no_vision_learnable_tokens)
        if local_config.use_learned_tokens:
            self.learnable_token_norm_weights.load_state_dict(torch.load("norm_weights.pt", weights_only=True))
            self.learnable_token_norm_weights = self.learnable_token_norm_weights
        self.learnable_token_norm_weights.to(local_config.device)

        if local_config.use_learned_tokens:
            self.learnable_token_language_weights = torch.load("learnable_token_language_weights.pt", weights_only=False)
            self.learnable_token_language_no_object = torch.load("learnable_token_language_no_object.pt", weights_only=False)
        else:
            self.learnable_token_language_weights = torch.nn.Embedding(300, 1).to(local_config.device)
            self.learnable_token_language_no_object = torch.nn.Embedding(1, 2304).to(local_config.device)
        

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
        }
    
    @torch.no_grad()
    def precompute_prompts(self, labels: List[str]):
        prompts = [local_config.default_prompt + " " + local_config.subprompt.format(target=t) for t in labels]
        embs, _, _ = self.embed_condition(prompts)
        return embs  # (C, 1, 300, D)
    
    def embed_condition(self, prompts):
        caption_token = self.tokenizer(
            prompts, max_length=506, padding="max_length", truncation=True, return_tensors="pt"
        ).to(local_config.device)
        select_index = [0] + list(range(-300 + 1, 0))
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
        device = language_weights.device
        out = []
        for i in range(0, language_tokens.shape[0], chunk_size):
            chunk = language_tokens[i:i + chunk_size].to(device, non_blocking=True)
            out.append((language_weights * chunk).sum(dim=-2))
        return torch.cat(out, dim=0)

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
    def clasify_masks(visual_tokens, language_targets, language_no_object):
        # visual_tokens -> (B, layers, Q, D), language_targets -> (C, 1, D)
        B = visual_tokens.shape[0]
        visual_tokens = visual_tokens.sum(dim=1) # sum across layers TODO: try other ways of aggregating visual features for comparison with language
        language_targets = torch.cat([language_targets, language_no_object[None,:,:]], dim=0).permute(1,2,0).expand(B, -1, -1)
        visual_tokens = visual_tokens.permute(0,2,1)
        my_norm = lambda x,d: torch.nn.functional.normalize(x, dim=d)
        classes = my_norm(visual_tokens, 1).permute(0,2,1) @ my_norm(language_targets, 1)
        return classes
    
    def call_model(self, image_tensor, dataset_name):
        segment_data = {}
        with torch.enable_grad():
            self.run_diffusion_model(image_tensor, segment_data)
        
        visual_features = segment_data["all_vision_features"]
        vl_features = segment_data["vl_learned"]

        learned_vision = visual_features[:,:,:local_config.no_vision_learnable_tokens,:]
        dense_vision = visual_features[:,:,local_config.no_vision_learnable_tokens:,:]
        h, w = segment_data["hw"]

        my_norm = lambda x: x if not local_config.norm_before_dot else torch.nn.functional.normalize(x, dim=-1)
        mask_pred = my_norm(dense_vision) @ my_norm(learned_vision).permute(0, 1, 3, 2)

        # masks (b, l, p, t) * (b, l, 1, t)
        # weighted sum to aggregate masks of all layers
        mask_pred = self.learnable_token_norm_weights(mask_pred)
        mask_pred = mask_pred.sum(dim=1)

        mask_logits = SAN.clasify_masks(
            vl_features,
            self.get_language_targets(dataset_name),
            self.learnable_token_language_no_object.weight,
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
            padding_constraints={"square_size": 1024},   # recent detectron2 versions
        )
        images_tensor = images.tensor
        # print("image tensor stats", images_tensor.shape, images_tensor.min(), images_tensor.max(), images_tensor.mean())
        # print("ids",  [x["file_name"]for x in batched_inputs])
        mask_logits, mask_preds = self.call_model(images_tensor, self._dataset_name(batched_inputs))
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

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
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

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    @property
    def device(self):
        return torch.device(local_config.device)