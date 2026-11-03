import random
import os
import torch
from PIL import Image
import numpy as np
import argparse
from typing import Optional, List
from dataclasses import dataclass, field
import pyrallis
from argparse import Namespace
import json

import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio

import config as local_config

from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.config import PixDiTConfig, model_init_config
from diffusion.utils.logger import get_root_logger

from diffusion import DPMS
from diffusion.model.builder import MODELS
from pixdit_core.pixeldit_t2i import PixDiT_T2I
import pixdit_core.compute_masks as compute_masks_utils

@dataclass
class PixelDiTInference(PixDiTConfig):
    config: Optional[str] = "configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml"
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
    args = Namespace(config='configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml')
    config = args = pyrallis.parse(config_class=PixelDiTInference, config_path=args.config)

    from tools.download import resolve_checkpoint
    args.model_path = resolve_checkpoint(args.model_path or "pixeldit_t2i_v1.pth")

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
    null_caption_embs = text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[0]

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
    state_dict = torch.load("pixeldit_t2i_v1.pth", map_location=lambda storage, loc: storage)
    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    return model.eval().to(torch.bfloat16), tokenizer, text_encoder.eval()

class Pix_Segmentor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model, self.tokenizer, self.text_encoder = get_model()
        with open("prompts/prompts.json") as f:
            self.prompts_dict = json.load(f)
        with open(local_config.ds_config_path) as f:
            self.labels = json.load(f)
        self.clean_labels = [x for x in self.labels if x != "unknown"]
        
        self.emb, self.emb_mask, self.tokens_count = self.precompute_prompts()
        self.emb = self.emb.to(torch.bfloat16)
        
        del self.tokenizer
        del self.text_encoder

        self.learnable_token_norm_weights = torch.nn.LayerNorm(local_config.no_vision_learnable_tokens)
        if local_config.use_learned_tokens:
            self.learnable_token_norm_weights.load_state_dict(torch.load("norm_weights.pt", weights_only=True))
            self.learnable_token_norm_weights = self.learnable_token_norm_weights.to(torch.bfloat16)
        self.learnable_token_norm_weights.to(local_config.device)

        if local_config.use_learned_tokens:
            self.learnable_token_language_weights = torch.load("learnable_token_language_weights.pt", weights_only=False)
            self.learnable_token_language_no_object = torch.load("learnable_token_language_no_object.pt", weights_only=False)
        else:
            self.learnable_token_language_weights = torch.nn.Embedding(300, 1).to(local_config.device)
            self.learnable_token_language_no_object = torch.nn.Embedding(1, 2304).to(local_config.device)
        
        require_grad = local_config.save_learned_tokens
        self.learnable_token_norm_weights.requires_grad_(require_grad)
        self.learnable_token_language_no_object.requires_grad_(require_grad)
        self.learnable_token_language_weights.requires_grad_(require_grad)


    def precompute_prompts(self):
        precomputed_embs = {}
        prompts = [local_config.default_prompt + " " + local_config.subprompt.format(target=current_target) for current_target in self.clean_labels]
        # default prompt for model inference
        prompts.append(
            local_config.default_prompt
        )
        return self.embed_condition(prompts)

    @staticmethod
    def img_id_from_path(x):
        return x[0]["file_name"].split("/")[-1].replace(".jpg", "")


    def add_extra_labels(self, gt_labels):
        extra_labels_count = local_config.extra_labels_count
        all_extra_labels = [x for x in self.labels if x not in gt_labels]
        samples_extra_labels = random.sample(all_extra_labels, extra_labels_count)
        return gt_labels + samples_extra_labels
    
    @staticmethod
    def select_labels(current_labels):
        # TODO
        if len(current_labels) <= local_config.select_labels_count:
            return current_labels
        else:
            return current_labels[:local_config.select_labels_count]
        try:
            # will break if sample size < label size
            return random.sample(current_labels, local_config.select_labels_count)
        except ValueError:
            return current_labels
    
    @staticmethod
    def resize_maps(maps, new_shape):
        return F.interpolate(maps, new_shape, mode="bilinear", align_corners=False).squeeze(0)

    def embed_condition(self, prompts):
        caption_token = self.tokenizer(
            prompts, max_length=506, padding="max_length", truncation=True, return_tensors="pt"
        ).to(local_config.device)
        select_index = [0] + list(range(-300 + 1, 0))
        caption_embs_list = [self.text_encoder(caption_token.input_ids[[i]], caption_token.attention_mask[[i]])[0][:, None][
            :, :, select_index
        ] for i in range(caption_token.input_ids.shape[0])]
        caption_embs = torch.cat(caption_embs_list, dim=0).cpu()
        tokens_count = caption_token.attention_mask.sum(dim=-1)
        emb_masks = caption_token.attention_mask[:, select_index]
        return caption_embs, emb_masks, tokens_count
    
    def run_diffusion_model(self, image, target_labels, segment_data):
        # TODO
        emb = self.emb[-1].unsqueeze(0)
        emb_mask = self.emb_mask[-1].unsqueeze(0)
        tokens_count = self.tokens_count[-1]
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
            image.unsqueeze(0),
            segment_data
        )

    def call_model(self, file_id, image_tensor, label_indexes, target_labels, gt_ids, original_shape, gt):
        segment_data = {"labels": target_labels, "img_id": file_id, "label_ids": label_indexes, \
            "gt_ids": gt_ids, "targets": target_labels, "gt": gt}
        with torch.enable_grad():
            self.run_diffusion_model(image_tensor, target_labels, segment_data)
        
        visual_features = segment_data["all_vision_features"]
        vl_features = segment_data["vl_learned"]
        learned_vision = visual_features[:,:,:local_config.no_vision_learnable_tokens,:]
        dense_vision = visual_features[:,:,local_config.no_vision_learnable_tokens:,:]
        h, w = segment_data["hw"]
        prediction = None
        loss = None
        dice_loss = None
        if local_config.save_learned_tokens:
            with torch.enable_grad():
                prediction, loss, dice_loss = compute_masks_utils.compute_loss(dense_vision, learned_vision, vl_features, self.emb, self.learnable_token_norm_weights, self.learnable_token_language_weights.weight, self.learnable_token_language_no_object.weight, gt, label_indexes, h, w)
        else:
            prediction = compute_masks_utils.compute_masks(dense_vision, learned_vision, vl_features, self.emb, self.learnable_token_norm_weights,  self.learnable_token_language_weights.weight, self.learnable_token_language_no_object.weight,label_indexes)
            prediction = prediction.view(1, 171, h, w)
        return prediction, loss, dice_loss

    def single_batch(self, file_id, image_tensor, label_indexes, target_labels, gt_ids, original_shape, gt):
        return self.call_model(file_id, image_tensor, label_indexes, target_labels, gt_ids, original_shape, gt)

    def single_class(self, file_id, image_tensor, label_indexes, target_labels, gt_ids, original_shape, gt):
        prediction, loss, loss_dice = self.call_model(file_id, image_tensor, label_indexes, target_labels, gt_ids, original_shape, gt)
        return prediction, loss, loss_dice

    @torch.no_grad()
    def forward_no_grad(self, x):
        image_tensor = x[0]["image"].to(local_config.device, dtype=torch.bfloat16) / 127.5 - 1
        original_shape = image_tensor.shape[-2:]

        h, w = x[0]["height"], x[0]["width"]
        gt = x[0]["sem_seg"]
        file_id = Pix_Segmentor.img_id_from_path(x)
        label_indexes = [el.item() for el in torch.unique(x[0]["sem_seg"]) if el.item() != 255]
        if local_config.select_labels_count and local_config.save_learned_tokens:
            label_indexes = self.select_labels(label_indexes)
        if local_config.run_on_extra_labels:
            target_labels = self.add_extra_labels(target_labels)
            label_indexes = [self.labels.index(x) for x in target_labels]
        target_labels = [self.clean_labels[el] for el in label_indexes]
        gt_ids = [self.labels.index(el) for el in target_labels]

        prediction, loss, loss_dice = self.single_batch(file_id, image_tensor, label_indexes, target_labels, gt_ids, original_shape, gt) if False and local_config.save_learned_tokens else \
            self.single_class(file_id, image_tensor, label_indexes, target_labels, gt_ids, original_shape, gt)

        prediction = Pix_Segmentor.resize_maps(prediction, (h, w))

        # TODO: FIX names and rename and commit trash dirs
        if os.path.isfile("print2"):
            predictions_all = torch.zeros(172 if "coco" in local_config.eval_dataset.value else len(self.labels) + 1, h, w).to(prediction.device)
            for i, lid in enumerate(label_indexes):
                predictions_all[lid] += prediction[i]
            seeem = F.interpolate(x[0]["sem_seg"][None, None].to(torch.float64), (h,w), mode="nearest").detach().cpu()[0,0].to(torch.int64)
            seeem[seeem == 255] = 171
            self.plot_segmentation_maps(torch.argmax(predictions_all[None,:,:,:], dim=1).detach().cpu()[0], seeem, "pred", "gt", f"trash2/{file_id}.png")
        
        return [{"sem_seg": prediction, "learnable_token_loss": loss, "loss_dice": loss_dice}]

    def forward(self, x):
        return self.forward_no_grad(x)

    @staticmethod
    def plot_segmentation_maps(
        mask1: np.ndarray,
        mask2: np.ndarray,
        prediction_id1: str,
        prediction_id2: str,
        save_path: str = "segmentation_comparison.png",
    ):
        """
        Plot two semantic segmentation maps side by side and save the figure.

        Args:
            mask1: First 2D segmentation map, with class IDs in [0, 171].
            mask2: Second 2D segmentation map, with class IDs in [0, 171].
            prediction_id1: ID/name of the first prediction.
            prediction_id2: ID/name of the second prediction.
            save_path: Output image file path.
        """
        mask1 = np.asarray(mask1)
        mask2 = np.asarray(mask2)

        if mask1.ndim != 2 or mask2.ndim != 2:
            raise ValueError("Both segmentation maps must be 2D arrays.")

        if mask1.shape != mask2.shape:
            raise ValueError(
                f"Mask shapes must match: {mask1.shape} vs {mask2.shape}"
            )

        for name, mask in [("mask1", mask1), ("mask2", mask2)]:
            if not np.issubdtype(mask.dtype, np.integer):
                raise ValueError(f"{name} must contain integer class IDs.")
            if mask.min() < 0 or mask.max() > 172:
                print(mask.min(), mask.max())
                raise ValueError(f"{name} must contain class IDs from 0 to 171.")

        # Use the same discrete color mapping for both masks.
        cmap = plt.get_cmap("turbo", 172)
        norm = plt.Normalize(vmin=-0.5, vmax=172.5)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im1 = axes[0].imshow(mask1, cmap=cmap, norm=norm, interpolation="nearest")
        axes[0].set_title(f"Prediction ID: {prediction_id1}")
        axes[0].axis("off")

        im2 = axes[1].imshow(mask2, cmap=cmap, norm=norm, interpolation="nearest")
        axes[1].set_title(f"Prediction ID: {prediction_id2}")
        axes[1].axis("off")

        # Shared color bar showing class IDs.
        cbar = fig.colorbar(
            im2,
            ax=axes,
            ticks=np.arange(0, 172, 10),
            fraction=0.025,
            pad=0.03,
        )
        cbar.set_label("Class ID")

        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def __del__(self):
        if local_config.save_learned_tokens:
            torch.save(self.learnable_token_norm_weights.state_dict(), "norm_weights.pt")
            torch.save(self.learnable_token_language_weights, "learnable_token_language_weights.pt")
            torch.save(self.learnable_token_language_no_object, "learnable_token_language_no_object.pt")
            # torch.save(self.learnable_token_language_weights.state_dict(), f"learnable_token_language_weights.pt")