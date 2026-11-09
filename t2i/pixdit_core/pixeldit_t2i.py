import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json, einops
from typing import Tuple

from .pixeldit_c2i import PatchTokenEmbedder, PixelTokenEmbedder, PiTBlock
from .modules import (
    FinalLayer,
    FeedForward,
    RMSNorm,
    TimestepConditioner,
    apply_adaln,
    apply_rotary_emb,
    precompute_freqs_cis_2d,
)

import t2i.config as local_config

class MMDiTJointAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            block_id: int = -1

    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)

        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)
        self.block_id = block_id

        if local_config.use_vision_mlp:
            self.learnable_token_vision_mlp = FeedForward(dim, 1 * dim)

        self.learnable_token_vl_mlp = FeedForward(dim, 1 * dim)

        if local_config.use_learned_tokens:
            self.learnable_token_vision = torch.load(f"learnable_token_vision_{self.block_id}.pt", weights_only=False)
            self.learnable_token_vl = torch.load(f"learnable_token_vl_{self.block_id}.pt", weights_only=False)
        else:
            self.learnable_token_vision = nn.Embedding(local_config.no_vision_learnable_tokens, dim)
            self.learnable_token_vl = nn.Embedding(local_config.no_vision_learnable_tokens, dim)
        
        if local_config.use_learned_tokens:
            if local_config.use_vision_mlp:
                vision_state = torch.load(f"learnable_token_vision_mlp_{self.block_id}.pt", map_location=local_config.device)
                self.learnable_token_vision_mlp.load_state_dict(vision_state)
            
            weight_vision_state = torch.load(f"learnable_token_vl_mlp_{self.block_id}.pt", map_location=local_config.device)
            self.learnable_token_vl_mlp.load_state_dict(weight_vision_state)
        
        with open(local_config.ds_config_path) as f:
            self.labels = json.load(f)

    def precompute_language_embs(self, y, pos_txt):
        By, Ny, Cy = y.shape

        qkv_y = self.qkv_y(y).reshape(By, Ny, 3, self.num_heads, Cy // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        qy = qy.transpose(1, 2)
        ky = ky.transpose(1, 2)
        vy = vy.transpose(1, 2)

        ky = einops.rearrange(ky, "b h p d -> b p (h d)")
        return ky

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            pos_img: torch.Tensor,
            pos_txt: torch.Tensor = None,
            attn_mask: torch.Tensor = None,
            segment_data=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, C = y.shape

        ####################################################################################################
        all_vision_tokens_count = x.shape[1] + local_config.no_vision_learnable_tokens
        qkv_x_l = self.qkv_x(
            torch.cat([self.learnable_token_vision.weight.expand(B, -1, -1), x], dim=1)
        ).reshape(B, all_vision_tokens_count, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qxl, kxl, vxl = qkv_x_l[0], qkv_x_l[1], qkv_x_l[2]
        qxl = self.q_norm_x(qxl)
        kxl = self.k_norm_x(kxl)

        qxl = qxl.transpose(1, 2)
        kxl = kxl.transpose(1, 2)
        vxl = vxl.transpose(1, 2)

        if local_config.use_vision_attention_bias:
            attention_bias = torch.ones((all_vision_tokens_count, all_vision_tokens_count), dtype=torch.bool, device=local_config.device)
            attention_bias[:, :local_config.weight_tokens] = False
        else:
            attention_bias = None
        out_joint_l = F.scaled_dot_product_attention(qxl, kxl, vxl, dropout_p=0.0, attn_mask=attention_bias)
        out_joint_l = out_joint_l.transpose(1, 2).reshape(B, all_vision_tokens_count, C)
        out_joint_l = self.proj_x(out_joint_l)

        segment_data["vision_learned"][self.block_id] = self.learnable_token_vision_mlp(out_joint_l)
        
        ####################################################################################################
        # vl
        if self.block_id == 0:
            all_vision_tokens_count = x.shape[1] + local_config.no_vision_learnable_tokens
            qkv_x_l = self.qkv_x(
                torch.cat([self.learnable_token_vl.weight.expand(B, -1, -1), x], dim=1)
            ).reshape(B, all_vision_tokens_count, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
            qxl, kxl, vxl = qkv_x_l[0], qkv_x_l[1], qkv_x_l[2]
            qxl = self.q_norm_x(qxl)
            kxl = self.k_norm_x(kxl)
            segment_data["language_targets"][self.block_id] = kxl[:,local_config.no_vision_learnable_tokens:,:,:]

            qxl = qxl.transpose(1, 2)
            kxl = kxl.transpose(1, 2)
            vxl = vxl.transpose(1, 2)

            if local_config.use_vision_attention_bias:
                attention_bias = torch.ones((all_vision_tokens_count, all_vision_tokens_count), dtype=torch.bool, device=local_config.device)
                attention_bias[:, :local_config.weight_tokens] = False
            else:
                attention_bias = None
            out_joint_l = F.scaled_dot_product_attention(qxl, kxl, vxl, dropout_p=0.0, attn_mask=attention_bias)
            out_joint_l = out_joint_l.transpose(1, 2).reshape(B, all_vision_tokens_count, C)
            out_joint_l = self.proj_x(out_joint_l)

            segment_data["vl_learned"][self.block_id] = self.learnable_token_vl_mlp(out_joint_l)
        
        ####################################################################################################
        

        B, Nx, C = x.shape
        By, Ny, Cy = y.shape
        assert B == By and C == Cy, "x and y must share batch and channel dims"

        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        attn_mk = torch.zeros((B, Nx), device=x.device, dtype=x.dtype)
        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        qx = qx.transpose(1, 2)
        kx = kx.transpose(1, 2)
        vx = vx.transpose(1, 2)

        qy = qy.transpose(1, 2)
        ky = ky.transpose(1, 2)
        vy = vy.transpose(1, 2)

        q_joint = torch.cat([qy, qx], dim=2)
        k_joint = torch.cat([ky, kx], dim=2)
        v_joint = torch.cat([vy, vx], dim=2)

        out_joint = F.scaled_dot_product_attention(q_joint, k_joint, v_joint, dropout_p=0.0, attn_mask=attn_mask)
        out_y = out_joint[:, :, :Ny, :]
        out_x = out_joint[:, :, Ny:, :]

        out_y = out_y.transpose(1, 2).reshape(B, Ny, C)
        out_x = out_x.transpose(1, 2).reshape(B, Nx, C)

        out_x_learned = self.proj_x(out_x.clone())
        if local_config.use_vision_mlp:
            out_x_learned = self.learnable_token_vision_mlp(out_x_learned)

        out_x = self.proj_drop_x(self.proj_x(out_x))
        out_y = self.proj_drop_y(self.proj_y(out_y))

        return out_x, out_y, attn_mk

    def __del__(self):
        if local_config.save_learned_tokens:
            if local_config.use_vision_mlp:
                torch.save(self.learnable_token_vision_mlp.state_dict(), f"learnable_token_vision_mlp_{self.block_id}.pt")
            torch.save(self.learnable_token_vl_mlp.state_dict(), f"learnable_token_vl_mlp_{self.block_id}.pt")
            torch.save(self.learnable_token_vision, f"learnable_token_vision_{self.block_id}.pt")
            torch.save(self.learnable_token_vl, f"learnable_token_vl_{self.block_id}.pt")

class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, adaLN_modulation_img=None, adaLN_modulation_txt=None, block_id=-1):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups

        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        # self.learnable_token_norm_weights = RMSNorm(hidden_size, eps=1e-6)

        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False, block_id=block_id)

        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation_img = adaLN_modulation_img if adaLN_modulation_img is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = adaLN_modulation_txt if adaLN_modulation_txt is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.block_id = block_id

    def precompute_language_embs(self, y, c, pos_txt):
        shift_msa_y, scale_msa_y, _, _, _, _ = self.adaLN_modulation_txt(c).chunk(6, dim=-1)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_y = self.attn.precompute_language_embs(y_norm, pos_txt)
        return attn_y

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None, segment_data=None):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)

        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        with torch.enable_grad():
            attn_x, attn_y, attn_mk = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask, segment_data=segment_data)

        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y, attn_mk

class PixDiT_T2I(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=64,
        pixel_attn_hidden_size=None,
        pixel_num_groups=None,
        patch_depth=26,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=4096,
        txt_max_length=1024,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        repa_encoder_index: int = -1,
        use_pixel_abs_pos: bool = True,
        pit_adaln_post_modulation: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.txt_embed_dim = int(txt_embed_dim)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.repa_encoder_index = int(repa_encoder_index)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT_T2I expects pixel_depth > 0 to retain the pixel pathway")

        self.pixel_embedder = PixelTokenEmbedder(in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = PatchTokenEmbedder(self.txt_embed_dim, hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, hidden_size))
        # self.register_buffer("learnable_token_target", torch.ones(1, 1, hidden_size), persistent=False)

        self._shared_cond_adaln = None
        self._shared_cond_adaln_img = None
        self._shared_cond_adaln_txt = None
        self.patch_blocks = nn.ModuleList([
            MMDiTBlockT2I(
                self.hidden_size,
                self.num_groups,
                adaLN_modulation_img=self._shared_cond_adaln_img,
                adaLN_modulation_txt=self._shared_cond_adaln_txt,
                block_id=i
            )
            for i in range(min(self.patch_depth, local_config.layer_count))
        ])
        self.text_refine_blocks = None
        self.pixel_attn_hidden_size = (
            int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        )
        self.pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        # self.pixel_blocks = nn.ModuleList(
        #     [
        #         PiTBlock(
        #             self.pixel_hidden_size,
        #             self.hidden_size,
        #             patch_size=self.patch_size,
        #             num_heads=self.num_groups,
        #             mlp_ratio=4.0,
        #             attn_hidden_size=self.pixel_attn_hidden_size,
        #             attn_num_heads=self.pixel_num_groups,
        #             rope_fn=precompute_freqs_cis_2d,
        #             adaln_post_modulation=self.pit_adaln_post_modulation,
        #         )
        #         for _ in range(self.pixel_depth)
        #     ]
        # )

        # self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)

        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()
        self.last_repa_tokens = None

        self.initialize_weights()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def fetch_pos_text(self, length, device):
        if length in self.precompute_pos_txt:
            return self.precompute_pos_txt[length].to(device)
        head_dim = self.hidden_size // self.num_groups
        freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(0, length, device=device).float().unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.precompute_pos_txt[length] = freqs_cis
        return freqs_cis

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # nn.init.zeros_(self.final_layer.linear.weight)
        # nn.init.zeros_(self.final_layer.linear.bias)

    # @staticmethod
    # def regularization_loss(tokens):
    #     l2_penalty = tokens.pow(2).mean()
    #     tokens = F.normalize(tokens, p=2, dim=-1)

    #     # Pairwise cosine similarity
    #     cos_dist = tokens @ tokens.transpose(-1, -2)
    #     K = tokens.shape[-2]

    #     # Remove diagonal without using bool negation
    #     eye = torch.eye(K, device=tokens.device)
    #     off_diag = 1.0 - eye
    #     diversity_loss = (cos_dist.pow(2) * off_diag).sum() / (K * (K - 1))
    #     return local_config.l2_regularization_weight * l2_penalty + local_config.diversity_regularization_weight * diversity_loss

    def precompute_language_embs(self, t, y):
        B = y.shape[0]
        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], self.txt_max_length)  # same as upstream: keep all 300 tokens
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y.to(local_config.device)).view(1, Ltxt, self.hidden_size).expand((B, -1, -1))
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        condition = torch.nn.functional.silu(t_emb)

        pos_txt = self.fetch_pos_text(Ltxt, local_config.device) if self.use_text_rope else None
        return self.patch_blocks[0].precompute_language_embs(y_emb, condition, pos_txt)

    def forward(self, x, t, y, s=None, mask=None, segment_data=None):
        B, _, H, W = x.shape
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        pos = self.fetch_pos(Hs, Ws, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], self.txt_max_length)  # same as upstream: keep all 300 tokens
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y.to(local_config.device, dtype=x.dtype)).view(1, Ltxt, self.hidden_size).expand((B, -1, -1))
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        condition = torch.nn.functional.silu(t_emb)
        if segment_data is not None:
            segment_data["hw"] = (Hs,Ws)
            segment_data["language_targets"] = {}
            segment_data["vision_learned"] = {}
            segment_data["vl_learned"] = {}
        
        if s is None:
            s0 = self.s_embedder(x_patches)
            pos_txt = self.fetch_pos_text(Ltxt, x.device) if self.use_text_rope else None
            attn_mask_joint = None
            if mask is not None and isinstance(mask, torch.Tensor):
                m = mask
                while m.dim() > 2 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 3 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 2:
                    pad = (m == 0)
                    pad_img = torch.zeros((B, L), dtype=torch.bool, device=x.device)
                    attn_mask_joint = torch.cat([pad[:, :Ltxt], pad_img], dim=1).view(B, 1, 1, Ltxt + L)
            self.last_repa_tokens = None
            s = s0

            with torch.enable_grad():
                for i in range(min(self.patch_depth, local_config.layer_count)):
                    s, y_emb, attn_mk = self.patch_blocks[i](s, y_emb, condition, pos, pos_txt, attn_mask_joint, segment_data=segment_data)
                    if 0 < self.repa_encoder_index == (i + 1):
                        self.last_repa_tokens = s
                    
                    # import os
                    # attn_mk_disp = attn_mk.to(torch.float32)[-1,:].view(Hs, Ws).detach().cpu()
                    # plt.imshow(attn_mk_disp, cmap="viridis")
                    # plt.savefig(f"trash/{i}_{len(os.listdir('trash'))}.png")
                    # plt.close()

                    # if segment_data is not None:
                    #     segment_data["mask"][i] = attn_mk.view(B, Hs, Ws)

                stack_features = lambda k: torch.stack([segment_data[k][i] for i in range(len(segment_data[k].keys()))], dim=1)
                segment_data["all_vision_features"] = stack_features("vision_learned")
                segment_data["vl_learned"] = stack_features("vl_learned")
                segment_data["language_targets"] = stack_features("language_targets")
        return x
