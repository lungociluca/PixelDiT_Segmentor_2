"""Shape and gradient check for the PixelDiT side adapter, without the PixelDiT checkpoint.

Run from the repo root:  python tools/smoke_test_side_adapter.py
It checks:
  1. the PixDiT_T2I feature hook (tiny randomly initialised model) returns the
     requested layers with the right shapes, and runs only the blocks it needs;
  2. the side adapter fuses (B, 1536, h, w) features, returns deep-supervision
     outputs in training and one output at inference, and the class-agnostic
     set criterion back-propagates into the fusion layers, side ViT and heads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.projects.deeplab import add_deeplab_config

import t2i.config as local_config
from san.config import add_san_config
from san.model.criterion import SetCriterion
from san.model.matcher import HungarianMatcher
from san.model.side_adapter import build_side_adapter_network
from t2i.pixdit_core.pixeldit_t2i import PixDiT_T2I

device = "cuda" if torch.cuda.is_available() else "cpu"
local_config.device = device
torch.manual_seed(0)


def check_feature_hook():
    local_config.layer_count = 3
    core = PixDiT_T2I(
        hidden_size=96, num_groups=4, patch_depth=14, txt_embed_dim=32, txt_max_length=8
    ).to(device).eval()
    assert len(core.patch_blocks) == 3, len(core.patch_blocks)
    x = torch.randn(2, 3, 128, 96, device=device)
    t = torch.full((2,), 2.0, device=device)
    y = torch.randn(1, 8, 32, device=device)

    calls = []
    hooks = [b.register_forward_hook(lambda m, i, o, k=k: calls.append(k)) for k, b in enumerate(core.patch_blocks)]
    seg = {"feature_layers": [0, 2]}
    with torch.no_grad():
        core(x, t, y, segment_data=seg)
    for h in hooks:
        h.remove()
    assert seg["hw"] == (8, 6), seg["hw"]
    assert sorted(seg["features"]) == [0, 2], seg["features"].keys()
    for k, v in seg["features"].items():
        assert v.shape == (2, 48, 96), (k, v.shape)
    assert calls == [0, 1], f"expected blocks 0,1 to run for deepest layer 2, ran {calls}"

    seg = {}  # old behaviour: all blocks stacked
    with torch.no_grad():
        core(x, t, y, segment_data=seg)
    assert seg["patch_features"].shape == (2, 3, 48, 96), seg["patch_features"].shape
    print("feature hook: ok")


def check_side_adapter():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file("configs/san_clip_vit_res4_coco.yaml")
    layers = list(cfg.MODEL.PIXELDIT.FEATURE_LAYERS)
    hidden = 1536

    net = build_side_adapter_network(cfg, {k: ShapeSpec(channels=hidden) for k in layers}).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"side adapter: {n_params / 1e6:.2f}M trainable params")

    B, H, W = 2, 512, 384  # smaller than the 1024 crops; also exercises pos-embed resizing
    h, w = H // 16, W // 16
    image = torch.randn(B, 3, H, W, device=device)
    feats = {k: torch.randn(B, hidden, h, w, device=device) for k in layers}

    net.train()
    use_amp = device == "cuda"
    with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
        mask_preds, logits = net(image, feats)
    n_ds = len(cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS)
    Q = cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES
    assert len(mask_preds) == len(logits) == n_ds
    for m, c in zip(mask_preds, logits):
        assert m.shape == (B, Q, h, w), m.shape
        assert c.shape == (B, Q, 2), c.shape

    matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0, num_points=1024)
    weight_dict = {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}
    for i in range(n_ds - 1):
        weight_dict.update({f"{k}_{i}": v for k, v in list(weight_dict.items())[:3]})
    criterion = SetCriterion(
        num_classes=1, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1,
        losses=["labels", "masks"], num_points=1024, oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ).to(device)
    targets = [
        {
            "labels": torch.zeros(3, dtype=torch.long, device=device),
            "masks": (torch.rand(3, H, W, device=device) > 0.5),
        }
        for _ in range(B)
    ]
    outputs = {
        "pred_logits": logits[-1].float(),
        "pred_masks": mask_preds[-1].float(),
        "aux_outputs": [
            {"pred_logits": c.float(), "pred_masks": m.float()}
            for c, m in zip(logits[:-1], mask_preds[:-1])
        ],
    }
    losses = criterion(outputs, targets)
    total = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
    assert torch.isfinite(total), losses
    total.backward()
    for name in ["fusion_layers.layer_0", "vit_model.blocks.0", "query_embed",
                 "mask_decoder.class_embed", "mask_decoder.pix_mlp"]:
        grads = [p.grad for n, p in net.named_parameters() if n.startswith(name)]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
    print(f"train: losses {sorted(losses)}  total {total.item():.3f}")

    net.eval()
    with torch.no_grad():
        mask_preds, logits = net(image, feats)
    assert len(mask_preds) == len(logits) == 1
    print("eval: ok")


if __name__ == "__main__":
    check_feature_hook()
    check_side_adapter()
    print("all checks passed")
