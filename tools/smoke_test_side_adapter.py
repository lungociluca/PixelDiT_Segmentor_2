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


def check_classifier():
    from san.model.pixeldit_classifier import PixelDiTRegionClassifier

    local_config.layer_count = 5
    core = PixDiT_T2I(
        hidden_size=96, num_groups=4, patch_depth=14, txt_embed_dim=32, txt_max_length=12
    ).to(device).eval()
    H_img, W_img = 128, 96
    x = torch.randn(1, 3, H_img, W_img, device=device)
    y = torch.randn(1, 12, 32, device=device)
    t = torch.tensor(2.0, device=device)

    with torch.no_grad():
        seg = {"feature_layers": [1, 2, 3]}
        core(x, t, y, segment_data=seg)
        L = seg["hw"][0] * seg["hw"][1]
        assert seg["condition"].shape == (1, 1, 96)
        cond_init = core.timestep_condition(t)
        assert torch.allclose(cond_init, seg["condition"], atol=1e-5)

        # Text-only keys for 3 "classes" with 2-4 real tokens each, right-padded.
        C, Ltxt = 3, 12
        caps = torch.randn(C, Ltxt, 32, device=device)
        attn = torch.zeros(C, Ltxt, dtype=torch.long, device=device)
        for i, n in enumerate([3, 5, 4]):
            attn[i, :n] = 1
        select = attn.bool().clone()
        select[:, 0] = False  # drop BOS
        for mode in ["soft", "hard", "none"]:
            clf = PixelDiTRegionClassifier(readout_block=5, num_masked_blocks=1, mask_attention=mode)
            assert clf.input_layer == 3 and clf.masked_blocks == [4]
            keys, valid = clf.build_text_keys(core, caps, attn, select, cond_init)
            assert keys.shape == (C, 4, 4, 24) and valid.sum(1).tolist() == [2, 4, 3]

            # Padding must not change a class's keys.
            caps2 = caps.clone()
            caps2[0, 3:] = torch.randn_like(caps2[0, 3:])
            keys2, _ = clf.build_text_keys(core, caps2, attn, select, cond_init)
            assert torch.allclose(keys[0][valid[0]], keys2[0][valid[0]], atol=1e-4), "padding leaked"

            regions = torch.zeros(L, 2, device=device)
            regions[: L // 2, 0] = 1.0
            regions[L // 2 : L - 4, 1] = 1.0  # last 4 tokens have no region
            scores, image_mean = clf(core, seg["features"][3], seg["condition"], seg["pos"],
                         regions, regions.T, keys, valid)
            assert scores.shape == (2, C) and torch.isfinite(scores).all(), mode
            assert image_mean.shape == (C,), image_mean.shape

        # Region restriction: with hard masks, changing one region's tokens must not
        # affect the other region's scores.
        clf = PixelDiTRegionClassifier(readout_block=5, num_masked_blocks=1, mask_attention="hard")
        tokens = seg["features"][3].clone()
        s1, _ = clf(core, tokens, seg["condition"], seg["pos"], regions, regions.T, keys, valid)
        tokens[:, L // 2 : L - 4] += torch.randn_like(tokens[:, L // 2 : L - 4])
        s2, _ = clf(core, tokens, seg["condition"], seg["pos"], regions, regions.T, keys, valid)
        assert torch.allclose(s1[0], s2[0], atol=1e-4) and not torch.allclose(s1[1], s2[1]), "regions not isolated"
    print("classifier: ok")


def check_concept_readout():
    from san.model.pixeldit_classifier import PixelDiTRegionClassifier

    local_config.layer_count = 3
    core = PixDiT_T2I(
        hidden_size=96, num_groups=4, patch_depth=14, txt_embed_dim=32, txt_max_length=12
    ).to(device).eval()
    x = torch.randn(1, 3, 128, 96, device=device)
    y = torch.randn(1, 12, 32, device=device)
    t = torch.tensor(500.0, device=device)
    with torch.no_grad():
        seg = {"feature_layers": [0, 3]}
        core(x, t, y, segment_data=seg)
        s0, y0 = seg["features"][0], seg["text_features"][0]
        assert y0.shape == (1, 12, 96), y0.shape
        L = s0.shape[1]
        pos_txt = core.fetch_pos_text(12, device)

        # One-directional: concept tokens must not change the image / prompt streams.
        cpt = torch.randn(1, 5, 96, device=device)
        blk = core.patch_blocks[0]
        xa, ya = blk(s0, y0, seg["condition"], seg["pos"], pos_txt)
        xb, yb, cb, o_x, o_c = blk.forward_with_concepts(
            s0, y0, cpt, seg["condition"], seg["pos"], pos_txt, pos_txt[2:7]
        )
        assert torch.allclose(xa, xb, atol=1e-5) and torch.allclose(ya, yb, atol=1e-5), "concepts leak"
        assert cb.shape == (1, 5, 96) and o_x.shape == (1, L, 96) and o_c.shape == (1, 5, 96)

        # Concept readout over all blocks, with and without region-restricted blocks.
        C = 3
        caps = torch.randn(C, 12, 32, device=device)
        select = torch.zeros(C, 12, dtype=torch.bool, device=device)
        select[0, 4:5] = True
        select[1, 4:7] = True
        select[2, 4:6] = True
        regions = torch.zeros(L, 2, device=device)
        regions[: L // 2, 0] = 1.0
        regions[L // 2 :, 1] = 1.0
        for masked, mode in [([], "soft"), ([2, 3], "soft"), ([2, 3], "hard")]:
            clf = PixelDiTRegionClassifier(
                readout_block=3, readout="concept", concept_num_blocks=3,
                concept_masked_blocks=masked, mask_attention=mode,
            )
            assert clf.input_layer == 0 and clf.deepest_block == 3
            concepts = clf.build_concepts(core, caps, select)
            assert concepts[0].shape == (6, 96) and concepts[2].shape == (C, 3)
            res = clf.concept_scores(
                core, s0, y0, seg["condition"], seg["pos"], concepts, regions, regions.T
            )
            assert [r[0] for r in res] == [1, 2, 3, "avg"], [r[0] for r in res]
            for _, scores, image_mean in res:
                assert scores.shape == (2, C) and image_mean.shape == (C,)
                assert torch.isfinite(scores).all()
                assert torch.allclose(scores.sum(-1), torch.ones(2, device=device), atol=1e-4)
    print("concept readout: ok")


if __name__ == "__main__":
    check_feature_hook()
    check_side_adapter()
    check_classifier()
    check_concept_readout()
    print("all checks passed")
