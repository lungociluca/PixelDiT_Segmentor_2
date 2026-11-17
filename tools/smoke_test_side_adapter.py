"""Shape and gradient check for the PixelDiT side adapter, without the PixelDiT checkpoint.

Run from the repo root:  python tools/smoke_test_side_adapter.py
It checks:
  1. the PixDiT_T2I feature hook (tiny randomly initialised model) returns the
     requested layers with the right shapes, and runs only the blocks it needs;
  2. the side adapter fuses (B, 1536, h, w) features, returns deep-supervision
     outputs in training and one output at inference, and the class-agnostic
     set criterion back-propagates into the fusion layers, side ViT and heads;
  3. the concept stream (ConceptAttention) leaves the image features untouched,
     returns per-layer scores of the right shape, and the mask classifier turns
     them into per-mask class probabilities;
  4. the concept LoRA starts as a no-op, never changes the image features, and
     the cross-entropy on mask-pooled logits back-propagates into it (and only
     into it and the learned head).
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
from san.model.concept_classifier import ConceptLoRA, ConceptMaskClassifier
from san.model.criterion import SetCriterion
from san.model.matcher import HungarianMatcher
from san.model.side_adapter import build_side_adapter_network
from san.utils.amp import resolve_amp_dtype
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
    amp_dtype = resolve_amp_dtype(cfg) if use_amp else None  # fp16 on pre-Ampere GPUs
    with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
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
    # Matching of the final output is kept for the joint concept-classification loss.
    assert len(criterion.last_indices) == B
    for (src, tgt), tg in zip(criterion.last_indices, targets):
        assert len(src) == len(tgt) == len(tg["labels"]), (src, tgt)
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


def check_concept_stream():
    local_config.layer_count = 4
    core = PixDiT_T2I(
        hidden_size=96, num_groups=4, patch_depth=14, txt_embed_dim=32, txt_max_length=8
    ).to(device).eval()
    assert len(core.patch_blocks) == 4, len(core.patch_blocks)
    B, K, hw = 2, 5, (8, 6)
    x = torch.randn(B, 3, 128, 96, device=device)
    t = torch.full((B,), 2.0, device=device)
    y = torch.randn(1, 8, 32, device=device)
    concepts = torch.randn(K, 32, device=device)

    calls = []
    hooks = [b.register_forward_hook(lambda m, i, o, k=k: calls.append(k)) for k, b in enumerate(core.patch_blocks)]
    plain = {"feature_layers": [1, 2, 3]}
    with torch.no_grad():
        core(x, t, y, segment_data=plain)
    assert calls == [0, 1, 2], calls
    assert "concept_scores" not in plain

    calls.clear()
    seg = {
        "feature_layers": [1, 2, 3],
        "concepts": concepts,
        "concept_layers": [1, 2, 3, 4],
        "concept_position": 2,
    }
    with torch.no_grad():
        core(x, t, y, segment_data=seg)
    for h in hooks:
        h.remove()
    assert calls == [0, 1, 2, 3], f"concept layer 4 needs block 3 to run, ran {calls}"
    # Concepts must not leak into the image stream: features are bit-identical.
    for k in [1, 2, 3]:
        assert torch.equal(plain["features"][k], seg["features"][k]), f"layer {k} changed"
    scores = seg["concept_scores"]
    assert sorted(scores) == [1, 2, 3, 4], scores.keys()
    for k, v in scores.items():
        assert v.shape == (B, hw[0] * hw[1], K) and v.dtype == torch.float32, (k, v.shape, v.dtype)
        assert torch.isfinite(v).all(), k

    # The order of the concepts carries no meaning: permuting them permutes the scores.
    perm = torch.randperm(K, device=device)
    seg_p = dict(seg, concepts=concepts[perm])
    with torch.no_grad():
        core(x, t, y, segment_data=seg_p)
    for k in scores:
        assert torch.allclose(seg_p["concept_scores"][k], scores[k][..., perm], atol=1e-4, rtol=1e-4), k

    # Bad layers are rejected.
    try:
        with torch.no_grad():
            core(x, t, y, segment_data=dict(seg, concept_layers=[5]))
    except ValueError:
        pass
    else:
        raise AssertionError("concept layer 5 > 4 built blocks should raise")

    Q = 7
    mask_logits = torch.randn(B, Q, *hw, device=device)
    for pixel_softmax in (True, False):
        clf = ConceptMaskClassifier(temperature=0.5, pixel_softmax=pixel_softmax)
        probs = clf(scores, mask_logits, hw)
        assert probs.shape == (B, Q, K), probs.shape
        assert torch.allclose(probs.sum(-1), torch.ones(B, Q, device=device), atol=1e-4), probs.sum(-1)
        pix = clf.pixel_map(scores, hw)
        assert pix.shape == (B, K, *hw), pix.shape
        assert torch.allclose(pix.sum(1), torch.ones(B, *hw, device=device), atol=1e-4)
    # A mask covering exactly one patch gets that patch's class distribution.
    clf = ConceptMaskClassifier(binarize_masks=True)
    one = torch.full((B, 1, *hw), -20.0, device=device)
    one[:, :, 3, 4] = 20.0
    idx = 3 * hw[1] + 4
    assert torch.allclose(clf(scores, one, hw)[:, 0], clf.pixel_scores(scores)[:, idx], atol=1e-5)
    # Empty binarized mask -> all-zero row (no object).
    assert clf(scores, torch.full((B, 1, *hw), -20.0, device=device), hw).abs().sum() == 0
    # Concept-only pass (no side-adapter features), as used for the noised pass.
    only = {"feature_layers": [], "concepts": concepts, "concept_layers": [2, 4], "concept_position": 2}
    with torch.no_grad():
        core(x, t, y, segment_data=only)
    assert only["features"] == {} and sorted(only["concept_scores"]) == [2, 4]
    for k in (2, 4):
        assert torch.allclose(only["concept_scores"][k], scores[k], atol=1e-4, rtol=1e-4), k
    # Pooling with explicit per-patch weights (GT masks), unbatched.
    clf = ConceptMaskClassifier()
    wts = torch.zeros(2, hw[0] * hw[1], device=device)
    wts[0, idx] = 1.0
    wts[1] = 1.0
    pooled = clf.pool(clf.pixel_scores(scores)[0], wts)
    assert pooled.shape == (2, K), pooled.shape
    assert torch.allclose(pooled[0], clf.pixel_scores(scores)[0, idx], atol=1e-5)
    assert torch.allclose(pooled[1], clf.pixel_scores(scores)[0].mean(0), atol=1e-5)
    # Masks on a different grid than the PixelDiT patches.
    probs = ConceptMaskClassifier()(scores, torch.randn(B, Q, 16, 12, device=device), hw)
    assert probs.shape == (B, Q, K), probs.shape
    print("concept stream + mask classifier: ok")


def check_concept_lora():
    local_config.layer_count = 4
    core = PixDiT_T2I(
        hidden_size=96, num_groups=4, patch_depth=14, txt_embed_dim=32, txt_max_length=8
    ).to(device).eval()
    for p in core.parameters():
        p.requires_grad = False
    B, K, hw = 2, 5, (8, 6)
    P = hw[0] * hw[1]
    x = torch.randn(B, 3, 128, 96, device=device)
    t = torch.full((B,), 400.0, device=device)
    y = torch.randn(1, 8, 32, device=device)
    concepts = torch.randn(K, 32, device=device)
    layers = [1, 2, 3, 4]
    base = {"feature_layers": [1, 2, 3], "concepts": concepts, "concept_layers": layers, "concept_position": 2}

    plain = dict(base)
    with torch.no_grad():
        core(x, t, y, segment_data=plain)

    lora = ConceptLoRA(96, num_blocks=4, rank=4, alpha=4.0).to(device)
    # Zero-initialised up projections: identical to the frozen model.
    seg = dict(base, concept_adapter=lora)
    with torch.no_grad():
        core(x, t, y, segment_data=seg)
    for k in layers:
        assert torch.equal(seg["concept_scores"][k], plain["concept_scores"][k]), k

    # Training step: cosine scores, learned head, CE on mask-pooled logits.
    clf = ConceptMaskClassifier(learned=True, num_layers=len(layers)).to(device)
    seg = dict(base, concept_adapter=lora, concept_cosine=True)
    core(x, t, y, segment_data=seg)  # grad enabled
    for k in layers:
        s = seg["concept_scores"][k]
        assert s.requires_grad and s.abs().max() <= 1 + 1e-4, (k, s.abs().max())
    for k in [1, 2, 3]:
        assert not seg["features"][k].requires_grad
        assert torch.equal(seg["features"][k], plain["features"][k]), f"layer {k} changed"
    S = 3
    weights = torch.rand(S, P, device=device)
    logits = clf.pool_logits(clf.pixel_scores(seg["concept_scores"])[0], weights)
    assert logits.shape == (S, K), logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits, torch.randint(0, K, (S,), device=device), label_smoothing=0.1
    )
    assert torch.isfinite(loss), loss
    loss.backward()
    # Up projections get gradients right away (the down ones only once up != 0).
    for name, p in lora.named_parameters():
        if "_up" in name:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert any(p.grad.abs().sum() > 0 for n, p in lora.named_parameters() if "_up" in n)
    assert clf.logit_scale.grad is not None and clf.layer_logits.grad is not None
    assert all(p.grad is None for p in core.parameters())

    # A non-zero adapter changes the concept scores but never the image features.
    with torch.no_grad():
        for p in lora.parameters():
            p.normal_(std=0.1)
        seg = dict(base, concept_adapter=lora)
        core(x, t, y, segment_data=seg)
    assert not torch.allclose(seg["concept_scores"][4], plain["concept_scores"][4])
    for k in [1, 2, 3]:
        assert torch.equal(seg["features"][k], plain["features"][k]), f"layer {k} changed"
    probs = clf.pool(clf.pixel_scores(seg["concept_scores"])[0], weights)
    assert torch.allclose(probs.sum(-1), torch.ones(S, device=device), atol=1e-4)
    print("concept LoRA: ok")


if __name__ == "__main__":
    check_feature_hook()
    check_concept_stream()
    check_concept_lora()
    check_side_adapter()
    print("all checks passed")
