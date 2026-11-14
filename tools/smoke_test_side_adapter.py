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


def check_concept_classifier():
    from san.model.concept_classifier import ConceptClassifier

    local_config.layer_count = 4
    core = PixDiT_T2I(
        hidden_size=96, num_groups=4, patch_depth=14, txt_embed_dim=32, txt_max_length=12
    ).to(device).eval()
    B, H, W = 2, 128, 96
    x = torch.randn(B, 3, H, W, device=device)
    y = torch.randn(1, 12, 32, device=device)
    t = torch.full((B,), 500.0, device=device)  # the DPM wrapper passes one t per image
    C, Q = 3, 5
    caps = torch.randn(C, 12, 32, device=device)
    select = torch.zeros(C, 12, dtype=torch.bool, device=device)
    select[0, 4:5] = True
    select[1, 4:7] = True
    select[2, 4:6] = True

    with torch.no_grad():
        ref = {"feature_layers": [1, 2, 3]}
        core(x, t, y, segment_data=ref)
        emb = {"feature_layers": [0]}
        core(x, t, y, segment_data=emb)
    h, w = ref["hw"]
    L = h * w

    for adapter, detach, pooling in [
        (False, False, "probs"), (False, False, "scores"), (True, True, "scores")
    ]:
        clf = ConceptClassifier(
            hidden_size=96, readout_blocks=[2, 4], image_adapter=adapter, detach_masks=detach,
            pooling=pooling,
        ).to(device)
        # One prompt per class (the per-prompt name tokens and their positions).
        name_embs = [caps[c][select[c]] for c in range(C)]
        name_pos = [select[c].nonzero().flatten() for c in range(C)]
        concepts = clf.build_concepts(core, name_embs, name_pos, list(range(C)), C)
        assert concepts.tokens.shape == (6, 96) and concepts.pool.shape == (6, C)
        assert torch.allclose(concepts.pool.sum(0), torch.ones(C, device=device))
        features, readout = clf.run_backbone(
            core, emb["features"][0], emb["text_features"][0], emb["condition"], emb["pos"],
            concepts, [1, 2, 3],
        )
        # The concept stream is one-directional: same image features as a normal pass.
        for k in (1, 2, 3):
            assert torch.allclose(features[k], ref["features"][k], atol=1e-4), f"layer {k} changed"
        assert len(readout) == 2

        probs = clf.token_outputs(readout, concepts)
        assert probs.shape == (B, L, C)
        if pooling == "probs":
            assert torch.allclose(probs.sum(-1), torch.ones(B, L, device=device), atol=1e-5)
        else:  # block-weighted standardised scores: zero mean across classes
            assert torch.allclose(probs.mean(-1), torch.zeros(B, L, device=device), atol=1e-4)
        if not adapter:
            # zscore: invariant to the scale of the dot products, so the softmax
            # cannot saturate however large the raw scores are.
            big = [(o_x * 100.0, o_c * 10.0) for o_x, o_c in readout]
            assert torch.allclose(clf.token_outputs(big, concepts), probs, atol=1e-4), "zscore not scale-free"
            probs = clf.token_outputs(readout, concepts)  # restore stats of the real scores
        st = clf.stats()
        assert {"temperature", "block_weight_2", "block_weight_4", "raw_score_std_2"} <= set(st), st

        mask_preds = torch.randn(B, Q, h, w, device=device, requires_grad=True)
        objectness = torch.randn(B, Q, 2, device=device, requires_grad=True)
        logits = clf.mask_class_logits(probs, mask_preds, objectness)
        assert logits.shape == (B, Q, C + 1)
        # Log-probabilities of a proper C + 1 distribution.
        assert torch.allclose(logits.logsumexp(-1), torch.zeros(B, Q, device=device), atol=1e-4)

        target = torch.randint(0, C + 1, (B * Q,), device=device)
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), target).backward()
        trainable = {n: p for n, p in clf.named_parameters()}
        assert {"block_logits", "log_temperature"} <= set(trainable), trainable.keys()
        for n, p in trainable.items():
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert objectness.grad is not None
        assert (mask_preds.grad is None) == detach, "DETACH_MASKS not respected"
    print("concept classifier: ok")


class _WordTokenizer:
    """Stand-in for the Gemma tokenizer: BOS + one id per word / punctuation mark."""
    bos_token_id = 1

    def __call__(self, text):
        import re
        words = re.findall(r"\w+|[^\w\s]", text)
        return {"input_ids": [self.bos_token_id] + [hash(w) % 10000 + 2 for w in words]}


def check_vocabulary():
    from san.model.concept_classifier import ConceptClassifier

    tok = _WordTokenizer()
    clf = ConceptClassifier(
        hidden_size=96, readout_blocks=[2],
        prompt_templates=("A picture of a {target}", "There is {target} in the scene."),
    )
    specs = clf.prompt_specs(["dog", "road, route", "sidewalk, pavement, footway"])
    assert len(specs) == 2 * (1 + 2 + 3), len(specs)
    assert [s.owner for s in specs] == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    assert {s.name for s in specs if s.owner == 1} == {"road", "route"}
    L = 16
    mask = torch.zeros(len(specs), L, dtype=torch.long)
    for i, sp in enumerate(specs):
        mask[i, : len(tok(sp.text)["input_ids"])] = 1
    sel = clf.name_token_mask(tok, specs, mask)
    # "A picture of a dog": BOS A picture of a | dog -> position 5 only
    assert sel[0].nonzero().flatten().tolist() == [5], sel[0]
    # "There is road in the scene.": BOS There is | road | in the scene . -> position 3
    assert sel[3].nonzero().flatten().tolist() == [3], sel[3]
    print("vocabulary: ok")


def check_isolated_concepts():
    """With isolate_concepts, adding classes to the vocabulary must not change the
    scores of the others (without it, it does)."""
    from san.model.concept_classifier import ConceptClassifier

    local_config.layer_count = 3
    core = PixDiT_T2I(
        hidden_size=96, num_groups=4, patch_depth=14, txt_embed_dim=32, txt_max_length=12
    ).to(device).eval()
    x = torch.randn(1, 3, 64, 64, device=device)
    y = torch.randn(1, 12, 32, device=device)
    with torch.no_grad():
        emb = {"feature_layers": [0]}
        core(x, torch.full((1,), 500.0, device=device), y, segment_data=emb)
    embs = [torch.randn(n, 32, device=device) for n in (1, 3, 2, 2)]
    pos = [torch.arange(4, 4 + n) for n in (1, 3, 2, 2)]
    for isolate in (True, False):
        clf = ConceptClassifier(hidden_size=96, readout_blocks=[2, 3], isolate_concepts=isolate).to(device)
        outs = []
        for n in (3, 4):  # vocabulary of 3 classes, then the same plus one more
            concepts = clf.build_concepts(core, embs[:n], pos[:n], list(range(n)), n)
            _, readout = clf.run_backbone(
                core, emb["features"][0], emb["text_features"][0], emb["condition"], emb["pos"],
                concepts, [],
            )
            # raw output-space scores of the first 3 classes at the last readout block
            o_x, o_c = readout[-1]
            outs.append((o_x[0] @ o_c[0].T)[:, : 1 + 3 + 2])
        same = torch.allclose(outs[0], outs[1], atol=1e-4)
        assert same == isolate, f"isolate_concepts={isolate}: vocabulary-invariant={same}"
    print("isolated concepts: ok")


if __name__ == "__main__":
    check_feature_hook()
    check_side_adapter()
    check_concept_classifier()
    check_vocabulary()
    check_isolated_concepts()
    print("all checks passed")
