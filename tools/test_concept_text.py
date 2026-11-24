"""CPU checks of the concept text side (names, synonyms, templates, reduction).

    python tools/test_concept_text.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from san.model.concept_classifier import (  # noqa: E402
    class_name_variants,
    clean_class_name,
    encode_concepts,
    instruction_prefix,
    reduce_synonyms,
)


def test_names():
    assert clean_class_name("diningtable") == "dining table"
    assert clean_class_name("wall-other-merged") == "wall"
    assert clean_class_name("tree-merged") == "tree"
    assert clean_class_name("door-stuff") == "door"
    assert clean_class_name("playingfield") == "playing field"
    assert clean_class_name("window-blind") == "window blind"
    assert class_name_variants("chest of drawers, chest, bureau, dresser") == [
        "chest of drawers", "chest", "bureau", "dresser"]
    assert class_name_variants("motorbike") == ["motorbike", "motorcycle"]
    assert class_name_variants("motorbike", clean=False) == ["motorbike"]
    assert class_name_variants("rock, stone", split=False) == ["rock, stone"]
    assert class_name_variants("window ") == ["window"]
    assert class_name_variants("sky, Sky") == ["sky"]


def test_reduce():
    variant_class = torch.tensor([0, 1, 1, 2, 2, 2])
    s = torch.randn(2, 5, 6, requires_grad=True)
    mx = reduce_synonyms(s, variant_class, 3, "max")
    assert mx.shape == (2, 5, 3)
    assert torch.allclose(mx[..., 0], s[..., 0])
    assert torch.allclose(mx[..., 1], s[..., 1:3].amax(-1))
    assert torch.allclose(mx[..., 2], s[..., 3:6].amax(-1))
    mn = reduce_synonyms(s, variant_class, 3, "mean")
    assert torch.allclose(mn[..., 2], s[..., 3:6].mean(-1))
    (mx.sum() + mn.sum()).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    one = torch.arange(3)
    t = torch.randn(1, 4, 3)
    assert reduce_synonyms(t, one, 3) is t  # no synonyms: unchanged


class FakeTokenizer:
    """Whitespace tokenizer with a BOS token; ids from a growing vocabulary."""

    def __init__(self):
        self.vocab = {"<bos>": 0}

    def _ids(self, text):
        return [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = ([0] if add_special_tokens else []) + self._ids(text)

        class Out:
            pass

        out = Out()
        if return_tensors == "pt":
            out.input_ids = torch.tensor([ids])
            out.attention_mask = torch.ones_like(out.input_ids)
            out.to = lambda device: out
        else:
            out.input_ids = ids
        return out


class FakeEncoder:
    """hidden state of a token = one-hot of its id (dim 64)."""

    def __call__(self, input_ids, attention_mask):
        return (torch.nn.functional.one_hot(input_ids, 64).float(),)


def test_encode():
    tok, enc = FakeTokenizer(), FakeEncoder()
    names = ["cat", "rock, stone", "diningtable"]
    templates = ["A picture of a {target}", "A photo of the {target} ."]
    embs, cls, start = encode_concepts(
        tok, enc, names, template=templates, device="cpu", clean_names=True, split_synonyms=True
    )
    assert cls.tolist() == [0, 1, 1, 2], cls
    assert embs.shape == (4, 64)
    assert start == 5  # <bos> A picture of a
    # "cat" pooled over its own token only, identical in both templates.
    assert embs[0, tok.vocab["cat"]] == 1.0 and embs[0].sum() == 1.0
    # "dining table": mean of its two tokens.
    assert embs[3, tok.vocab["dining"]] == 0.5 and embs[3, tok.vocab["table"]] == 0.5
    # Instruction context shifts the name position.
    ctx = "Some instruction here. User Prompt: "
    _, _, start_ctx = encode_concepts(tok, enc, ["cat"], template=templates, device="cpu", context=ctx)
    assert start_ctx == start + len(ctx.split())


class PaddingFakeTokenizer(FakeTokenizer):
    """FakeTokenizer that also takes a batch of prompts padded to max_length."""

    def __call__(self, text, add_special_tokens=True, return_tensors=None, max_length=None,
                 padding=None, truncation=False):
        if isinstance(text, str):
            return super().__call__(text, add_special_tokens, return_tensors)
        rows = [([0] + self._ids(t))[:max_length] for t in text]
        ids = torch.zeros(len(rows), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, r in enumerate(rows):
            ids[i, : len(r)], mask[i, : len(r)] = torch.tensor(r), 1

        class Out:
            pass

        out = Out()
        out.input_ids, out.attention_mask = ids, mask
        out.to = lambda device: out
        return out


def test_text_cache(tmp_dir="/tmp/text_cache_test"):
    from san.model.text_cache import build_text_spec, compute_text_cache, load_text_cache, save_text_cache

    tok, enc = PaddingFakeTokenizer(), FakeEncoder()
    spec = build_text_spec(
        "fake", "Do things. User Prompt: a scene", ["A picture of a {target}", "A photo of the {target}."],
        "mean", True, True, True, {"coco": ["cat", "dining table", "person-other"], "voc": ["cat", "dog"]},
    )
    cache = compute_text_cache(tok, enc, spec, "cpu", log=lambda *_: None)
    assert cache["default"]["emb"].shape == (1, 1, 300, 64) and cache["default"]["mask"].shape == (1, 300)
    assert int(cache["default"]["tokens_count"]) == 1 + len(spec["default_prompt"].split())
    # Same numbers as encoding directly.
    embs, cls, pos = encode_concepts(
        tok, enc, ["cat", "dining table", "person-other"], template=spec["templates"], device="cpu",
        context=instruction_prefix(spec["default_prompt"]), clean_names=True, split_synonyms=True,
    )
    c = cache["concepts"]["coco"]
    assert torch.equal(c["embs"], embs) and torch.equal(c["variant_class"], cls) and c["position"] == pos

    path = os.path.join(tmp_dir, "cache.pt")
    save_text_cache(path, cache)
    loaded = load_text_cache(path, spec)
    assert torch.equal(loaded["concepts"]["voc"]["embs"], cache["concepts"]["voc"]["embs"])
    # A subset of the datasets is fine; other settings or class names are refused.
    load_text_cache(path, dict(spec, datasets={"voc": ["cat", "dog"]}), datasets=["voc"])
    for bad, kwargs in [
        (dict(spec, pool="last"), {}),
        (dict(spec, datasets={"voc": ["cat", "bird"]}), {"datasets": ["voc"]}),
        (dict(spec, datasets={"ade": ["wall"]}), {"datasets": ["ade"]}),
    ]:
        try:
            load_text_cache(path, bad, **kwargs)
            raise AssertionError(f"stale cache accepted for {kwargs}")
        except ValueError:
            pass
    try:
        load_text_cache(os.path.join(tmp_dir, "missing.pt"), spec)
        raise AssertionError("missing cache accepted")
    except FileNotFoundError:
        pass


def test_instruction_prefix():
    p = "Do things. User Prompt: A cat -> x. More. User Prompt: A picture with sky."
    assert instruction_prefix(p) == "Do things. User Prompt: A cat -> x. More. User Prompt: "


if __name__ == "__main__":
    test_names()
    test_reduce()
    test_encode()
    test_instruction_prefix()
    test_text_cache()
    print("all concept text tests passed")
