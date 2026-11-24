"""Precomputed text-encoder (Gemma) outputs (MODEL.TEXT_CACHE).

The text encoder is only used when the model is built, for
  * the default prompt that conditions every PixelDiT pass
    (embedding (1, 1, 300, D), attention mask (1, 300), token count), and
  * the class-name concept tokens of every dataset (MODEL.CONCEPT / MODEL.REC):
    (N, D) embeddings, the (N,) class index of every synonym variant and the
    text position of the concepts.
tools/precompute_text_embeddings.py computes these once and saves them; with
MODEL.TEXT_CACHE.PATH set the model loads them instead of Gemma.

The cache stores the settings it was computed with (the "spec": encoder name,
default prompt, templates, pooling, name cleaning, and the class names of each
dataset). Loading checks them against the current config, so a cache made with
other text settings is refused instead of silently used.
"""
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional

import torch

from .concept_classifier import encode_concepts, instruction_prefix

CACHE_VERSION = 1
PROMPT_LENGTH = 300


def build_text_spec(
    text_encoder_name: str,
    default_prompt: str,
    templates: List[str],
    pool: str,
    instruction_context: bool,
    clean_names: bool,
    split_synonyms: bool,
    dataset_labels: Dict[str, List[str]],
) -> Dict:
    """Everything the cached text-encoder outputs depend on."""
    return {
        "text_encoder_name": text_encoder_name,
        "default_prompt": default_prompt,
        "templates": list(templates),
        "pool": pool,
        "instruction_context": bool(instruction_context),
        "clean_names": bool(clean_names),
        "split_synonyms": bool(split_synonyms),
        "datasets": {k: list(v) for k, v in dataset_labels.items()},
    }


def spec_digest(spec: Dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


@torch.no_grad()
def embed_prompts(tokenizer, text_encoder, prompts, device, with_chi_window=False):
    """Text-encoder embeddings of `prompts` as PixelDiT takes them:
    (B, 1, 300, D), attention mask (B, 300), token counts (B,)."""
    max_length = 506 if with_chi_window else PROMPT_LENGTH
    caption_token = tokenizer(
        prompts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    select_index = (
        [0] + list(range(-PROMPT_LENGTH + 1, 0)) if with_chi_window else list(range(PROMPT_LENGTH))
    )
    caption_embs = torch.cat(
        [
            text_encoder(caption_token.input_ids[[i]], caption_token.attention_mask[[i]])[0][:, None][
                :, :, select_index
            ]
            for i in range(caption_token.input_ids.shape[0])
        ],
        dim=0,
    )
    tokens_count = caption_token.attention_mask.sum(dim=-1)
    emb_masks = caption_token.attention_mask[:, select_index]
    return caption_embs, emb_masks, tokens_count


@torch.no_grad()
def compute_text_cache(tokenizer, text_encoder, spec: Dict, device, log=print) -> Dict:
    """Run the text encoder for everything in `spec` -> cache dict (CPU tensors)."""
    emb, mask, count = embed_prompts(tokenizer, text_encoder, [spec["default_prompt"]], device)
    context = instruction_prefix(spec["default_prompt"]) if spec["instruction_context"] else ""
    concepts = {}
    for name, labels in spec["datasets"].items():
        embs, variant_class, position = encode_concepts(
            tokenizer,
            text_encoder,
            labels,
            template=spec["templates"],
            pool=spec["pool"],
            device=device,
            context=context,
            clean_names=spec["clean_names"],
            split_synonyms=spec["split_synonyms"],
        )
        concepts[name] = {
            "embs": embs.float().cpu(),
            "variant_class": variant_class.cpu(),
            "position": int(position),
        }
        log(f"text cache: {name}: {len(labels)} classes -> {embs.shape[0]} concepts")
    return {
        "version": CACHE_VERSION,
        "spec": spec,
        "digest": spec_digest(spec),
        "default": {
            "emb": emb.float().cpu(),
            "mask": mask.cpu(),
            "tokens_count": count[0].cpu(),
        },
        "concepts": concepts,
    }


def save_text_cache(path: str, cache: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(cache, tmp)
    os.replace(tmp, path)  # never leave a half-written cache behind


def _mismatch(name: str, cached, current) -> str:
    def short(v):
        s = json.dumps(v)
        return s if len(s) <= 120 else s[:117] + "..."

    return f"  {name}: cache {short(cached)} vs config {short(current)}"


def load_text_cache(
    path: str, spec: Dict, datasets: Optional[Iterable[str]] = None, precompute_hint: str = ""
) -> Dict:
    """Load a cache made by compute_text_cache and check it against `spec`.
    `datasets`: the datasets whose concepts are needed (default: all of spec's).
    The cache may hold more datasets than needed, but every needed one must be
    there with the same class names."""
    hint = f"\nRecompute it with:\n  {precompute_hint}" if precompute_hint else ""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MODEL.TEXT_CACHE.PATH {path!r} does not exist.{hint}")
    cache = torch.load(path, map_location="cpu")
    if cache.get("version") != CACHE_VERSION:
        raise ValueError(
            f"Text cache {path!r} has version {cache.get('version')}, expected {CACHE_VERSION}.{hint}"
        )
    cached = cache["spec"]
    errors = [
        _mismatch(k, cached.get(k), v) for k, v in spec.items() if k != "datasets" and cached.get(k) != v
    ]
    for name in (spec["datasets"] if datasets is None else datasets):
        if name not in cache["concepts"]:
            errors.append(f"  dataset {name!r} is not in the cache (has {sorted(cache['concepts'])})")
        elif cached["datasets"][name] != spec["datasets"][name]:
            errors.append(f"  dataset {name!r}: class names differ")
    if errors:
        raise ValueError(
            f"Text cache {path!r} was computed with other text settings:\n" + "\n".join(errors) + hint
        )
    return cache
