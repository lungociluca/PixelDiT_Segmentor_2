"""Run the text encoder (Gemma) once and save its outputs for MODEL.TEXT_CACHE.

Saves the default-prompt embedding that conditions PixelDiT and the class-name
concept tokens of every dataset in DATASETS.TRAIN / DATASETS.TEST (plus
--extra-datasets), with the text settings of the given config. Training and
evaluation with MODEL.TEXT_CACHE.PATH set then load this file and never load
Gemma.

Run from the repository root, with the same config (and overrides) you train with:

  python tools/precompute_text_embeddings.py \
      --config-file configs/san_pixeldit_rec_attn_bias.yaml \
      MODEL.TEXT_CACHE.PATH text_cache/rec_attn_bias.pt

  python train_net.py --config-file configs/san_pixeldit_rec_attn_bias.yaml --num-gpus 1 \
      MODEL.TEXT_CACHE.PATH text_cache/rec_attn_bias.pt OUTPUT_DIR output_rec

The output path is MODEL.TEXT_CACHE.PATH (or --output). Recompute when you
change the templates, pooling, name cleaning / synonyms, instruction context,
the default prompt or the datasets; loading a stale cache fails and says what
differs.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

import san  # noqa: F401  (registers the datasets)
import t2i.config as local_config
from san.config import add_san_config
from san.model.san import get_dataset_labels, load_pixeldit_config, load_text_encoder
from san.model.text_cache import build_text_spec, compute_text_cache, load_text_cache, save_text_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output", default="", help="default: MODEL.TEXT_CACHE.PATH")
    parser.add_argument(
        "--extra-datasets", nargs="*", default=[],
        help="more registered dataset names to include (e.g. to evaluate on later)",
    )
    parser.add_argument("--overwrite", action="store_true", help="recompute even if a matching cache exists")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="config overrides, as for train_net.py")
    args = parser.parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output = args.output or cfg.MODEL.TEXT_CACHE.PATH
    if not output:
        parser.error("give the output file as MODEL.TEXT_CACHE.PATH <file> or --output <file>")

    names = list(dict.fromkeys(list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST) + args.extra_datasets))
    c = cfg.MODEL.CONCEPT
    pixeldit_config = load_pixeldit_config()
    spec = build_text_spec(
        text_encoder_name=pixeldit_config.text_encoder.text_encoder_name,
        default_prompt=local_config.default_prompt,
        templates=list(c.TEMPLATES) or [c.TEMPLATE],
        pool=c.POOL,
        instruction_context=c.INSTRUCTION_CONTEXT,
        clean_names=c.CLEAN_NAMES,
        split_synonyms=c.SPLIT_SYNONYMS,
        dataset_labels={n: get_dataset_labels(n) for n in names},
    )

    if os.path.isfile(output) and not args.overwrite:
        try:
            cache = load_text_cache(output, spec)
            print(f"{output} already matches this config (spec {cache['digest']}); nothing to do "
                  "(--overwrite to recompute).")
            return
        except ValueError as e:
            print(f"Recomputing {output}:\n{e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()
    tokenizer, text_encoder = load_text_encoder(pixeldit_config, device)
    cache = compute_text_cache(tokenizer, text_encoder, spec, device)
    save_text_cache(output, cache)
    print(
        f"Saved {output} (spec {cache['digest']}, {os.path.getsize(output) / 2**20:.1f} MiB, "
        f"{time.time() - start:.0f} s): default prompt + concepts of {names}.\n"
        f"Train / evaluate with:  MODEL.TEXT_CACHE.PATH {output}"
    )


if __name__ == "__main__":
    main()
