"""Break down an evaluation's per-class results by (seen in training | unseen) and
(single name | synonym list), to tell vocabulary overfitting apart from problems
with how class names are encoded.

Usage (from the repo root):
  python tools/class_breakdown.py output/adapter_20k/metrics.json
  python tools/class_breakdown.py output/adapter_20k/metrics.json \
      --dataset ade20k_sem_seg_val --train-names t2i/catseg_configs/coco.json

Reads the last record of metrics.json that has per-class IoU for the dataset (keys
"<dataset>/sem_seg/IoU-<class>", or "sem_seg/IoU-<class>" when it was the only test
set). A class is "seen" if any of its synonyms equals one of a training class's
synonyms after normalisation (lower case; COCO-Stuff suffixes "-merged", "-other",
"-stuff" removed). The matching is approximate, so the seen classes are printed for
a sanity check.
"""
import argparse
import json
import re
import statistics


def synonyms(name: str):
    out = set()
    for part in name.split(","):
        s = part.strip().lower()
        s = re.sub(r"-(merged|other|stuff)\b", "", s).replace("-", " ").strip()
        if s:
            out.add(s)
    return out


def load_metrics(path: str, dataset: str):
    prefixes = [f"{dataset}/sem_seg/", "sem_seg/"]
    record = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for p in prefixes:
                if any(k.startswith(p + "IoU-") for k in r):
                    record, prefix = r, p
                    break
    if record is None:
        raise SystemExit(f"No per-class IoU for {dataset} in {path}")
    iou = {k[len(prefix) + 4:]: v for k, v in record.items() if k.startswith(prefix + "IoU-")}
    acc = {k[len(prefix) + 4:]: v for k, v in record.items() if k.startswith(prefix + "ACC-")}
    return record.get("iteration"), prefix, iou, acc, record.get(prefix + "mIoU")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--dataset", default="ade20k_sem_seg_val")
    ap.add_argument("--train-names", default="t2i/catseg_configs/coco.json",
                    help="JSON list of training class names (CAT-Seg style)")
    args = ap.parse_args()

    with open(args.train_names) as f:
        train = [x for x in json.load(f) if x != "unknown"]
    seen_vocab = set().union(*(synonyms(n) for n in train))

    it, prefix, iou, acc, miou = load_metrics(args.metrics, args.dataset)
    groups = {}
    seen_list = []
    for name, v in iou.items():
        seen = bool(synonyms(name) & seen_vocab)
        multi = "," in name
        groups.setdefault((seen, multi), []).append((name, v, acc.get(name, float("nan"))))
        if seen:
            seen_list.append(name)

    print(f"{args.dataset} at iteration {it}: mIoU {miou:.2f} over {len(iou)} classes "
          f"(keys {prefix}*)\n")
    print(f"{'group':<32}{'classes':>8}{'mIoU':>8}{'mACC':>8}{'IoU=0':>8}")
    for seen in (True, False):
        for multi in (False, True):
            g = groups.get((seen, multi), [])
            if not g:
                continue
            label = f"{'seen' if seen else 'unseen'}, {'synonym list' if multi else 'single name'}"
            zero = sum(1 for _, v, _ in g if v < 0.5)
            print(f"{label:<32}{len(g):>8}{statistics.mean(v for _, v, _ in g):>8.2f}"
                  f"{statistics.mean(a for _, _, a in g):>8.2f}{zero:>8}")
    for seen in (True, False):
        g = [x for m in (False, True) for x in groups.get((seen, m), [])]
        if g:
            print(f"{'all ' + ('seen' if seen else 'unseen'):<32}{len(g):>8}"
                  f"{statistics.mean(v for _, v, _ in g):>8.2f}{statistics.mean(a for _, _, a in g):>8.2f}")

    print(f"\nClasses counted as seen ({len(seen_list)}):\n  " + "; ".join(sorted(seen_list)))
    worst = sorted((v, n) for g in groups.values() for n, v, _ in g)[:15]
    print("\nLowest IoU:\n  " + "; ".join(f"{n} {v:.1f}" for v, n in worst))


if __name__ == "__main__":
    main()