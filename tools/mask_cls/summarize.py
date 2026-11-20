"""Summary table: teacher vs student, seen / unseen, per dataset.

    python tools/mask_cls/summarize.py [--metrics top1 miou] [--filter ridge]

Reads <results_dir>/results.jsonl (latest row per run and dataset), writes
summary.csv and summary.md next to it and prints the markdown table.
"""
import csv
import json
import logging
import os

import _common

log = logging.getLogger("mask_cls")

SHORT = {
    "coco_2017_test_stuff_sem_seg": "COCO",
    "ade20k_sem_seg_val": "A-150",
    "ade20k_full_sem_seg_val": "A-847",
    "voc_sem_seg_val": "VOC",
    "pcontext_sem_seg_val": "PC-59",
    "pcontext_full_sem_seg_val": "PC-459",
}


def main():
    def extra(ap):
        ap.add_argument("--metrics", nargs="*", default=["top1", "miou"])
        ap.add_argument("--filter", default=None, help="substring of run_name to keep")

    args, cfg = _common.parse(__doc__, extra)
    path = os.path.join(cfg.paths.results_dir, "results.jsonl")
    rows = [json.loads(l) for l in open(path)]
    latest = {}
    for r in rows:
        if args.filter and args.filter not in r["run_name"]:
            continue
        latest[(r["run_name"], r["dataset"])] = r
    runs = list(dict.fromkeys(k[0] for k in latest))
    datasets = [d for d in cfg.data.eval if any(k[1] == d for k in latest)]
    # Teachers first, then students.
    runs.sort(key=lambda n: (next(r for k, r in latest.items() if k[0] == n)["kind"] != "teacher", n))

    cols = []
    for d in datasets:
        for m in args.metrics:
            subs = ["seen"] if d.startswith("coco") else ["seen", "unseen"]
            cols += [(d, f"{m}_{s}") for s in subs]
    header = ["run", "kind", "method", "target", "features", "scoring"] + [f"{SHORT.get(d, d)} {m}" for d, m in cols]
    table = []
    for n in runs:
        any_r = next(r for k, r in latest.items() if k[0] == n)
        line = [n, any_r.get("kind"), any_r.get("method"), any_r.get("target"), any_r.get("features"),
                any_r.get("scoring", "raw")]
        for d, m in cols:
            v = latest.get((n, d), {}).get(m)
            line.append("" if v is None else f"{v:.1f}")
        table.append(line)

    out_csv = os.path.join(cfg.paths.results_dir, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(x) for x in line) + " |" for line in table]
    out_md = os.path.join(cfg.paths.results_dir, "summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    log.info("wrote %s and %s", out_csv, out_md)


if __name__ == "__main__":
    main()
