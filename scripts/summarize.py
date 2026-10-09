"""Summarise a results JSONL file by (name, n): mean ± SEM of key metrics.

Usage: python scripts/summarize.py results/e1.jsonl [--split best_natural] [--compare-workshop DIR]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

KEYS = ["acc", "dprime", "hit", "fa", "fa_nonlure", "auc"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--split", default="best_natural")
    p.add_argument("--compare-workshop", default="", help="workshop repo scripts/ dir")
    a = p.parse_args()
    rows = [json.loads(l) for l in open(a.path)]
    g = defaultdict(list)
    for r in rows:
        g[(r["name"], r["config"]["n"])].append(r)
    ref = {}
    if a.compare_workshop:
        d = json.load(open(f"{a.compare_workshop}/trained_models/1_layer_1_head_all_results.json"))
        for n in range(1, 7):
            v = np.array([x["test_accuracy"] for k, x in d.items() if k.startswith(f"{n}back_")])
            ref[n] = (v.mean(), v.std() / np.sqrt(len(v)), len(v))
    print(f"{'name':<12}{'N':>3}{'runs':>5}{'steps':>8}  " + "  ".join(f"{k:>14}" for k in KEYS)
          + ("   workshop_acc(all pos)  ours  diff/SE" if ref else ""))
    for (name, n), rs in sorted(g.items()):
        cells = []
        for k in KEYS:
            v = np.array([r[a.split].get(k, np.nan) for r in rs], float)
            cells.append(f"{np.nanmean(v):7.3f}±{np.nanstd(v) / np.sqrt(len(v)):.3f}")
        line = f"{name:<12}{n:>3}{len(rs):>5}{np.mean([r['steps'] for r in rs]):>8.0f}  " + "  ".join(f"{c:>14}" for c in cells)
        if ref and f"{a.split.split('_')[0]}_workshop_test" in rs[0]:
            ours = np.array([r[f"{a.split.split('_')[0]}_workshop_test"]["acc_all_positions"] for r in rs])
            m, se, k = ref[n]
            se_c = np.sqrt(se ** 2 + (ours.std() / np.sqrt(len(ours))) ** 2)
            line += f"   {m:.4f}±{se:.4f} (n={k})  {ours.mean():.4f}  {(ours.mean() - m) / se_c:+.2f}"
        print(line)


if __name__ == "__main__":
    main()
