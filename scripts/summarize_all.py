"""Reduce every result part to one compact summary file (results/summary.json).

Sections:
  toy        per (experiment tag, N): run count, mean and SEM of behaviour (natural and
             lure test sets) and of layer-0 head-0 attention statistics, solution share
  e1_seeds   one compact row per E1 run (the G1 gate statistics)
  e1_curves  mean test d' against training step per N (E1)
  l1         per model: pieces that are not single tokens (empty = prompt format is valid)
  l2         LLM behaviour rows (metrics only)
  l6_behave  Pythia checkpoint behaviour rows
  l6_heads   Pythia checkpoint head summaries (top heads by N-back score and patching)

Usage: python scripts/summarize_all.py [--parts results/parts] [--out results/summary.json]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

BEH = ["acc", "dprime", "hit", "fa", "fa_nonlure", "auc", "fa_lure_1", "fa_lure_2", "fa_lure_3", "fa_lure_4",
       "fa_lure_5", "fa_lure_6", "fa_lure_7", "fa_lure_8"]
ATT = ["target_att_match", "target_att_nonmatch", "entropy_match", "entropy_nonmatch", "keff_match",
       "keff_nonmatch", "entropy_all", "mi_attn_label", "total_entropy_avgmatrix"]


def r4(x):
    if isinstance(x, float):
        return float(f"{x:.4g}") if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r4(v) for v in x]
    return x


def ms(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    return [float(v.mean()), float(v.std() / np.sqrt(len(v))) if len(v) > 1 else 0.0, int(len(v))]


def solution(b):
    m, nm = b.get("L0H0_target_att_match"), b.get("L0H0_target_att_nonmatch")
    if m is None or nm is None:
        return None
    return "match" if m > nm else "nonmatch"


def toy_rows(parts):
    rows = []
    for p in sorted(glob.glob(os.path.join(parts, "*.jsonl"))):
        name = os.path.basename(p)
        if re.match(r"(l\d|l6_)", name):
            continue
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarise_toy(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["name"], r["config"]["n"])].append(r)
    out = []
    for (name, n), rs in sorted(g.items()):
        d = {"tag": name, "n": n, "runs": len(rs), "steps": ms([r["steps"] for r in rs]),
             "n_params": rs[0].get("n_params"), "length": rs[0]["config"].get("length", 24)}
        for split in ("best_natural", "best_lure"):
            for k in BEH:
                v = ms([r[split].get(k) for r in rs])
                if v:
                    d[f"{split.split('_')[1]}_{k}"] = v
        for k in ATT:
            v = ms([r["best_natural"].get(f"L0H0_{k}") for r in rs])
            if v:
                d[f"att_{k}"] = v
        sols = [solution(r["best_natural"]) for r in rs]
        sols = [s for s in sols if s]
        if sols:
            d["share_attend_on_match"] = sols.count("match") / len(sols)
            for s in ("match", "nonmatch"):
                sub = [r for r in rs if solution(r["best_natural"]) == s]
                if sub:
                    d[f"dprime_{s}_solution"] = ms([r["best_natural"]["dprime"] for r in sub])
        out.append(d)
    return out


def e1_seeds(rows):
    out = []
    for r in rows:
        if r["name"] != "e1":
            continue
        b, l = r["best_natural"], r["best_lure"]
        out.append({"n": r["config"]["n"], "seed": r["seed"], "steps": r["steps"], "best_step": r["best_step"],
                    "acc": b["acc"], "dprime": b["dprime"], "hit": b["hit"], "fa": b["fa"], "auc": b["auc"],
                    "solution": solution(b), "tam": b.get("L0H0_target_att_match"),
                    "tanm": b.get("L0H0_target_att_nonmatch"), "hm": b.get("L0H0_entropy_match"),
                    "hnm": b.get("L0H0_entropy_nonmatch"), "mi": b.get("L0H0_mi_attn_label"),
                    "tot_h": b.get("L0H0_total_entropy_avgmatrix"), "lure_dprime": l["dprime"],
                    "lure_fa_nonlure": l.get("fa_nonlure"),
                    **{f"lure_fa_{k.split('_')[-1]}": v for k, v in l.items() if k.startswith("fa_lure_")}})
    return out


def e1_curves(rows):
    by_n = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["name"] != "e1":
            continue
        c = r["curves"]
        for s, v in zip(c["step"], c["test_dprime"]):
            by_n[r["config"]["n"]][s].append(v)
    out = {}
    for n, d in by_n.items():
        steps = sorted(d)
        keep = [s for s in steps if len(d[s]) >= 10]
        idx = np.unique(np.linspace(0, len(keep) - 1, min(40, len(keep))).round().astype(int)) if keep else []
        out[n] = [[keep[i], float(np.mean(d[keep[i]])), len(d[keep[i]])] for i in idx]
    return out


def read_jsonl(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def l1(parts):
    out = {}
    for p in sorted(glob.glob(os.path.join(parts, "l1_*.json"))):
        try:
            d = json.load(open(p))
        except Exception as e:  # the file holds printed JSON; tolerate stray log lines
            txt = open(p).read()
            d = json.loads(txt[txt.index("{"):])
        out[d["model"]] = [k for k, v in d["tokens"].items() if v != 1]
    return out


def l6_heads(rows):
    out = []
    for r in rows:
        nb = np.array(r["nback_score"]); pt = np.array(r["prev_token"]); ind = np.array(r["induction"])
        top = np.dstack(np.unravel_index(np.argsort(-nb, axis=None), nb.shape))[0][:5]
        best_patch = sorted(r["patch_top"].items(), key=lambda kv: -kv[1])[:5]
        out.append({"model": r["model"], "revision": r["revision"], "n": r["n"],
                    "ld_clean": r["logit_diff_clean"], "ld_corrupt": r["logit_diff_corrupt"],
                    "top_nback": [[int(i), int(j), float(nb[i, j])] for i, j in top],
                    "max_prev_token": float(pt.max()), "max_induction": float(ind.max()),
                    "top_patch": [[h, v] for h, v in best_patch]})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", default="results/parts")
    p.add_argument("--out", default="results/summary.json")
    a = p.parse_args()
    rows = toy_rows(a.parts)
    l2 = read_jsonl(os.path.join(a.parts, "l2_*.jsonl"))
    for r in l2:
        r.pop("kback_fit_raw", None)
    summary = {"toy": summarise_toy(rows), "e1_seeds": e1_seeds(rows), "e1_curves": e1_curves(rows),
               "l1": l1(a.parts), "l2": l2,
               "l6_behave": read_jsonl(os.path.join(a.parts, "l6_behave_*.jsonl")),
               "l6_heads": l6_heads(read_jsonl(os.path.join(a.parts, "l6_heads_*.jsonl")))}
    s = json.dumps(r4(summary), separators=(",", ":"))
    open(a.out, "w").write(s)
    print(f"toy rows {len(rows)}, summary {len(s) / 1e3:.0f} kB -> {a.out}")


if __name__ == "__main__":
    main()
