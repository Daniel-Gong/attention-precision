"""L5: causal interventions on the heads found in L3.

Arms (each scored with ap.llm.behave on the same 1,000 lure-controlled sequences):
  sharpen-top    the top-k heads by L3 patching effect, logits x 1/tau
  sharpen-random k random heads matched to the top heads' layers (5 draws)
  sharpen-all    every head, logits x 1/tau
  ablate-top     the top-k heads mean-ablated at every position (necessity)
  ablate-random  matched random heads mean-ablated
The content-suppression arm from Xiong et al. (2026) is added once their method is
implemented from the paper (see the plan's L5 second arm).

Usage: python -m ap.llm.intervene --model EleutherAI/pythia-410m --l3 results/l3.jsonl --ns 2 3 --out results/l5.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ap.llm.behave import calibrate, make_sets, score
from ap.llm.hooks import control, load, n_layers_heads
from ap.llm.prompts import build
from ap.metrics import behaviour

TAUS = [0.5, 0.67, 0.8, 1.25, 1.5, 2.0]


def top_heads(l3_path, model, n, k):
    rows = [json.loads(l) for l in open(l3_path)]
    r = [x for x in rows if x["model"] == model and x["n"] == n][-1]
    ranked = sorted(r["patch_top"].items(), key=lambda kv: -kv[1])[:k]
    return [tuple(map(int, h.split("."))) for h, _ in ranked]


def matched_random(heads, nh, rng, exclude):
    out = []
    for l, _ in heads:
        choices = [h for h in range(nh) if (l, h) not in exclude and (l, h) not in out]
        out.append((l, int(rng.choice(choices))))
    return out


@torch.no_grad()
def mean_head_outputs(model, tok, n, x, t, demos, device, batch=16):
    nl, _ = n_layers_heads(model)
    acc, cnt = None, 0
    for s in range(0, len(x), batch):
        built = [build(tok, n, x[b], t[b], demos) for b in range(s, min(s + batch, len(x)))]
        ids = torch.tensor([b.ids for b in built], device=device)
        lp = torch.tensor(built[0].letter_pos, device=device)
        with control(cache_outputs=True) as c:
            model(ids)
        m = torch.stack([c.outputs[l][:, lp].mean((0, 1)) for l in range(nl)])     # (layers, H, dh)
        acc = m if acc is None else acc + m
        cnt += 1
    return acc / cnt


def run_arm(model, tok, n, sets, demos, device, ctl_kwargs, batch):
    (xc, tc, lc), (x, t, l) = sets
    with control(**ctl_kwargs):
        scc = score(model, tok, n, xc, tc, demos, "feedback", batch, device)
        sc = score(model, tok, n, x, t, demos, "feedback", batch, device)
    thr = calibrate(scc, tc, n)
    return behaviour(sc, t, l, n, threshold=thr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--l3", required=True)
    p.add_argument("--ns", type=int, nargs="+", default=[2, 3])
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 10])
    p.add_argument("--random-draws", type=int, default=5)
    p.add_argument("--n-seq", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--out", default="results/l5.jsonl")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(a.model, dtype=getattr(torch, a.dtype), device=device)
    nl, nh = n_layers_heads(model)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def emit(row):
        with open(a.out, "a") as f:
            f.write(json.dumps({"model": a.model, **row}) + "\n")
        print(row["n"], row["arm"], row.get("k"), row.get("tau"), f"d'={row['dprime']:.3f}", flush=True)

    for n in a.ns:
        demos, cal, test = make_sets(n, a.n_seq, 200, 3)
        sets = (cal, test)
        emit({"n": n, "arm": "baseline", **run_arm(model, tok, n, sets, demos, device, {}, a.batch)})
        means = mean_head_outputs(model, tok, n, cal[0], cal[1], demos, device)
        for k in a.k:
            top = top_heads(a.l3, a.model, n, k)
            rng = np.random.default_rng(100 + n * 10 + k)
            rands = [matched_random(top, nh, rng, set(top)) for _ in range(a.random_draws)]
            for tau in TAUS:
                emit({"n": n, "arm": "sharpen-top", "k": k, "tau": tau, "heads": top,
                      **run_arm(model, tok, n, sets, demos, device, {"temps": {h: tau for h in top}}, a.batch)})
                for d, rh in enumerate(rands):
                    emit({"n": n, "arm": "sharpen-random", "k": k, "tau": tau, "draw": d, "heads": rh,
                          **run_arm(model, tok, n, sets, demos, device, {"temps": {h: tau for h in rh}}, a.batch)})
            emit({"n": n, "arm": "ablate-top", "k": k, "heads": top,
                  **run_arm(model, tok, n, sets, demos, device, {"ablate": {h: means[h[0], h[1]] for h in top}}, a.batch)})
            for d, rh in enumerate(rands):
                emit({"n": n, "arm": "ablate-random", "k": k, "draw": d, "heads": rh,
                      **run_arm(model, tok, n, sets, demos, device, {"ablate": {h: means[h[0], h[1]] for h in rh}}, a.batch)})
        for tau in TAUS:
            allh = {(l, h): tau for l in range(nl) for h in range(nh)}
            emit({"n": n, "arm": "sharpen-all", "tau": tau,
                  **run_arm(model, tok, n, sets, demos, device, {"temps": allh}, a.batch)})


if __name__ == "__main__":
    main()
