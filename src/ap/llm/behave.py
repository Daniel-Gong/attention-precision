"""L1/L2: N-back behaviour of pretrained LLMs.

For each model and N, scores 1,000 lure-controlled sequences (generator v2, 4 planted lures
per sequence) after 3 demonstrations. Reports, on positions with an N-back letter:
accuracy, hits, false alarms, d', AUC, false alarms by lure offset, at a threshold
calibrated on 200 held-out sequences; plus the task-drift diagnostic: how well the model's
answers fit the k-back rule for every k (Hu & Lewis 2025).

Usage (Misha GPU node):
  python -m ap.llm.behave --model EleutherAI/pythia-410m --ns 1 2 3 4 5 6 --out results/l2.jsonl
  python -m ap.llm.behave --model gpt2 --report-tokenization        # L1 check only
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ap.data import GenConfig, generate
from ap.llm.hooks import load
from ap.llm.prompts import build, k_back_labels, tokenization_report
from ap.metrics import auc, behaviour


def make_sets(n, n_seq, n_cal, n_demo, length=24, seed=0):
    rng = np.random.default_rng(20_000 + 31 * n + seed)
    demos = [generate(GenConfig(n, length, n_lure=2, clean=True), 1, rng) for _ in range(n_demo)]
    demos = [(d[0][0], d[1][0]) for d in demos]
    cal = generate(GenConfig(n, length, n_lure=4, clean=True), n_cal, rng)
    test = generate(GenConfig(n, length, n_lure=4, clean=True), n_seq, rng)
    return demos, cal, test


@torch.no_grad()
def score(model, tok, n, x, t, demos, condition="feedback", batch=32, device="cpu"):
    """Returns scores (B, L): logit(m) - logit(-) at each item's letter token."""
    out = np.zeros(x.shape, dtype=np.float32)
    for s in range(0, len(x), batch):
        xs, ts = x[s:s + batch], t[s:s + batch]
        answers = ts.copy() if condition == "feedback" else np.zeros_like(ts)
        steps = [None] if condition == "feedback" else range(x.shape[1])
        for step in steps:
            built = [build(tok, n, xs[b], answers[b], demos) for b in range(len(xs))]
            ids = torch.tensor([b.ids for b in built], device=device)
            logits = model(ids).logits.float()
            pos = torch.tensor(built[0].letter_pos, device=device)
            sc = (logits[:, pos, built[0].m_id] - logits[:, pos, built[0].dash_id]).cpu().numpy()
            if step is None:
                out[s:s + batch] = sc
            else:
                out[s:s + batch, step] = sc[:, step]
                answers[:, step] = sc[:, step] > 0       # the model's own answer feeds forward
    return out


def calibrate(sc, t, n):
    s, y = sc[:, n:].ravel(), t[:, n:].ravel().astype(bool)
    cands = np.quantile(s, np.linspace(0.01, 0.99, 197))
    accs = [((s > c) == y).mean() for c in cands]
    return float(cands[int(np.argmax(accs))])


def drift(sc, x, thr, n, kmax=6):
    pred = sc > thr
    start = kmax
    fits = {}
    for k in range(1, kmax + 1):
        lab = np.stack([k_back_labels(s, k) for s in x])
        fits[k] = float((pred[:, start:] == lab[:, start:]).mean())
    best = max(fits, key=fits.get)
    return {"kback_fit": fits, "best_k": best, "drifting": best != n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None, help="e.g. step143000 for Pythia checkpoints")
    p.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--n-seq", type=int, default=1000)
    p.add_argument("--n-cal", type=int, default=200)
    p.add_argument("--n-demo", type=int, default=3)
    p.add_argument("--conditions", nargs="+", default=["feedback", "own"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--out", default="results/l2.jsonl")
    p.add_argument("--save-scores", default="")
    p.add_argument("--report-tokenization", action="store_true")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(a.model, dtype=getattr(torch, a.dtype), device=device, revision=a.revision)
    if a.report_tokenization:
        print(json.dumps({"model": a.model, "tokens": tokenization_report(tok)}, indent=1))
        return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for n in a.ns:
        demos, (xc, tc, lc), (x, t, l) = make_sets(n, a.n_seq, a.n_cal, a.n_demo)
        for cond in a.conditions:
            t0 = time.time()
            scc = score(model, tok, n, xc, tc, demos, cond, a.batch, device)
            sc = score(model, tok, n, x, t, demos, cond, a.batch, device)
            thr = calibrate(scc, tc, n)
            row = {"model": a.model, "revision": a.revision, "n": n, "condition": cond, "threshold": thr,
                   **behaviour(sc, t, l, n, threshold=thr), "auc_raw": auc(sc[:, n:].ravel(), t[:, n:].ravel()),
                   **drift(sc, x, thr, n), "seconds": round(time.time() - t0, 1)}
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            if a.save_scores:
                os.makedirs(a.save_scores, exist_ok=True)
                tag = a.model.replace("/", "_") + (f"@{a.revision}" if a.revision else "")
                np.savez_compressed(os.path.join(a.save_scores, f"{tag}_n{n}_{cond}.npz"), score=sc, x=x, t=t, l=l)
            print(f"{a.model} N={n} {cond}: d'={row['dprime']:.2f} auc={row['auc']:.3f} hit={row['hit']:.3f} "
                  f"fa={row['fa']:.3f} best_k={row['best_k']} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
