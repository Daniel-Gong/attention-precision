"""B1 + X1: sequence-level bootstrap confidence intervals for the main LLM interventions.

For one model and each N, scores the same calibration (200) and test (1,000) sequences under:
  baseline
  sharpen-top3@0.8, sharpen-top3@0.67   (top-3 L3 heads, logits x 1/tau)
  ablate-top10, ablate-random10          (mean ablation; random heads matched by layer, one draw)
  suppress@0.5, suppress@1               (distractor letter-subspace removal, ap.llm.suppress)
  suppress-random@1                      (random subspace control)
Head conditions are skipped when the model has no L3 results (e.g. a new, larger model).
Each condition's threshold is calibrated on the calibration sequences; then 2,000 bootstrap
resamples of the 1,000 test sequences (the same resample for every condition, so differences
are paired) give 95% intervals for d' and for each condition's change from baseline.
Per-sequence scores are saved as .npz so figures can recompute anything.

Usage: python -m ap.llm.boot --model gpt2 --l3 results/l3.jsonl --ns 2 3 4 --out results/b1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import norm

from ap.llm.behave import calibrate, make_sets, score
from ap.llm.hooks import control, load, n_layers_heads
from ap.llm import intervene as I
from ap.llm import suppress as S


def dprime_counts(h, ns, f, nn):
    return norm.ppf((h + 0.5) / (ns + 1)) - norm.ppf((f + 0.5) / (nn + 1))


def boot_dprime(sc, t, thr, n, idx):
    """d' for each bootstrap resample. sc, t: (B, L); idx: (R, B) sequence indices."""
    P = (sc[:, n:] > thr); T = t[:, n:].astype(bool)
    hit_s, sig_s = (P & T).sum(1), T.sum(1)
    fa_s, noi_s = (P & ~T).sum(1), (~T).sum(1)
    return dprime_counts(hit_s[idx].sum(1), sig_s[idx].sum(1), fa_s[idx].sum(1), noi_s[idx].sum(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--l3", default="")
    p.add_argument("--ns", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--n-seq", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--out", default="results/b1.jsonl")
    p.add_argument("--scores-dir", default="")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(a.model, dtype=getattr(torch, a.dtype), device=device)
    nl, nh = n_layers_heads(model)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for n in a.ns:
        demos, cal, test = make_sets(n, a.n_seq, 200, 3)
        (xc, tc, lc), (x, t, l) = cal, test
        conds = {}

        def full(name, ctl):
            with control(**ctl):
                conds[name] = (score(model, tok, n, xc, tc, demos, "feedback", a.batch, device),
                               score(model, tok, n, x, t, demos, "feedback", a.batch, device))
            print(a.model, n, name, flush=True)

        full("baseline", {})
        try:
            top3 = I.top_heads(a.l3, a.model, n, 3) if a.l3 else None
            top10 = I.top_heads(a.l3, a.model, n, 10) if a.l3 else None
        except (IndexError, FileNotFoundError):
            top3 = top10 = None
        if top3:
            for tau in (0.8, 0.67):
                full(f"sharpen-top3@{tau}", {"temps": {h: tau for h in top3}})
            means = I.mean_head_outputs(model, tok, n, xc, tc, demos, device)
            full("ablate-top10", {"ablate": {h: means[h[0], h[1]] for h in top10}})
            try:
                rnd = I.matched_random(top10, nh, np.random.default_rng(100 + n * 10 + 10), set(top10))
                full("ablate-random10", {"ablate": {h: means[h[0], h[1]] for h in rnd}})
            except ValueError:                       # a layer has no spare heads to draw from
                pass
        spaces = S.fit_subspaces(model, tok, n, xc, tc, demos, device)
        for name, sp, alpha in (("suppress@0.5", spaces, 0.5), ("suppress@1", spaces, 1.0),
                                ("suppress-random@1", S.random_spaces(spaces), 1.0)):
            cond = "random" if "random" in name else "distractors"
            sup = S.Suppressor(model, sp, alpha)
            try:
                conds[name] = (S.score_cond(model, tok, n, xc, tc, demos, device, sup, cond, a.batch),
                               S.score_cond(model, tok, n, x, t, demos, device, sup, cond, a.batch))
            finally:
                sup.remove()
            print(a.model, n, name, flush=True)

        idx = np.random.default_rng(7).integers(0, len(x), size=(a.reps, len(x)))
        thr = {k: calibrate(c, tc, n) for k, (c, _) in conds.items()}
        bd = {k: boot_dprime(s, t, thr[k], n, idx) for k, (_, s) in conds.items()}
        full_idx = np.arange(len(x))[None]
        point = {k: float(boot_dprime(s, t, thr[k], n, full_idx)[0]) for k, (_, s) in conds.items()}
        with open(a.out, "a") as f:
            for k in conds:
                d = bd[k] - bd["baseline"]
                f.write(json.dumps({"model": a.model, "n": n, "cond": k, "dprime": point[k],
                                    "ci": [float(np.percentile(bd[k], 2.5)), float(np.percentile(bd[k], 97.5))],
                                    "delta": point[k] - point["baseline"],
                                    "delta_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                    "p_le0": float((d <= 0).mean()), "reps": a.reps}) + "\n")
        if a.scores_dir:
            os.makedirs(a.scores_dir, exist_ok=True)
            np.savez_compressed(os.path.join(a.scores_dir, f"{a.model.replace('/', '_')}_n{n}.npz"),
                                x=x, t=t, l=l, xc=xc, tc=tc,
                                **{f"{k}__test": s for k, (_, s) in conds.items()},
                                **{f"{k}__cal": c for k, (c, _) in conds.items()})
        print(a.model, n, " ".join(f"{k}={point[k]:.2f}" for k in conds), flush=True)


if __name__ == "__main__":
    main()
