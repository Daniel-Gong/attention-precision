"""L5 second arm: content suppression (does removing interfering item content help?).

The interference account (Xiong et al. 2026) predicts that N-back errors come from the content of
other items leaking into the target's readout, so removing that content should help. The precision
account predicts no gain once attention already lands on i - N (L7). This study tests it directly.

Letter subspace: for every residual-stream site (embedding output and each layer's output), the
20 letter class means of the hidden state at test-item letter tokens (calibration sequences) span
a 19-dimensional subspace U. Suppressing an item replaces the letter-specific part of the hidden
state at every token of its line (letter, answer, newline) by the average:
    h <- h - alpha * ((h - mu) U) U^T
at every site, so no layer can read that item's identity.

Each query i is scored on its own prefix (the model is causal), so the suppressed set can depend
on i. Conditions:
  baseline       nothing suppressed
  distractors    every earlier item except i - N (alpha = 1 and 0.5)
  lures          only i - N - 1 and i - N + 1
  random         every earlier item except i - N, in a random 19-dim subspace (control)
  target         only i - N (negative control: should destroy performance)
Thresholds are calibrated per condition on 200 held-out sequences, as in L2/L5.

Usage: python -m ap.llm.suppress --model gpt2 --ns 2 3 --out results/l5s.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ap.llm.behave import calibrate, make_sets
from ap.llm.hooks import load
from ap.llm.prompts import build
from ap.metrics import behaviour

N_LETTERS = 20


def decoder_layers(model):
    for path in ("model.layers", "gpt_neox.layers", "transformer.h"):
        m = model
        try:
            for p in path.split("."):
                m = getattr(m, p)
            return m
        except AttributeError:
            continue
    raise ValueError("cannot find decoder layers")


def _site_states(model):
    """Hooks recording every residual site exactly where Suppressor edits it (hidden_states from
    the model output would give the final-norm output at the last site instead)."""
    layers, store = decoder_layers(model), {}
    def pre(module, args, kwargs):
        store[0] = args[0] if args else kwargs["hidden_states"]
    def make(site):
        def post(module, args, output):
            store[site] = output[0] if isinstance(output, tuple) else output
        return post
    hs = [layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
    hs += [layer.register_forward_hook(make(i + 1)) for i, layer in enumerate(layers)]
    return store, hs


@torch.no_grad()
def fit_subspaces(model, tok, n, x, t, demos, device, batch=16):
    """Per site s (0 = input to layer 0, l + 1 = output of layer l): mean mu (d,) and basis U (d, 19)."""
    sums, counts = None, np.zeros(N_LETTERS)
    store, handles = _site_states(model)
    try:
        for s in range(0, len(x), batch):
            built = [build(tok, n, x[b], t[b], demos) for b in range(s, min(s + batch, len(x)))]
            ids = torch.tensor([b.ids for b in built], device=device)
            lp = torch.tensor(built[0].letter_pos, device=device)
            model(ids)
            hs = [store[k] for k in sorted(store)]
            lab = torch.as_tensor(x[s:s + len(built)], device=device).reshape(-1)      # (B * L,)
            if sums is None:
                sums = [torch.zeros(N_LETTERS, h.shape[-1], device=device, dtype=torch.float64) for h in hs]
            for k, h in enumerate(hs):
                sums[k].index_add_(0, lab, h[:, lp].reshape(-1, h.shape[-1]).double())
            counts += np.bincount(lab.cpu().numpy(), minlength=N_LETTERS)
    finally:
        for h in handles:
            h.remove()
    out = []
    c = torch.as_tensor(counts, device=device, dtype=torch.float64).clamp(min=1)[:, None]
    for S in sums:
        means = S / c
        mu = means.mean(0)
        U, _, _ = torch.linalg.svd((means - mu).T, full_matrices=False)  # (d, 20)
        out.append((mu.float(), U[:, : N_LETTERS - 1].float().contiguous()))
    return out


class Suppressor:
    """Forward hooks that remove the letter subspace at masked positions of every residual site."""
    def __init__(self, model, spaces, alpha=1.0, sites=None):
        self.layers, self.spaces, self.alpha, self.mask = decoder_layers(model), spaces, alpha, None
        self.sites = sites                     # None = every site
        self.handles = [self.layers[0].register_forward_pre_hook(self._pre, with_kwargs=True)]
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._make_post(li + 1)))

    def _edit(self, h, site):
        if self.mask is None or self.alpha == 0 or (self.sites is not None and site not in self.sites):
            return h
        mu, U = self.spaces[site]
        mu, U = mu.to(h.dtype), U.to(h.dtype)
        proj = ((h - mu) @ U) @ U.T
        m = self.mask[..., None].to(h.dtype)
        if m.dim() == 2:                       # (T, 1): same positions in every row
            m = m[: h.shape[1]]
        return h - self.alpha * proj * m

    def _pre(self, module, args, kwargs):
        if args:
            return (self._edit(args[0], 0), *args[1:]), kwargs
        kwargs["hidden_states"] = self._edit(kwargs["hidden_states"], 0)
        return args, kwargs

    def _make_post(self, site):
        def post(module, args, output):
            if isinstance(output, tuple):
                return (self._edit(output[0], site), *output[1:])
            return self._edit(output, site)
        return post

    def remove(self):
        for h in self.handles:
            h.remove()


def random_spaces(spaces, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for mu, U in spaces:
        Q, _ = torch.linalg.qr(torch.randn(U.shape[0], U.shape[1], generator=g))
        out.append((mu, Q.to(U)))
    return out


def suppressed_items(cond, i, n):
    if cond == "baseline":
        return []
    if cond in ("distractors", "random"):
        return [j for j in range(i) if j != i - n]
    if cond == "lures":
        return [j for j in (i - n - 1, i - n + 1) if 0 <= j < i]
    if cond == "target":
        return [i - n]
    raise ValueError(cond)


@torch.no_grad()
def score_cond(model, tok, n, x, t, demos, device, sup, cond, batch=32):
    """Scores (B, L): each query i >= n on its own prefix, with that query's items suppressed."""
    out = np.zeros(x.shape, dtype=np.float32)
    for s in range(0, len(x), batch):
        built = [build(tok, n, x[b], t[b], demos) for b in range(s, min(s + batch, len(x)))]
        ids = torch.tensor([b.ids for b in built], device=device)
        lp = built[0].letter_pos
        span = lp[1] - lp[0]
        for i in range(n, len(lp)):
            pre = ids[:, : lp[i] + 1]
            m = torch.zeros(pre.shape[1], dtype=torch.bool, device=device)
            for j in suppressed_items(cond, i, n):
                m[lp[j]: lp[j] + span] = True
            sup.mask = m[None].expand(pre.shape[0], -1) if m.any() else None
            logits = model(pre).logits[:, -1].float()
            out[s:s + len(built), i] = (logits[:, built[0].m_id] - logits[:, built[0].dash_id]).cpu().numpy()
    sup.mask = None
    return out


def xiong_sweep(model, tok, n, cal, test, demos, device, spaces, batch, n_dirs=5, alphas=(0.3, 0.5, 1.0)):
    """Replication of Xiong et al. (2026, App. A.5.13): one letter-identity principal direction at
    one of the two earliest depths, applied only at the positions that produce answers (here each
    test item's letter token), h <- h - alpha (proj_B(h) - mu_proj). One full pass per sequence, so
    earlier edited positions stay in context, as in their turn-by-turn evaluation. Unlike their
    best-in-sweep summary, the configuration is also chosen on calibration sequences and scored on
    held-out test sequences."""
    from ap.llm.behave import score
    nl = len(decoder_layers(model))
    depths = sorted({max(1, round(0.1 * nl)), max(2, round(0.25 * nl))})   # sites = outputs of these layers
    (xc, tc, lc), (x, t, l) = cal, test
    lp = build(tok, n, x[0], t[0], demos).letter_pos
    rows = []
    for site in depths:
        mu, U = spaces[site]
        for k in range(n_dirs):
            one = [None] * len(spaces)
            one[site] = (mu, U[:, k:k + 1].contiguous())
            for a in alphas:
                sup = Suppressor(model, one, a, sites={site})
                T = len(build(tok, n, x[0], t[0], demos).ids)
                m = torch.zeros(T, dtype=torch.bool, device=device)
                m[torch.as_tensor(lp, device=device)] = True
                sup.mask = m
                try:
                    scc = score(model, tok, n, xc, tc, demos, "feedback", batch, device)
                    sc = score(model, tok, n, x, t, demos, "feedback", batch, device)
                finally:
                    sup.remove()
                thr = calibrate(scc, tc, n)
                rows.append({"cond": "xiong", "site": site, "dir": k, "alpha": a,
                             "cal_dprime": behaviour(scc, tc, lc, n, threshold=thr)["dprime"],
                             **behaviour(sc, t, l, n, threshold=thr)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ns", type=int, nargs="+", default=[2, 3])
    p.add_argument("--n-seq", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--conds", nargs="+", default=["baseline", "distractors:1", "distractors:0.5", "lures:1",
                                                    "random:1", "target:1"])
    p.add_argument("--out", default="results/l5s.jsonl")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(a.model, dtype=getattr(torch, a.dtype), device=device)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for n in a.ns:
        demos, (xc, tc, lc), (x, t, l) = make_sets(n, a.n_seq, 200, 3)
        spaces = fit_subspaces(model, tok, n, xc, tc, demos, device)
        if "xiong" in a.conds:
            from ap.llm.behave import score
            base_c = score(model, tok, n, xc, tc, demos, "feedback", a.batch, device)
            base_t = score(model, tok, n, x, t, demos, "feedback", a.batch, device)
            thr = calibrate(base_c, tc, n)
            b = behaviour(base_t, t, l, n, threshold=thr)
            bc = behaviour(base_c, tc, lc, n, threshold=thr)["dprime"]
            rows = xiong_sweep(model, tok, n, (xc, tc, lc), (x, t, l), demos, device, spaces, a.batch)
            pick = max(rows, key=lambda r: r["cal_dprime"])
            summ = [{"cond": "xiong_baseline", "cal_dprime": bc, **b},
                    {"cond": "xiong_heldout", **{k: pick[k] for k in ("site", "dir", "alpha", "cal_dprime")},
                     **{k: v for k, v in pick.items() if k not in ("cond", "site", "dir", "alpha", "cal_dprime")}},
                    {"cond": "xiong_best_in_sweep", **max(rows, key=lambda r: r["dprime"]), "note": "selected on test"}]
            summ[2]["cond"] = "xiong_best_in_sweep"
            with open(a.out, "a") as f:
                for r in rows + summ:
                    f.write(json.dumps({"model": a.model, "n": n, **r}) + "\n")
            print(f"{a.model} N={n} xiong: base d'={b['dprime']:.3f} heldout={summ[1]['dprime']:.3f} "
                  f"best-in-sweep={summ[2]['dprime']:.3f}", flush=True)
        for spec in a.conds:
            if spec == "xiong":
                continue
            cond, _, alpha = spec.partition(":")
            alpha = float(alpha or 1.0)
            sup = Suppressor(model, random_spaces(spaces) if cond == "random" else spaces, alpha)
            t0 = time.time()
            try:
                scc = score_cond(model, tok, n, xc, tc, demos, device, sup, cond, a.batch)
                sc = score_cond(model, tok, n, x, t, demos, device, sup, cond, a.batch)
            finally:
                sup.remove()
            row = {"model": a.model, "n": n, "cond": cond, "alpha": alpha,
                   **behaviour(sc, t, l, n, threshold=calibrate(scc, tc, n)), "seconds": round(time.time() - t0, 1)}
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{a.model} N={n} {cond}:{alpha} d'={row['dprime']:.3f} hit={row['hit']:.3f} fa={row['fa']:.3f} "
                  f"({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
