"""Lure-position probe: does the model still know *where* the matching letter was when it
false-alarms on a lure?

On a match trial the same letter sits at i - N; on a lure trial it sits at i - N +- 1 (or another
lure offset). Both trials contain a same-letter item, so content alone cannot separate them; only
position can. For every residual-stream site (the query token, each layer's output) and for the
concatenated outputs of the top L3 heads, a logistic probe is trained to classify match vs lure
items, with 5-fold cross-validation over sequences. The probe's held-out predictions are then
split by what the model did:
  lure false alarms   the model said "match" on a lure
  lure correct rejects
If the probe still labels most lure false alarms as lures (from some layer on), the position
information was present and the model failed to use it: a readout failure. If position is not
decodable for those items, the representation itself lost it.

Usage: python -m ap.llm.probe_lure --model gpt2 --l3 results/l3.jsonl --l2 results/l2.jsonl --ns 2 3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ap.llm.behave import make_sets
from ap.llm.hooks import control, load
from ap.llm.locate import l2_threshold, top_heads
from ap.llm.prompts import build
from ap.llm.suppress import _site_states


@torch.no_grad()
def collect(model, tok, n, x, t, demos, heads, device, sites, batch=16):
    """Scores (B, L); per site the query-token state (B, L, d); head outputs (B, L, k*dh)."""
    layers = sorted({l for l, _ in heads})
    store, handles = _site_states(model)
    sc, feats, outs = [], {s: [] for s in sites}, []
    try:
        for s in range(0, len(x), batch):
            built = [build(tok, n, x[b], t[b], demos) for b in range(s, min(s + batch, len(x)))]
            ids = torch.tensor([b.ids for b in built], device=device)
            lp = torch.tensor(built[0].letter_pos, device=device)
            with control(record_layers=layers, cache_outputs=True) as c:
                logits = model(ids).logits.float()
            b0 = built[0]
            sc.append((logits[:, lp, b0.m_id] - logits[:, lp, b0.dash_id]).cpu().numpy())
            for site in sites:
                feats[site].append(store[site][:, lp].float().cpu().numpy().astype(np.float16))
            outs.append(torch.cat([c.outputs[l][:, lp, h].float().cpu() for l, h in heads], -1).numpy())
    finally:
        for h in handles:
            h.remove()
    return np.concatenate(sc), {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(outs)


def cv_probe(X, y, groups, device, folds=5, epochs=200, wd=1e-2, seed=0):
    """Held-out logits from a logistic probe, folds split by sequence (groups)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); rng.shuffle(ug)
    fold_of = {g: i % folds for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups])
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    out = np.zeros(len(y), dtype=np.float32)
    for k in range(folds):
        tr, te = torch.tensor(f != k, device=device), torch.tensor(f == k, device=device)
        mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-4
        A, B = (Xt[tr] - mu) / sd, (Xt[te] - mu) / sd
        w = torch.zeros(X.shape[1], device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.02, weight_decay=wd)
        for _ in range(epochs):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt[tr])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            out[f == k] = (B @ w + b).cpu().numpy()
    return out


def auc(s, y):
    from ap.metrics import auc as _auc
    return float(_auc(s, y))


def analyse(n, t, l, sc, thr, feats, outs, device):
    T, Lu = t[:, n:].astype(bool), l[:, n:] > 0
    P = sc[:, n:] > thr
    keep = T | (Lu & ~T)                                   # match items and lure items
    idx = np.argwhere(keep)
    y = T[keep].astype(np.float32)                         # 1 = match, 0 = lure
    groups = idx[:, 0]
    said = P[keep]
    lure_fa, lure_cr = (y == 0) & said, (y == 0) & ~said
    hit, miss = (y == 1) & said, (y == 1) & ~said
    res = {"n_match": int(y.sum()), "n_lure": int((y == 0).sum()), "n_lure_fa": int(lure_fa.sum()),
           "model_auc": auc(sc[:, n:][keep], y), "sites": {}}
    sources = {f"site{k}": v[:, n:][keep].astype(np.float32) for k, v in feats.items()}
    sources["heads"] = outs[:, n:][keep].astype(np.float32)
    for name, X in sources.items():
        z = cv_probe(X, y, groups, device)
        pred_match = z > 0
        res["sites"][name] = {
            "auc": auc(z, y),
            "lure_fa_called_lure": float((~pred_match[lure_fa]).mean()) if lure_fa.any() else None,
            "lure_cr_called_lure": float((~pred_match[lure_cr]).mean()) if lure_cr.any() else None,
            "hit_called_match": float(pred_match[hit].mean()) if hit.any() else None,
            "miss_called_match": float(pred_match[miss].mean()) if miss.any() else None,
        }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--l3", required=True)
    p.add_argument("--l2", required=True)
    p.add_argument("--ns", type=int, nargs="+", default=[2, 3])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-seq", type=int, default=1000)
    p.add_argument("--n-sites", type=int, default=9, help="residual sites probed, evenly spaced over depth")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--out", default="results/l8.jsonl")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(a.model, dtype=getattr(torch, a.dtype), device=device)
    nl = model.config.num_hidden_layers
    sites = sorted({int(round(v)) for v in np.linspace(0, nl, a.n_sites)})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for n in a.ns:
        heads = top_heads(a.l3, a.model, n, a.k)
        demos, _, (x, t, l) = make_sets(n, a.n_seq, 200, 3)     # same test set as L2
        sc, feats, outs = collect(model, tok, n, x, t, demos, heads, device, sites, batch=a.batch)
        res = analyse(n, t, l, sc, l2_threshold(a.l2, a.model, n), feats, outs, device)
        row = {"model": a.model, "n": n, "heads": heads, "n_layers": nl, **res}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        best = max(res["sites"].items(), key=lambda kv: kv[1]["auc"])
        print(a.model, n, f"model AUC={res['model_auc']:.3f} best probe {best[0]} AUC={best[1]['auc']:.3f} "
              f"lureFA called lure={best[1]['lure_fa_called_lure']}", flush=True)


if __name__ == "__main__":
    main()
