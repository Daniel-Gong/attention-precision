"""L7: does an LLM's N-back error come from looking at the wrong place, or from reading
the right place wrongly? (H8: precision versus interference.)

For each model and N, using the top-k N-back heads from L3 and the L2 test sequences
(feedback condition):
  1. Score every test item (logit m minus logit -), threshold at the L2 calibration value.
  2. At each item's letter token, record the top heads' attention to the letter at i - N,
     to the lure letters at i - N +- 1, and the heads' outputs.
  3. Locate test: an item is "located" when the heads give the whole line of item i - N
     (letter, answer, newline) at least as much attention as either neighbouring line
     (i - N - 1, i - N + 1). This does not depend on the heads' absolute attention level.
  4. Readout test: a logistic probe trained on correctly answered items predicts the
     match label from the concatenated head outputs; applied to error items it tells
     whether the heads' output still carried the right answer.
Each error is then classified:
  mislocated          attention not on i - N (precision failure)
  located, probe right  heads carried the answer, later layers lost it (downstream interference)
  located, probe wrong  heads attended correctly but their output was wrong (readout interference)

Usage: python -m ap.llm.locate --model gpt2 --l3 results/l3.jsonl --l2 results/l2.jsonl --ns 2 3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ap.llm.behave import make_sets
from ap.llm.hooks import control, load
from ap.llm.prompts import build


def top_heads(l3_path, model, n, k):
    rows = [json.loads(l) for l in open(l3_path) if l.strip()]
    r = [x for x in rows if x["model"] == model and x["n"] == n and x.get("revision") is None][-1]
    pt = {h: v for h, v in r["patch_top"].items() if v == v}            # drop NaN
    ranked = sorted(pt.items(), key=lambda kv: -kv[1])[:k]
    return [tuple(map(int, h.split("."))) for h, _ in ranked]


def l2_threshold(l2_path, model, n):
    rows = [json.loads(l) for l in open(l2_path) if l.strip()]
    r = [x for x in rows if x["model"] == model and x["n"] == n and x["condition"] == "feedback"][-1]
    return r["threshold"]


@torch.no_grad()
def collect(model, tok, n, x, t, demos, heads, device, batch=16):
    """Per item i >= n and per head: attention from item i's letter token to the whole line of
    item j (letter, answer and newline tokens) for j = i-n-1, i-n, i-n+1 (offsets -1, 0, +1).
    Returns scores (B, L), line attention (B, L-n, 3, k) and head outputs (B, L-n, k*dh)."""
    layers = sorted({l for l, _ in heads})
    scores, att, outs = [], [], []
    for s in range(0, len(x), batch):
        built = [build(tok, n, x[b], t[b], demos) for b in range(s, min(s + batch, len(x)))]
        ids = torch.tensor([b.ids for b in built], device=device)
        lp = np.array(built[0].letter_pos); span = int(lp[1] - lp[0])
        with control(record_layers=layers, cache_outputs=True) as c:
            logits = model(ids).logits.float()
        b0 = built[0]
        scores.append((logits[:, lp, b0.m_id] - logits[:, lp, b0.dash_id]).cpu().numpy())
        q = lp[n:]
        per_off = []
        for off in (-1, 0, 1):
            cols = []
            for i in range(n, len(lp)):
                j = i - n + off
                cols.append(np.arange(lp[j], lp[j] + span) if 0 <= j < i else None)
            hs = []
            for l, h in heads:
                A = c.attn[l][:, h]                                         # (B, T, T)
                hs.append(np.stack([A[:, lp[i], cols[i - n]].sum(-1).numpy() if cols[i - n] is not None
                                    else np.full(A.shape[0], np.nan) for i in range(n, len(lp))], 1))
            per_off.append(np.stack(hs, -1))                                # (B, L-n, k)
        att.append(np.stack(per_off, 2))                                    # (B, L-n, 3, k)
        outs.append(torch.cat([c.outputs[l][:, torch.as_tensor(q, device=device), h].float().cpu()
                               for l, h in heads], -1).numpy())
    return np.concatenate(scores), np.concatenate(att), np.concatenate(outs)


def probe(Xtr, ytr, Xte, epochs=300, wd=1e-3):
    Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=wd)
    y = torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (Xte @ w + b).numpy()


def analyse(n, x, t, l, sc, att, outs, thr):
    T = t[:, n:].astype(bool); P = sc[:, n:] > thr; Lu = l[:, n:] > 0
    A = np.nanmean(att, -1)                                   # (B, L-n, 3): offsets -1, 0, +1, mean over heads
    a_m1, a_t, a_p1 = A[..., 0], A[..., 1], A[..., 2]
    nb = np.fmax(np.nan_to_num(a_m1, nan=-1), np.nan_to_num(a_p1, nan=-1))
    located = a_t >= nb                                       # target line gets at least as much as either neighbour
    correct, err = P == T, P != T
    rng = np.random.default_rng(0)
    idx = np.argwhere(correct); rng.shuffle(idx)
    tr = idx[: min(len(idx), 4000)]
    X = outs.reshape(-1, outs.shape[-1]).astype(np.float32)
    flat = lambda ij: ij[:, 0] * T.shape[1] + ij[:, 1]
    ei = np.argwhere(err)
    pz = probe(X[flat(tr)], T[tr[:, 0], tr[:, 1]].astype(np.float32), X[flat(ei)]) if len(ei) else np.array([])
    probe_right = (pz > 0) == T[ei[:, 0], ei[:, 1]] if len(ei) else np.array([], bool)
    loc_e = located[ei[:, 0], ei[:, 1]] if len(ei) else np.array([], bool)
    cats = {"correct_match": correct & T, "correct_nonmatch": correct & ~T, "miss": T & err,
            "false_alarm": ~T & err, "lure_false_alarm": ~T & err & Lu, "lure_correct_reject": ~T & correct & Lu}
    line = lambda m: {k: (float(np.nanmean(v[m])) if m.any() else None) for k, v in
                      {"target": a_t, "minus1": a_m1, "plus1": a_p1}.items()}
    out = {"n_items": int(T.size), "n_errors": int(err.sum()),
           "line_attention": {k: line(m) for k, m in cats.items()},
           "located_rate": {k: (float(located[m].mean()) if m.any() else None) for k, m in cats.items()},
           "error_split": {"mislocated": float((~loc_e).mean()) if len(loc_e) else None,
                           "located_probe_right": float((loc_e & probe_right).mean()) if len(loc_e) else None,
                           "located_probe_wrong": float((loc_e & ~probe_right).mean()) if len(loc_e) else None}}
    for kind in ("miss", "false_alarm", "lure_false_alarm"):
        sel = cats[kind][ei[:, 0], ei[:, 1]] if len(ei) else np.array([], bool)
        if sel.any():
            out[f"split_{kind}"] = {"mislocated": float((~loc_e[sel]).mean()),
                                    "located_probe_right": float((loc_e[sel] & probe_right[sel]).mean()),
                                    "located_probe_wrong": float((loc_e[sel] & ~probe_right[sel]).mean()),
                                    "count": int(sel.sum())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--l3", required=True)
    p.add_argument("--l2", required=True)
    p.add_argument("--ns", type=int, nargs="+", default=[2, 3])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-seq", type=int, default=1000)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--out", default="results/l7.jsonl")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(a.model, dtype=getattr(torch, a.dtype), device=device)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for n in a.ns:
        heads = top_heads(a.l3, a.model, n, a.k)
        demos, _, (x, t, l) = make_sets(n, a.n_seq, 200, 3)     # same test set as L2
        sc, att, outs = collect(model, tok, n, x, t, demos, heads, device)
        res = analyse(n, x, t, l, sc, att, outs, l2_threshold(a.l2, a.model, n))
        row = {"model": a.model, "n": n, "heads": heads, **res}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(a.model, n, json.dumps(res["error_split"]), flush=True)


if __name__ == "__main__":
    main()
