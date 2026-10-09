"""L3: locate the heads that do N-back in a pretrained LLM.

Four per-head measures, all at the letter token of item i (the position that predicts
the answer), written to one JSONL row per (model, N):

  nback_score  mean attention from item i's letter to item (i - N)'s letter, i >= N
  lure_score   the same for the lure offsets N - 1 and N + 1
  prev_token   standard previous-token score on repeated random tokens
  induction    standard induction score on a random sequence repeated twice
  atp          attribution-patching estimate of each head's indirect effect on the
               match logit difference (Kramar et al. 2024), clean = match, corrupt = the
               letter at i - N replaced so the item no longer matches
  patch        exact activation-patching effect for the top heads by |atp| (--top)

Usage: python -m ap.llm.heads --model EleutherAI/pythia-410m --ns 1 2 3 --out results/l3.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ap.data import ALPHABET
from ap.llm.behave import make_sets
from ap.llm.hooks import control, load, n_layers_heads
from ap.data import lure_offsets
from ap.llm.prompts import build


def batch_ids(tok, n, x, t, demos, device):
    built = [build(tok, n, x[b], t[b], demos) for b in range(len(x))]
    return torch.tensor([b.ids for b in built], device=device), np.array(built[0].letter_pos), built[0]


@torch.no_grad()
def attention_scores(model, tok, n, x, t, demos, device, batch=16):
    nl, nh = n_layers_heads(model)
    nb = np.zeros((nl, nh)); lure = np.zeros((nl, nh)); cnt = 0
    offs = [d for d in lure_offsets(n) if d in (n - 1, n + 1)]
    for s in range(0, len(x), batch):
        ids, lp, _ = batch_ids(tok, n, x[s:s + batch], t[s:s + batch], demos, device)
        with control(record_layers=range(nl)) as c:
            model(ids)
        q = lp[n:]
        for l in range(nl):
            A = c.attn[l]                                            # (B, H, T, T)
            nb[l] += A[:, :, q, lp[:-n]].mean((0, 2)).numpy()
            lure[l] += np.mean([A[:, :, lp[i], lp[i - d]].mean(0).numpy()
                                for i in range(n, len(lp)) for d in offs if i - d >= 0], axis=0)
        cnt += 1
    return nb / cnt, lure / cnt


@torch.no_grad()
def standard_head_scores(model, device, vocab_lo=1000, length=50, reps=8, seed=0, vocab_span=5000):
    """Previous-token and induction scores on random tokens (Olsson et al. 2022 style)."""
    nl, nh = n_layers_heads(model)
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(vocab_lo, vocab_lo + vocab_span, (reps, length), generator=g)
    ids = torch.cat([r, r], 1).to(device)
    with control(record_layers=range(nl)) as c:
        model(ids)
    prev, ind = np.zeros((nl, nh)), np.zeros((nl, nh))
    L = ids.shape[1]
    idx = torch.arange(1, L)
    second = torch.arange(length, L)
    for l in range(nl):
        A = c.attn[l]
        prev[l] = A[:, :, idx, idx - 1].mean((0, 2)).numpy()
        ind[l] = A[:, :, second, second - length + 1].mean((0, 2)).numpy()
    return prev, ind


def counterfactual_pairs(x, t, n, rng):
    """For each sequence pick one match item i; corrupt the letter at i - N so it no longer
    matches (and creates no other match or lure at i)."""
    clean, corrupt, items = [], [], []
    for s, tt in zip(x, t):
        ms = np.flatnonzero(tt[n:]) + n
        if len(ms) == 0:
            continue
        i = int(rng.choice(ms))
        c = s.copy()
        bad = {s[i]} | {s[j] for j in (i - n - 1, i - n + 1) if 0 <= j < len(s)}
        choices = [k for k in range(len(ALPHABET)) if k not in bad]
        c[i - n] = int(rng.choice(choices))
        clean.append(s); corrupt.append(c); items.append(i)
    return np.array(clean), np.array(corrupt), np.array(items)


def logit_diff(logits, pos, item, b):
    rows = torch.arange(len(item), device=logits.device)
    q = torch.as_tensor(pos[item], device=logits.device)
    return logits[rows, q, b.m_id] - logits[rows, q, b.dash_id]


def attribution_patching(model, tok, n, clean, corrupt, items, demos, device, batch=8):
    """AtP: effect_h ~ <grad of corrupt logit-diff wrt head output, clean - corrupt output>
    at the query position, summed over the batch. Labels for the prompt answers: the clean
    answers in both runs, so only the letter at i - N differs."""
    nl, nh = n_layers_heads(model)
    eff = np.zeros((nl, nh)); base = []
    for s in range(0, len(clean), batch):
        sl = slice(s, s + batch)
        tc = np.zeros_like(clean[sl], dtype=np.int64)
        for k, (seq, it) in enumerate(zip(clean[sl], items[sl])):
            tc[k, n:] = (seq[n:] == seq[:-n]).astype(np.int64)
        ids_c, lp, b = batch_ids(tok, n, clean[sl], tc, demos, device)
        ids_x, _, _ = batch_ids(tok, n, corrupt[sl], tc, demos, device)
        q = torch.as_tensor(lp[items[sl]], device=device)
        with torch.no_grad(), control(cache_outputs=True) as cc:
            ld_c = logit_diff(model(ids_c).logits.float(), lp, items[sl], b)
        with torch.enable_grad(), control(cache_outputs=True, keep_grad=True) as cx:
            ld_x = logit_diff(model(ids_x).logits.float(), lp, items[sl], b)
            ld_x.sum().backward()
        rows = torch.arange(len(q), device=device)
        for l in range(nl):
            o_c, o_x = cc.outputs[l][rows, q], cx.outputs[l][rows, q]          # (B, H, dh)
            gx = cx.outputs[l].grad[rows, q]
            eff[l] += ((o_c - o_x.detach()) * gx).sum((0, 2)).float().cpu().numpy()
        base.append(torch.stack([ld_c, ld_x.detach()], 1).cpu().numpy())
        model.zero_grad(set_to_none=True)
    base = np.concatenate(base)
    return eff / len(clean), float(base[:, 0].mean()), float(base[:, 1].mean())


@torch.no_grad()
def exact_patching(model, tok, n, clean, corrupt, items, demos, device, heads, batch=8):
    """Patch each head's clean output into the corrupt run at the query position; report the
    recovered fraction of the clean - corrupt logit difference."""
    out = {h: [] for h in heads}
    for s in range(0, len(clean), batch):
        sl = slice(s, s + batch)
        tc = np.zeros_like(clean[sl], dtype=np.int64)
        for k, seq in enumerate(clean[sl]):
            tc[k, n:] = (seq[n:] == seq[:-n]).astype(np.int64)
        ids_c, lp, b = batch_ids(tok, n, clean[sl], tc, demos, device)
        ids_x, _, _ = batch_ids(tok, n, corrupt[sl], tc, demos, device)
        with control(cache_outputs=True) as cc:
            ld_c = logit_diff(model(ids_c).logits.float(), lp, items[sl], b)
        ld_x = logit_diff(model(ids_x).logits.float(), lp, items[sl], b)
        for (l, h) in heads:
            src = cc.outputs[l][:, :, h].clone()
            pos_mask = torch.zeros(ids_x.shape, dtype=torch.bool)          # (B, T): each row's own item
            pos_mask[torch.arange(len(items[sl])), torch.as_tensor(lp[items[sl]])] = True
            with control(patch={(l, h): src}, patch_positions=pos_mask):
                ld_p = logit_diff(model(ids_x).logits.float(), lp, items[sl], b)
            out[(l, h)].append(((ld_p - ld_x) / (ld_c - ld_x).clamp(min=1e-3)).cpu().numpy())
    return {f"{l}.{h}": float(np.concatenate(v).mean()) for (l, h), v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n-seq", type=int, default=200)
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--out", default="results/l3.jsonl")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(a.model, dtype=getattr(torch, a.dtype), device=device, revision=a.revision)
    prev, ind = standard_head_scores(model, device)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for n in a.ns:
        demos, _, (x, t, l) = make_sets(n, a.n_seq, 1, 3)
        nb, lure = attention_scores(model, tok, n, x, t, demos, device)
        rng = np.random.default_rng(7 + n)
        clean, corrupt, items = counterfactual_pairs(x[: a.n_pairs], t[: a.n_pairs], n, rng)
        atp, ld_c, ld_x = attribution_patching(model, tok, n, clean, corrupt, items, demos, device)
        order = np.dstack(np.unravel_index(np.argsort(-np.abs(atp), axis=None), atp.shape))[0][: a.top]
        heads = [(int(i), int(j)) for i, j in order]
        patch = exact_patching(model, tok, n, clean, corrupt, items, demos, device, heads)
        row = {"model": a.model, "revision": a.revision, "n": n, "logit_diff_clean": ld_c, "logit_diff_corrupt": ld_x,
               "nback_score": nb.round(5).tolist(), "lure_score": lure.round(5).tolist(),
               "prev_token": prev.round(5).tolist(), "induction": ind.round(5).tolist(),
               "atp": atp.round(6).tolist(), "patch_top": patch}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        best = max(patch, key=patch.get)
        print(f"{a.model} N={n}: clean-corrupt logit diff {ld_c:.2f} vs {ld_x:.2f}; top head {best} "
              f"recovers {patch[best]:.2f}", flush=True)


if __name__ == "__main__":
    main()
