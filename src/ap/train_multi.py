"""E10: one model, every N. The task's N is given by a cue token at the start of the sequence.

A model trained on a single N can hard-wire the look-back gap into its position weights.
Here the gap must be read from the cue and applied at run time, as a pretrained LLM must do
with an instruction. The question is whether that makes retrieval positionally imprecise,
which would show up as false alarms on near-target lures (H1), as in LLMs.

Input: [cue_N, x_1, ..., x_L] with cue token id = 20 + N - 1 (vocabulary 26). Loss and
metrics use positions 1..L only. One layer cannot use the cue to choose where to attend
(its query sees only token and position), so the default model has two layers with a
residual stream.

Usage: python -m ap.train_multi configs/e10.yaml [--set key=value ...] --seeds 0-4 --out results/e10.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import yaml

from ap.data import GenConfig, generate
from ap.metrics import attention_stats, behaviour
from ap.models.attn import AttnControl, AttnModel, ModelConfig, n_params
from ap.train import _set, git_hash, parse_seeds

CUE0 = 20


@dataclass
class MultiConfig:
    name: str = "e10"
    ns: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    length: int = 24
    model: dict = field(default_factory=dict)
    max_steps: int = 60000
    patience: int = 4000
    eval_every: int = 500
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.0
    n_val: int = 300          # per N
    n_test: int = 1000        # per N, per test set
    threads: int = 2
    save_ckpt: str = ""


def with_cue(x, n):
    return np.concatenate([np.full((len(x), 1), CUE0 + n - 1), x], 1)


def batch(ns, length, size, rng):
    xs, ts = [], []
    for n in rng.choice(ns, size=size):
        x, t, _ = generate(GenConfig(int(n), length, clean=False), 1, rng)
        xs.append(with_cue(x, int(n))[0]); ts.append(t[0])
    return np.stack(xs), np.stack(ts)


def eval_sets(cfg):
    out = {}
    for n in cfg.ns:
        erng = np.random.default_rng(2_000_000 + 97 * n + cfg.length)
        val = generate(GenConfig(n, cfg.length, clean=False), cfg.n_val, erng)
        nat = generate(GenConfig(n, cfg.length, clean=False), cfg.n_test, erng)
        lure = generate(GenConfig(n, cfg.length, n_lure=4, clean=True), cfg.n_test, erng)
        out[n] = {"val": val, "natural": nat, "lure": lure}
    return out


@torch.no_grad()
def evaluate(model, x, t, l, n, full=False):
    model.eval()
    ctl = AttnControl(record=True) if full else None
    logits = model(torch.as_tensor(with_cue(x, n)), ctl)[:, 1:]
    loss = nn.functional.cross_entropy(logits.reshape(-1, 2), torch.as_tensor(t).reshape(-1)).item()
    score = (logits[..., 1] - logits[..., 0]).numpy()
    out = {"loss": loss, **behaviour(score, t, l, n)}
    if full:
        for li, A in enumerate(ctl.store["attn"]):
            A = A[:, :, 1:, 1:].numpy()           # drop the cue row and column (rows no longer sum to 1)
            for h in range(A.shape[1]):
                for k, v in attention_stats(A[:, h], t, n).items():
                    out[f"L{li}H{h}_{k}"] = v
            out[f"L{li}_cue_attention"] = float(ctl.store["attn"][li][:, :, 1 + n:, 0].mean())
    model.train()
    return out


def run(cfg: MultiConfig, seed: int) -> dict:
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(30_000 + seed)
    mdl = {"n_layers": 2, "residual": True, **cfg.model}
    mcfg = ModelConfig(**{"vocab": CUE0 + max(cfg.ns), "max_len": cfg.length + 1, **mdl})
    model = AttnModel(mcfg)
    opt = (torch.optim.AdamW if cfg.weight_decay > 0 else torch.optim.Adam)(
        model.parameters(), lr=cfg.lr, **({"weight_decay": cfg.weight_decay} if cfg.weight_decay > 0 else {}))
    ce = nn.CrossEntropyLoss()
    sets = eval_sets(cfg)
    best = {"loss": float("inf"), "step": 0, "state": None}
    curves = {"step": [], "train_loss": [], "val_loss": [], **{f"val_dprime_n{n}": [] for n in cfg.ns}}
    running, nrun, step, t0 = 0.0, 0, 0, time.time()
    while step < cfg.max_steps:
        x, t = batch(cfg.ns, cfg.length, cfg.batch_size, rng)
        logits = model(torch.as_tensor(x))[:, 1:]
        loss = ce(logits.reshape(-1, 2), torch.as_tensor(t).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item(); nrun += 1; step += 1
        if step % cfg.eval_every == 0:
            vs = {n: evaluate(model, *sets[n]["val"], n) for n in cfg.ns}
            vloss = float(np.mean([v["loss"] for v in vs.values()]))
            curves["step"].append(step); curves["train_loss"].append(round(running / nrun, 5))
            curves["val_loss"].append(round(vloss, 5))
            for n in cfg.ns:
                curves[f"val_dprime_n{n}"].append(round(vs[n]["dprime"], 4))
            running, nrun = 0.0, 0
            if vloss < best["loss"] - 1e-5:
                best = {"loss": vloss, "step": step, "state": copy.deepcopy(model.state_dict())}
            elif step - best["step"] >= cfg.patience:
                break
    model.load_state_dict(best["state"] or model.state_dict())
    row = {"name": cfg.name, "seed": seed, "git": git_hash(), "config": asdict(cfg), "model": mcfg.to_dict(),
           "n_params": n_params(model), "steps": step, "best_step": best["step"],
           "seconds": round(time.time() - t0, 1), "curves": curves, "per_n": {}}
    for n in cfg.ns:
        row["per_n"][n] = {"natural": evaluate(model, *sets[n]["natural"], n, full=True),
                           "lure": evaluate(model, *sets[n]["lure"], n, full=True)}
    if cfg.save_ckpt:
        os.makedirs(cfg.save_ckpt, exist_ok=True)
        torch.save({"model": mcfg.to_dict(), "state": model.state_dict()},
                   os.path.join(cfg.save_ckpt, f"{cfg.name}_s{seed}.pt"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    raw = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, _, v = kv.partition("=")
        _set(raw, k, v)
    cfg = MultiConfig(**raw)
    out = a.out or f"results/{cfg.name}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for s in parse_seeds(a.seeds):
        row = run(cfg, s)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        msg = " ".join(f"N{n}:d'={row['per_n'][n]['natural']['dprime']:.2f}" for n in cfg.ns)
        print(f"{cfg.name} seed={s} steps={row['steps']} {msg} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
