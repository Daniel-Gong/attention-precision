"""Config-driven training for N-back models.

Two data regimes:
  fixed : a fixed training set iterated for `epochs` (workshop replication), or of
          size `train_size` (data-size sweep), with early stopping if `max_steps` set
  fresh : new sequences every step (default for sweeps), early stopping on val loss

Every run returns one result row: config, seed, git hash, curves and final metrics on
two test sets (natural statistics; planted lures), measured at the best-validation step.

Usage:  python -m ap.train configs/e1.yaml [--set key=value ...] [--seeds 0-9] [--out results/e1.jsonl]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import yaml

from ap.data import GenConfig, generate, load_workshop
from ap.metrics import attention_stats, behaviour
from ap.models.attn import AttnControl, AttnModel, ModelConfig, n_params
from ap.models.recurrent import FAMILIES, RecurrentConfig, RecurrentModel


@dataclass
class TrainConfig:
    name: str = "run"
    n: int = 1
    length: int = 24
    model: dict = field(default_factory=dict)
    regime: str = "fresh"            # fresh | fixed | workshop
    train_size: int | None = None    # fixed regime: number of training sequences
    train_lures: int = 0             # planted lures in training data (default: natural)
    epochs: int = 10                 # fixed/workshop regime when max_steps is None
    max_steps: int | None = 30000
    patience: int = 2000             # steps without val-loss improvement before stopping
    eval_every: int = 250
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    n_val: int = 1000
    n_test: int = 2000
    workshop_dir: str = ""           # for regime=workshop
    save_ckpt: str = ""              # directory for best checkpoints ("" = don't save)
    threads: int = 2
    device: str = "cpu"              # "cuda" when a GPU is available (Modal runs)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)).strip()
    except Exception:
        return "unknown"


def evaluate(model, x, t, l, n, full=False, ctl=None):
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        c = ctl if ctl is not None else (AttnControl(record=True) if full else None)
        logits = model(torch.as_tensor(x, device=dev), c).float()
        loss = nn.functional.cross_entropy(logits.reshape(-1, 2), torch.as_tensor(t, device=dev).reshape(-1)).item()
        score = (logits[..., 1] - logits[..., 0]).cpu().numpy()
    out = {"loss": loss, **behaviour(score, t, l, n)}
    if full and c is not None and c.store.get("attn"):
        for li, A in enumerate(c.store["attn"]):
            for h in range(A.shape[1]):
                for k, v in attention_stats(A[:, h].detach().cpu().numpy(), t, n).items():
                    out[f"L{li}H{h}_{k}"] = v
    model.train()
    return out


def run(cfg: TrainConfig, seed: int) -> dict:
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(10_000 + seed)
    if cfg.model.get("family", "attn") in FAMILIES:
        keep = RecurrentConfig.__dataclass_fields__
        mcfg = RecurrentConfig(**{"max_len": cfg.length, **{k: v for k, v in cfg.model.items() if k in keep}})
        model = RecurrentModel(mcfg)
    else:
        keep = ModelConfig.__dataclass_fields__
        mcfg = ModelConfig(**{"max_len": cfg.length, **{k: v for k, v in cfg.model.items() if k in keep}})
        model = AttnModel(mcfg)
    dev = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt_cls = torch.optim.AdamW if cfg.weight_decay > 0 else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    # evaluation sets: fixed per (n, length), independent of the run seed
    erng = np.random.default_rng(1_000_000 + 97 * cfg.n + cfg.length)
    val = generate(GenConfig(cfg.n, cfg.length, clean=False), cfg.n_val, erng)
    test_nat = generate(GenConfig(cfg.n, cfg.length, clean=False), cfg.n_test, erng)
    test_lure = generate(GenConfig(cfg.n, cfg.length, n_lure=4, clean=True), cfg.n_test, erng)

    if cfg.regime == "workshop":
        xtr, ttr, _, _ = load_workshop(f"{cfg.workshop_dir}/nback_{cfg.n}_train.json", cfg.n)
        xte, tte_given, lte, _ = load_workshop(f"{cfg.workshop_dir}/nback_{cfg.n}_test.json", cfg.n)
        test_ws = (xte, tte_given, lte)
    elif cfg.regime == "fixed":
        xtr, ttr, _ = generate(GenConfig(cfg.n, cfg.length, n_lure=cfg.train_lures, clean=cfg.train_lures > 0),
                               cfg.train_size, rng)

    gcfg = GenConfig(cfg.n, cfg.length, n_lure=cfg.train_lures, clean=cfg.train_lures > 0)
    if cfg.regime in ("workshop", "fixed") and cfg.max_steps is None:
        total_steps = cfg.epochs * int(np.ceil(len(xtr) / cfg.batch_size))
    else:
        total_steps = cfg.max_steps

    def batches():
        if cfg.regime == "fresh":
            while True:
                x, t, _ = generate(gcfg, cfg.batch_size, rng)
                yield x, t
        g = torch.Generator().manual_seed(seed)
        while True:  # epochs over the fixed set, reshuffled like DataLoader(shuffle=True)
            perm = torch.randperm(len(xtr), generator=g).numpy()
            for s in range(0, len(xtr), cfg.batch_size):
                idx = perm[s : s + cfg.batch_size]
                yield xtr[idx], ttr[idx]

    curves = {"step": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_dprime": [],
              "test_acc": [], "test_dprime": []}
    best = {"val_loss": float("inf"), "step": 0, "state": None}
    running, nrun, step, t0 = 0.0, 0, 0, time.time()
    it = batches()
    while step < total_steps:
        x, t = next(it)
        logits = model(torch.as_tensor(x, device=dev))
        loss = ce(logits.reshape(-1, 2), torch.as_tensor(t, device=dev).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item(); nrun += 1; step += 1
        if step % cfg.eval_every == 0 or step == total_steps:
            v = evaluate(model, *val, cfg.n)
            te = evaluate(model, *test_nat, cfg.n)
            for k, val_ in (("step", step), ("train_loss", running / nrun), ("val_loss", v["loss"]),
                            ("val_acc", v["acc"]), ("val_dprime", v["dprime"]),
                            ("test_acc", te["acc"]), ("test_dprime", te["dprime"])):
                curves[k].append(round(float(val_), 5))
            running, nrun = 0.0, 0
            if v["loss"] < best["val_loss"] - 1e-5:
                best = {"val_loss": v["loss"], "step": step, "state": copy.deepcopy(model.state_dict())}
            elif cfg.max_steps is not None and step - best["step"] >= cfg.patience:
                break

    final_state = copy.deepcopy(model.state_dict())
    row = {"name": cfg.name, "seed": seed, "git": git_hash(), "config": asdict(cfg), "model": mcfg.to_dict(),
           "n_params": n_params(model), "steps": step, "best_step": best["step"],
           "seconds": round(time.time() - t0, 1), "curves": curves}
    row["device"] = str(dev)
    for tag, state in (("final", final_state), ("best", best["state"] or final_state)):
        model.load_state_dict(state)
        row[f"{tag}_natural"] = evaluate(model, *test_nat, cfg.n, full=True)
        row[f"{tag}_lure"] = evaluate(model, *test_lure, cfg.n, full=True)
        if cfg.regime == "workshop":
            row[f"{tag}_workshop_test"] = evaluate(model, *test_ws, cfg.n)
    if cfg.save_ckpt:
        os.makedirs(cfg.save_ckpt, exist_ok=True)
        torch.save({"model": mcfg.to_dict(), "state": best["state"] or final_state},
                   os.path.join(cfg.save_ckpt, f"{cfg.name}_n{cfg.n}_s{seed}.pt"))
    return row


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def _set(d: dict, key: str, value: str):
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. n=3 model.pe=rope")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    raw = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, _, v = kv.partition("=")
        _set(raw, k, v)
    cfg = TrainConfig(**raw)
    out = a.out or f"results/{cfg.name}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for s in parse_seeds(a.seeds):
        row = run(cfg, s)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        b = row["best_natural"]
        print(f"{cfg.name} n={cfg.n} seed={s} steps={row['steps']} acc={b['acc']:.4f} "
              f"d'={b['dprime']:.2f} hit={b['hit']:.3f} fa={b['fa']:.3f} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
