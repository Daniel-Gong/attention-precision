"""Test-time interventions on saved checkpoints (E5 temperature, E7 causal tests).

For each checkpoint in a directory, evaluates on the fixed natural and lure test sets:
  - baseline
  - temperature sweep: attention logits / tau for tau in TAUS (E5)
  - oracle attention: one-hot on i - N in every head (E7; tests the readout alone)
  - term knockouts for 1-layer learned-PE models: drop the content (EE) or position (PP)
    part of the logits by hooking the logits (E7)
and writes one JSONL row per (checkpoint, condition).

Usage: python scripts/analyze_ckpt.py runs/e1 --out results/e1_interventions.jsonl
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from ap.circuits import qk_terms
from ap.data import GenConfig, generate
from ap.models.attn import AttnControl, AttnModel, ModelConfig
from ap.train import evaluate

TAUS = [0.25, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 4.0]


def test_sets(n, length=24, n_test=2000):
    erng = np.random.default_rng(1_000_000 + 97 * n + length)   # same as ap.train
    generate(GenConfig(n, length, clean=False), 1000, erng)      # skip the val set
    nat = generate(GenConfig(n, length, clean=False), n_test, erng)
    lure = generate(GenConfig(n, length, n_lure=4, clean=True), n_test, erng)
    return nat, lure


class Knockout(torch.nn.Module):
    """Wraps layer-0 attention so the content (EE) or position (PP) term is removed."""
    def __init__(self, model, drop):
        super().__init__()
        self.model, self.drop = model, drop
        self.T = {k: torch.as_tensor(v) for k, v in qk_terms(model).items()}

    def forward(self, x, ctl=None):
        at = self.model.layers[0].self_attn
        T, drop = self.T, self.drop
        orig = at._normalise

        def patched(logits, future, L):
            idx = x
            if drop == "content":
                sub = T["EE"][:, idx[:, :, None], idx[:, None, :]].permute(1, 0, 2, 3)
            else:
                sub = T["PP"][:, :L, :L].unsqueeze(0)
            return orig(logits - sub, future, L)
        at._normalise = patched
        try:
            return self.model(x, ctl)
        finally:
            at._normalise = orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--conds", default="all", help="all | oracle2 (only the split-oracle and denoise conditions)")
    p.add_argument("--shard", default="0/1", help="i/k: process every k-th checkpoint starting at i")
    a = p.parse_args()
    si, sk = map(int, a.shard.split("/"))
    torch.set_num_threads(2)
    with open(a.out, "a") as f:
        for path in sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")))[si::sk]:
            ck = torch.load(path, map_location="cpu")
            if "family" in ck["model"]:
                continue
            cfg = ModelConfig(**ck["model"])
            m = AttnModel(cfg); m.load_state_dict(ck["state"]); m.eval()
            n = int(re.search(r"_n(\d+)_s", path).group(1)); seed = int(re.search(r"_s(\d+)\.pt", path).group(1))
            sets = dict(zip(("natural", "lure"), test_sets(n, cfg.max_len)))
            conds = [("tau", t, AttnControl(temperature=t, record=True)) for t in TAUS]
            conds.append(("oracle", n, AttnControl(oracle_offset=n, record=True)))
            # E7 readout test, redone: the 1-layer model has no residual stream, so the output must see
            # both the current item and i - N. Split oracle: weight w on i - N and 1 - w on i itself.
            # Denoise: keep the model's own attention on i and i - N only (all other mass removed).
            split = [("oracle_mix", w, AttnControl(oracle_offset=n, oracle_mix=w, record=True)) for w in (0.25, 0.5, 0.75)]
            split.append(("denoise", n, AttnControl(denoise_offset=n, record=True)))
            conds = split if a.conds == "oracle2" else conds + split
            for kind, val, ctl in conds:
                for sname, (x, t, l) in sets.items():
                    ctl.store.clear()
                    r = evaluate(m, x, t, l, n, ctl=ctl)
                    f.write(json.dumps({"ckpt": os.path.basename(path), "n": n, "seed": seed, "set": sname,
                                        "cond": kind, "value": val, **r}) + "\n")
            if cfg.pe == "learned" and not cfg.ln and a.conds == "all":
                for drop in ("content", "position"):
                    ko = Knockout(m, drop)
                    for sname, (x, t, l) in sets.items():
                        r = evaluate(ko, x, t, l, n)
                        f.write(json.dumps({"ckpt": os.path.basename(path), "n": n, "seed": seed, "set": sname,
                                            "cond": "knockout", "value": drop, **r}) + "\n")
            print("done", path, flush=True)


if __name__ == "__main__":
    main()
