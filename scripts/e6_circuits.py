"""E6: circuit statistics for every saved 1-layer, learned-position checkpoint.

Per checkpoint (one JSONL row):
  solution        attend-on-match or attend-on-nonmatch (from attention on the test set)
  term_shares     covariance share of the row-centred logit variance carried by each QK
                  term (EE content, EP, PE, PP position)
  delta_content   mean content bonus for a repeated letter: EE[a, a] - mean_b!=a EE[a, b]
  delta_pos       positional margin of the target offset over its neighbours, averaged over
                  query positions i >= N + 1: PP[i, i-N] - max(PP[i, i-N-1], PP[i, i-N+1])
  pos_profile     mean PP[i, i-d] over i for offsets d = 0..12 (row-centred)
  readout         for the no-residual model: corr. of cE with "letter identity" is not
                  meaningful, so we report the spread of cE and cP (their std)
These are the quantities the theory (T1-T2) is fit to.

Usage: python scripts/e6_circuits.py runs/e1 --out results/e6_circuits.jsonl
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from ap.circuits import ov_readout, qk_terms, strategy, term_variance_shares
from ap.data import GenConfig, generate
from ap.metrics import attention_stats
from ap.models.attn import AttnControl, AttnModel, ModelConfig


def pos_stats(PP, n, L):
    PPc = PP - np.array([PP[i, : i + 1].mean() for i in range(L)])[:, None]
    margins = []
    for i in range(n + 1, L):
        nb = [PPc[i, i - n - 1]] + ([PPc[i, i - n + 1]] if n > 1 else [PPc[i, i]])
        margins.append(PPc[i, i - n] - max(nb))
    prof = [float(np.mean([PPc[i, i - d] for i in range(d, L)])) for d in range(0, 13)]
    return float(np.mean(margins)), prof


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--n-seq", type=int, default=500)
    a = p.parse_args()
    torch.set_num_threads(2)
    with open(a.out, "w") as f:
        for path in sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt"))):
            ck = torch.load(path, map_location="cpu")
            cfg = ModelConfig(**ck["model"])
            if cfg.pe != "learned" or cfg.ln or cfg.n_layers != 1 or cfg.n_heads != 1:
                continue
            m = AttnModel(cfg); m.load_state_dict(ck["state"]); m.eval()
            n = int(re.search(r"_n(\d+)_s", path).group(1)); seed = int(re.search(r"_s(\d+)\.pt", path).group(1))
            tag = os.path.basename(path).split(f"_n{n}_")[0]
            x, t, l = generate(GenConfig(n, cfg.max_len, clean=False), a.n_seq, np.random.default_rng(5 + n))
            ctl = AttnControl(record=True)
            with torch.no_grad():
                m(torch.as_tensor(x), ctl)
            st = attention_stats(ctl.store["attn"][0][:, 0].numpy(), t, n)
            T = qk_terms(m)
            EE = T["EE"][0]
            dcont = float(np.mean([EE[k, k] - np.delete(EE[k], k).mean() for k in range(EE.shape[0])]))
            dpos, prof = pos_stats(T["PP"][0], n, cfg.max_len)
            row = {"tag": tag, "n": n, "seed": seed, "d_model": cfg.d_model,
                   "solution": strategy({"L0H0_target_att_match": st["target_att_match"],
                                         "L0H0_target_att_nonmatch": st["target_att_nonmatch"]}),
                   "target_att_match": st["target_att_match"], "target_att_nonmatch": st["target_att_nonmatch"],
                   "term_shares": term_variance_shares(T, x[:200]), "delta_content": dcont,
                   "delta_pos": dpos, "pos_profile": prof}
            if not cfg.residual and not cfg.ffn:
                r = ov_readout(m)
                row["readout_std"] = {"cE": float(np.std(r["cE"])), "cP": float(np.std(r["cP"]))}
            f.write(json.dumps(row) + "\n")
            print(tag, n, seed, row["solution"], f"dpos={dpos:.2f} dcont={dcont:.2f}", flush=True)


if __name__ == "__main__":
    main()
