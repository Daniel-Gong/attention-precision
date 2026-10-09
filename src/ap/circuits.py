"""E6: exact circuit decomposition for 1-layer attention models with learned positions.

For the first layer, x_i = E[t_i] + P[i] (no LayerNorm), so every query-key logit splits
exactly into four terms (biases folded into the position terms):

    s_ij = EE[t_i, t_j] + EP[t_i, j] + PE[i, t_j] + PP[i, j]      (per head, already / sqrt(d_h))

EE carries "same letter?", PP carries "which offset?", and the cross terms mix the two.
For the no-residual, no-FFN model, the readout is linear in the attended values, so the
match logit difference also splits into a letter part and a position part:

    logit_m - logit_- = sum_j A_ij (cE[t_j] + cP[j]) + c0      (single head)
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ap.models.attn import AttnModel


@torch.no_grad()
def qk_terms(model: AttnModel, layer: int = 0) -> dict[str, np.ndarray]:
    cfg = model.cfg
    assert cfg.pe == "learned" and not cfg.ln and layer == 0, "exact split needs layer 0, learned PE, no LN"
    at = model.layers[layer].self_attn
    d, h = cfg.d_model, cfg.n_heads
    dh = d // h
    Wq, Wk, _ = at.in_proj_weight.chunk(3)
    bq, bk, _ = at.in_proj_bias.chunk(3)
    E, P = model.embedding.weight, model.positional_encoding.weight
    qE, kE = (E @ Wq.T).view(-1, h, dh), (E @ Wk.T).view(-1, h, dh)
    qP, kP = (P @ Wq.T + bq).view(-1, h, dh), (P @ Wk.T + bk).view(-1, h, dh)
    s = 1 / math.sqrt(dh)
    ein = lambda a, b: (torch.einsum("ahd,bhd->hab", a, b) * s).numpy()
    return {"EE": ein(qE, kE), "EP": ein(qE, kP), "PE": ein(qP, kE), "PP": ein(qP, kP)}  # (h, ., .)


def term_variance_shares(terms: dict, x: np.ndarray, head: int = 0) -> dict[str, float]:
    """Share of the variance of the visible logits s_ij (j <= i, rows centred) carried by each
    term, over sequences x (B, L). Row-centring removes what softmax ignores."""
    B, L = x.shape
    i, j = np.tril_indices(L)
    parts = {
        "EE": terms["EE"][head][x[:, i], x[:, j]],
        "EP": terms["EP"][head][x[:, i], j],
        "PE": terms["PE"][head][i, x[:, j]],
        "PP": np.broadcast_to(terms["PP"][head][i, j], (B, len(i))),
    }
    def centre(v):  # subtract each row's mean over its visible keys
        out = np.empty_like(v, dtype=float)
        for r in range(L):
            m = i == r
            out[:, m] = v[:, m] - v[:, m].mean(1, keepdims=True)
        return out
    c = {k: centre(np.asarray(v, float)) for k, v in parts.items()}
    total = sum(c.values())
    var = total.var()
    # covariance-based share: each term's covariance with the total (shares sum to 1)
    return {k: float(((v - v.mean()) * (total - total.mean())).mean() / var) for k, v in c.items()}


@torch.no_grad()
def ov_readout(model: AttnModel) -> dict[str, np.ndarray]:
    """Letter, position and constant contributions to logit(m) - logit(-) for the
    single-head, no-residual, no-FFN model."""
    cfg = model.cfg
    assert cfg.n_layers == 1 and cfg.n_heads == 1 and not (cfg.residual or cfg.ffn or cfg.ln)
    at = model.layers[0].self_attn
    _, _, Wv = at.in_proj_weight.chunk(3)
    _, _, bv = at.in_proj_bias.chunk(3)
    u = model.unembed.weight[1] - model.unembed.weight[0]
    g = u @ at.out_proj.weight
    E, P = model.embedding.weight, model.positional_encoding.weight
    return {"cE": ((E @ Wv.T) @ g).numpy(), "cP": ((P @ Wv.T) @ g).numpy(),
            "c0": float(bv @ g + u @ at.out_proj.bias + model.unembed.bias[1] - model.unembed.bias[0])}


def strategy(stats: dict, head: str = "L0H0") -> str:
    """Name the solution a head implements from its per-trial-type attention statistics:
    'attend-on-match' if it puts more weight on i - N when the letters match, else
    'attend-on-nonmatch' (the inverted solution seen at N = 4 in the workshop retrain)."""
    m, nm = stats[f"{head}_target_att_match"], stats[f"{head}_target_att_nonmatch"]
    return "attend-on-match" if m > nm else "attend-on-nonmatch"
