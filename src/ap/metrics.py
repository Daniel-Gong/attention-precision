"""Metric suite used by every experiment.

Behaviour: accuracy, hit rate, false-alarm rate, d' (log-linear corrected), AUC,
false alarms by lure offset.
Attention (per trial type, never on averaged matrices): attention on the target
position i - N, row entropy, K_eff = exp(entropy), and the mutual information between
the attended position and the trial label.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def dprime(hits: int, n_signal: int, fas: int, n_noise: int) -> float:
    """Log-linear correction (Hautus 1995): add 0.5 to counts, 1 to totals."""
    h = (hits + 0.5) / (n_signal + 1)
    f = (fas + 0.5) / (n_noise + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank statistic (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rpos = ranks[: len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def behaviour(score: np.ndarray, target: np.ndarray, lure: np.ndarray | None, n: int,
              threshold: float = 0.0) -> dict:
    """score: logit(match) - logit(no match), shape (B, L). Positions < n are excluded
    (no N-back letter exists), matching the paper's analyses."""
    s, t = score[:, n:], target[:, n:].astype(bool)
    pred = s > threshold
    hits, fas = int((pred & t).sum()), int((pred & ~t).sum())
    ns, nn_ = int(t.sum()), int((~t).sum())
    out = {
        "acc": float((pred == t).mean()),
        "acc_all_positions": float(((score > threshold) == target.astype(bool)).mean()),
        "hit": hits / max(ns, 1), "fa": fas / max(nn_, 1),
        "dprime": dprime(hits, ns, fas, nn_), "auc": auc(s.ravel(), t.ravel()),
        "n_signal": ns, "n_noise": nn_,
    }
    if lure is not None:
        lu = lure[:, n:]
        clean = (~t) & (lu == 0)
        out["fa_nonlure"] = float(pred[clean].mean()) if clean.any() else float("nan")
        for d in np.unique(lu[lu > 0]):
            m = lu == d
            out[f"fa_lure_{int(d)}"] = float(pred[m].mean())
            out[f"n_lure_{int(d)}"] = int(m.sum())
    return out


def _entropy(p: np.ndarray, axis=-1) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=axis)


def attention_stats(A: np.ndarray, target: np.ndarray, n: int) -> dict:
    """A: attention probabilities (B, L, L) for one head (rows sum to 1 over j <= i).
    Statistics are computed per row and split by trial type."""
    B, L, _ = A.shape
    rows = np.arange(n, L)
    tgt_att = A[:, rows, rows - n]                       # (B, L - n)
    H = _entropy(A[:, rows, :])                          # (B, L - n)
    t = target[:, n:].astype(bool)
    out = {}
    for name, m in (("match", t), ("nonmatch", ~t)):
        if m.any():
            out[f"target_att_{name}"] = float(tgt_att[m].mean())
            out[f"entropy_{name}"] = float(H[m].mean())
            out[f"keff_{name}"] = float(np.exp(H[m]).mean())
    out["entropy_all"] = float(H.mean())
    # MI between attended position J and label Y, per query row, averaged over rows:
    # I(J;Y) = H(p(j)) - sum_y p(y) H(p(j|y)), with p(j|y) the mean attention row given y
    mis = []
    for r_idx, r in enumerate(rows):
        y = t[:, r_idx]
        if y.all() or (~y).all():
            continue
        p1, p0 = A[y, r, : r + 1].mean(0), A[~y, r, : r + 1].mean(0)
        py = y.mean()
        pm = py * p1 + (1 - py) * p0
        mis.append(_entropy(pm) - py * _entropy(p1) - (1 - py) * _entropy(p0))
    out["mi_attn_label"] = float(np.mean(mis)) if mis else float("nan")
    # paper's original statistic, for comparison: entropy of the batch-averaged matrix
    Am = A.mean(0)
    out["total_entropy_avgmatrix"] = float(sum(_entropy(Am[i, : i + 1]) for i in range(L)))
    return out
