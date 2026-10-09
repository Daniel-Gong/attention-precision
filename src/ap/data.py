"""N-back stimulus generator v2.

Extends the workshop generator (same alphabet, 8 matches per 24 letters) with
controlled lures, variable length and exact post-hoc labels.

Labels are always recomputed from the final sequence, so they are correct by
construction:
  target[i] = 1  iff i >= N and seq[i] == seq[i - N]
  lure[i]   = d  if target[i] == 0 and seq[i] == seq[i - d] for the smallest d in
                 LURE_OFFSETS(N) (d != N, i - d >= 0); else 0

Note on the workshop generator: it only rejected accidental matches for i > N,
so position i == N could repeat seq[0] while labelled as a non-match. This
generator rejects accidental matches at every i >= N.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

ALPHABET = "bcdfghjklnpqrstvwxyz"
C2I = {c: i for i, c in enumerate(ALPHABET)}


def lure_offsets(n: int) -> list[int]:
    """Offsets treated as near-target lures for an N-back task."""
    return [d for d in (n - 1, n + 1, n + 2) if d >= 1]


@dataclass
class GenConfig:
    n: int
    length: int = 24
    n_match: int | None = None      # default: length // 3 (8 for 24)
    n_lure: int = 0                 # planted lures per sequence
    clean: bool = True              # non-lure positions avoid all lure offsets
    seed: int | None = None
    extra: dict = field(default_factory=dict)

    def matches(self) -> int:
        return self.length // 3 if self.n_match is None else self.n_match


def _rand_letter(rng, avoid: set[int]) -> int:
    while True:
        c = int(rng.integers(0, len(ALPHABET)))
        if c not in avoid:
            return c


def generate_one(cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    n, L = cfg.n, cfg.length
    offs = lure_offsets(n)
    k = cfg.matches()
    eligible = np.arange(n, L)
    match_pos = set(rng.choice(eligible, size=k, replace=False).tolist())
    lure_plan: dict[int, int] = {}
    if cfg.n_lure:
        cand = [i for i in range(L) if i not in match_pos and any(i - d >= 0 for d in offs)]
        rng.shuffle(cand)
        for i in cand[: cfg.n_lure]:
            ds = [d for d in offs if i - d >= 0]
            lure_plan[i] = int(rng.choice(ds))
    s: list[int] = []
    for i in range(L):
        if i in match_pos:
            s.append(s[i - n])
            continue
        avoid = {s[i - n]} if i >= n else set()
        if i in lure_plan:
            # try the planned offset first, then the others; the lure letter must not
            # create a match or coincide with a different lure offset
            ds = [lure_plan[i]] + [d for d in offs if d != lure_plan[i] and i - d >= 0]
            placed = False
            for d in ds:
                others = {s[i - e] for e in offs if e != d and i - e >= 0}
                if s[i - d] not in avoid and s[i - d] not in others:
                    s.append(s[i - d])
                    placed = True
                    break
            if placed:
                continue
            # no valid lure letter here: move this lure to a later free position
            later = [j for j in range(i + 1, L) if j not in match_pos and j not in lure_plan]
            if later:
                j = int(rng.choice(later))
                lure_plan[j] = int(rng.choice([d for d in offs if j - d >= 0]))
        if cfg.clean:
            avoid |= {s[i - d] for d in offs if i - d >= 0}
        s.append(_rand_letter(rng, avoid))
    return np.array(s, dtype=np.int64)


def labels(seq: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    L = len(seq)
    tgt = np.zeros(L, dtype=np.int64)
    lure = np.zeros(L, dtype=np.int64)
    tgt[n:] = (seq[n:] == seq[:-n]).astype(np.int64)
    for i in range(L):
        if tgt[i]:
            continue
        for d in sorted(lure_offsets(n)):
            if i - d >= 0 and seq[i] == seq[i - d]:
                lure[i] = d
                break
    return tgt, lure


def generate(cfg: GenConfig, count: int, rng: np.random.Generator | None = None):
    """Returns (x, target, lure) arrays of shape (count, length)."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    x = np.stack([generate_one(cfg, rng) for _ in range(count)])
    t, l = zip(*(labels(s, cfg.n) for s in x))
    return x, np.stack(t), np.stack(l)


def load_workshop(path: str, n: int):
    """Load a workshop JSON file (nback_{n}_{split}.json) with exact relabelling.

    Returns (x, target_as_given, lure, target_exact)."""
    d = json.load(open(path))
    x = np.array([[C2I[c] for c in e["input"]] for e in d], dtype=np.int64)
    given = np.array([[1 if c == "m" else 0 for c in e["target"]] for e in d], dtype=np.int64)
    t, l = zip(*(labels(s, n) for s in x))
    return x, given, np.stack(l), np.stack(t)


def decode(seq) -> str:
    return "".join(ALPHABET[int(i)] for i in seq)
