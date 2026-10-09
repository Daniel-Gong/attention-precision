"""N-back prompts for base LLMs, built token by token so every answer slot is known.

Line format (one item per line, answer after the letter):

    In an N-back task, answer m when the letter matches the letter N lines earlier, otherwise -.
    N = 2

    k -
    t -
    k m
    ...

Demonstration sequences (same N, correct answers) come first, separated by blank lines.
The test sequence follows. In the "feedback" condition the correct answer for each earlier
item is shown; scoring a whole sequence then needs one forward pass. The score at item i is
logit(" m") - logit(" -") at the letter token of item i, which predicts the next token.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ap.data import ALPHABET

HEADER = ("In an N-back task, answer m when the letter matches the letter N lines earlier, "
          "otherwise -.\nN = {n}\n\n")


@dataclass
class Built:
    ids: list[int]
    letter_pos: list[int]     # token index of each test item's letter (query positions)
    m_id: int
    dash_id: int


def single(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{s!r} is {len(ids)} tokens for this tokenizer")
    return ids[0]


def tokenization_report(tok) -> dict:
    """L1 check: which pieces are single tokens."""
    out = {}
    for s in [c for c in ALPHABET] + ["\n" + c for c in ALPHABET[:3]] + [" m", " -", "\n"]:
        out[repr(s)] = len(tok.encode(s, add_special_tokens=False))
    return out


def build(tok, n: int, test_seq: np.ndarray, test_answers: np.ndarray, demos: list[tuple[np.ndarray, np.ndarray]],
          bos: bool = True) -> Built:
    m_id, dash_id, nl = single(tok, " m"), single(tok, " -"), tok.encode("\n", add_special_tokens=False)
    letter_ids = [tok.encode(c, add_special_tokens=False) for c in ALPHABET]
    if any(len(x) != 1 for x in letter_ids):
        raise ValueError("a letter is not a single token at line start")
    ids: list[int] = []
    if bos and tok.bos_token_id is not None:
        ids.append(tok.bos_token_id)
    ids += tok.encode(HEADER.format(n=n), add_special_tokens=False)

    def add_seq(seq, ans, record):
        pos = []
        for c, a in zip(seq, ans):
            pos.append(len(ids))
            ids.extend(letter_ids[int(c)])
            ids.append(m_id if a else dash_id)
            ids.extend(nl)
        ids.extend(nl)
        return pos

    for s, a in demos:
        add_seq(s, a, False)
    letter_pos = add_seq(test_seq, test_answers, True)
    return Built(ids, letter_pos, m_id, dash_id)


def k_back_labels(seq: np.ndarray, k: int) -> np.ndarray:
    t = np.zeros(len(seq), dtype=bool)
    t[k:] = seq[k:] == seq[:-k]
    return t
