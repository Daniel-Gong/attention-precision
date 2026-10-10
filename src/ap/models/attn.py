"""Attention model family for N-back.

One configurable decoder covers the workshop model and its extensions:

  workshop model: residual=False, ffn=False, ln=False, pe="learned", 1 layer, 1 head
  standard block: residual=True,  ffn=True,  ln=True  (pre-LN)

Attention is implemented by hand (not nn.MultiheadAttention) so experiments can read
and change the logits: temperature, attention variants, oracle attention and term
knockouts all go through `AttnControl`. With default controls and the workshop config
the model is numerically identical to the workshop code (see tests/test_models.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

PE_TYPES = ("learned", "sinusoidal", "rope", "alibi", "none")
ATTN_FNS = ("softmax", "sigmoid", "ssmax", "topk")


@dataclass
class ModelConfig:
    vocab: int = 20
    d_model: int = 512
    n_layers: int = 1
    n_heads: int = 1
    max_len: int = 24
    pe: str = "learned"
    residual: bool = False
    ffn: bool = False
    ln: bool = False
    ffn_mult: int = 4
    attn_fn: str = "softmax"     # softmax | sigmoid | ssmax (scalable softmax) | topk
    topk: int = 4
    n_out: int = 2

    def to_dict(self):
        return asdict(self)


@dataclass
class AttnControl:
    """Runtime interventions, applied to every layer/head unless `heads` narrows it."""
    temperature: float = 1.0                  # logits / temperature
    heads: list[tuple[int, int]] | None = None  # (layer, head) pairs to apply to; None = all
    oracle_offset: int | None = None          # replace attention with one-hot on i - offset
    oracle_mix: float | None = None           # with oracle_offset: weight on i - offset, rest on i itself
    denoise_offset: int | None = None         # keep only the model's own mass on i and i - offset, renormalised
    record: bool = False                      # keep attention probs and logits
    store: dict = field(default_factory=dict)

    def applies(self, layer: int, head: int) -> bool:
        return self.heads is None or (layer, head) in self.heads


def sinusoidal_table(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


def rope_cache(n: int, dh: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2).float() / dh))
    ang = torch.arange(n).float().unsqueeze(1) * inv.unsqueeze(0)  # (n, dh/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: (B, H, L, dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.size(-2)], sin[: x.size(-2)]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


def alibi_slopes(h: int) -> torch.Tensor:
    return torch.tensor([2 ** (-8 * (i + 1) / h) for i in range(h)])


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        d, h = cfg.d_model, cfg.n_heads
        assert d % h == 0
        self.dh = d // h
        # same parameterisation and init as nn.MultiheadAttention
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
        self.out_proj = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)
        if cfg.pe == "rope":
            cos, sin = rope_cache(4 * cfg.max_len, self.dh)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        if cfg.pe == "alibi":
            self.register_buffer("slopes", alibi_slopes(h), persistent=False)

    def forward(self, x, ctl: AttnControl | None = None):  # x: (B, L, d)
        B, L, d = x.shape
        h, dh = self.cfg.n_heads, self.dh
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.view(B, L, h, dh).transpose(1, 2) for t in (q, k, v))  # (B, h, L, dh)
        if self.cfg.pe == "rope":
            q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        logits = q @ k.transpose(-1, -2) / math.sqrt(dh)                   # (B, h, L, L)
        i = torch.arange(L, device=x.device)
        rel = (i.unsqueeze(1) - i.unsqueeze(0)).float()                   # i - j
        if self.cfg.pe == "alibi":
            logits = logits - self.slopes.view(1, h, 1, 1) * rel.clamp(min=0)
        future = rel < 0
        if ctl is not None and ctl.temperature != 1.0:
            scale = torch.ones(h, device=x.device)
            for hh in range(h):
                if ctl.applies(self.layer, hh):
                    scale[hh] = 1.0 / ctl.temperature
            logits = logits * scale.view(1, h, 1, 1)
        A = self._normalise(logits, future, L)
        if ctl is not None and ctl.oracle_offset is not None:
            off = ctl.oracle_offset
            oracle = torch.zeros(L, L, device=x.device)
            w = 1.0 if ctl.oracle_mix is None else ctl.oracle_mix
            for r in range(L):
                oracle[r, r] += 1.0 - w
                oracle[r, max(0, r - off)] += w
            for hh in range(h):
                if ctl.applies(self.layer, hh):
                    A[:, hh] = oracle
        if ctl is not None and ctl.denoise_offset is not None:
            keep = (rel == 0) | (rel == ctl.denoise_offset)
            keep = keep | ((i < ctl.denoise_offset).unsqueeze(1) & (i == 0).unsqueeze(0))  # early rows: i - N clipped to 0
            for hh in range(h):
                if ctl.applies(self.layer, hh):
                    Ah = A[:, hh] * keep
                    A[:, hh] = Ah / Ah.sum(-1, keepdim=True).clamp(min=1e-9)
        if ctl is not None and ctl.record:
            ctl.store.setdefault("attn", []).append(A.detach())
            ctl.store.setdefault("logits", []).append(logits.masked_fill(future, float("-inf")).detach())
        out = (A @ v).transpose(1, 2).reshape(B, L, d)
        return self.out_proj(out)

    def _normalise(self, logits, future, L):
        fn = self.cfg.attn_fn
        if fn == "softmax":
            return logits.masked_fill(future, float("-inf")).softmax(-1)
        if fn == "ssmax":  # scalable softmax: scale logits by log(number of visible keys)
            nkeys = torch.arange(1, L + 1, device=logits.device).float().log().clamp(min=1.0)
            return (logits * nkeys.view(1, 1, L, 1)).masked_fill(future, float("-inf")).softmax(-1)
        if fn == "sigmoid":  # unnormalised; bias -log(L) as in sigmoid-attention practice
            return torch.sigmoid(logits - math.log(L)).masked_fill(future, 0.0)
        if fn == "topk":
            masked = logits.masked_fill(future, float("-inf"))
            kth = masked.topk(min(self.cfg.topk, L), dim=-1).values[..., -1:]
            return masked.masked_fill(masked < kth, float("-inf")).softmax(-1)
        raise ValueError(fn)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.cfg = cfg
        self.self_attn = Attention(cfg, layer)
        if cfg.ln:
            self.ln1 = nn.LayerNorm(cfg.d_model)
        if cfg.ffn:
            if cfg.ln:
                self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.ffn_mult * cfg.d_model), nn.GELU(),
                                     nn.Linear(cfg.ffn_mult * cfg.d_model, cfg.d_model))

    def forward(self, x, ctl=None):
        a = self.self_attn(self.ln1(x) if self.cfg.ln else x, ctl)
        x = x + a if self.cfg.residual else a
        if self.cfg.ffn:
            f = self.mlp(self.ln2(x) if self.cfg.ln else x)
            x = x + f if self.cfg.residual else f
        return x


class AttnModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.pe in PE_TYPES and cfg.attn_fn in ATTN_FNS
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.d_model)
        if cfg.pe == "learned":
            self.positional_encoding = nn.Embedding(cfg.max_len, cfg.d_model)
        elif cfg.pe == "sinusoidal":
            self.register_buffer("pe_table", sinusoidal_table(4 * cfg.max_len, cfg.d_model), persistent=False)
        self.layers = nn.ModuleList([Block(cfg, l) for l in range(cfg.n_layers)])
        if cfg.ln:
            self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.n_out)

    def forward(self, x, ctl: AttnControl | None = None):  # x: (B, L) -> logits (B, L, n_out)
        L = x.size(1)
        h = self.embedding(x)
        if self.cfg.pe == "learned":
            h = h + self.positional_encoding(torch.arange(L, device=x.device))
        elif self.cfg.pe == "sinusoidal":
            h = h + self.pe_table[:L]
        for blk in self.layers:
            h = blk(h, ctl)
        if self.cfg.ln:
            h = self.ln_f(h)
        return self.unembed(h)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
