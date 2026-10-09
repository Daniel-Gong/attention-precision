"""Non-attention baselines for E9: same input/output interface as AttnModel.

  lstm      embedding -> LSTM -> linear readout
  lru       linear recurrent unit: diagonal complex recurrence (Orvieto et al. 2023 style)
  selective input-dependent diagonal recurrence (a minimal Mamba-style selective scan)
  linattn   causal linear attention (elu + 1 feature map) with learned positions

Each forward takes (x, ctl=None) and returns logits (B, L, n_out); `ctl` is ignored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

FAMILIES = ("lstm", "lru", "selective", "linattn")


@dataclass
class RecurrentConfig:
    family: str = "lstm"
    vocab: int = 20
    d_model: int = 128      # embedding / residual width
    state: int = 128        # hidden or state size (the capacity knob for E9)
    n_layers: int = 1
    max_len: int = 24
    n_out: int = 2

    def to_dict(self):
        return asdict(self)


class LRULayer(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        u1, u2 = torch.rand(n), torch.rand(n)
        r_min, r_max = 0.4, 0.99
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)))
        self.theta_log = nn.Parameter(torch.log(u2 * math.pi * 2))
        self.B_re = nn.Parameter(torch.randn(n, d) / math.sqrt(2 * d))
        self.B_im = nn.Parameter(torch.randn(n, d) / math.sqrt(2 * d))
        self.C_re = nn.Parameter(torch.randn(d, n) / math.sqrt(n))
        self.C_im = nn.Parameter(torch.randn(d, n) / math.sqrt(n))
        self.D = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):  # (B, L, d)
        lam = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))
        gamma = torch.sqrt(1 - lam.abs() ** 2)
        Bu = torch.complex(x @ self.B_re.T, x @ self.B_im.T) * gamma      # (B, L, n)
        h = torch.zeros(x.size(0), lam.numel(), dtype=torch.cfloat, device=x.device)
        ys = []
        for t in range(x.size(1)):
            h = lam * h + Bu[:, t]
            ys.append((h @ torch.complex(self.C_re, self.C_im).T).real)
        y = torch.stack(ys, 1) + self.D * x
        return x + self.out(F.gelu(self.norm(y)))


class SelectiveLayer(nn.Module):
    """h_t = exp(-dt_t * A) h_{t-1} + dt_t * B_t x_t ;  y_t = C_t h_t + D x_t  (per channel, real diagonal)."""
    def __init__(self, d, n):
        super().__init__()
        self.n = n
        self.in_proj = nn.Linear(d, 2 * d)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(d, 1))  # (d, n)
        self.dt_proj = nn.Linear(d, d)
        self.BC = nn.Linear(d, 2 * n)
        self.D = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        u, gate = self.in_proj(self.norm(x)).chunk(2, -1)
        dt = F.softplus(self.dt_proj(u))                                  # (B, L, d)
        Bm, Cm = self.BC(u).chunk(2, -1)                                  # (B, L, n)
        A = -torch.exp(self.A_log)                                        # (d, n)
        h = torch.zeros(x.size(0), x.size(2), self.n, device=x.device)
        ys = []
        for t in range(x.size(1)):
            dA = torch.exp(dt[:, t, :, None] * A)
            h = dA * h + dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, 1) + self.D * u
        return x + self.out(y * F.silu(gate))


class LinAttnLayer(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d, n), nn.Linear(d, n), nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        q, k, v = F.elu(self.q(x)) + 1, F.elu(self.k(x)) + 1, self.v(x)
        S = torch.cumsum(k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)        # (B, L, n, d)
        z = torch.cumsum(k, dim=1)                                        # (B, L, n)
        num = (q.unsqueeze(-1) * S).sum(-2)
        den = (q * z).sum(-1, keepdim=True).clamp(min=1e-6)
        return self.out(num / den)


class RecurrentModel(nn.Module):
    def __init__(self, cfg: RecurrentConfig):
        super().__init__()
        assert cfg.family in FAMILIES
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.d_model)
        if cfg.family == "lstm":
            self.rnn = nn.LSTM(cfg.d_model, cfg.state, num_layers=cfg.n_layers, batch_first=True)
            self.unembed = nn.Linear(cfg.state, cfg.n_out)
            return
        if cfg.family == "linattn":
            self.positional_encoding = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = {"lru": LRULayer, "selective": SelectiveLayer, "linattn": LinAttnLayer}[cfg.family]
        self.layers = nn.ModuleList([layer(cfg.d_model, cfg.state) for _ in range(cfg.n_layers)])
        self.unembed = nn.Linear(cfg.d_model, cfg.n_out)

    def forward(self, x, ctl=None):
        h = self.embedding(x)
        if self.cfg.family == "lstm":
            return self.unembed(self.rnn(h)[0])
        if self.cfg.family == "linattn":
            h = h + self.positional_encoding(torch.arange(x.size(1), device=x.device))
        for layer in self.layers:
            h = layer(h)
        return self.unembed(h)
