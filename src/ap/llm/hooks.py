"""Attention control for Hugging Face models (GPT-2, Pythia/GPT-NeoX, Qwen2, Llama, ...).

Registers an attention implementation "ap_eager" that computes the same eager attention
as transformers and, through a module-level `CONTROL`, can:
  - record attention probabilities (and logits) for chosen layers
  - scale chosen heads' logits by 1/tau (head-specific temperature, L5)
  - mean- or zero-ablate chosen heads' outputs (L5 necessity)
  - overwrite chosen heads' outputs with cached ones (activation patching, L3)
  - cache every head's output (to build patching sources)

Use:  model = load(name)  # sets attn_implementation="ap_eager"
      with control(record_layers=[3], temps={(3, 5): 0.5}): model(ids)
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask


@dataclass
class Control:
    record_layers: set[int] = field(default_factory=set)
    record_logits: bool = False
    temps: dict[tuple[int, int], float] = field(default_factory=dict)        # (layer, head) -> tau
    ablate: dict[tuple[int, int], torch.Tensor | None] = field(default_factory=dict)  # None = zero
    patch: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)  # (B, L, dh) head output
    patch_positions: torch.Tensor | None = None   # bool (L,) positions to patch; None = all
    cache_outputs: bool = False
    keep_grad: bool = False
    attn: dict[int, torch.Tensor] = field(default_factory=dict)
    logits: dict[int, torch.Tensor] = field(default_factory=dict)
    outputs: dict[int, torch.Tensor] = field(default_factory=dict)            # layer -> (B, L, H, dh)


CONTROL = Control()


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def ap_eager(module: nn.Module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    c = CONTROL
    layer = getattr(module, "layer_idx", None)
    n_rep = query.shape[1] // key.shape[1]
    key, value = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    w = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : key.shape[-2]]
    if c.temps and layer is not None:
        s = torch.ones(w.shape[1], dtype=w.dtype, device=w.device)
        for (l, h), tau in c.temps.items():
            if l == layer:
                s[h] = 1.0 / tau
        w = w * s.view(1, -1, 1, 1)
    if layer in c.record_layers and c.record_logits:
        c.logits[layer] = w.detach().float().cpu()
    p = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    p = nn.functional.dropout(p, p=dropout, training=module.training)
    if layer in c.record_layers:
        c.attn[layer] = p.detach().float().cpu()
    out = torch.matmul(p, value).transpose(1, 2).contiguous()          # (B, L, H, dh)
    if c.cache_outputs and layer is not None:
        if c.keep_grad:                      # attribution patching: keep the graph
            out.retain_grad()
            c.outputs[layer] = out
        else:
            c.outputs[layer] = out.detach().clone()
    if layer is not None and (c.ablate or c.patch):
        out = out.clone()
        B, T = out.shape[:2]
        if c.patch_positions is None:
            m = torch.ones(B, T, dtype=torch.bool, device=out.device)
        else:   # (T,) shared across rows, or (B, T) per row
            m = c.patch_positions.to(out.device)
            m = m[None].expand(B, T) if m.dim() == 1 else m
        m = m[..., None]
        for (l, h), mean in c.ablate.items():
            if l == layer:
                fill = torch.zeros_like(out[:, :, h]) if mean is None else mean.to(out).expand_as(out[:, :, h])
                out[:, :, h] = torch.where(m, fill, out[:, :, h])
        for (l, h), src in c.patch.items():
            if l == layer:
                out[:, :, h] = torch.where(m, src.to(out), out[:, :, h])
    return out, p


AttentionInterface.register("ap_eager", ap_eager)
ALL_MASK_ATTENTION_FUNCTIONS.register("ap_eager", eager_mask)


@contextlib.contextmanager
def control(**kw):
    """Temporarily set CONTROL fields; recorded tensors stay readable on the returned object."""
    global CONTROL
    old = CONTROL
    CONTROL = Control(**{k: (set(v) if k == "record_layers" else v) for k, v in kw.items()})
    try:
        yield CONTROL
    finally:
        CONTROL = old


def load(name: str, dtype=torch.float32, device: str = "cpu", revision: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype,
                                                 attn_implementation="ap_eager").to(device).eval()
    return model, tok


def n_layers_heads(model) -> tuple[int, int]:
    cfg = model.config
    return cfg.num_hidden_layers, cfg.num_attention_heads
