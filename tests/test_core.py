import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.stats import norm

from ap.data import GenConfig, generate, labels, lure_offsets
from ap.metrics import auc, behaviour, dprime, attention_stats
from ap.models.attn import ATTN_FNS, PE_TYPES, AttnControl, AttnModel, ModelConfig


# ---------------- generator ----------------
@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_match_count_and_labels(n):
    x, t, l = generate(GenConfig(n=n, seed=0), 500)
    assert (t.sum(1) == 8).all()                      # exactly 8 matches per sequence
    assert not t[:, :n].any()                         # no match before position n
    for s, tt in zip(x[:50], t[:50]):                 # labels agree with the definition
        for i in range(n, 24):
            assert tt[i] == int(s[i] == s[i - n])


@pytest.mark.parametrize("n", [1, 3, 5])
def test_lure_rates(n):
    x, t, l = generate(GenConfig(n=n, n_lure=4, clean=True, seed=1), 2000)
    per_seq = (l > 0).sum(1).mean()
    assert abs(per_seq - 4) / 4 < 0.10                # within 10% of 4 planted lures
    assert set(np.unique(l[l > 0])) <= set(lure_offsets(n))
    assert not (l[t == 1]).any()                      # a match is never also a lure
    x0, t0, l0 = generate(GenConfig(n=n, n_lure=0, clean=True, seed=2), 500)
    assert (l0 == 0).all()                            # clean + no planted lures = no lures


def test_lengths():
    x, t, l = generate(GenConfig(n=2, length=96, seed=0), 50)
    assert x.shape == (50, 96) and (t.sum(1) == 32).all()


# ---------------- metrics ----------------
def test_dprime_matches_scipy():
    h, f = (30 + 0.5) / 41, (5 + 0.5) / 61
    assert math.isclose(dprime(30, 40, 5, 60), norm.ppf(h) - norm.ppf(f))


def test_auc():
    assert auc(np.array([3, 2, 1, 0]), np.array([1, 1, 0, 0])) == 1.0
    assert auc(np.array([0, 1, 2, 3]), np.array([1, 1, 0, 0])) == 0.0
    assert auc(np.array([1, 1, 1, 1]), np.array([1, 0, 1, 0])) == 0.5


def test_behaviour_and_attention_stats():
    n = 2
    x, t, l = generate(GenConfig(n=n, n_lure=4, seed=3), 200)
    perfect = np.where(t == 1, 5.0, -5.0)
    b = behaviour(perfect, t, l, n)
    assert b["acc"] == 1.0 and b["fa"] == 0.0 and b["hit"] == 1.0
    # oracle attention: one-hot on i - n gives target attention 1, entropy 0
    A = np.zeros((200, 24, 24))
    for i in range(24):
        A[:, i, max(0, i - n)] = 1
    s = attention_stats(A, t, n)
    assert s["target_att_match"] == 1.0 and s["entropy_all"] == 0.0 and s["mi_attn_label"] == 0.0


# ---------------- models ----------------
class WorkshopLayer(nn.Module):        # copied from the workshop notebook
    def __init__(self, d, h):
        super().__init__(); self.self_attn = nn.MultiheadAttention(embed_dim=d, num_heads=h)
    def forward(self, x, mask):
        return self.self_attn(x, x, x, attn_mask=mask)[0]


class WorkshopModel(nn.Module):        # copied from the workshop notebook
    def __init__(self, vocab, d, h, nl, out, L):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d); self.positional_encoding = nn.Embedding(L, d)
        self.layers = nn.ModuleList([WorkshopLayer(d, h) for _ in range(nl)]); self.unembed = nn.Linear(d, out)
    def forward(self, x, mask):
        pos = torch.arange(x.size(1)).unsqueeze(0).transpose(0, 1)
        x = self.embedding(x.transpose(0, 1)) + self.positional_encoding(pos)
        for layer in self.layers:
            x = layer(x, mask)
        return self.unembed(x)


@pytest.mark.parametrize("h,nl", [(1, 1), (2, 1), (4, 2)])
def test_workshop_equivalence(h, nl):
    torch.manual_seed(0)
    ref = WorkshopModel(20, 64, h, nl, 2, 24)
    ours = AttnModel(ModelConfig(d_model=64, n_heads=h, n_layers=nl))
    missing = ours.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randint(0, 20, (8, 24))
    mask = torch.triu(torch.ones(24, 24), diagonal=1).bool()
    with torch.no_grad():
        a = ref(x, mask).transpose(0, 1)
        b = ours(x)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("pe", PE_TYPES)
@pytest.mark.parametrize("fn", ATTN_FNS)
def test_variants_are_causal(pe, fn):
    torch.manual_seed(0)
    m = AttnModel(ModelConfig(d_model=32, n_heads=2, pe=pe, attn_fn=fn, residual=True, ffn=True, ln=True)).eval()
    x = torch.randint(0, 20, (4, 24))
    y = x.clone(); y[:, 15:] = (y[:, 15:] + 1) % 20     # change only the future
    with torch.no_grad():
        assert torch.allclose(m(x)[:, :15], m(y)[:, :15], atol=1e-6)


def test_controls():
    torch.manual_seed(0)
    m = AttnModel(ModelConfig(d_model=32, n_heads=2)).eval()
    x = torch.randint(0, 20, (4, 24))
    c = AttnControl(oracle_offset=3, record=True)
    with torch.no_grad():
        m(x, c)
    A = c.store["attn"][0]
    assert torch.all(A[:, :, 10, 7] == 1)
    c2 = AttnControl(temperature=0.5, heads=[(0, 1)], record=True)
    with torch.no_grad():
        m(x, c2); base = AttnControl(record=True); m(x, base)
    assert torch.allclose(c2.store["attn"][0][:, 0], base.store["attn"][0][:, 0])   # head 0 untouched
    assert not torch.allclose(c2.store["attn"][0][:, 1], base.store["attn"][0][:, 1])


# ---------------- recurrent baselines ----------------
from ap.models.recurrent import FAMILIES, RecurrentConfig, RecurrentModel


@pytest.mark.parametrize("fam", FAMILIES)
def test_recurrent_causal(fam):
    torch.manual_seed(0)
    m = RecurrentModel(RecurrentConfig(family=fam, d_model=32, state=16)).eval()
    x = torch.randint(0, 20, (4, 24))
    y = x.clone(); y[:, 15:] = (y[:, 15:] + 1) % 20
    with torch.no_grad():
        assert m(x).shape == (4, 24, 2)
        assert torch.allclose(m(x)[:, :15], m(y)[:, :15], atol=1e-5)


def test_train_smoke(tmp_path):
    from ap.train import TrainConfig, run
    for model in ({"d_model": 32, "pe": "rope", "residual": True}, {"family": "lstm", "d_model": 32, "state": 16, "pe": "learned"}):
        row = run(TrainConfig(n=2, model=model, max_steps=40, eval_every=20, n_val=50, n_test=50, threads=1), 0)
        assert row["steps"] >= 20 and 0 <= row["best_natural"]["acc"] <= 1


# ---------------- circuits ----------------
def test_qk_terms_reconstruct_logits():
    from ap.circuits import qk_terms, ov_readout, term_variance_shares
    torch.manual_seed(0)
    m = AttnModel(ModelConfig(d_model=32, n_heads=2)).eval()
    x = torch.randint(0, 20, (3, 24))
    c = AttnControl(record=True)
    with torch.no_grad():
        m(x, c)
    T = qk_terms(m)
    xi = x.numpy()
    for h in range(2):
        rec = T["EE"][h][xi[:, :, None], xi[:, None, :]] + T["EP"][h][xi][:, :, :] + \
              T["PE"][h][:, xi].transpose(1, 0, 2) + T["PP"][h][None]
        got = c.store["logits"][0][:, h].numpy()
        vis = np.tril(np.ones((24, 24), bool))
        assert np.allclose(rec[:, vis], got[:, vis], atol=1e-4)
    shares = term_variance_shares(T, xi, head=0)
    assert abs(sum(shares.values()) - 1) < 1e-6
    m1 = AttnModel(ModelConfig(d_model=32)).eval()
    c1 = AttnControl(record=True)
    with torch.no_grad():
        out = m1(x, c1)
    r = ov_readout(m1); A = c1.store["attn"][0][:, 0].numpy()
    pred = (A * (r["cE"][xi][:, None, :] + r["cP"][None, None, :])).sum(-1) + r["c0"]
    assert np.allclose(pred, (out[..., 1] - out[..., 0]).numpy(), atol=1e-4)
