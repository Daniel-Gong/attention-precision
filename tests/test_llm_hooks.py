import pytest
import torch

from ap.llm import hooks
from ap.llm.hooks import control
from transformers import AutoModelForCausalLM, GPT2Config, GPTNeoXConfig, Qwen2Config


def tiny(kind):
    if kind == "gpt2":
        return GPT2Config(vocab_size=50, n_positions=64, n_embd=32, n_layer=2, n_head=4)
    if kind == "neox":
        return GPTNeoXConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                             intermediate_size=64, max_position_embeddings=64)
    return Qwen2Config(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, intermediate_size=64, max_position_embeddings=64)


@pytest.mark.parametrize("kind", ["gpt2", "neox", "qwen2"])
def test_ap_eager_matches_eager_and_controls(kind):
    torch.manual_seed(0)
    ref = AutoModelForCausalLM.from_config(tiny(kind), attn_implementation="eager").eval()
    ours = AutoModelForCausalLM.from_config(tiny(kind), attn_implementation="ap_eager").eval()
    ours.load_state_dict(ref.state_dict())
    ids = torch.randint(0, 50, (2, 20))
    with torch.no_grad():
        a = ref(ids).logits
        with control(record_layers=[0, 1], cache_outputs=True) as c:
            b = ours(ids).logits
        assert torch.allclose(a, b, atol=1e-5)
        A = c.attn[1]
        assert A.shape == (2, 4, 20, 20) and torch.allclose(A.sum(-1), torch.ones(2, 4, 20), atol=1e-5)
        assert torch.all(A.triu(1) == 0)                                   # causal
        # patching a head with its own cached output changes nothing
        with control(patch={(1, 2): c.outputs[1][:, :, 2]}):
            assert torch.allclose(ours(ids).logits, b, atol=1e-5)
        # temperature on one head changes only that head's attention
        with control(record_layers=[1], temps={(1, 2): 0.5}) as c2:
            ours(ids)
        assert torch.allclose(c2.attn[1][:, [0, 1, 3]], A[:, [0, 1, 3]], atol=1e-6)
        assert not torch.allclose(c2.attn[1][:, 2], A[:, 2])
        with control(ablate={(1, 0): None}):
            assert not torch.allclose(ours(ids).logits, b)


class FakeTok:
    """Character-level tokenizer with ' m' and ' -' as single tokens."""
    bos_token_id = 0
    def __init__(self):
        self.vocab = {" m": 1, " -": 2}
        for ch in "abcdefghijklmnopqrstuvwxyzN=0123456789 .,-\n":
            self.vocab.setdefault(ch, len(self.vocab) + 3)
    def encode(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            if s[i:i + 2] in (" m", " -"):
                out.append(self.vocab[s[i:i + 2]]); i += 2
            else:
                out.append(self.vocab.get(s[i].lower(), 3)); i += 1
        return out


def test_behaviour_pipeline_on_random_model():
    import numpy as np
    from ap.llm.behave import calibrate, drift, make_sets, score
    from ap.llm.prompts import build, k_back_labels
    tok = FakeTok()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=64, n_positions=1024, n_embd=32, n_layer=2, n_head=4),
                                             attn_implementation="ap_eager").eval()
    demos, (xc, tc, lc), (x, t, l) = make_sets(2, 8, 8, 2)
    b = build(tok, 2, x[0], t[0], demos)
    assert all(b.ids[p + 1] in (b.m_id, b.dash_id) for p in b.letter_pos)       # answer follows each letter
    assert [b.ids[p + 1] == b.m_id for p in b.letter_pos] == list(t[0].astype(bool))
    for cond in ("feedback", "own"):
        sc = score(model, tok, 2, x, t, demos, cond, batch=4)
        assert sc.shape == x.shape and np.isfinite(sc).all()
    # own-answer scoring is causal: item i's score does not depend on answers after i
    sc_f = score(model, tok, 2, x[:2], t[:2], demos, "feedback")
    t2 = t[:2].copy(); t2[:, 10:] = 1 - t2[:, 10:]
    sc_g = score(model, tok, 2, x[:2], t2, demos, "feedback")
    assert np.allclose(sc_f[:, :11], sc_g[:, :11], atol=1e-5)
    thr = calibrate(sc, t, 2); d = drift(sc, x, thr, 2)
    assert set(d["kback_fit"]) == set(range(1, 7))


def test_heads_pipeline_on_random_model():
    import numpy as np
    from ap.llm import heads as H
    from ap.llm.behave import make_sets
    tok = FakeTok()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=64, n_positions=1024, n_embd=32, n_layer=2, n_head=4),
                                             attn_implementation="ap_eager").eval()
    n = 2
    demos, _, (x, t, l) = make_sets(n, 8, 1, 2)
    nb, lure = H.attention_scores(model, tok, n, x, t, demos, "cpu", batch=4)
    assert nb.shape == (2, 4) and (nb >= 0).all() and (nb <= 1).all()
    prev, ind = H.standard_head_scores(model, "cpu", vocab_lo=5, length=10, reps=2, vocab_span=40)
    assert prev.shape == (2, 4)
    rng = np.random.default_rng(0)
    clean, corrupt, items = H.counterfactual_pairs(x, t, n, rng)
    assert all(c[i] != c[i - n] for c, i in zip(corrupt, items)) and all(c[i] == c[i - n] for c, i in zip(clean, items))
    atp, ldc, ldx = H.attribution_patching(model, tok, n, clean, corrupt, items, demos, "cpu", batch=4)
    assert atp.shape == (2, 4) and np.isfinite(atp).all()
    eff = H.exact_patching(model, tok, n, clean, corrupt, items, demos, "cpu", [(1, 0), (0, 3)], batch=4)
    assert set(eff) == {"1.0", "0.3"}
    # a random model's clean-corrupt gap is tiny, so the effect is NaN by design; otherwise finite
    assert all(np.isnan(v) for v in eff.values()) or all(np.isfinite(v) for v in eff.values())


def test_patching_all_heads_recovers_clean():
    import numpy as np
    from ap.llm import heads as H
    from ap.llm.behave import make_sets
    tok = FakeTok()
    torch.manual_seed(1)
    for cfg in (GPT2Config(vocab_size=64, n_positions=1024, n_embd=32, n_layer=2, n_head=4),
                Qwen2Config(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, intermediate_size=64, max_position_embeddings=1024),
                GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                              intermediate_size=64, max_position_embeddings=1024)):
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="ap_eager").eval()
        n = 3
        demos, _, (x, t, l) = make_sets(n, 6, 1, 1)
        clean, corrupt, items = H.counterfactual_pairs(x, t, n, np.random.default_rng(0))
        tc = np.stack([np.r_[np.zeros(n, int), (s[n:] == s[:-n]).astype(int)] for s in clean])
        with torch.no_grad():
            ids_c, lp, b = H.batch_ids(tok, n, clean, tc, demos, "cpu")
            ids_x, _, _ = H.batch_ids(tok, n, corrupt, tc, demos, "cpu")
            with control(cache_outputs=True) as cc:
                ld_c = H.logit_diff(model(ids_c).logits, lp, items, b)
            mask = torch.zeros(ids_x.shape, dtype=torch.bool)
            mask[torch.arange(len(items)), torch.as_tensor(lp[items])] = True
            allp = {(L_, h): cc.outputs[L_][:, :, h] for L_ in range(2) for h in range(4)}
            with control(patch=allp, patch_positions=mask):
                ld_p = H.logit_diff(model(ids_x).logits, lp, items, b)
        assert torch.allclose(ld_p, ld_c, atol=1e-4)


def test_intervene_arms_on_random_model(tmp_path):
    import json
    import numpy as np
    from ap.llm import intervene as I
    from ap.llm.behave import make_sets
    tok = FakeTok()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=64, n_positions=1024, n_embd=32, n_layer=2, n_head=4),
                                             attn_implementation="ap_eager").eval()
    n = 2
    demos, cal, test = make_sets(n, 6, 6, 1)
    l3 = tmp_path / "l3.jsonl"
    l3.write_text(json.dumps({"model": "m", "n": 2, "patch_top": {"1.2": 0.5, "0.1": 0.2, "1.0": 0.1}}) + "\n")
    top = I.top_heads(str(l3), "m", 2, 2)
    assert top == [(1, 2), (0, 1)]
    rh = I.matched_random(top, 4, np.random.default_rng(0), set(top))
    assert [l for l, _ in rh] == [1, 0] and not set(rh) & set(top)
    means = I.mean_head_outputs(model, tok, n, cal[0], cal[1], demos, "cpu")
    assert means.shape == (2, 4, 8)
    base = I.run_arm(model, tok, n, (cal, test), demos, "cpu", {}, 4)
    sharp = I.run_arm(model, tok, n, (cal, test), demos, "cpu", {"temps": {h: 0.5 for h in top}}, 4)
    abl = I.run_arm(model, tok, n, (cal, test), demos, "cpu", {"ablate": {h: means[h] for h in top}}, 4)
    assert all("dprime" in r for r in (base, sharp, abl))


def test_locate_pipeline_on_random_model(tmp_path):
    import json
    import numpy as np
    from ap.llm import locate as Lc
    from ap.llm.behave import make_sets
    tok = FakeTok()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=64, n_positions=1024, n_embd=32, n_layer=2, n_head=4),
                                             attn_implementation="ap_eager").eval()
    n = 2
    (tmp_path / "l3.jsonl").write_text(json.dumps({"model": "m", "n": 2, "revision": None,
                                                   "patch_top": {"1.2": 0.5, "0.1": 0.2, "1.0": float("nan")}}) + "\n")
    (tmp_path / "l2.jsonl").write_text(json.dumps({"model": "m", "n": 2, "condition": "feedback", "threshold": 0.0}) + "\n")
    heads = Lc.top_heads(str(tmp_path / "l3.jsonl"), "m", 2, 5)
    assert heads == [(1, 2), (0, 1)]
    demos, _, (x, t, l) = make_sets(n, 12, 4, 1)
    sc, att, outs = Lc.collect(model, tok, n, x, t, demos, heads, "cpu", batch=4)
    assert sc.shape == x.shape and att.shape == (12, 24 - n, 3, 2) and outs.shape == (12, 24 - n, 2 * 8)
    # target-line attention of a head is a probability mass over the 3-4 tokens of that line
    assert np.nanmax(att) <= 1.0 + 1e-5 and np.nanmin(att) >= 0
    res = Lc.analyse(n, x, t, l, sc, att, outs, 0.0)
    split = res["error_split"]
    if res["n_errors"]:
        assert abs(sum(split.values()) - 1) < 1e-6


class FakeChatTok(FakeTok):
    """Adds a minimal chat template: <u> content <a> answer ..."""
    def __init__(self):
        super().__init__()
        for k in ("<u>", "<a>", "m", "-"):
            self.vocab.setdefault(k, len(self.vocab) + 3)
    def encode(self, s, add_special_tokens=False):
        if s in ("m", "-"):
            return [self.vocab[s]]
        return super().encode(s)
    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
        ids = []
        for m in msgs:
            if m["role"] == "user":
                ids += [self.vocab["<u>"]] + super().encode(m["content"])
            else:
                ids += [self.vocab["<a>"], self.vocab[m["content"]]]
        if add_generation_prompt:
            ids += [self.vocab["<a>"]]
        return ids


def test_chat_builder_and_scoring():
    import numpy as np
    from ap.llm.behave import make_sets, score
    from ap.llm.prompts import build_chat
    tok = FakeChatTok()
    demos, _, (x, t, l) = make_sets(2, 4, 2, 1)
    b = build_chat(tok, 2, x[0], t[0])
    assert len(b.letter_pos) == 24
    assert [b.ids[p + 1] == b.m_id for p in b.letter_pos] == list(t[0].astype(bool))
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=80, n_positions=2048, n_embd=32, n_layer=2, n_head=4),
                                             attn_implementation="ap_eager").eval()
    sc = score(model, tok, 2, x, t, demos, "feedback", batch=2, builder=build_chat)
    assert sc.shape == x.shape and np.isfinite(sc).all()


def test_suppress_on_random_model(tmp_path):
    import numpy as np
    from ap.llm import suppress as S
    from ap.llm.behave import make_sets
    from ap.llm.prompts import build
    tok = FakeTok()
    torch.manual_seed(0)
    for cfg in (GPT2Config(vocab_size=64, n_positions=1024, n_embd=32, n_layer=2, n_head=4),
                Qwen2Config(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, intermediate_size=64, max_position_embeddings=1024),
                GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                              intermediate_size=64, max_position_embeddings=1024)):
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="ap_eager").eval()
        n = 2
        demos, (xc, tc, lc), (x, t, l) = make_sets(n, 4, 8, 1)
        spaces = S.fit_subspaces(model, tok, n, xc, tc, demos, "cpu", batch=4)
        assert len(spaces) == 3 and spaces[1][1].shape == (32, 19)
        U = spaces[1][1]
        assert torch.allclose(U.T @ U, torch.eye(19), atol=1e-4)
        sup = S.Suppressor(model, spaces, 1.0)
        try:
            base = S.score_cond(model, tok, n, x, t, demos, "cpu", sup, "baseline", 4)
            # per-prefix scoring equals one full-sequence pass when nothing is suppressed
            built = [build(tok, n, x[b], t[b], demos) for b in range(len(x))]
            with torch.no_grad():
                lg = model(torch.tensor([b.ids for b in built])).logits
            lp = built[0].letter_pos
            full = (lg[:, lp, built[0].m_id] - lg[:, lp, built[0].dash_id]).numpy()
            assert np.allclose(base[:, n:], full[:, n:], atol=1e-4)
            dis = S.score_cond(model, tok, n, x, t, demos, "cpu", sup, "distractors", 4)
            assert not np.allclose(dis[:, n + 2:], base[:, n + 2:])
            assert sup.mask is None
        finally:
            sup.remove()
        # hooks removed: model back to baseline
        with torch.no_grad():
            assert torch.allclose(model(torch.tensor([built[0].ids])).logits, lg[:1], atol=1e-5)
    assert S.suppressed_items("distractors", 5, 2) == [0, 1, 2, 4]
    assert S.suppressed_items("lures", 5, 2) == [2, 4]
    assert S.suppressed_items("target", 5, 2) == [3]


def test_xiong_sweep_runs():
    from ap.llm import suppress as S
    from ap.llm.behave import make_sets
    tok = FakeTok()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=64, n_positions=1024, n_embd=32, n_layer=4, n_head=4),
                                             attn_implementation="ap_eager").eval()
    n = 2
    demos, cal, test = make_sets(n, 4, 6, 1)
    spaces = S.fit_subspaces(model, tok, n, cal[0], cal[1], demos, "cpu", batch=4)
    rows = S.xiong_sweep(model, tok, n, cal, test, demos, "cpu", spaces, 4, n_dirs=2, alphas=(1.0,))
    assert len(rows) == 4 and all("dprime" in r and "cal_dprime" in r for r in rows)
