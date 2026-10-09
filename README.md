# attention-precision

Code for the full paper extending *Self-attention limits working memory capacity of
transformer-based models* (Gong & Zhang, NeurIPS 2024 Behavioral ML workshop,
[arXiv 2409.10715](https://arxiv.org/abs/2409.10715)).

**Thesis.** Working-memory limits on the N-back task come from the precision of
position-based retrieval under softmax attention. The repo has three parts: theory
(`src/ap/theory`), small-model experiments (E1–E9), and pretrained-LLM studies (L1–L6).
The research plan and execution log live in the project's plan doc.

## Layout

```
src/ap/data.py          N-back generator v2: lures, lengths, exact labels
src/ap/models/attn.py   attention family: residual/FFN/LayerNorm flags, 5 position
                        encodings, softmax/sigmoid/scalable-softmax/top-k attention,
                        runtime controls (temperature, oracle attention, recording)
src/ap/metrics.py       accuracy, hits, false alarms, d', AUC, lure profile,
                        target attention, entropy, K_eff, MI(attention; label)
src/ap/train.py         config-driven trainer (fresh / fixed / workshop data)
configs/                one YAML per experiment
results/                one JSONL row per run (config, seed, git hash, curves, metrics)
scripts/summarize.py    mean ± SEM tables from a results file
tests/                  generator, metric and model-equivalence tests
```

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests
PYTHONPATH=src python -m ap.train configs/e1.yaml --set n=3 --seeds 0-4 --out results/e1.jsonl
python scripts/summarize.py results/e1.jsonl
```

`--set` overrides any config key, including nested model keys (`model.pe=rope`).

## Reproducing the workshop model

`configs/workshop.yaml` trains the paper's exact model (1 layer, 1 head, d = 512, no
residual, FFN or LayerNorm, learned absolute positions; Adam 1e-4, batch 32, 10 epochs)
on the workshop data. `tests/test_core.py::test_workshop_equivalence` checks that the
model here produces the same outputs as the workshop notebook's model given the same
weights.

**Note on the workshop data.** The original generator only rejected accidental matches
at positions after N, so a few true matches at position N are labelled as non-matches
(0.1–0.2% of positions). The new generator labels every position exactly.
