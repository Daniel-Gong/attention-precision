# Handoff: where the project stands (2026-10-10)

Read this first when picking the project up in a new (local) Claude Code session.

## Pointers
- Repo: github.com/Daniel-Gong/attention-precision (everything below is committed on `main`).
- Plan doc (Claude Doc): "Attention Precision Research Plan" — https://claude.ai/code/artifact/37ddb091-8084-41c2-bbf5-b5c8868bb9ad
  - "Revised framing" section at the top = current claims C1–C4, scorecard, outline.
  - Execution log is up to date (all phases Done except the open paper items below).
- Paper draft: `paper/main.tex` (NeurIPS 2026 style, anonymous), `paper/main.pdf`. Build: `cd paper && latexmk -pdf main.tex` (runs bibtex).
- Target venue: NeurIPS 2027.

## Story (one paragraph)
Transformers *find* the item but *misread* it. Retrieval heads locate the N-back target on most
trials (L7), and a linear probe separates match from lure almost perfectly (L8: AUC ≈1.0 vs model's own
0.58–0.63), so errors are readout interference, not attention imprecision. Removing distractor content
(L5s suppression) raises d′ in 42/43 cells (median +0.60); sharpening attention does nothing (+0.04);
ablating the located heads hurts (−0.96). Toy models agree: depth removes the decline; position margin
does not predict d′ but the attention mass gap does (T2, ρ = 0.62).

## Key results (all in `results/summary.json`; `python scripts/summarize_all.py` regenerates)
- L2 behaviour: d′ falls with N in all 15 LLMs (Qwen2.5-14B 4.28 → 1.15 over N = 1–6).
- L3 patching → top heads; L5 sharpen ≈ 0; ablate-top10 −0.96 (39/40 negative), random ablation −0.01.
- L5s / B1 (paired bootstrap, 2,000 resamples): suppress@0.5 +0.60 median, random-subspace −0.03,
  target suppression −1.54. Wilcoxon p < 1e−8. Degenerate cells excluded: Pythia-70m, GPT-2 XL at N = 4.
- L5x (Xiong et al. replication): held-out +0.11, best-in-sweep +0.12 — much smaller than ours.
- L7: heads locate target on 88–91% of correct matches; misses with attention elsewhere 6–15%.
- L8: probe calls 86–95% of the model's own lure FAs "lure".
- Toy: E1 2.67 → 2.15; E5 sigmoid/top-k flatten decline; E3b/E10 depth removes it; E9 linear attn declines;
  E6 position margin doesn't shrink with N; E7b every attention edit lowers d′.

## How to run
- Local/CPU: `pip install -r requirements.txt && pip install -e .`; `pytest tests` (55 pass).
- Modal (account daniel-gong): `modal run scripts/modal_run.py::main --study <l5|l7|l3|l2|l5s|l8|b1> --gpu best`
  (gpu: small=L40S, big=H100, best=B200). Toy: `modal run scripts/modal_run.py::toy --tag ...`.
  Outputs land in `results/parts/`, scores in `results/scores/<study>/`.
- Misha (Yale): repo at `~/project/attention-precision`; `module load miniconda && conda activate ap`;
  SLURM scripts in `scripts/slurm/`.
- Figures: `python scripts/figures.py` → `figures/`. Tables: `python scripts/paper_tables.py` → `paper/tables/`.

## Open items (next steps)
1. Fill in total GPU-hours in the appendix / checklist (currently a placeholder "~400 GPU-hours").
2. Verify bib entries in `paper/refs.bib` (several use "and others"; check arXiv IDs of 2026 papers).
3. Deduplicate e9-selective-512 rows (13 runs found, expected 10) in summarize/figures.
4. Fig 3: smallest-model labels missing.
5. ~2 pages of spare main-text room — candidates: theory (P1–P3) expansion, Qwen-14B suppression detail.
6. Optional: L8 for Qwen2.5-14B.

## Gotchas learned
- Modal warm containers can mix outputs across models — `_run` clears /tmp outputs and filters rows by model.
- Collect Modal results in completion order (already implemented).
- On Misha, use tmux; the OnDemand web shell disconnects.
- 1-layer toy models have no residual stream; oracle edits must be split (E7b design).
