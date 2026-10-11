# Handoff: where the project stands (2026-10-10)

Read this first when picking the project up in a new (local) Claude Code session.

## Pointers
- Repo: github.com/Daniel-Gong/attention-precision (everything below is committed on `main`).
- Plan doc (Claude Doc): "Attention Precision Research Plan" — https://claude.ai/code/artifact/37ddb091-8084-41c2-bbf5-b5c8868bb9ad
  - "Revised framing" section at the top = current claims C1–C4, scorecard, outline.
  - Execution log is up to date (all phases Done except the open paper items below).
- Previous chat (for reference): ~/attention-precision-chat.jsonl
- Paper draft: `paper/main.tex` (NeurIPS 2026 style, anonymous), `paper/main.pdf`. Build: `cd paper && latexmk -pdf main.tex` (runs bibtex).
- Target venue: NeurIPS 2027.

## Story (one paragraph)
Transformers *find* the item but *misread* it. Retrieval heads locate the N-back target on most
trials (L7), and a linear probe separates match from lure almost perfectly (L8: AUC ≈1.0 vs model's own
0.58–0.63), so errors are readout interference, not attention imprecision. Removing distractor content
(L5s suppression) raises d′ in 42/43 cells (median +0.60); sharpening attention barely helps (+0.05);
ablating the located heads hurts (−0.93). Toy models agree: depth removes the decline; position margin
does not predict d′ but the attention mass gap does (T2, ρ = 0.62).

## Key results (all in `results/summary.json`; `python scripts/summarize_all.py` regenerates)
- L2 behaviour: d′ falls with N in all 15 LLMs (Qwen2.5-14B 4.28 → 1.15 over N = 1–6).
- L3 patching → top heads; sharpen-top3 +0.05 (small but consistent, p = 0.04); ablate-top10 −0.93 (42/43 negative), random ablation +0.00.
- L5s / B1 (paired bootstrap, 2,000 resamples): suppress@0.5 +0.60 median, random-subspace −0.03,
  target suppression −1.54. Wilcoxon p < 1e−8. Degenerate cells excluded: Pythia-70m, GPT-2 XL at N = 4.
- L5x (Xiong et al. replication): held-out +0.11, best-in-sweep +0.12 — much smaller than ours.
- L7: heads locate target on 88–91% of correct matches; misses with attention elsewhere 7–16% (14 models).
- L8: probe calls 86–95% of the model's own lure FAs "lure" (15 models incl. Qwen2.5-14B).
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
Done 2026-10-10 (second session):
- Bib verified against arXiv/ACL/OpenReview/Crossref; 11 entries fixed (full authors, published venues). None hallucinated.
- `summarize_all.py` keeps one row per (tag, N, seed) and warns on unequal run counts across N; `summary.json`
  regenerated on Misha (only E9 selective-512 changed: 10 seeds per N; N = 1 d′ 8.12 → 8.15).
- Fig 3 smallest-model labels fixed (they were clipped: d′ < 0 at N = 6); Fig 3 no longer double-counts L2 rows.
- Spare room used for Fig 2 d–f (position margin vs mass gap, 300 E1 checkpoints) and Fig 5b + a "Scale"
  paragraph (ablation cost shrinks with size, ρ = 0.68, p = 0.007; suppression gain grows, ρ = 0.57, p = 0.03).
- Compute: `\gpuhours` = 51 (Misha H100 13.2 via `scripts/compute_hours.sh`; Modal B200 13.0, H100 12.2,
  L40S 13.0 from the usage dashboard, Oct 2026 dollars / list rate). Misha CPU ≈ 1,060 core-hours.
- L3 + L8 for Qwen2.5-14B run on Modal B200 (logs/modal_l3_q14.out, logs/modal_l8_q14.out on Misha): probe AUC
  1.00 / 0.997 vs model 0.81 / 0.64; 99.8% / 95% of lure FAs called lures. In Table 3, Fig 4 and §6; medians unchanged.

- Qwen2.5-14B now has every LLM study (L2, L3, L5, L5s, L7, L8, B1); runs on Modal B200, logs on Misha
  (logs/modal_{b1_q14h,l5_q14,l7_q14}.out). The old suppression-only 14B B1 file is in results/partial/.
  Section 7 numbers recomputed (also fixed: sharpening p was misquoted as 0.23; flattening cost 0.32;
  whole-model temperature 19 of 30). `summarize_all.py` now dedups L5 rows (H100/Q0.5B reruns won).

- Theory expanded (§5): Corollary 1 (content-independent attention gives no mean signal, however precise;
  confirmed by E7 |d′| ≤ 0.07), Corollary 2 (the precision account's lure prediction; rejected by the data),
  Proposition 4 (linear readout under interference: suppression vs sharpening vs random subspace), Table 1
  mapping each prediction to its test; proofs in App. A. Reviewed by a fresh-eyes agent; its 12 points fixed
  (notably: Cor. 1 gives equal means, not equal distributions; Prop. 4 allows lure-dependent distractor bias).
  Main text now ends on page 9 (limit reached).

Still open:
1. Misha checkout is behind origin and has an untracked results/summary.json that blocks `git pull`
   (move it aside first).

## Gotchas learned
- Modal warm containers can mix outputs across models — `_run` clears /tmp outputs and filters rows by model.
- Collect Modal results in completion order (already implemented).
- On Misha, use tmux; the OnDemand web shell disconnects.
- 1-layer toy models have no residual stream; oracle edits must be split (E7b design).
