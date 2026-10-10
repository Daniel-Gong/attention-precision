#!/bin/bash
# Phase 2 CPU runs: E3 with a 2-layer residual model, E10 (cued multi-N) and its single-N controls, E6.
set -euo pipefail
mkdir -p logs results/parts
A=scripts/slurm/train_array.sh; M=scripts/slurm/multi_array.sh
NS="1 2 3 4 5 6"
# E3b: position encodings in a 2-layer residual model (ALiBi and no-position need depth to use position)
for pe in learned sinusoidal rope alibi none; do
  sbatch --job-name=ap-e3b-$pe --array=0-5 $A configs/sweep.yaml e3b-$pe "$NS" 10 1 \
    model.pe=$pe model.n_layers=2 model.residual=true
done
# E10: cued multi-N training, three architectures, 5 seeds each
sbatch --job-name=ap-e10-h1   --array=0-4 $M configs/e10.yaml e10-h1   5 model.n_heads=1
sbatch --job-name=ap-e10-h4   --array=0-4 $M configs/e10.yaml e10-h4   5 model.n_heads=4
sbatch --job-name=ap-e10-full --array=0-4 $M configs/e10.yaml e10-full 5 model.n_heads=4 model.ffn=true model.ln=true
# E10 controls: the same models trained on one N (cue constant), 5 seeds per N
for arch in "h1 model.n_heads=1" "h4 model.n_heads=4" "full model.n_heads=4 model.ffn=true model.ln=true"; do
  set -- $arch; a=$1; shift
  NSETS="$NS" sbatch --job-name=ap-e10s-$a --array=0-29 $M configs/e10.yaml e10s-$a 5 "$@"
done
# E6: circuit statistics on every E1 checkpoint
sbatch --job-name=ap-e6 --partition=day --time=4:00:00 --cpus-per-task=2 --mem=8G --output=logs/%x_%j.out \
  --wrap "set +u; module load miniconda; conda activate ap; PYTHONPATH=src python scripts/e6_circuits.py runs/e1 --out results/e6_circuits.jsonl"
echo "phase 2 submitted"
