#!/bin/bash
# E7 readout test, redone with split-oracle and denoised attention, on the E1 checkpoints (12 shards).
#   sbatch scripts/slurm/e7_oracle2.sh
#SBATCH --job-name=ap-e7-oracle2
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=2:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
set +u; module load miniconda; conda activate ap; set -u
mkdir -p results/e7_oracle2
PYTHONPATH=src python scripts/analyze_ckpt.py runs/e1 --conds ${CONDS:-oracle2} --shard ${SLURM_ARRAY_TASK_ID}/12 \
  --out results/e7_oracle2/${CONDS:-oracle2}_s${SLURM_ARRAY_TASK_ID}.jsonl
