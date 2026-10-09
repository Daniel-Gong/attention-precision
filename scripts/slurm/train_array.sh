#!/bin/bash
# CPU array job: one (N, seed) training run per task.
# Usage (repo root):  sbatch --array=0-299 scripts/slurm/train_array.sh configs/e1.yaml "1 2 3 4 5 6" 50 [extra --set args]
#   task id -> N = Ns[id / SEEDS], seed = id % SEEDS
#SBATCH --job-name=ap-train
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
CONFIG=$1; NS=($2); SEEDS=$3; shift 3
N=${NS[$((SLURM_ARRAY_TASK_ID / SEEDS))]}
SEED=$((SLURM_ARRAY_TASK_ID % SEEDS))
module load miniconda; conda activate ap
NAME=$(basename "$CONFIG" .yaml)
mkdir -p results/parts logs
PYTHONPATH=src python -m ap.train "$CONFIG" --set n=$N threads=2 "$@" --seeds $SEED \
  --out results/parts/${NAME}_n${N}_s${SEED}.jsonl
