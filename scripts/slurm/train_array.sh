#!/bin/bash
# CPU array job. Each task trains SEEDS_PER_TASK seeds for one N, sequentially.
#   sbatch --array=0-$((NNS*NBLOCKS-1)) scripts/slurm/train_array.sh CONFIG TAG "NS" SEEDS_PER_TASK NBLOCKS [--set ...]
#   task id -> N = NS[id / NBLOCKS], seeds = block*SPT .. block*SPT+SPT-1 with block = id % NBLOCKS
#SBATCH --job-name=ap
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
CONFIG=$1; TAG=$2; NS=($3); SPT=$4; NB=$5; shift 5
N=${NS[$((SLURM_ARRAY_TASK_ID / NB))]}
B=$((SLURM_ARRAY_TASK_ID % NB)); S0=$((B * SPT)); S1=$((S0 + SPT - 1))
module load miniconda; conda activate ap
mkdir -p results/parts logs
PYTHONPATH=src python -m ap.train "$CONFIG" --set n=$N threads=2 name=$TAG "$@" --seeds $S0-$S1 \
  --out results/parts/${TAG}_n${N}_b${B}.jsonl
