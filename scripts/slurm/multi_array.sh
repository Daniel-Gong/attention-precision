#!/bin/bash
# CPU array for E10 (ap.train_multi): one seed per task.
#   sbatch --array=0-K scripts/slurm/multi_array.sh CONFIG TAG SEEDS_PER_SET [key=value ...]
#   task id -> set = id / SEEDS_PER_SET (index into NSETS if given), seed = id % SEEDS_PER_SET
#   NSETS env var (space-separated, e.g. "1 2 3") runs single-N controls: ns=[NSETS[set]]
#SBATCH --job-name=ap-multi
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
CONFIG=$1; TAG=$2; SPS=$3; shift 3
SEED=$((SLURM_ARRAY_TASK_ID % SPS)); SET=$((SLURM_ARRAY_TASK_ID / SPS))
EXTRA=()
if [ -n "${NSETS:-}" ]; then NS=($NSETS); N=${NS[$SET]}; EXTRA=("ns=[$N]"); TAG=${TAG}-n$N; fi
set +u; module load miniconda; conda activate ap; set -u
mkdir -p results/parts logs
PYTHONPATH=src python -m ap.train_multi "$CONFIG" --set name=$TAG threads=2 "${EXTRA[@]}" "$@" --seeds $SEED \
  --out results/parts/${TAG}_s${SEED}.jsonl
