#!/bin/bash
# GPU array job: one model (one line of configs/llm_models.txt) per task.
#   sbatch --array=0-13 scripts/slurm/llm_array.sh behave        # L1 tokenization + L2 behaviour
#   sbatch --array=0-13 scripts/slurm/llm_array.sh heads         # L3 localization
#   sbatch --array=0-13 scripts/slurm/llm_array.sh intervene     # L5 (needs results/l3.jsonl)
#SBATCH --job-name=ap-llm
#SBATCH --partition=gpu
#SBATCH --qos=qos_yildirim
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
STUDY=$1; shift
LINE=$(grep -v '^#' configs/llm_models.txt | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r MODEL DTYPE BATCH <<< "$LINE"
module load miniconda; conda activate ap
export HF_HOME=${HF_HOME:-$HOME/project/hf_cache}
mkdir -p results/parts logs
TAG=$(echo "$MODEL" | tr '/' '_')
case $STUDY in
  behave)
    PYTHONPATH=src python -m ap.llm.behave --model "$MODEL" --report-tokenization > results/parts/l1_${TAG}.json
    PYTHONPATH=src python -m ap.llm.behave --model "$MODEL" --dtype $DTYPE --batch $BATCH \
      --out results/parts/l2_${TAG}.jsonl --save-scores runs/l2_scores "$@" ;;
  heads)
    PYTHONPATH=src python -m ap.llm.heads --model "$MODEL" --dtype $DTYPE --ns 1 2 3 4 \
      --out results/parts/l3_${TAG}.jsonl "$@" ;;
  intervene)
    cat results/parts/l3_*.jsonl > results/l3.jsonl
    PYTHONPATH=src python -m ap.llm.intervene --model "$MODEL" --dtype $DTYPE --batch $BATCH --l3 results/l3.jsonl \
      --out results/parts/l5_${TAG}.jsonl "$@" ;;
esac
