#!/bin/bash
# L6: behaviour and head scores across Pythia training checkpoints (10 log-spaced steps).
#   sbatch --array=0-19 scripts/slurm/pythia_ckpts.sh      # 2 models x 10 checkpoints
#SBATCH --job-name=ap-l6
#SBATCH --partition=gpu
#SBATCH --qos=qos_yildirim
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
MODELS=(EleutherAI/pythia-160m EleutherAI/pythia-1.4b)
STEPS=(0 512 1000 2000 4000 8000 16000 33000 66000 143000)
M=${MODELS[$((SLURM_ARRAY_TASK_ID / 10))]}; S=${STEPS[$((SLURM_ARRAY_TASK_ID % 10))]}
module load miniconda; conda activate ap
export HF_HOME=${HF_HOME:-$HOME/project/hf_cache}
mkdir -p results/parts logs
TAG=$(echo "$M" | tr '/' '_')_step$S
PYTHONPATH=src python -m ap.llm.behave --model $M --revision step$S --ns 1 2 3 --conditions feedback \
  --out results/parts/l6_behave_${TAG}.jsonl
PYTHONPATH=src python -m ap.llm.heads --model $M --revision step$S --ns 1 2 3 --top 10 \
  --out results/parts/l6_heads_${TAG}.jsonl
