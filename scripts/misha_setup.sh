#!/bin/bash
# One-time setup on Misha (run on a login node from the repo root).
set -euo pipefail
module load miniconda
conda create -y -n ap python=3.11
conda activate ap
pip install -r requirements.txt
mkdir -p logs results/parts runs ~/project/hf_cache
PYTHONPATH=src python -m pytest -q tests
