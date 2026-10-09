#!/bin/bash
# One-time setup on Misha (run on a login node from the repo root).
set -euo pipefail
set +u; module load miniconda
conda create -y -n ap python=3.11
conda activate ap
pip install -r requirements.txt
# Misha's GPU driver supports CUDA 12.8: use the matching PyTorch build
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
mkdir -p logs results/parts runs ~/project/hf_cache
PYTHONPATH=src python -m pytest -q tests
