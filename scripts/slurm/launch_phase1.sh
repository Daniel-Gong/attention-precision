#!/bin/bash
# Submits every Phase 1 toy-model run (E1, E3, E4, E5, E8, E9 and the reviewer sweeps).
# 37 submissions, 246 array tasks, 2 CPUs each (fits the 512-CPU per-user limit on `day`).
# Run from the repo root on a Misha login node, after scripts/misha_setup.sh.
set -euo pipefail
A=scripts/slurm/train_array.sh
NS="1 2 3 4 5 6"
sub() { local tag=$1 cfg=$2 spt=$3 nb=$4; shift 4
  sbatch --job-name=ap-$tag --array=0-$((6 * nb - 1)) $A $cfg $tag "$NS" $spt $nb "$@" ; }

# E1: the paper's model (d = 512) to convergence, 50 seeds per N (5 blocks of 10)
sub e1 configs/e1.yaml 10 5 regime=fresh
# E1 reviewer sweeps, 10 seeds each: weight decay, training-set size, width
sub e1-wd01  configs/e1.yaml 10 1 weight_decay=0.01
sub e1-wd1   configs/e1.yaml 10 1 weight_decay=0.1
sub e1-n800  configs/e1.yaml 10 1 regime=fixed train_size=800
sub e1-n8k   configs/e1.yaml 10 1 regime=fixed train_size=8000
sub e1-n80k  configs/e1.yaml 10 1 regime=fixed train_size=80000
sub e1-d64   configs/e1.yaml 10 1 model.d_model=64
sub e1-d128  configs/e1.yaml 10 1 model.d_model=128
# E3: position encodings (d = 128)
for pe in learned sinusoidal rope alibi none; do sub e3-$pe configs/sweep.yaml 10 1 model.pe=$pe; done
# E4: block components, added one at a time
sub e4-res       configs/sweep.yaml 10 1 model.residual=true
sub e4-res-ffn   configs/sweep.yaml 10 1 model.residual=true model.ffn=true
sub e4-res-ffn-ln configs/sweep.yaml 10 1 model.residual=true model.ffn=true model.ln=true
# E5: attention normalisation, trained that way (test-time temperature runs later on E1 checkpoints)
for fn in sigmoid ssmax topk; do sub e5-$fn configs/sweep.yaml 10 1 model.attn_fn=$fn; done
# E8: sequence length (competitor count) x position encoding
for L in 48 96; do for pe in learned rope alibi; do
  sub e8-L$L-$pe configs/sweep.yaml 10 1 length=$L model.pe=$pe; done; done
# E9: non-attention baselines x state size
for fam in lstm lru selective linattn; do for st in 32 128 512; do
  sub e9-$fam-$st configs/sweep.yaml 10 1 model.family=$fam model.state=$st; done; done
echo "submitted; monitor with: squeue -u \$USER"
