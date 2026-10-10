#!/bin/bash
# Re-run the E9 groups that failed in the first Phase 1 launch:
#   linear attention at state 128 and 512 (out of memory; fixed in code)
#   selective SSM at state 512 (time limit; now one seed per task)
# Partial selective-512 output is moved to results/partial/ so seeds are not duplicated.
set -euo pipefail
mkdir -p logs results/parts results/partial
mv results/parts/e9-selective-512_n*_b*.jsonl results/partial/ 2>/dev/null || true
A=scripts/slurm/train_array.sh
NS="1 2 3 4 5 6"
for st in 128 512; do
  sbatch --job-name=ap-e9-linattn-$st --array=0-5 $A configs/sweep.yaml e9-linattn-$st "$NS" 10 1 \
    model.family=linattn model.state=$st
done
sbatch --job-name=ap-e9-selective-512 --time=10:00:00 --array=0-59 $A configs/sweep.yaml e9-selective-512 "$NS" 1 10 \
  model.family=selective model.state=512
echo "submitted E9 reruns"
