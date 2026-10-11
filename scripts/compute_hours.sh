#!/bin/bash
# Compute used by this project on Misha (Slurm accounting): GPU-hours and CPU-core-hours by job name.
# Run on Misha:  bash scripts/compute_hours.sh [START_DATE]   (default 2026-09-01)
# Modal GPU time is not in Slurm; read it from the Modal dashboard (Usage, workspace daniel-gong) and add it.
START=${1:-2026-09-01}
sacct -X -n -P -S "$START" --format=JobName,ElapsedRaw,AllocTRES,State 2>/dev/null |
  awk -F'|' '$1 ~ /^ap/ && $4 !~ /PENDING/ {
    gpus = 0; cpus = 0
    n = split($3, t, ",")
    for (i = 1; i <= n; i++) {
      if (t[i] ~ /^gres\/gpu=/) { split(t[i], kv, "="); gpus = kv[2] }
      if (t[i] ~ /^cpu=/) { split(t[i], kv, "="); cpus = kv[2] }
    }
    name = $1
    gh[name] += gpus * $2 / 3600; ch[name] += cpus * $2 / 3600; jobs[name]++
    G += gpus * $2 / 3600; Cc += cpus * $2 / 3600
  }
  END {
    printf "%-26s %6s %10s %12s\n", "job name", "tasks", "GPU-hours", "CPU-core-h"
    for (k in jobs) printf "%-26s %6d %10.1f %12.1f\n", k, jobs[k], gh[k], ch[k]
    printf "%-26s %6s %10.1f %12.1f\n", "TOTAL", "", G, Cc
  }'
