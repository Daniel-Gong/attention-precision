#!/bin/bash
# One-screen status of every attention-precision job: Slurm (Misha) and Modal runs.
# Usage: bash scripts/status.sh            (once)
#        bash scripts/status.sh --loop     (rewrite logs/status.txt every 5 minutes)
cd "$(dirname "$0")/.."
report() {
  echo "== status at $(date '+%Y-%m-%d %H:%M')"
  echo "-- Slurm queue (tasks by job name and state)"
  squeue -u "$USER" -h -r -o "%j %T" 2>/dev/null | sort | uniq -c | awk '{printf "  %-24s %-9s %s\n",$2,$3,$1}'
  echo "-- Slurm finished since Oct 10 09:00 (non-completed states only)"
  sacct -X -S 2026-10-10T09:00 -n --format=JobName%26,State%14 2>/dev/null | grep -vE "COMPLETED|RUNNING|PENDING|ood-desk|CANCELLED" \
    | sort | uniq -c | awk '{printf "  %-26s %-14s %s\n",$2,$3,$1}'
  echo "-- Modal runs (tmux session, models finished, failures)"
  for f in logs/modal_*.out; do
    [ -e "$f" ] || continue
    s=$(basename "$f" .out); alive=$(tmux has-session -t "$s" 2>/dev/null && echo running || echo ended)
    done_n=$(grep -c ": exit " "$f"); bad=$(grep ": exit " "$f" | grep -vc ": exit 0,")
    printf "  %-12s %-8s finished %-3s nonzero-exit %s\n" "$s" "$alive" "$done_n" "$bad"
  done
  echo "-- Result files by experiment"
  ls results/parts 2>/dev/null | sed -E 's/_(n[0-9]+_b[0-9]+|s[0-9]+)\.jsonl$//; s/_(EleutherAI|gpt2|Qwen).*//' | sort | uniq -c \
    | awk '{printf "  %-22s %s\n",$2,$1}'
}
if [ "${1:-}" = "--loop" ]; then
  while true; do report > logs/status.tmp 2>&1 && mv logs/status.tmp logs/status.txt; sleep 300; done
else
  report
fi
