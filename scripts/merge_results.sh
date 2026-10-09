#!/bin/bash
# Merge per-task result parts into one file per experiment tag: results/<tag>.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
for tag in $(ls results/parts/*.jsonl | sed -E 's#.*/(.*)_n[0-9]+_b[0-9]+\.jsonl#\1#' | sort -u); do
  cat results/parts/${tag}_n*_b*.jsonl > results/$tag.jsonl; echo "$tag: $(wc -l < results/$tag.jsonl) runs"
done
