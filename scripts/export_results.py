"""Pack every result part into one compact gzip file for transfer off the cluster.

Keeps all metrics and attention statistics; downsamples training curves to at most 40
points per run. LLM rows (l1/l2/l6) are kept whole. Output: results/export.jsonl.gz
plus base64 chunks results/export_b64/part_XXX.txt (each <= 180 kB) for text transfer.

Usage: python scripts/export_results.py
"""
import base64
import glob
import gzip
import json
import os

import numpy as np

OUT = "results/export.jsonl.gz"


def slim(row):
    c = row.get("curves")
    if c and c.get("step"):
        idx = np.unique(np.linspace(0, len(c["step"]) - 1, min(40, len(c["step"]))).round().astype(int))
        row["curves"] = {k: [v[i] for i in idx] for k, v in c.items()}
    return row


def main():
    n = 0
    with gzip.open(OUT, "wt") as f:
        for p in sorted(glob.glob("results/parts/*.jsonl")) + sorted(glob.glob("results/parts/*.json")):
            src = os.path.basename(p)
            if p.endswith(".json"):
                f.write(json.dumps({"_src": src, "_l1": json.load(open(p))}) + "\n"); n += 1
                continue
            for line in open(p):
                line = line.strip()
                if line:
                    f.write(json.dumps({"_src": src, **slim(json.loads(line))}) + "\n"); n += 1
    data = base64.b64encode(open(OUT, "rb").read()).decode()
    os.makedirs("results/export_b64", exist_ok=True)
    for old in glob.glob("results/export_b64/*.txt"):
        os.remove(old)
    size = 180_000
    for i in range(0, len(data), size):
        open(f"results/export_b64/part_{i // size:03d}.txt", "w").write(data[i:i + size])
    print(f"{n} rows, {os.path.getsize(OUT) / 1e6:.1f} MB gzip, {len(data) // size + 1} chunks")


if __name__ == "__main__":
    main()
