"""LaTeX tables for the paper appendix, from results/summary.json.

Writes paper/tables/{l2_dprime,b1_cells,l7_l8}.tex
Usage: python scripts/paper_tables.py
"""
import json
import os

import numpy as np

S = json.load(open("results/summary.json"))
OUT = "paper/tables"
os.makedirs(OUT, exist_ok=True)
ORDER = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
         "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "gpt2", "gpt2-medium",
         "gpt2-large", "gpt2-xl", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"]
NAME = {m: m.split("/")[-1].replace("pythia", "Pythia").replace("gpt2", "GPT-2").replace("GPT-2-", "GPT-2 ")
        for m in ORDER}


def l2_table():
    rows = {(r["model"], r["n"]): r for r in S["l2"]
            if (r.get("format") or "lines") == "lines" and r["condition"] == "feedback" and r.get("revision") is None}
    lines = [r"\begin{tabular}{l" + "r" * 6 + "r}", r"\toprule",
             r"Model & $N{=}1$ & 2 & 3 & 4 & 5 & 6 & lure FA, $N{=}2$ \\", r"\midrule"]
    for m in ORDER:
        if (m, 1) not in rows:
            continue
        ds = [f"{rows[(m, n)]['dprime']:.2f}" if (m, n) in rows else "--" for n in range(1, 7)]
        r2 = rows.get((m, 2), {})
        near = [r2[k] for k in r2 if k.startswith("fa_lure_") and abs(int(k.split("_")[-1]) - 2) == 1]
        lf = f"{np.mean(near):.2f} / {r2['fa_nonlure']:.2f}" if near else "--"
        lines.append(f"{NAME[m]} & " + " & ".join(ds) + f" & {lf} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/l2_dprime.tex", "w").write("\n".join(lines) + "\n")


def b1_table():
    b = {(r["model"], r["n"], r["cond"]): r for r in S["b1"]}
    conds = [("sharpen-top3@0.67", r"sharpen"), ("ablate-top10", "ablate top"), ("ablate-random10", "ablate rand."),
             ("suppress@0.5", "suppress"), ("suppress-random@1", "supp. rand.")]
    lines = [r"\begin{tabular}{lrr" + "r" * len(conds) + "}", r"\toprule",
             "Model & $N$ & base $d'$ & " + " & ".join(c for _, c in conds) + r" \\", r"\midrule"]
    for m in ORDER:
        for n in (2, 3, 4):
            base = b.get((m, n, "baseline"))
            if not base:
                continue
            cells = []
            for c, _ in conds:
                r = b.get((m, n, c))
                if not r:
                    cells.append("--"); continue
                sig = r["delta_ci"][0] > 0 or r["delta_ci"][1] < 0
                v = f"{r['delta']:+.2f}"
                cells.append(r"\textbf{" + v + "}" if sig else v)
            flag = r"$^\dagger$" if base["dprime"] <= 0.3 else ""
            lines.append(f"{NAME[m] if n == 2 else ''} & {n} & {base['dprime']:.2f}{flag} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/b1_cells.tex", "w").write("\n".join(lines) + "\n")


def l7_l8_table():
    l7 = {(r["model"], r["n"]): r for r in S["l7"]}
    l8 = {(r["model"], r["n"]): r for r in S["l8"]}
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"& & \multicolumn{3}{c}{L7: share of misses} & \multicolumn{3}{c}{L8: match vs.\ lure} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Model & $N$ & elsewhere & lost later & head wrong & model AUC & probe AUC & FA called lure \\", r"\midrule"]
    for m in ORDER:
        for n in (2, 3):
            a, p = l7.get((m, n)), l8.get((m, n))
            if not a and not p:
                continue
            s = a["split_miss"] if a else None
            c1 = [f"{s[k]:.2f}" for k in ("mislocated", "located_probe_right", "located_probe_wrong")] if s else ["--"] * 3
            if p:
                best = max(((k, v) for k, v in p["sites"].items() if k.startswith("site")), key=lambda kv: kv[1]["auc"])
                c2 = [f"{p['model_auc']:.2f}", f"{best[1]['auc']:.2f}", f"{best[1]['lure_fa_called_lure']:.2f}"]
            else:
                c2 = ["--"] * 3
            lines.append(f"{NAME[m] if n == 2 else ''} & {n} & " + " & ".join(c1 + c2) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/l7_l8.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    l2_table(); b1_table(); l7_l8_table()
    print("wrote", os.listdir(OUT))
