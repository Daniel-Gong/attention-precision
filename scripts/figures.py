"""Paper figures from results/summary.json (built by scripts/summarize_all.py).

  Fig 2  minimal models: the decline (E1), attention variants (E5), depth removes it (E3b, E10)
  Fig 3  LLM behaviour: d' versus N by family, lure false alarms, chat format, 14B
  Fig 4  locating versus readout: L7 error split, L8 probe by depth
  Fig 5  causal tests on one axis (B1 bootstrap) + Xiong et al. replication
  Fig 6  architectures (E9) and development (L6)

Usage: python scripts/figures.py [--summary results/summary.json] [--out figures]
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.stats import spearmanr

# reference categorical palette (dataviz skill), fixed order; text and grid tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAM = {"pythia": C[0], "gpt2": C[1], "qwen": C[2]}
SIZE = {"pythia-70m": 0.07, "pythia-160m": 0.16, "pythia-410m": 0.41, "pythia-1b": 1.0, "pythia-1.4b": 1.4,
        "pythia-2.8b": 2.8, "pythia-6.9b": 6.9, "gpt2": 0.124, "gpt2-medium": 0.355, "gpt2-large": 0.774,
        "gpt2-xl": 1.5, "Qwen2.5-0.5B": 0.5, "Qwen2.5-1.5B": 1.5, "Qwen2.5-7B": 7.0, "Qwen2.5-14B": 14.0}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7,
    "lines.linewidth": 1.5, "lines.markersize": 4, "savefig.dpi": 300, "figure.dpi": 150,
})


def short(m):
    return m.split("/")[-1]


def family(m):
    m = m.lower()
    return "pythia" if "pythia" in m else "qwen" if "qwen" in m else "gpt2"


def shade(hexc, t):
    """t in [0, 1]: light (0) to the full hue (1), for a size ramp inside one family."""
    r = np.array(to_rgb(hexc)); w = np.ones(3)
    return tuple(w + (r - w) * (0.35 + 0.65 * t))


def size_t(m, fam_models):
    s = np.log([SIZE[short(x)] for x in fam_models]); v = np.log(SIZE[short(m)])
    return 0.0 if s.max() == s.min() else (v - s.min()) / (s.max() - s.min())


def end_label(ax, x, y, text, color, dy=0):
    lo, hi = ax.get_ylim()                      # keep the anchor inside the axes so the label is not clipped
    y = min(max(y, lo + 0.06 * (hi - lo)), hi - 0.04 * (hi - lo))
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", fontsize=7, color=INK,
                annotation_clip=False)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight")
    plt.close(fig)


def toy_curve(S, tag, key="natural_dprime"):
    pts = sorted((r["n"], r[key][0], r[key][1]) for r in S["toy"] if r["tag"] == tag and key in r)
    return np.array(pts) if pts else None


def e10_curve(S, tag, key="natural_dprime"):
    pts = sorted((r["n"], r[key][0], r[key][1]) for r in S["e10"] if r["tag"] == tag and key in r)
    return np.array(pts) if pts else None


def series(ax, arr, color, label, marker="o", ls="-", direct=True, dy=0):
    ax.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], color=color, marker=marker, ls=ls, capsize=0,
                elinewidth=1, label=label, mec="white", mew=0.8)
    if direct:
        end_label(ax, arr[-1, 0], arr[-1, 1], label, color, dy)


def e1_checkpoints(S, e6_path="results/e6_slim.json"):
    """E1 checkpoints joined with their E6 logit decomposition: (n, d', delta_P, |mass gap|, attend-on-match)."""
    if not os.path.exists(e6_path):
        return None
    E = json.load(open(e6_path))
    rows = [(r["n"], r["dprime"], e["dP"], abs(e["tam"] - e["tanm"]), e["on"])
            for r in S["e1_seeds"] if (e := E.get(f"e1|{r['n']}|{r['seed']}"))]
    return np.array(rows, float) if rows else None


def spearman(x, y):
    return float(spearmanr(x, y)[0])


def fig2(S, out):
    ck = e1_checkpoints(S)
    fig, grid = plt.subplots(2 if ck is not None else 1, 3, figsize=(7.2, 4.7 if ck is not None else 2.3),
                             squeeze=False)
    axs = grid[0]
    ax = axs[0]
    e1 = toy_curve(S, "e1")
    series(ax, e1, C[0], "1-layer, softmax", direct=False)
    ax.set_title("a  The decline at convergence (E1)", loc="left")
    ax.set_xlabel("N"); ax.set_ylabel("d′ (test, 50 seeds, ± SEM)"); ax.set_ylim(0, 3.2)
    ax = axs[1]
    for i, (tag, lab, dy) in enumerate([("e1", "softmax", -5), ("e5-ssmax", "scalable softmax", 5),
                                        ("e5-topk", "top-k", -2), ("e5-sigmoid", "sigmoid", 4)]):
        a = toy_curve(S, tag)
        if a is not None:
            series(ax, a, C[i], lab, dy=dy)
    ax.set_title("b  Normalization sets the slope (E5)", loc="left")
    ax.set_xlabel("N"); ax.set_ylim(0, 4.2); ax.set_xlim(0.7, 7.6)
    ax = axs[2]
    rows = [(toy_curve(S, "e1"), "1 layer, no residual", C[0]), (toy_curve(S, "e3b-learned"), "2 layers + residual", C[1]),
            (e10_curve(S, "e10-full"), "2 layers, all N (cued)", C[2])]
    for a, lab, col in rows:
        if a is not None:
            series(ax, a, col, lab)
    ax.set_title("c  Depth removes it (E3b, E10)", loc="left")
    ax.set_xlabel("N"); ax.set_ylim(0, 8.5); ax.set_xlim(0.7, 8.2)
    for ax in axs:
        ax.set_xticks(range(1, 7))
    if ck is not None:
        lo, hi = np.array(to_rgb("#b3cdee")), np.array(to_rgb("#0f3b78"))   # one hue, light (N = 1) to dark (N = 6)
        nshade = lambda n: tuple(lo + (hi - lo) * (n - 1) / 5)
        ncol = [nshade(n) for n in ck[:, 0]]
        for ax, col, lab, letter, title in (
                (grid[1, 0], 2, r"position margin $\delta_P$ (logit)", "d", "Position margin"),
                (grid[1, 1], 3, "|target attention, match − non-match|", "e", "Mass gap")):
            ax.scatter(ck[:, col], ck[:, 1], s=9, c=ncol, linewidths=0.3, edgecolors="white", zorder=3)
            rho = f"{spearman(ck[:, col], ck[:, 1]):.2f}".replace("-", "−")
            ax.set_title(f"{letter}  {title} (ρ = {rho})", loc="left")
            ax.set_xlabel(lab)
        grid[1, 0].set_ylabel("d′ (one checkpoint)")
        for n in range(1, 7):
            grid[1, 1].scatter([], [], s=9, color=nshade(n), label=f"N = {n}")
        grid[1, 1].legend(loc="upper left", fontsize=6.5, handletextpad=0.1, borderaxespad=0.2, ncol=2,
                          columnspacing=0.6)
        ax = grid[1, 2]
        ns = np.arange(1, 7)
        share = [ck[ck[:, 0] == n, 4].mean() for n in ns]
        ax.bar(ns, share, 0.62, color=[nshade(n) for n in ns])
        ax.set_ylim(0, 1); ax.set_xticks(ns); ax.set_xlabel("N"); ax.set_ylabel("share of seeds")
        ax.set_title("f  Seeds attending to the match", loc="left"); ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    save(fig, out, "fig2_minimal_models")


def l2_rows(S, fmt="lines", cond="feedback"):
    rows = {(r["model"], r["n"]): r for r in S["l2"]          # later rows win (re-run duplicates)
            if (r.get("format") or "lines") == fmt and r["condition"] == cond and r.get("revision") is None}
    return list(rows.values())


def fig3(S, out):
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.25), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    rows = l2_rows(S)
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append((r["n"], r["dprime"]))
    fams = defaultdict(list)
    for m in by:
        if "Instruct" not in m:
            fams[family(m)].append(m)
    titles = {"pythia": "a  Pythia 70M–6.9B", "gpt2": "b  GPT-2 124M–1.5B", "qwen": "c  Qwen2.5 0.5B–14B"}
    for ax, fam in zip(axs[:3], ["pythia", "gpt2", "qwen"]):
        ms = sorted(fams[fam], key=lambda m: SIZE[short(m)])
        for m in ms:
            a = np.array(sorted(by[m]))
            col = shade(FAM[fam], size_t(m, ms))
            ax.plot(a[:, 0], a[:, 1], color=col, marker="o", mec="white", mew=0.6)
        ax.set_ylim(0, 4.5); ax.set_xlim(0.7, 7.3)
        for m, dy in ((ms[0], -2), (ms[-1], 6)):
            a = np.array(sorted(by[m]))
            lab = short(m).replace("Qwen2.5-", "").replace("pythia-", "").upper().replace("GPT2-XL", "XL").replace("GPT2", "124M")
            end_label(ax, a[-1, 0], a[-1, 1], lab, None, dy)
        ax.text(0.98, 0.97, "darker = larger", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
        ax.set_title(titles[fam], loc="left"); ax.set_xlabel("N"); ax.set_xticks(range(1, 7))
    axs[0].set_ylabel("d′ (feedback condition)")
    ax = axs[3]
    lure, non = defaultdict(list), defaultdict(list)
    for r in rows:
        if "Instruct" in r["model"]:
            continue
        ks = [k for k in r if k.startswith("fa_lure_")]
        near = [r[k] for k in ks if abs(int(k.split("_")[-1]) - r["n"]) == 1]
        if near and r.get("fa_nonlure") is not None:
            lure[r["n"]].append(np.mean(near)); non[r["n"]].append(r["fa_nonlure"])
    ns = sorted(n for n in lure if n >= 2)
    x = np.arange(len(ns)); w = 0.36
    ml = [np.median(lure[n]) for n in ns]; mn = [np.median(non[n]) for n in ns]
    ax.bar(x - w / 2 - 0.01, ml, w, color=C[7], label="lure at N ± 1")
    ax.bar(x + w / 2 + 0.01, mn, w, color="#b8b7b1", label="no lure")
    for xi, n in zip(x, ns):
        ax.scatter(np.full(len(lure[n]), xi - w / 2) + np.random.default_rng(n).uniform(-0.08, 0.08, len(lure[n])),
                   lure[n], s=5, color=INK2, zorder=3, linewidths=0)
    ax.set_xticks(x, [str(n) for n in ns]); ax.set_xlabel("N"); ax.set_ylabel("false-alarm rate")
    ax.set_title("d  Lures drive false alarms", loc="left")
    ax.legend(loc="upper right", handlelength=1, bbox_to_anchor=(1.02, 1.0), fontsize=6.5)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig3_llm_behaviour")


def fig4(S, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})
    ax = axs[0]
    parts = [("located_probe_wrong", "attended, head output wrong", C[7]),
             ("located_probe_right", "attended, lost downstream", C[3]),
             ("mislocated", "attention elsewhere", "#b8b7b1")]
    ns = [2, 3, 4]
    dedup = {(r["model"], r["n"]): r for r in S["l7"]}          # later rows win (reused-container duplicates)
    good = [r for r in dedup.values() if "gpt2-large" not in r["model"] and r.get("split_miss")
            and r["split_miss"]["count"] < 0.9 * r["n_items"] * 0.3]
    bottom = np.zeros(len(ns))
    for key, lab, col in parts:
        v = np.array([np.median([r["split_miss"][key] for r in good if r["n"] == n]) for n in ns])
        ax.bar(ns, v, 0.62, bottom=bottom, color=col, label=lab, edgecolor="white", linewidth=1)
        bottom += v
    ax.set_ylim(0, 1); ax.set_xticks(ns); ax.set_xlabel("N"); ax.set_ylabel("share of misses (median model)")
    ax.set_title("a  Misses: attention is in place (L7)", loc="left"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, fontsize=6.5, handlelength=1)
    for ax, n, letter in ((axs[1], 2, "b"), (axs[2], 3, "c")):
        rows = [r for r in S["l8"] if r["n"] == n]
        grid = np.linspace(0, 1, 9)
        cur, fa = [], []
        for r in rows:
            sites = sorted((int(k[4:]) / r["n_layers"], v) for k, v in r["sites"].items() if k.startswith("site"))
            d = np.array([s for s, _ in sites])
            cur.append(np.interp(grid, d, [v["auc"] for _, v in sites]))
            fa.append(np.interp(grid, d, [v["lure_fa_called_lure"] or 0 for _, v in sites]))
            ax.plot(d, [v["auc"] for _, v in sites], color=shade(C[0], 0.15), lw=0.8, zorder=1)
        cur, fa = np.array(cur), np.array(fa)
        ax.plot(grid, np.median(cur, 0), color=C[0], lw=2, label="probe: match vs lure (AUC)", zorder=3)
        ax.plot(grid, np.median(fa, 0), color=C[1], lw=2, ls="--", label="model's lure false alarms\nthe probe calls lures", zorder=3)
        mauc = np.median([r["model_auc"] for r in rows])
        ax.axhline(mauc, color=INK2, lw=1.2, ls=":", zorder=2)
        ax.annotate(f"model's own answer, AUC {mauc:.2f}", (1.0, mauc), xytext=(0, -8), textcoords="offset points",
                    ha="right", fontsize=6.5, color=INK2)
        ax.set_ylim(0.0, 1.03); ax.set_xlabel("relative depth (0 = input, 1 = last layer)")
        ax.set_title(f"{letter}  Residual-stream probe, N = {n} (L8)", loc="left")
        if n == 2:
            ax.set_ylabel("AUC / share of items")
    fig.tight_layout(w_pad=1.0)
    # b/c legend sits just under their x labels, level with panel a's legend
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    lab = inv.transform(axs[1].xaxis.label.get_window_extent())
    x0, x1 = axs[1].get_position().x0, axs[2].get_position().x1
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, [x.replace("\n", " ") for x in l], loc="upper center", bbox_to_anchor=((x0 + x1) / 2, lab[0, 1] - 0.01),
               ncol=2, fontsize=6.5)
    save(fig, out, "fig4_locate_vs_readout")


def fig5(S, out):
    b = [r for r in S["b1"]]
    base = {(r["model"], r["n"]): r["dprime"] for r in b if r["cond"] == "baseline"}
    ok = {k for k, v in base.items() if v > 0.3}
    conds = [("sharpen-top3@0.8", "Sharpen top-3 heads, τ = 0.8", C[0]),
             ("sharpen-top3@0.67", "Sharpen top-3 heads, τ = 0.67", C[0]),
             ("ablate-random10", "Ablate 10 random heads", "#8d8c86"),
             ("ablate-top10", "Ablate top-10 heads", C[7]),
             ("suppress-random@1", "Suppress random subspace", "#8d8c86"),
             ("xiong", "Xiong et al. intervention (held-out)", C[3]),
             ("suppress@1", "Suppress distractor content, full", C[2]),
             ("suppress@0.5", "Suppress distractor content, half", C[2])]
    xi = defaultdict(dict)
    for r in S["l5s"]:
        if r["cond"] in ("xiong_baseline", "xiong_heldout"):
            xi[(r["model"], r["n"])][r["cond"]] = r["dprime"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.75, 1]})
    rng = np.random.default_rng(0)
    for y, (c, lab, col) in enumerate(conds):
        if c == "xiong":
            d = np.array([v["xiong_heldout"] - v["xiong_baseline"] for k, v in xi.items()
                          if len(v) == 2 and k in ok])
            lo = hi = d
        else:
            rs = [r for r in b if r["cond"] == c and (r["model"], r["n"]) in ok]
            d = np.array([r["delta"] for r in rs]); lo = np.array([r["delta_ci"][0] for r in rs])
            hi = np.array([r["delta_ci"][1] for r in rs])
        yy = y + rng.uniform(-0.22, 0.22, len(d))
        ax.hlines(yy, lo, hi, color=col, lw=0.7, alpha=0.6)
        ax.scatter(d, yy, s=9, color=col, linewidths=0, zorder=3, alpha=0.85)
        med = np.median(d)
        ax.plot([med, med], [y - 0.34, y + 0.34], color=INK, lw=1.8, zorder=4)
        ax.text(2.2, y, f"median {med:+.2f}".replace("-", "−"), va="center", fontsize=7, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(conds)), [l for _, l, _ in conds]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("change in d′ from baseline\n(dot = one model × N, bar = 95% bootstrap CI, black line = median)")
    ax.set_xlim(-2.4, 2.1)
    ax.set_title("a  Causal tests, 15 LLMs × N = 2–4", loc="left", fontsize=8.5)
    # b: per-model median over N = 2-4 against size; the head account weakens with scale, interference does not
    models = sorted({m for m, _ in ok}, key=lambda m: SIZE[short(m)])
    for c, lab, col, dy in (("suppress@0.5", "suppress distractors", C[2], 0), ("ablate-top10", "ablate top-10 heads", C[7], 0)):
        pts, fams = [], []
        for m in models:
            d = [r["delta"] for r in b if r["cond"] == c and r["model"] == m and (m, r["n"]) in ok]
            if d:
                pts.append((SIZE[short(m)], np.median(d))); fams.append(family(m))
        pts, fams = np.array(pts), np.array(fams)
        for fam, mk in (("pythia", "o"), ("gpt2", "s"), ("qwen", "^")):
            sel = fams == fam
            bx.scatter(pts[sel, 0], pts[sel, 1], s=16, marker=mk, color=col, edgecolors="white", linewidths=0.5, zorder=3)
        k = np.polyfit(np.log10(pts[:, 0]), pts[:, 1], 1)
        xs = np.array([pts[:, 0].min(), pts[:, 0].max()])
        bx.plot(xs, np.polyval(k, np.log10(xs)), color=col, lw=1.2, alpha=0.8)
        rho = f"{spearman(pts[:, 0], pts[:, 1]):.2f}".replace("-", "−")
        bx.annotate(f"{lab}\nρ = {rho}", (xs[-1], np.polyval(k, np.log10(xs[-1]))), xytext=(5, dy),
                    textcoords="offset points", va="center", fontsize=6.5, color=INK, annotation_clip=False)
    for fam, mk in (("pythia", "o"), ("gpt2", "s"), ("qwen", "^")):
        bx.scatter([], [], marker=mk, s=16, color=INK2, label={"pythia": "Pythia", "gpt2": "GPT-2", "qwen": "Qwen2.5"}[fam])
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6.5, handletextpad=0.1, columnspacing=0.8)
    bx.axhline(0, color=INK2, lw=1)
    bx.set_xscale("log"); bx.set_xlim(0.05, 20)
    bx.set_xlabel("parameters (billions)"); bx.set_ylabel("median change in d′ (N = 2–4)")
    bx.set_title("b  Scale weakens the heads, not interference", loc="left", fontsize=8.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig5_causal_tests")


def fig6(S, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    fams = [("lstm", "LSTM", C[0]), ("lru", "LRU", C[1]), ("linattn", "linear attention", C[2]), ("selective", "selective SSM", C[3])]
    for ax, st, letter in ((axs[0], 32, "a"), (axs[1], 512, "b")):
        for f, lab, col in fams:
            a = toy_curve(S, f"e9-{f}-{st}")
            if a is not None:
                series(ax, a, col, lab, direct=False)
        a = toy_curve(S, "e1")
        series(ax, a, "#8d8c86", "1-layer attention", ls="--", direct=False)
        ax.set_title(f"{letter}  State size {st} (E9)", loc="left"); ax.set_xlabel("N"); ax.set_xticks(range(1, 7))
    axs[0].set_ylabel("d′ (test, ± SEM)")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.36, -0.08), ncol=5, fontsize=6.5)
    ax = axs[2]
    rows = [r for r in S["l6_behave"] if r["condition"] == "feedback" and r["n"] == 2]
    for i, m in enumerate(sorted({r["model"] for r in rows}, key=lambda m: SIZE[short(m)])):
        pts = sorted((int(r["revision"][4:]), r["dprime"]) for r in rows if r["model"] == m)
        a = np.array(pts); a[:, 0] = np.maximum(a[:, 0], 100)
        ax.plot(a[:, 0], a[:, 1], color=C[i], marker="o", mec="white", mew=0.6, label=short(m))
        end_label(ax, a[-1, 0], a[-1, 1], short(m).replace("pythia-", "Pythia-"), None)
    ax.set_xscale("log"); ax.set_xlabel("training step (step 0 shown at 10²)"); ax.set_xlim(80, 6e5)
    ax.set_title("c  N-back emerges early in Pythia (L6, N = 2)", loc="left")
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig6_architectures_development")


def tint(hexc, a):
    """Mix a colour with white: a = 0 is white, a = 1 the full hue."""
    return tuple(1 - a + a * np.array(to_rgb(hexc)))


def fig1(S, out):
    """Schematic: the task, the two accounts drawn as mechanisms, and the causal answer (B1 cells)."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    TGT, TGT_D, LURE = C[3], "#9a6800", C[7]                   # target amber (dark variant for text), lure red
    PREC, PREC_D, INTF, INTF_D = C[0], "#1d5aa3", C[2], "#0d7a52"  # precision / interference accounts (as in Fig 5)
    PAPER = "#f1f0ec"
    W, H = 7.2, 2.62
    fig = plt.figure(figsize=(W, H))
    cv = fig.add_axes([0, 0, 1, 1]); cv.axis("off")
    cv.set_xlim(0, 72); cv.set_ylim(0, H * 10); cv.set_aspect("equal")   # 1 unit = 0.1 inch

    def box(x, y, w, h, fc, ec="none", lw=0.0, r=0.35, ls="-", z=1):
        cv.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                    ls=ls, zorder=z))

    def arrow(p, q, col, rad=0.0, ls="-", lw=1.0, ms=7, z=3):
        cv.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", color=col, lw=lw,
                                     ls=ls, mutation_scale=ms, shrinkA=0, shrinkB=0, zorder=z))

    def title(x, s):
        cv.text(x, H * 10 - 1.0, s, fontsize=8.5, color=INK, va="center", weight="bold")

    letters = ["k", "t", "k", "r", "t", "q", "r"]
    LI, TI, QI = 3, 4, 6                                       # lure, target, query indices

    def tile_style(j):
        if j == TI:
            return tint(TGT, 0.30), TGT, 0.9
        if j == LI:
            return tint(LURE, 0.22), LURE, 0.9
        if j == QI:
            return "white", INK, 1.2
        return PAPER, "none", 0.0

    # ---- a: the task -------------------------------------------------------------------------------------------
    title(0.2, "a  Task: 2-back with a lure")
    w, step, x0, y0, h = 2.25, 2.62, 0.5, 12.6, 3.0
    cx = [x0 + j * step + w / 2 for j in range(7)]
    for j, c in enumerate(letters):
        fc, ec, lw = tile_style(j)
        box(x0 + j * step, y0, w, h, fc, ec, lw, r=0.45)
        cv.text(cx[j], y0 + h / 2, c, ha="center", va="center", fontsize=12, family="DejaVu Sans Mono",
                color=INK, zorder=4)
    top, bot = y0 + h + 0.3, y0 - 0.3                         # lure arc above the tiles, target arc below
    arrow((cx[QI] - 0.2, top), (cx[LI], top), LURE, rad=0.42, lw=1.1, ls=(0, (3, 2)))
    arrow((cx[QI] - 0.2, bot), (cx[TI], bot), TGT_D, rad=-0.55, lw=1.3)
    cv.text((cx[QI] + cx[LI]) / 2 - 0.1, top + 2.65, "lure: same letter, 3 back", ha="center", va="center",
            fontsize=6.5, color=LURE, weight="bold")
    cv.text((cx[QI] + cx[TI]) / 2 - 0.1, bot - 2.35, "target: 2 back", ha="center", va="center", fontsize=6.5,
            color=TGT_D, weight="bold")
    cv.text(cx[QI] + 0.15, top + 0.75, "query", ha="left", va="center", fontsize=6, color=INK2)
    cv.text(9.9, 6.3, "Does the query match the item 2 back?", ha="center", va="center", fontsize=6.5, color=INK)
    cv.text(9.9, 4.6, "No: “r” ≠ “t”. Answering “match” because", ha="center", va="center", fontsize=6.5, color=INK2)
    cv.text(9.9, 3.3, "of the 3-back “r” is a lure false alarm.", ha="center", va="center", fontsize=6.5, color=INK2)

    # ---- b: two accounts, drawn as mechanisms ----------------------------------------------------------------
    bx0, bx1 = 21.6, 53.2
    title(bx0, "b  Two accounts of the same false alarm")
    rows = [  # (card top, header, account, fix, colour, dark, attention over positions 1-6, read-out, ghosts, note)
        (24.4, "Locating failure", "precision", "sharpen attention", PREC, PREC_D,
         [0.03, 0.05, 0.07, 0.42, 0.30, 0.13], ("r", LURE), [("t", TGT, 0.8)], "attention spreads;\nmostly the lure is read"),
        (12.4, "Readout failure", "interference", "remove distractors", INTF, INTF_D,
         [0.01, 0.02, 0.02, 0.04, 0.88, 0.03], ("t", TGT_D), [("r", LURE, 0.9), ("k", INK2, 0.75), ("q", INK2, 0.75)],
         "target is read, but other\nitems' content interferes"),
    ]
    for ytop, head, acct, fix, col, dark, att, main_l, ghosts, note in rows:
        ch = 11.2
        box(bx0, ytop - ch, bx1 - bx0, ch, tint(col, 0.07), tint(col, 0.45), 0.6, r=0.5, z=0)
        cv.text(bx0 + 0.6, ytop - 1.15, head, fontsize=7.5, weight="bold", color=dark, va="center")
        cv.text(bx0 + 0.6 + 0.62 * len(head) + 0.4, ytop - 1.15, f"({acct} account)", fontsize=6.5, color=INK2,
                va="center")
        # mini sequence with the query's attention over it
        mw, ms, mx0, my0, mh = 1.3, 1.55, bx0 + 0.9, ytop - ch + 1.1, 1.75
        mc = [mx0 + j * ms + mw / 2 for j in range(7)]
        for j, c in enumerate(letters):
            fc, ec, lw = tile_style(j)
            box(mx0 + j * ms, my0, mw, mh, fc, ec, lw * 0.8, r=0.25)
            cv.text(mc[j], my0 + mh / 2, c, ha="center", va="center", fontsize=7.5, family="DejaVu Sans Mono",
                    color=INK, zorder=4)
        base, amax = my0 + mh + 0.35, 5.3
        cv.plot([mx0 - 0.2, mc[5] + mw / 2 + 0.2], [base, base], color=INK2, lw=0.6, zorder=2, solid_capstyle="butt")
        for j, a in enumerate(att):
            hh = max(a / max(att) * amax * (0.95 if max(att) < 0.5 else 1.0), 0.12)
            cv.add_patch(plt.Rectangle((mc[j] - 0.5, base), 1.0, hh, fc=col, ec="none", zorder=3))
        # read-out: what reaches the decision
        ox, oy, R = 37.4, my0 + 3.6, 2.15
        cv.text(mc[QI] + 0.2, oy + 0.9, "attention\nfrom query", ha="center", va="bottom", fontsize=6.2,
                color=INK2, linespacing=1.0)
        arrow((mc[QI] - 0.3, oy), (ox - R - 0.35, oy), INK2, lw=0.9, ms=6)
        cv.add_patch(Circle((ox, oy), R, fc="white", ec=tint(col, 0.6), lw=0.9, zorder=3))
        cv.text(ox, oy, main_l[0], ha="center", va="center", fontsize=13, family="DejaVu Sans Mono", weight="bold",
                color=main_l[1], zorder=5)
        spots = [(-1.15, 0.95), (1.15, 0.95), (1.15, -0.95), (-1.15, -0.95)]
        for (g, gc, al), (dx, dy) in zip(ghosts, spots):
            cv.text(ox + dx, oy + dy, g, ha="center", va="center", fontsize=7.5, family="DejaVu Sans Mono",
                    color=gc, alpha=al, zorder=4)
        cv.text(ox, oy - R - 0.75, "read out", ha="center", va="center", fontsize=6.2, color=INK2)
        # decision, and the fix this account implies
        dx0 = ox + R + 0.35
        arrow((dx0, oy), (dx0 + 1.9, oy), INK2, lw=0.9, ms=6)
        cv.text(dx0 + 2.3, oy + 0.6, "“match”", ha="left", va="center", fontsize=7.5, color=INK, weight="bold")
        cv.text(dx0 + 7.3, oy + 0.6, "✗", ha="left", va="center", fontsize=9, color=LURE, weight="bold")
        cv.text(dx0 + 2.3, oy - 1.2, note, ha="left", va="center", fontsize=6.2, color=INK2, linespacing=1.05)
        fw = 0.52 * len("fix: " + fix) + 1.0
        box(bx1 - 0.6 - fw, my0 - 0.35, fw, 1.75, tint(col, 0.22), "none", r=0.85, z=2)
        cv.text(bx1 - 0.6 - fw / 2, my0 + 0.53, f"fix: {fix}", fontsize=6.5, color=dark, va="center", ha="center",
                weight="bold", zorder=4)

    # ---- c: which fix works (B1 cells) ------------------------------------------------------------------------
    title(55.6, "c  Which fix works?")
    cv.text(55.6, H * 10 - 2.6, "15 LLMs × N = 2–4; dot = cell, bar = median", fontsize=6.2, color=INK2, va="center")
    b = S["b1"]; base_d = {(r["model"], r["n"]): r["dprime"] for r in b if r["cond"] == "baseline"}
    ok = {k for k, v in base_d.items() if v > 0.3}
    ax = fig.add_axes([0.825, 0.16, 0.165, 0.66])
    rng = np.random.default_rng(1)
    for x, (c, col, dark) in enumerate((("sharpen-top3@0.67", PREC, PREC_D), ("suppress@0.5", INTF, INTF_D))):
        d = np.array([r["delta"] for r in b if r["cond"] == c and (r["model"], r["n"]) in ok])
        med = np.median(d)
        ax.bar(x, med, 0.62, color=tint(col, 0.28), zorder=1)
        ax.scatter(x + rng.uniform(-0.2, 0.2, len(d)), d, s=6, color=col, linewidths=0, alpha=0.9, zorder=3)
        ax.plot([x - 0.31, x + 0.31], [med, med], color=dark, lw=1.6, zorder=4, solid_capstyle="butt")
        ax.text(x, max(d.max(), med) + 0.1, f"{med:+.2f}", ha="center", va="bottom", fontsize=7.5, color=dark,
                weight="bold")
        if c.startswith("suppress"):
            ax.text(x + 0.44, med, f"{(d > 0).sum()}/{len(d)}\ncells\nimprove", ha="left", va="center", fontsize=6.2,
                    color=INK2, linespacing=1.0)
    ax.axhline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_xticks([0, 1], ["sharpen\nattention", "remove\ndistractors"], fontsize=6.5)
    for t, dark in zip(ax.get_xticklabels(), (PREC_D, INTF_D)):
        t.set_color(dark)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(-0.45, 2.15); ax.set_yticks([0, 1, 2])
    ax.grid(axis="x", visible=False); ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("change in d′", fontsize=7, labelpad=1)
    save(fig, out, "fig1_overview")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary.json")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    S = json.load(open(a.summary))
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f(S, a.out)
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
