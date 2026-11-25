#!/usr/bin/env python3
"""Regenerate every number, table and figure in the paper from results/ (no number is typed by hand).

    python scripts/make_paper_assets.py --out paper_assets
Writes gen_macros.tex, gen_runs.tex, gen_extra.tex and figures/fig1.pdf into --out.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import check_predictions as cp
from toolcomp.horizon import rule_expected_utility

ORDER = ["primary_0p5b", "q35_0p8b", "q35_0p8b_seed20261120", "q35_0p8b_incremental", "q35_2b", "q35_4b", "q35_9b"]
NAME = {"primary_0p5b": "Qwen2.5-0.5B", "q35_0p8b": "Qwen3.5-0.8B", "q35_0p8b_seed20261120": "Qwen3.5-0.8B (fresh pool)",
        "q35_0p8b_incremental": "Qwen3.5-0.8B (hidden operands)", "q35_2b": "Qwen3.5-2B", "q35_4b": "Qwen3.5-4B", "q35_9b": "Qwen3.5-9B"}
TBL = {"primary_0p5b": "0.5B (Qwen2.5)", "q35_0p8b": "0.8B", "q35_0p8b_seed20261120": "0.8B fresh pool", "q35_0p8b_incremental": "0.8B hidden operands", "q35_2b": "2B", "q35_4b": "4B", "q35_9b": "9B"}
SHORT = {"primary_0p5b": "0.5B", "q35_0p8b": "0.8B", "q35_0p8b_seed20261120": "0.8B fresh", "q35_0p8b_incremental": "0.8B hidden", "q35_2b": "2B", "q35_4b": "4B", "q35_9b": "9B"}
COL = {"primary_0p5b": "#a23b2a", "q35_0p8b": "#d98324", "q35_0p8b_seed20261120": "#e3b341", "q35_0p8b_incremental": "#8a6d3b", "q35_2b": "#2a78d6", "q35_4b": "#1baf7a", "q35_9b": "#4a3aa7"}
M = {}


def macro(k, v):
    assert k.isalpha(), k
    M[k] = v


def f1(x):
    return f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper_assets"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    out, res = Path(a.out), Path(a.results)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    R = {t: json.loads((res / "scaling" / f"{t}.json").read_text()) for t in ORDER if (res / "scaling" / f"{t}.json").exists()}
    extra = sorted(p.stem for p in (res / "scaling").glob("*.json") if p.stem not in ORDER)
    C = {t: cp.cells(R[t]) for t in R}
    allc = [c for t in R for c in C[t]]
    S = json.loads((res / "predictions_scored.json").read_text()) if (res / "predictions_scored.json").exists() else None

    # ------------------------------------------------------------ headline statistics
    x = np.array([c["x"] for c in allc]); r = np.array([c["best_per_step_regret"] for c in allc]); pr = np.array([c["predicted_best_per_step"] for c in allc])
    dpf = np.array([c["dp_vs_label_free"] for c in allc]); dpa = np.array([c["dp_vs_all"] for c in allc])
    dpr = np.array([c["u"]["belief_dp"] - c["u"]["thr_retuned"] for c in allc])
    macro("NRuns", str(len(R)))
    macro("NCells", str(len(allc)))
    macro("RhoAll", f"{spearmanr(x, r)[0]:.2f}")
    macro("PearAll", f"{pearsonr(pr, r)[0]:.2f}")
    mid = (x >= 0.3) & (x < 3)
    macro("NMid", str(int(mid.sum())))
    macro("PearMid", f"{pearsonr(pr[mid], r[mid])[0]:.2f}")
    macro("MAEAll", f1(np.abs(pr - r).mean()))
    macro("MAEMid", f1(np.abs(pr - r)[mid].mean()))
    macro("DPFreeMax", f1(dpf.max()))
    macro("DPFreeLeOne", f"{100 * (dpf <= 1).mean():.0f}")
    macro("DPAllMax", f1(dpa.max()))
    macro("DPRetMean", f1(dpr.mean()))
    macro("DPRetMin", f1(dpr.min()))
    macro("DPRetBelow", f"{100 * (dpr < -1).mean():.0f}")
    macro("MaxBPS", f1(r.max()))
    low = x < 0.3
    macro("NLow", str(int(low.sum())))
    macro("LowMax", f"{r[low].max():.2f}")
    high = x > 1.5
    macro("NHigh", str(int(high.sum())))
    macro("HighFrac", f"{100 * (r[high] >= 8).mean():.0f}")
    macro("HighMed", f1(np.median(r[high])))
    if S:
        for key, nm in (("all_runs", "All"), ("new_runs_only", "New")):
            if S.get(key):
                s = S[key]
                macro(f"Rho{nm}", f"{s['P1']['spearman']:.2f}")
                macro(f"Pear{nm}", f"{s['P4']['pearson']:.2f}")
                macro(f"NCells{nm}", str(s["P1"]["n_cells"]))
        macro("PtwoHeld", "held" if S["new_runs_only"]["P2"]["held"] else "did not hold")
    # informative signals shrink the regret at fixed x (0.8B, T = 24)
    for sig, k in (("S0", "SigZero"), ("S4", "SigConf"), ("S0+S4", "SigFuse")):
        c = next(c for c in C["q35_0p8b"] if c["sig"] == sig and c["m"] == 8)
        macro(k, f1(c["best_per_step_regret"]))
    # pair counts
    for t, k in (("primary_0p5b", "A"), ("q35_0p8b", "B"), ("q35_2b", "C"), ("q35_4b", "D"), ("q35_9b", "E")):
        if t in R and R[t].get("second_differences"):
            macro(f"Comp{k}", str(R[t]["second_differences"]["complements"]))
            macro(f"Subs{k}", str(R[t]["second_differences"]["substitutes"]))
    # pre-registered Iteration-1 result and Iteration-1 ladder
    pre = json.loads((ROOT / "data" / "preregistered_0p5b.json").read_text())["contrasts"]
    for k, n in (("mup_vs_pp_step_direct", "PregA"), ("mup_vs_best_global_pattern", "PregB"), ("mup_vs_pp_first", "PregC")):
        macro(n, f"{pre[k]['diff']:+.2f}")
    macro("PregP", f"{max(v['p_holm'] for v in pre.values()):.4f}")
    # first horizon where best per-step regret exceeds 1 point, per run (S0)
    for t, k in (("q35_4b", "FourB"), ("q35_9b", "NineB"), ("q35_2b", "TwoB")):
        if t in R:
            cs = [c for c in C[t] if c["sig"] == "S0"]
            first = next((c for c in cs if c["best_per_step_regret"] > 1), None)
            macro(f"First{k}T", str(first["T"]) if first else "none")
            macro(f"First{k}x", f"{first['x']:.1f}" if first else "none")

    # ------------------------------------------------------------ Table: one row per run (signal S0, so every run is comparable)
    L = ["\\begin{tabular}{@{}lrrrrrr@{}}\\toprule", "Run & $x_{3}$ & $x_{24}$ & \\multicolumn{3}{c}{Regret at $T$=} & DP \\\\",
         " & & & 3 & 12 & 24 & gap \\\\ \\midrule"]
    for t in ORDER:
        if t not in R:
            continue
        cs = {c["m"]: c for c in C[t] if c["sig"] == "S0"}
        gap = max(c["dp_vs_label_free"] for c in cs.values())
        L.append(f"{TBL[t]} & {cs[1]['x']:.2f} & {cs[8]['x']:.1f} & {cs[1]['best_per_step_regret']:.1f} & {cs[4]['best_per_step_regret']:.1f} & {cs[8]['best_per_step_regret']:.1f} & {gap:.1f} \\\\")
    L.append("\\bottomrule\\end{tabular}")
    (out / "gen_runs.tex").write_text("\n".join(L))

    # ------------------------------------------------------------ deep runs (6-step arithmetic, MuSiQue), when present
    deep = {t: json.loads((res / "scaling" / f"{t}.json").read_text()) for t in extra if t.startswith(("arith6_", "musique_"))}
    if deep:
        L = ["\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule", "Run & $n$ & never & all calls & $x_{m=1}$ & \\multicolumn{3}{c}{Regret, $m$=} \\\\",
             " & & & & & 1 & 2 & 4 \\\\ \\midrule"]
        for t, d in sorted(deep.items()):
            cs = {c["m"]: c for c in cp.cells(d) if c["sig"] == "S0"}
            lab = t.replace("arith6_", "6-step arith., ").replace("musique_k", "MuSiQue $K$=").replace("_q35_", ", ").replace("_gpu_0p5b", ", 0.5B")
            lab = lab.replace("2b", "2B").replace("4b", "4B").replace("9b", "9B").replace("0p8b", "0.8B")
            L.append(f"{lab} & {d['n']} & {100 * d['never_accuracy']:.0f} & {100 * d['all_call_accuracy']:.0f} & {cs[1]['x']:.1f} & " + " & ".join(f"{cs[m]['best_per_step_regret']:.1f}" if m in cs else "--" for m in (1, 2, 4)) + " \\\\")
        L.append("\\bottomrule\\end{tabular}")
        (out / "gen_deep.tex").write_text("\n".join(L))
        macro("NDeep", str(len(deep)))
    # ------------------------------------------------------------ label efficiency summary (text + figure data)
    le = {}
    for f in sorted((res / "label_efficiency").glob("*.json")):
        d = json.loads(f.read_text())
        le[f.stem] = d
    names = {"q35_2b_S0_m4": "Two", "q35_0p8b_S0S4_m8": "Conf", "q35_0p8b_seed20261120_S0_m8": "Fresh"}
    for stem, k in names.items():
        if stem in le:
            d = le[stem]
            cur = d["curve"]
            ns = sorted(int(n) for n in cur)
            macro(f"LE{k}DP", f1(d["belief_dp"]))
            macro(f"LE{k}T", str(d["T"]))
            macro(f"LE{k}Full", f1(cur[str(ns[-1])]["mean"]))
            macro(f"LE{k}NFull", str(ns[-1]))
            cross = next((n for n in ns if cur[str(n)]["mean"] >= d["belief_dp"] - 0.5), None)
            macro(f"LE{k}Cross", str(cross) if cross else "none")
            macro(f"LE{k}Few", f1(cur[str(ns[min(range(len(ns)), key=lambda i: abs(ns[i] - 4))])]["mean"]))

    # ------------------------------------------------------------ figure 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    # (a) utility versus total price for a homogeneous chain
    ax = axs[0]
    T, p = 12, 0.9
    Cs = np.linspace(0.02, 2.0, 200)
    for rule, st, lab, col in (("optimal_open_loop", "-", "optimal", "#2a78d6"), ("optimistic", "--", "optimistic", "#eb6834"), ("pessimistic", ":", "pessimistic", "#4a3aa7")):
        ax.plot(Cs, [rule_expected_utility(rule, [p], T, c / T) for c in Cs], st, color=col, lw=1.5, label=lab)
    ax.set_xlabel("total price of calling every step, $Tc$")
    ax.set_ylabel("expected utility")
    ax.set_title(f"(a) chain with $T$={T}, $p$={p}", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    # (b) measured and predicted regret against x
    ax = axs[1]
    for t in ORDER:
        if t not in R:
            continue
        cs = C[t]
        ax.scatter([c["x"] for c in cs], [c["best_per_step_regret"] for c in cs], s=9, color=COL[t], label=SHORT[t], zorder=3, lw=0)
        ax.scatter([c["x"] for c in cs], [c["predicted_best_per_step"] for c in cs], s=11, facecolors="none", edgecolors=COL[t], lw=0.5, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("expected unaided errors $x$")
    ax.set_ylabel("per-step regret (pts)")
    ax.set_title("(b) measured (filled), predicted (open)", fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper left", handletextpad=0.1, columnspacing=0.6, labelspacing=0.2)
    # (c) label efficiency
    ax = axs[2]
    sty = {"q35_2b_S0_m4": ("#2a78d6", "2B, $T$=12"), "q35_0p8b_S0S4_m8": ("#d98324", "0.8B + conf., $T$=24"), "q35_0p8b_seed20261120_S0_m8": ("#e3b341", "0.8B fresh, $T$=24")}
    for stem, (col, lab) in sty.items():
        if stem not in le:
            continue
        d = le[stem]
        ns = sorted(int(n) for n in d["curve"])
        ax.errorbar(ns, [d["curve"][str(n)]["mean"] - d["belief_dp"] for n in ns], yerr=[d["curve"][str(n)]["sd"] for n in ns], color=col, marker="o", ms=2.5, lw=1.1, elinewidth=0.5, label=lab)
    ax.axhline(0, color="#0b0b0b", lw=0.8)
    ax.text(1.05, 0.5, "Belief-DP (0 labels)", fontsize=6, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("labelled long tasks used")
    ax.set_ylabel("re-tuned $-$ Belief-DP (pts)")
    ax.set_title("(c) label efficiency", fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.2, top=0.9, wspace=0.42)
    fig.savefig(out / "figures" / "fig1.pdf")
    fig.savefig(out / "figures" / "fig1.png", dpi=170)
    (out / "gen_macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
    print(f"{len(M)} macros, {len(R)} runs (+ extra: {extra}), {len(allc)} cells -> {out}")


if __name__ == "__main__":
    main()
