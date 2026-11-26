#!/usr/bin/env python3
"""Regenerate every number, table and figure in the paper from results/ (no number is typed by hand).

    python scripts/make_paper_assets.py --out paper_assets
Writes gen_macros.tex, gen_runs.tex, gen_preds.tex, gen_six.tex and figures/fig1.pdf into --out.
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

FRESH5 = ["gpu_0p5b_seed20261201", "q35_0p8b_seed20261201", "q35_2b_seed20261201", "q35_4b_seed20261201", "q35_9b_seed20261201"]       # one pool (generator seed 20261201), five models
ORDER = ["primary_0p5b", "gpu_0p5b_seed20261201", "q35_0p8b", "q35_0p8b_seed20261120", "q35_0p8b_seed20261201", "q35_0p8b_incremental", "q35_2b", "q35_4b", "q35_9b",
         "q35_2b_seed20261201", "q35_4b_seed20261201", "q35_9b_seed20261201"]                                                              # every 3-step run
SIX = ["arith6_gpu_0p5b", "arith6_q35_0p8b", "arith6_q35_2b", "arith6_q35_4b", "arith6_q35_9b"]                                            # 6-step chains, committed prices
SIXM = [t + "_matched" for t in SIX]                                                                                                       # the same at half the per-call price
ROW = {"gpu_0p5b_seed20261201": "0.5B", "q35_0p8b_seed20261201": "0.8B", "q35_2b_seed20261201": "2B", "q35_4b_seed20261201": "4B", "q35_9b_seed20261201": "9B"}


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
    R = {t: json.loads((res / "scaling" / f"{t}.json").read_text()) for t in ORDER + SIX if (res / "scaling" / f"{t}.json").exists()}
    extra = sorted(p.stem for p in (res / "scaling").glob("*.json") if p.stem not in R)
    C = {t: cp.cells(R[t]) for t in R}
    allc = [c for t in R for c in C[t]]
    S = json.loads((res / "predictions_scored.json").read_text()) if (res / "predictions_scored.json").exists() else None

    # ------------------------------------------------------------ headline statistics
    x = np.array([c["x"] for c in allc]); r = np.array([c["best_per_step_regret"] for c in allc]); pr = np.array([c["predicted_best_per_step"] for c in allc])
    dpf = np.array([c["dp_vs_label_free"] for c in allc]); dpa = np.array([c["dp_vs_all"] for c in allc])
    dpr = np.array([c["u"]["belief_dp"] - c["u"]["thr_retuned"] for c in allc])
    macro("NRuns", str(len(R)))
    macro("NThree", str(len([t for t in R if t in ORDER])))
    macro("NSix", str(len([t for t in R if t in SIX])))
    full = [t for t in R if t in ORDER and "S1" in R[t]["runs"]["partition_0"]]
    macro("NThreeAll", str(len(full)))
    macro("NThreeSurf", str(len([t for t in R if t in ORDER]) - len(full)))
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
    # prediction accuracy: closed form (signal-blind) against signal-aware (this study's second prediction set), on every arithmetic cell
    sa = np.array([c["signal_aware"] for c in allc]); inf = np.array([c["sig"] != "S0" for c in allc])
    macro("PearSA", f"{pearsonr(sa, r)[0]:.2f}")
    macro("MAESA", f1(np.abs(sa - r).mean()))
    macro("PearZero", f"{pearsonr(pr[~inf], r[~inf])[0]:.2f}")
    macro("PearInf", f"{pearsonr(pr[inf], r[inf])[0]:.2f}")
    macro("PearSAInf", f"{pearsonr(sa[inf], r[inf])[0]:.2f}")
    macro("MAEInf", f1(np.abs(pr - r)[inf].mean()))
    macro("MAESAInf", f1(np.abs(sa - r)[inf].mean()))
    macro("NInf", str(int(inf.sum())))
    if S:
        v2 = S["v2_unseen_runs"]
        macro("VNCells", str(v2["n_cells"]))
        macro("VPear", f"{v2['V1']['pearson']:.2f}")
        macro("VClosedPear", f"{v2['V1']['closed_form_pearson']:.2f}")
        macro("VMAE", f1(v2["V2"]["mae"]))
        macro("VClosedMAE", f1(v2["V2"]["closed_form_mae"]))
        macro("VInfMAE", f1(v2["V3"]["mae_signal_informative"]))
        macro("VInfClosedMAE", f1(v2["V3"]["mae_closed_informative"]))
        macro("VMidN", str(v2["V4"]["n"]))
        macro("VMidPear", f"{v2['V4']['pearson']:.2f}")
        for k, nm in (("v1_as_protocol_all_signals", "Arith"), ("v1_protocol_literal", "Lit")):
            macro(f"PHeld{nm}", str(sum(bool(v["held"]) for v in S[k].values()) + (1 if S["P6"]["held"] else 0)))
        macro("PLitRho", f"{S['v1_protocol_literal']['P1']['spearman']:.2f}")
        macro("PLitPear", f"{S['v1_protocol_literal']['P4']['pearson']:.2f}")
        macro("PLitDPMax", f1(S["v1_protocol_literal"]["P5"]["max"]))
        macro("PArithRho", f"{S['v1_as_protocol_all_signals']['P1']['spearman']:.2f}")
        macro("PArithPear", f"{S['v1_as_protocol_all_signals']['P4']['pearson']:.2f}")
        macro("PArithThreeFrac", f"{100 * S['v1_as_protocol_all_signals']['P3']['fraction_ge8']:.0f}")
        macro("PSixFrac", f"{100 * S['v1_plus_6step_committed_prices']['P3']['fraction_ge8']:.0f}")
    if S:
        for key, nm in (("v2_as_written_all_unseen", "VA"), ("v2_l40s_completely_unseen", "VL"), ("v2_l40s_matched_price", "VM"), ("v2_l40s_with_musique", "VQ")):
            v2 = S[key]
            macro(f"{nm}NCells", str(v2["n_cells"]))
            macro(f"{nm}Pear", f"{v2['V1']['pearson']:.2f}")
            macro(f"{nm}ClosedPear", f"{v2['V1']['closed_form_pearson']:.2f}")
            macro(f"{nm}MAE", f"{v2['V2']['mae']:.2f}")
            macro(f"{nm}ClosedMAE", f"{v2['V2']['closed_form_mae']:.2f}")
            macro(f"{nm}InfMAE", f"{v2['V3']['mae_signal_informative']:.2f}")
            macro(f"{nm}InfClosedMAE", f"{v2['V3']['mae_closed_informative']:.2f}")
            macro(f"{nm}DiffZero", f"{v2['V3']['mae_difference_on_S0']:.1f}")
            macro(f"{nm}MidN", str(v2["V4"]["n"]))
            macro(f"{nm}MidPear", f"{v2['V4']['pearson']:.2f}")
            macro(f"{nm}Held", str(sum(bool(v2[k]["held"]) for k in ("V1", "V2", "V3", "V4"))))
        def _spread(d):                       # largest range over the three re-partitions of the measured regret of one cell
            top = 0.0
            for sig in d["runs"]["partition_0"]:
                for m in d["ms"]:
                    rg = []
                    for i in range(3):
                        u = {r: d["runs"][f"partition_{i}"][sig]["m"][str(m)]["rules"][r]["u_auc"] for r in cp.ALL}
                        rg.append(max(u.values()) - max(u[r] for r in cp.PER_STEP))
                    top = max(top, max(rg) - min(rg))
            return top
        macro("VLSpread", f"{max(_spread(R[t]) for t in cp.L40S_RUNS if t in R):.0f}")
        sx = S["v1_plus_6step_committed_prices"]
        macro("PSixDPMax", f1(sx["P5"]["max"]))
        macro("PSixRho", f"{sx['P1']['spearman']:.2f}")
        macro("PFreshRho", f"{S['replication_pools_20261201']['P1']['spearman']:.2f}")
        macro("PFreshPear", f"{S['replication_pools_20261201']['P4']['pearson']:.2f}")
        macro("PFreshHigh", f"{100 * S['replication_pools_20261201']['P3']['fraction_ge8']:.0f}")
        macro("PFreshDPMax", f1(S["replication_pools_20261201"]["P5"]["max"]))
        macro("PSixMatchedHigh", f"{100 * S['v1_plus_6step_matched_prices']['P3']['fraction_ge8']:.0f}")
        macro("PSixMatchedRho", f"{S['v1_plus_6step_matched_prices']['P1']['spearman']:.2f}")
        if "P6_shared_pool" in S:
            for t, k in zip(FRESH5, "ABCDE"):
                cc, ss = S["P6_shared_pool"]["counts"][t]
                macro(f"Fr{k}Comp", str(cc))
                macro(f"Fr{k}Subs", str(ss))
                rr = cc / max(1, ss)
                macro(f"Fr{k}Rat", f"{rr:.0f}" if rr >= 10 else (f"{rr:.1f}" if rr >= 1 else f"{rr:.2f}"))
    # the shared fresh pool: how much an informative signal shrinks the regret at T = 24 (2B)
    if "q35_2b_seed20261201" in C:
        for sig, k in (("S0", "Zero"), ("S1", "Hid"), ("S4", "Conf"), ("S0+S4", "Fuse")):
            macro(f"FrTwo{k}", f1(next(c for c in C["q35_2b_seed20261201"] if c["sig"] == sig and c["m"] == 8)["best_per_step_regret"]))
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
    mags = [abs(pre[k]["diff"]) for k in ("mup_vs_pp_step_direct", "mup_vs_best_global_pattern", "mup_vs_pp_first")]
    macro("PregLo", f"{min(mags):.2f}")
    macro("PregHi", f"{max(mags):.2f}")
    # first horizon where best per-step regret exceeds 1 point, per run (S0)
    for t, k in (("q35_4b", "FourB"), ("q35_9b", "NineB"), ("q35_2b", "TwoB")):
        if t in R:
            cs = [c for c in C[t] if c["sig"] == "S0"]
            first = next((c for c in cs if c["best_per_step_regret"] > 1), None)
            macro(f"First{k}T", str(first["T"]) if first else "none")
            macro(f"First{k}x", f"{first['x']:.1f}" if first else "none")

    # 6-step chains: ceiling of the price grid and regret at one part (T = 6), committed and matched prices
    from toolcomp.horizon import auc as _auc

    from toolcomp.eval_harness import LAMBDAS as _L
    macro("CeilThree", f1(float(_auc(np.maximum(0, 1 - 3 * np.array(_L))))))
    macro("CeilSix", f1(float(_auc(np.maximum(0, 1 - 6 * np.array(_L))))))
    dpall = []
    for t, k in zip(SIXM, "ABCDE"):
        k = "Six" + k
        if t in extra:
            d = json.loads((res / "scaling" / f"{t}.json").read_text())
            cs = {(c["sig"], c["m"]): c for c in cp.cells(d)}
            macro(f"{k}x", f1(cs[("S0", 1)]["x"]))
            macro(f"{k}Zero", f1(cs[("S0", 1)]["best_per_step_regret"]))
            macro(f"{k}Fuse", f1(cs[("S0+S4", 1)]["best_per_step_regret"]))
            macro(f"{k}Long", f1(cs[("S0", 32)]["best_per_step_regret"]))
            macro(f"{k}DPMax", f1(max(c["dp_vs_label_free"] for c in cs.values())))
            macro(f"{k}Never", f"{100 * d['never_accuracy']:.0f}")
            dpall.append(max(c["dp_vs_label_free"] for c in cs.values()))
    macro("SixDPAll", f1(max(dpall)))
    before = sorted((res / "scaling_before_input_flag_fix").glob("arith6_*_matched.json"))        # the same runs with the colliding "missing input" flag
    if before:
        macro("SixBugGap", f1(max(c["dp_vs_label_free"] for f in before for c in cp.cells(json.loads(f.read_text())))))
    if all(t in extra for t in SIXM):
        L = ["\\begin{tabular}{@{}lrrrrr@{}}\\toprule", " & unaided & \\multicolumn{2}{c}{Loss, $T$=6} & $T$=192 & DP \\\\",
             "Model & acc.\\ (\\%) & surf. & +conf. & surf. & gap \\\\ \\midrule"]
        for t, nm in zip(SIXM, ("0.5B", "0.8B", "2B", "4B", "9B")):
            d = json.loads((res / "scaling" / f"{t}.json").read_text())
            cs = {(c["sig"], c["m"]): c for c in cp.cells(d)}
            L.append(f"{nm} & {100 * d['never_accuracy']:.0f} & {cs[('S0', 1)]['best_per_step_regret']:.1f} & {cs[('S0+S4', 1)]['best_per_step_regret']:.1f} & "
                     f"{cs[('S0', 32)]['best_per_step_regret']:.1f} & {max(c['dp_vs_label_free'] for c in cs.values()):.1f} \\\\")
        L.append("\\bottomrule\\end{tabular}")
        (out / "gen_six.tex").write_text("\n".join(L))
    macro("SixDPExB", f1(max(v for t_, v in zip(SIXM, dpall) if "0p8b" not in t_)))
    ms = res / "musique_summary.json"
    if ms.exists():
        mu = json.loads(ms.read_text())
        macro("MusNeverMax", f"{100 * max(v['never'] for v in mu.values()):.0f}")
        macro("MusAllMin", f"{100 * min(v['all_call'] for v in mu.values()):.0f}")
        macro("MusAllMax", f"{100 * max(v['all_call'] for v in mu.values()):.0f}")
        macro("MusWrongMin", f"{100 * min(v['share_correct_with_wrong_hop'] for v in mu.values()):.0f}")
        macro("MusWrongMax", f"{100 * max(v['share_correct_with_wrong_hop'] for v in mu.values()):.0f}")
        macro("MusDPShort", f1(max(v["belief_dp_worst_shortfall_vs_never"] for v in mu.values())))
        qs_ = [x for v in mu.values() for x in v["tool_right_given_correct_prefix"] if x is not None]
        macro("NMus", str(len(mu)))
        macro("MusToolMin", f"{100 * min(qs_):.0f}")
        macro("MusToolMax", f"{100 * max(qs_):.0f}")
        nz = [v["belief_dp_noisy"] for v in mu.values() if "belief_dp_noisy" in v]
        if nz:
            ncell = sum(v["cells"] for v in nz)
            macro("MusNZCells", str(ncell))
            macro("MusNZNever", f"{max(v['worst_shortfall_vs_never'] for v in nz):.2f}")
            macro("MusNZBest", f"{max(v['worst_shortfall_vs_best_of_nine'] for v in nz):.1f}")
            macro("MusNZNeverBest", f"{100 * sum(v['cells_never_is_best'] for v in nz) / ncell:.0f}")
            macro("MusNZBeat", f"{100 * sum(v['cells_noisy_beats_never_by_1'] for v in nz) / ncell:.0f}")
            macro("MusNZGain", f1(max(v["best_gain_over_never"] for v in nz)))
        macro("MusComp", str(sum(v["complements"] for v in mu.values())))
        macro("MusSubs", str(sum(v["substitutes"] for v in mu.values())))
    # ------------------------------------------------------------ Table: the shared fresh pool (one set of 1000 instances, five models, every signal)
    L = ["\\begin{tabular}{@{}lrrrrr@{}}\\toprule", "Model & $x$ & surface & hidden & conf. & surf.+conf. \\\\ \\midrule"]
    for t in FRESH5:
        if t not in R:
            continue
        cs = {(c["sig"], c["m"]): c for c in C[t]}
        L.append(f"{ROW[t]} & {cs[('S0', 8)]['x']:.1f} & " + " & ".join(f"{cs[(g, 8)]['best_per_step_regret']:.1f}" for g in ("S0", "S1", "S4", "S0+S4")) + " \\\\")
    L.append("\\bottomrule\\end{tabular}")
    (out / "gen_runs.tex").write_text("\n".join(L))

    # ------------------------------------------------------------ Table: both pre-committed prediction sets, held / not held with the measured value
    if S:
        def mark(v, val):
            return ("yes" if v else "\\textbf{no}") + (f" {val}" if val else "")

        A, B_ = S["v1_as_protocol_all_signals"], S["v1_protocol_literal"]
        r2 = lambda k: (lambda d: f"{d[k]:.2f}")
        first = [("P1", "loss rises with $x$: $\\rho\\ge0.7$", "P1", r2("spearman")),
                 ("P2", "$x<0.3$: max loss $\\le3$", "P2", r2("max")),
                 ("P3", "$x>1.5$: 90\\% lose $\\ge8$", "P3", lambda d: f"{100 * d['fraction_ge8']:.0f}\\%"),
                 ("P4", "closed form: $r\\ge0.8$", "P4", r2("pearson")),
                 ("P5", "gap: 90\\% $\\le1$, max $\\le3$", "P5", lambda d: f"{d['max']:.1f}"),
                 ("P6", "pair ratio 2B$>$4B$>$9B", None, None),
                 ("P7", "new pool: max gap $\\le3$", "P7", lambda d: f"{d['max']:.1f}")]
        L = ["\\begin{tabular}{@{}llll@{}}\\toprule", "& \\emph{First set} & 5 runs & +6-step, QA \\\\ \\midrule"]
        for pid, text, key, val in first:
            if key is None:
                L.append(f"{pid} & {text} & {mark(S['P6']['held'], '')} & {mark(S['P6']['held'], '')} \\\\")
            else:
                L.append(f"{pid} & {text} & {mark(A[key]['held'], val(A[key]))} & {mark(B_[key]['held'], val(B_[key]))} \\\\")
        U, N = S["v2_as_written_all_unseen"], S["v2_l40s_completely_unseen"]
        L.append("\\midrule & \\emph{Second set} & 11 runs & newest 6 \\\\ \\midrule")
        for pid, text, val in (("V1", "signal-aware: $r\\ge0.9$", r2("pearson")), ("V2", "mean error $\\le2$ points", r2("mae")),
                               ("V3", "error vs.\\ closed form", lambda d: f"{d['mae_signal_informative']:.1f}/{d['mae_closed_informative']:.1f}"),
                               ("V4", "below cap: $r\\ge0.85$", r2("pearson"))):
            L.append(f"{pid} & {text} & {mark(U[pid]['held'], val(U[pid]))} & {mark(N[pid]['held'], val(N[pid]))} \\\\")
        L.append("\\bottomrule\\end{tabular}")
        (out / "gen_preds.tex").write_text("\n".join(L))

    # ------------------------------------------------------------ deep runs (6-step arithmetic, MuSiQue), when present
    deep = {t: json.loads((res / "scaling" / f"{t}.json").read_text()) for t in extra if t.startswith("arith6_")}
    if deep:
        L = ["\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule", "Run & $n$ & never & all calls & $x_{m=1}$ & \\multicolumn{3}{c}{Regret, $m$=} \\\\",
             " & & & & & 1 & 2 & 4 \\\\ \\midrule"]
        for t, d in sorted(deep.items()):
            cs = {c["m"]: c for c in cp.cells(d) if c["sig"] == "S0"}
            lab = t.replace("arith6_", "").replace("gpu_0p5b", "0.5B").replace("q35_0p8b", "0.8B").replace("_matched", ", matched price")
            lab = lab + (", committed price" if "matched" not in lab else "")
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

    # ------------------------------------------------------------ figure 1 (drawn at the printed size, so font sizes are the printed ones)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
                         "axes.edgecolor": "#6b6a66", "xtick.color": "#2f2e2b", "ytick.color": "#2f2e2b"})
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.3))
    # (a) utility versus total price for a homogeneous chain
    ax = axs[0]
    T, p = 12, 0.9
    Cs = np.linspace(0.02, 2.0, 400)
    for rule, st, lab, col, lw in (("optimal_open_loop", "-", "optimal", "#2a78d6", 3.0), ("optimistic", "--", "optimistic", "#eb6834", 1.8), ("pessimistic", ":", "pessimistic", "#1a1a19", 1.8)):
        ax.plot(Cs, [rule_expected_utility(rule, [p], T, c / T) for c in Cs], st, color=col, lw=lw, label=lab)
    ax.set_xlabel("total price $Tc$")
    ax.set_ylabel("expected utility")
    ax.set_title(f"(a) chain, $T$={T}, $p$={p}", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", handlelength=1.8, borderaxespad=0.1)
    # (b), (c) measured regret of the best per-step rule against the two predictions, split by signal type
    for ax, key, title in ((axs[1], "predicted_best_per_step", "(b) closed form"), (axs[2], "signal_aware", "(c) signal-aware")):
        for informative, col, mk, lab in ((False, "#2a78d6", "o", "surface"), (True, "#eb6834", "^", "informative")):
            cs = [c for c in allc if (c["sig"] != "S0") == informative]
            xv, yv = [c[key] for c in cs], [c["best_per_step_regret"] for c in cs]
            ax.scatter(xv, yv, s=13, marker=mk, color=col, alpha=0.6, lw=0, label=f"{lab}: $r$ = {pearsonr(xv, yv)[0]:.2f}", zorder=3 if not informative else 2)
        ax.plot([0, 19], [0, 19], color="#1a1a19", lw=0.8, zorder=1)
        ax.set_xlim(-0.7, 19.5)
        ax.set_ylim(-0.7, 25.5)                      # headroom above the data for the legend
        ax.set_xticks([0, 5, 10, 15])
        ax.set_yticks([0, 5, 10, 15])
        ax.set_xlabel("predicted loss (points)")
        ax.set_title(title, fontsize=9)
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.1, borderaxespad=0.1, labelspacing=0.8, markerscale=1.4)
        for t_, col in zip(leg.get_texts(), ("#2a78d6", "#eb6834")):
            t_.set_color(col)
            t_.set_fontweight("bold")
    axs[1].set_ylabel("measured loss (points)")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.9, wspace=0.34)
    fig.savefig(out / "figures" / "fig1.pdf")
    fig.savefig(out / "figures" / "fig1.png", dpi=250)
    (out / "gen_macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
    print(f"{len(M)} macros, {len(R)} runs (+ extra: {extra}), {len(allc)} cells -> {out}")


if __name__ == "__main__":
    main()
