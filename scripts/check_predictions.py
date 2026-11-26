#!/usr/bin/env python3
"""Score the pre-specified predictions (configs/predictions.yaml, protocol in configs/predictions_protocol.yaml) on results/scaling/*.json.

    python scripts/check_predictions.py [--runs tag1 tag2 ...] [--json results/predictions_scored.json]
Cells are (run, signal, m). Utility = mean over the three re-partitions of test U-AUC (points).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolcomp.horizon import predicted_regret

ALL = ["always", "never", "all_or_nothing", "local", "optimistic", "pessimistic", "thr_transfer", "thr_retuned", "belief_dp"]
PER_STEP = ["local", "optimistic", "pessimistic"]
LABEL_FREE = ["never", "always", "all_or_nothing", "local", "optimistic", "pessimistic", "belief_dp"]
NEW_RUNS = ["q35_2b", "q35_4b", "q35_9b", "q35_0p8b_seed20261120", "q35_0p8b_incremental"]


def cells(res):
    out = []
    ms = res["ms"]
    sigs = list(res["runs"]["partition_0"])
    for sig in sigs:
        for m in ms:
            u = {r: float(np.mean([res["runs"][f"partition_{i}"][sig]["m"][str(m)]["rules"][r]["u_auc"] for i in range(3)])) for r in ALL}
            best = max(u.values())
            bps = best - max(u[r] for r in PER_STEP)
            bdp = max(u[r] for r in LABEL_FREE) - u["belief_dp"]
            pr = predicted_regret(res["step_accuracy"], m, price_scale=res.get("price_scale", 1.0))
            sa = float(np.mean([res["runs"][f"partition_{i}"][sig]["m"][str(m)]["signal_aware_regret"] for i in range(3)])) if "signal_aware_regret" in res["runs"]["partition_0"][sig]["m"][str(m)] else None
            out.append({"tag": res["tag"], "sig": sig, "m": m, "T": res["K"] * m, "x": res["x_by_m"][str(m)], "best_per_step_regret": bps,
                        "dp_vs_label_free": bdp, "dp_vs_all": best - u["belief_dp"], "predicted_best_per_step": min(pr.values()), "signal_aware": sa, "u": u})
    return out


def score(C, fresh_tag="q35_0p8b_seed20261120"):
    x = np.array([c["x"] for c in C]); r = np.array([c["best_per_step_regret"] for c in C]); pr = np.array([c["predicted_best_per_step"] for c in C])
    dp = np.array([c["dp_vs_label_free"] for c in C])
    P = {}
    rho = spearmanr(x, r)[0]
    P["P1"] = {"held": bool(rho >= 0.7), "spearman": float(rho), "n_cells": len(C)}
    lo = r[x < 0.3]
    P["P2"] = {"held": bool(len(lo) > 0 and (lo <= 3).mean() >= 0.9), "n": int(len(lo)), "fraction_le3": float((lo <= 3).mean()) if len(lo) else None, "max": float(lo.max()) if len(lo) else None}
    hi = r[x > 1.5]
    P["P3"] = {"held": bool(len(hi) > 0 and (hi >= 8).mean() >= 0.9), "n": int(len(hi)), "fraction_ge8": float((hi >= 8).mean()) if len(hi) else None, "median": float(np.median(hi)) if len(hi) else None}
    pe = pearsonr(pr, r)[0]
    P["P4"] = {"held": bool(pe >= 0.8), "pearson": float(pe)}
    P["P5"] = {"held": bool((dp <= 1).mean() >= 0.9 and dp.max() <= 3), "fraction_le1": float((dp <= 1).mean()), "max": float(dp.max()), "n": len(dp)}
    fr = [c["dp_vs_label_free"] for c in C if c["tag"] == fresh_tag]
    P["P7"] = {"held": bool(len(fr) > 0 and max(fr) <= 3.0), "max": float(max(fr)) if fr else None, "n": len(fr)}
    return P


def score_v2(C, ceiling=15.0):
    """Second prediction set (configs/predictions_v2.yaml): signal-aware prediction against the measured per-step regret."""
    C = [c for c in C if c["signal_aware"] is not None]
    r = np.array([c["best_per_step_regret"] for c in C]); sa = np.array([c["signal_aware"] for c in C]); cf = np.array([c["predicted_best_per_step"] for c in C])
    out = {"n_cells": len(C)}
    pe = pearsonr(sa, r)[0]
    out["V1"] = {"held": bool(pe >= 0.9), "pearson": float(pe), "closed_form_pearson": float(pearsonr(cf, r)[0])}
    out["V2"] = {"held": bool(np.abs(sa - r).mean() <= 2.0), "mae": float(np.abs(sa - r).mean()), "closed_form_mae": float(np.abs(cf - r).mean())}
    s0 = np.array([c["sig"] == "S0" for c in C])
    d_s0 = abs(np.abs(sa - r)[s0].mean() - np.abs(cf - r)[s0].mean()) if s0.any() else None
    better = bool((np.abs(sa - r)[~s0].mean() < np.abs(cf - r)[~s0].mean())) if (~s0).any() else None
    out["V3"] = {"held": bool(better and d_s0 is not None and d_s0 <= 1.0), "n_informative_cells": int((~s0).sum()), "mae_signal_informative": float(np.abs(sa - r)[~s0].mean()) if (~s0).any() else None,
                 "mae_closed_informative": float(np.abs(cf - r)[~s0].mean()) if (~s0).any() else None, "mae_difference_on_S0": float(d_s0) if d_s0 is not None else None}
    mid = (r > 0.3) & (r < ceiling)
    out["V4"] = {"held": bool(mid.sum() > 2 and pearsonr(sa[mid], r[mid])[0] >= 0.85), "n": int(mid.sum()), "pearson": float(pearsonr(sa[mid], r[mid])[0]) if mid.sum() > 2 else None}
    return out


V2_RUNS = ["primary_0p5b", "q35_0p8b", "q35_2b", "q35_4b", "q35_9b"]          # configs/predictions_v2.yaml: unseen_prediction_only
L40S_RUNS = ["arith6_q35_2b", "arith6_q35_4b", "arith6_q35_9b", "q35_2b_seed20261201", "q35_4b_seed20261201", "q35_9b_seed20261201"]    # unseen_completely: committed prices
L40S_MATCHED = ["arith6_q35_2b_matched", "arith6_q35_4b_matched", "arith6_q35_9b_matched"]
FRESH_POOLS = ["gpu_0p5b_seed20261201", "q35_0p8b_seed20261201", "q35_2b_seed20261201", "q35_4b_seed20261201", "q35_9b_seed20261201"]    # one shared pool (generator seed 20261201)
EXPLORATORY = ["arith6_gpu_0p5b", "arith6_q35_0p8b", "arith6_gpu_0p5b_matched", "arith6_q35_0p8b_matched", "gpu_0p5b_seed20261201", "q35_0p8b_seed20261201"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "scaling"))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--json", default=str(ROOT / "results" / "predictions_scored.json"))
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob("*.json"))
    R = {f.stem: json.loads(f.read_text()) for f in files}
    tags = a.runs or sorted(R)
    allc = [c for t in tags for c in cells(R[t])]
    new = [c for c in allc if c["tag"] in NEW_RUNS or not c["tag"].startswith(("q35_0p8b", "gpu_0p5b", "primary"))]
    out = {"all_runs": score(allc), "new_runs_only": score([c for c in allc if c["tag"] in NEW_RUNS]) if any(c["tag"] in NEW_RUNS for c in allc) else None, "runs": tags}
    byt = lambda tt: [c for t in tt if t in R for c in cells(R[t])]
    MUS = [t for t in R if t.startswith("musique_")]
    A6 = sorted(t for t in R if t.startswith("arith6_") and not t.endswith("_matched"))          # every 6-step run that returned, committed price
    A6M = sorted(t for t in R if t.startswith("arith6_") and t.endswith("_matched"))             # the same at half the per-call price
    out["v1_protocol_literal"] = score(byt(NEW_RUNS + A6 + MUS))          # protocol text: new runs + every gpu run that returned
    out["v1_as_protocol_all_signals"] = score(byt(NEW_RUNS))                                    # the five pre-registered runs, every signal available
    out["v1_plus_6step_committed_prices"] = score(byt(NEW_RUNS + A6))
    out["v1_plus_6step_matched_prices"] = score(byt(NEW_RUNS + A6M))
    out["replication_pools_20261201"] = score(byt(FRESH_POOLS))
    out["v2_unseen_runs"] = score_v2(byt(V2_RUNS))                                                  # unseen_prediction_only (the 0.5B-9B runs of the first study)
    out["v2_l40s_completely_unseen"] = score_v2(byt(L40S_RUNS))                                     # unseen_completely, committed prices
    out["v2_as_written_all_unseen"] = score_v2(byt(V2_RUNS + L40S_RUNS))                            # "all cells of the unseen runs": both groups
    out["v2_l40s_matched_price"] = score_v2(byt(L40S_MATCHED))                                      # not in the committed set: 6-step chains at half the price
    out["v2_l40s_with_musique"] = score_v2(byt(L40S_RUNS + MUS))                                    # if MuSiQue is counted as a run of the bundle
    out["v2_exploratory_runs_not_a_test"] = score_v2(byt(EXPLORATORY))
    # P6 (pair counts, K=3 runs only)
    sd = {t: R[t].get("second_differences") for t in tags if R[t].get("second_differences")}
    ratio = {t: sd[t]["complements"] / max(1, sd[t]["substitutes"]) for t in sd}
    if all(k in ratio for k in ("q35_2b", "q35_4b", "q35_9b")):
        out["P6"] = {"held": bool(ratio["q35_2b"] > ratio["q35_4b"] > ratio["q35_9b"]), "ratios": {k: ratio[k] for k in ("q35_2b", "q35_4b", "q35_9b")}}
    if all(t in ratio for t in FRESH_POOLS):                       # the same ordering on one pool shared by all five models (0.5B, 0.8B, 2B, 4B, 9B)
        rr = [ratio[t] for t in FRESH_POOLS]
        out["P6_shared_pool"] = {"held": bool(all(a_ > b_ for a_, b_ in zip(rr, rr[1:]))), "ratios": dict(zip(FRESH_POOLS, rr)),
                                 "counts": {t: [sd[t]["complements"], sd[t]["substitutes"]] for t in FRESH_POOLS}}
    Path(a.json).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, json.dumps(v)[:600])


if __name__ == "__main__":
    main()
