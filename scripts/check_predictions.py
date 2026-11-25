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
            pr = predicted_regret(res["step_accuracy"], m)
            out.append({"tag": res["tag"], "sig": sig, "m": m, "T": res["K"] * m, "x": res["x_by_m"][str(m)], "best_per_step_regret": bps,
                        "dp_vs_label_free": bdp, "dp_vs_all": best - u["belief_dp"], "predicted_best_per_step": min(pr.values()), "u": u})
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
    # P6 (pair counts, K=3 runs only)
    sd = {t: R[t].get("second_differences") for t in tags if R[t].get("second_differences")}
    ratio = {t: sd[t]["complements"] / max(1, sd[t]["substitutes"]) for t in sd}
    if all(k in ratio for k in ("q35_2b", "q35_4b", "q35_9b")):
        out["P6"] = {"held": bool(ratio["q35_2b"] > ratio["q35_4b"] > ratio["q35_9b"]), "ratios": {k: ratio[k] for k in ("q35_2b", "q35_4b", "q35_9b")}}
    Path(a.json).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, json.dumps(v)[:600])


if __name__ == "__main__":
    main()
