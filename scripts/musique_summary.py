#!/usr/bin/env python3
"""Descriptive statistics of the MuSiQue outcome trees (the exact-tool theory does not fit them; this records why).

    python scripts/musique_summary.py --deep out/deep [more dirs ...] --out results/musique_summary.json
Per run: n, never-call and all-call accuracy, probability the retrieval tool is right on a correct prefix, share of correct final answers whose chain
contained a wrong hop (so the task is not conjunctive), pair counts complement:substitute, and the largest shortfall of Belief-DP against never calling.
When results/scaling/<run>.json was computed with --noisy-tool, also the same for Belief-DP with the measured tool accuracy (rule belief_dp_noisy).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolcomp import scaling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "musique_summary.json"))
    a = ap.parse_args()
    out = {}
    for f in sorted(f for d in a.deep for f in Path(d).glob("musique_k*_*.npz")):
        K = int(f.stem.split("_k")[1][0])
        r = scaling.run_from_deep(f.stem, "", f)
        N = r.n
        ok = scaling.prefix_correct(r)
        q = []
        for lv in range(K):
            sl = slice(2 ** lv - 1, 2 ** (lv + 1) - 1)
            m = ok[:, sl]
            q.append(float(r.call_ok[:, sl][m].mean()) if m.any() else None)
        allok = np.zeros((N, 2 ** K), bool)
        for leaf in range(2 ** K):
            o, nid = np.ones(N, bool), 0
            for lv in range(K):
                act = (leaf >> (K - 1 - lv)) & 1
                o &= (r.call_ok if act else r.Y)[:, nid]
                nid = 2 * nid + 1 + act
            allok[:, leaf] = o
        lc = r.leaf.astype(int)
        pos = neg = 0
        for i in range(K):
            for j in range(i + 1, K):
                bi, bj = 1 << i, 1 << j
                for base in range(2 ** K):
                    if base & (bi | bj):
                        continue
                    d = lc[:, base | bi | bj] - lc[:, base | bi] - lc[:, base | bj] + lc[:, base]
                    pos, neg = pos + int((d > 0).sum()), neg + int((d < 0).sum())
        res = json.loads((ROOT / "results" / "scaling" / f"{f.stem}.json").read_text())
        # a cell is (signal, m) with each rule's utility averaged over the three re-partitions, as everywhere else in the study
        worst = -1e9
        nz = {"vs_never": -1e9, "vs_best_nine": -1e9, "gain_over_exact_dp": -1e9, "cells": 0, "within_1_of_never": 0, "never_is_best": 0, "noisy_beats_never_by_1": 0, "best_gain_over_never": -1e9}
        parts = list(res["runs"].values())
        for sig in parts[0]:
            for mm in parts[0][sig]["m"]:
                ru = {k: float(np.mean([pt[sig]["m"][mm]["rules"][k]["u_auc"] for pt in parts])) for k in parts[0][sig]["m"][mm]["rules"]}
                worst = max(worst, ru["never"] - ru["belief_dp"])
                if "belief_dp_noisy" in ru:
                    u = ru["belief_dp_noisy"]
                    nine = max(ru[k] for k in ("never", "always", "all_or_nothing", "local", "optimistic", "pessimistic", "thr_transfer", "thr_retuned", "belief_dp"))
                    nz["vs_never"] = max(nz["vs_never"], ru["never"] - u)
                    nz["vs_best_nine"] = max(nz["vs_best_nine"], nine - u)
                    nz["gain_over_exact_dp"] = max(nz["gain_over_exact_dp"], u - ru["belief_dp"])
                    nz["cells"] += 1
                    nz["within_1_of_never"] += int(ru["never"] - u <= 1.0)
                    nz["never_is_best"] += int(ru["never"] >= nine - 1e-9)
                    nz["noisy_beats_never_by_1"] += int(u - ru["never"] > 1.0)
                    nz["best_gain_over_never"] = max(nz["best_gain_over_never"], u - ru["never"])
        out[f.stem] = {"n": int(N), "K": K, "never": float(r.leaf[:, 0].mean()), "all_call": float(r.leaf[:, -1].mean()), "tool_right_given_correct_prefix": q,
                       "share_correct_with_wrong_hop": float((r.leaf & ~allok).sum() / max(1, r.leaf.sum())), "complements": pos, "substitutes": neg,
                       "belief_dp_worst_shortfall_vs_never": float(worst)}
        if nz["cells"]:
            out[f.stem]["belief_dp_noisy"] = {"q_by_level": res["runs"]["partition_0"]["S0"]["q_by_level"], "worst_shortfall_vs_never": float(nz["vs_never"]),
                                              "worst_shortfall_vs_best_of_nine": float(nz["vs_best_nine"]), "max_gain_over_exact_dp": float(nz["gain_over_exact_dp"]),
                                              "cells": nz["cells"], "cells_within_1_of_never": nz["within_1_of_never"], "cells_never_is_best": nz["never_is_best"],
                                              "cells_noisy_beats_never_by_1": nz["noisy_beats_never_by_1"], "best_gain_over_never": float(nz["best_gain_over_never"])}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "tool_right_given_correct_prefix"} for k, v in out.items()}, indent=0)[:1800])


if __name__ == "__main__":
    main()
