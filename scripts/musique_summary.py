#!/usr/bin/env python3
"""Descriptive statistics of the MuSiQue outcome trees (the exact-tool theory does not fit them; this records why).

    python scripts/musique_summary.py --deep out/deep --out results/musique_summary.json
Per run: n, never-call and all-call accuracy, probability the retrieval tool is right on a correct prefix, share of correct final answers whose chain
contained a wrong hop (so the task is not conjunctive), pair counts complement:substitute, and the largest shortfall of Belief-DP against never calling.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolcomp import scaling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "musique_summary.json"))
    a = ap.parse_args()
    out = {}
    for f in sorted(Path(a.deep).glob("musique_k*_*.npz")):
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
        worst = 0.0
        for part in res["runs"].values():
            for sig in part.values():
                for mm in sig["m"].values():
                    ru = mm["rules"]
                    worst = max(worst, ru["never"]["u_auc"] - ru["belief_dp"]["u_auc"])
        out[f.stem] = {"n": int(N), "K": K, "never": float(r.leaf[:, 0].mean()), "all_call": float(r.leaf[:, -1].mean()), "tool_right_given_correct_prefix": q,
                       "share_correct_with_wrong_hop": float((r.leaf & ~allok).sum() / max(1, r.leaf.sum())), "complements": pos, "substitutes": neg,
                       "belief_dp_worst_shortfall_vs_never": float(worst)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "tool_right_given_correct_prefix"} for k, v in out.items()}, indent=0)[:1800])


if __name__ == "__main__":
    main()
