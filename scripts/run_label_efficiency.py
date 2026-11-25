#!/usr/bin/env python3
"""How many labelled long tasks does a re-tuned threshold need before it matches Belief-DP (which needs none)?

For one run/signal/horizon m: fit the step signal as in the horizon study (partition 0), then for n_lab in a grid draw n_lab labelled composites
(disjoint groups of m instances from the calibration+gate pool), tune the confidence threshold for every price on them, and score it on the same
test composites used for Belief-DP. 30 random draws per n_lab.

    python scripts/run_label_efficiency.py --kind text --tag q35_2b --sidecar data/trees_text/q35_2b.json --seed 20261101 --sig S0 --m 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolcomp import scaling
from toolcomp.config import load_config
from toolcomp.envs.arith import generate_instances
from toolcomp.eval_harness import LAMBDAS
from toolcomp.horizon import (IsoSignal, StepSignal, auc, belief_dp, brier_ll, default_sizes, level_samples, make_tuples, rule_dp, rule_threshold,
                              simulate)
from toolcomp.splits import make_splits
from toolcomp.tree import TreeData

SEED = 20261111


def fit(run, sig, seed=SEED):
    sp = make_splits(run.ids, default_sizes(run.n), seed)
    X, Y = run.feats[sig], run.Y
    tr, ca, ga, te = sp["train"], sp["cal"], sp["gate"], sp["test"]
    tc = np.sort(np.concatenate([tr, ca]))
    a, b = StepSignal(run.C.get(sig, 1.0)).fit(X[tr], Y[tr], X[ca], Y[ca]), IsoSignal(run.C.get(sig, 1.0)).fit(X[tc], Y[tc])
    gate = {"platt": brier_ll(a(X[ga]), Y[ga])[1], "isotonic": brier_ll(b(X[ga]), Y[ga])[1]}
    pick = min(gate, key=gate.get)
    P = a(X) if pick == "platt" else b(X)
    fitP = a(X[tr]) if pick == "platt" else b.cal(b.oof)
    return sp, P, fitP


def u_auc_of(P, leaf, tuples, rule_for_lam, m):
    U = []
    for lam in LAMBDAS:
        ok, c = simulate(P, leaf, tuples, rule_for_lam(lam))
        U.append(float((ok - lam / m * c).mean()))
    return float(auc(np.array(U)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("text", "tree"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sidecar"), ap.add_argument("--seed", type=int), ap.add_argument("--tree"), ap.add_argument("--config"), ap.add_argument("--results")
    ap.add_argument("--sig", default="S0")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results" / "label_efficiency"))
    a = ap.parse_args()
    if a.kind == "text":
        run = scaling.run_from_text(a.tag, "", a.sidecar, a.seed)
    else:
        cfg = load_config(a.config)
        J = json.loads(Path(a.results).read_text())
        run = scaling.run_from_tree(a.tag, "", TreeData.load(Path(a.tree)), generate_instances(cfg["pool"]["n"], cfg["pool"]["seed"], cfg["pool"]["prefix"]),
                                    [J[f"partition_{r}"]["fit_info"]["mup"] for r in range(3)])
        run.feats, run.C = run.per_partition[0]["feats"], run.per_partition[0]["C"]
    sp, P, fitP = fit(run, a.sig)
    m, leaf = a.m, run.leaf
    pool = np.concatenate([sp["cal"], sp["gate"]])
    samples = level_samples(fitP)
    T = len(samples) * m
    Ws = {lam: belief_dp(samples, T, lam / m) for lam in LAMBDAS}
    rng = np.random.default_rng(1000 + SEED)
    test_t = make_tuples(sp["test"], m, rng, reps=40)
    dp = u_auc_of(P, leaf, test_t, lambda lam: rule_dp(Ws[lam], lam / m), m)
    grid = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(P[pool], np.linspace(0.01, 0.99, 60))]))

    def tune_on(tp, lam):
        best, bu = None, -np.inf
        for t in grid:
            ok, c = simulate(P, leaf, tp, rule_threshold(t), P)
            u = float((ok - lam / m * c).mean())
            if u > bu + 1e-12:
                best, bu = t, u
        return best

    max_lab = len(pool) // m
    n_grid = sorted({n for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, max_lab) if n <= max_lab})
    out = {"tag": a.tag, "sig": a.sig, "m": m, "T": T, "belief_dp": dp, "n_test_composites": int(len(test_t)), "pool_instances": int(len(pool)), "curve": {}}
    full = []
    for n_lab in n_grid:
        vals = []
        for d in range(a.draws):
            r = np.random.default_rng(d)
            q = r.permutation(pool)[:n_lab * m].reshape(n_lab, m)
            taus = {lam: tune_on(q, lam) for lam in LAMBDAS}
            vals.append(u_auc_of(P, leaf, test_t, lambda lam: rule_threshold(taus[lam]), m))
        out["curve"][str(n_lab)] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "min": float(np.min(vals)), "frac_below_dp": float(np.mean(np.array(vals) < dp))}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.tag}_{a.sig.replace('+', '')}_m{m}.json").write_text(json.dumps(out, indent=1))
    print(a.tag, a.sig, f"m={m} T={T} Belief-DP {dp:.2f} | retuned by #labelled composites:", {k: round(v["mean"], 1) for k, v in out["curve"].items()})


if __name__ == "__main__":
    main()
