"""Composite-horizon experiment on exact outcome trees: same signal, different decision rule (docs/09 predictions).

  python scripts/run_oring.py --tree <path/to/*_tree.npz> --config configs/q35_0p8b.yaml --results <results.json> --out results/oring_q35_0p8b.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcomp.config import load_config
from toolcomp.envs.arith import dataset_hash, generate_instances
from toolcomp.eval_harness import LAMBDAS
from toolcomp.features import draft_features, hidden_features
from toolcomp.horizon import (IsoSignal, StepSignal, rule_all_or_nothing, belief_dp, hindsight, level_samples, make_tuples, rule_dp, rule_local, rule_optimistic,
                       rule_pessimistic, rule_threshold, simulate, surface_nobin)
from toolcomp.splits import make_splits
from toolcomp.tree import LEVEL_OF, N_NODES, TreeData

MS = (1, 2, 4, 8)
RULES = ("never", "always", "all_or_nothing", "local", "optimistic", "pessimistic", "thr_transfer", "thr_retuned", "belief_dp", "hindsight")


def auc(U):
    lam = np.asarray(LAMBDAS)
    return 100 * trapezoid(U, lam, axis=-1) / (lam[-1] - lam[0])


def brier_ll(p, y):
    p, y = p.reshape(-1), y.reshape(-1).astype(float)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune(Pca, leaf_ca, m, price, rng, score):
    tp = make_tuples(np.arange(Pca.shape[0]), m, rng, reps=20)
    grid = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(score, np.linspace(0.01, 0.99, 60))]))
    best, bu = None, -np.inf
    for t in grid:
        ok, c = simulate(Pca, leaf_ca, tp, rule_threshold(t), score)
        u = float((ok - price * c).mean())
        if u > bu + 1e-12:
            best, bu = t, u
    return best


def partition(tree, ins, cfg, prior, seed, B, reps, calib=None):
    sp = make_splits(tree.instance_ids, cfg["splits"], seed)
    Y = tree.local_no_correct
    S0, S1, S4 = surface_nobin(tree, ins), hidden_features(tree, prior["layer"]), draft_features(tree)
    feats = {"S0": S0, "S1": S1, "S4": S4, "S0+S4": np.concatenate([S0, S4], -1)}
    Cs = {"S0": 1.0, "S1": prior["C"], "S4": 1.0, "S0+S4": 1.0}
    tr, ca, ga, te = sp["train"], sp["cal"], sp["gate"], sp["test"]
    tc = np.sort(np.concatenate([tr, ca]))
    leaf = tree.leaf_correct
    out = {}
    for sig, X in feats.items():
        # two calibrators; the one with the lower step log-loss on the GATE split is used (test is never consulted)
        sp_ = StepSignal(Cs[sig]).fit(X[tr], Y[tr], X[ca], Y[ca])
        si_ = IsoSignal(Cs[sig]).fit(X[tc], Y[tc])
        gate = {"platt": brier_ll(sp_(X[ga]), Y[ga])[1], "isotonic": brier_ll(si_(X[ga]), Y[ga])[1]}
        pick = calib or min(gate, key=gate.get)
        P = sp_(X) if pick == "platt" else si_(X)
        fitP = sp_(X[tr]) if pick == "platt" else si_.cal(si_.oof)
        samples = level_samples(fitP)
        means = [float(x.mean()) for x in samples]
        yt, pt = Y[te].reshape(-1).astype(float), P[te].reshape(-1)
        order = np.argsort(pt)
        r = np.empty(len(pt))
        r[order] = np.arange(1, len(pt) + 1)
        npos, nneg = yt.sum(), (1 - yt).sum()
        auroc = (r[yt == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        res = {"auroc_step_test": float(auroc), "brier_step_test": float(np.mean((pt - yt) ** 2)), "calibrator": pick, "gate_logloss": gate, "mean_p_by_level": means, "m": {}}
        rng = np.random.default_rng(1000 + seed)
        tau1 = {lam: tune(P[ca], leaf[ca], 1, lam, np.random.default_rng(7), P[ca]) for lam in LAMBDAS}
        rules = {}
        for m in MS:
            T = 3 * m
            for lam in LAMBDAS:
                pr = lam / m
                W = belief_dp(samples, T, pr)
                rules[(m, lam)] = {"never": rule_threshold(-np.inf), "always": rule_threshold(np.inf), "all_or_nothing": rule_all_or_nothing(pr, T), "local": rule_local(pr),
                                   "optimistic": rule_optimistic(pr), "pessimistic": rule_pessimistic(pr, means, T),
                                   "thr_transfer": rule_threshold(tau1[lam]),
                                   "thr_retuned": rule_threshold(tune(P[ca], leaf[ca], m, pr, np.random.default_rng(7), P[ca])),
                                   "belief_dp": rule_dp(W, pr)}
        for m in MS:
            tp = make_tuples(te, m, rng, reps=reps)
            block = {}
            for name in RULES:
                U, S, K = [], [], []
                for lam in LAMBDAS:
                    ok, c = hindsight(leaf, tp, lam / m) if name == "hindsight" else simulate(P, leaf, tp, rules[(m, lam)][name])
                    U.append(float((ok - lam / m * c).mean()))
                    S.append(float(ok.mean()))
                    K.append(float(c.mean() / m))
                block[name] = {"u_auc": float(auc(np.array(U))), "u_by_lam": U, "success_by_lam": S, "calls_per_part_by_lam": K}
            # instance-level bootstrap: resample test instances, rebuild disjoint tuples, paired across rules
            names = [n for n in RULES if n not in ("never", "always", "hindsight")]
            boot = {n: [] for n in names}
            brng = np.random.default_rng(31 + m)
            for _ in range(B):
                q = brng.choice(te, size=len(te), replace=True)
                g = len(q) // m
                tb = q[:g * m].reshape(g, m)
                for n in names:
                    u = [float((lambda r: (r[0] - lam / m * r[1]).mean())(simulate(P, leaf, tb, rules[(m, lam)][n]))) for lam in LAMBDAS]
                    boot[n].append(float(auc(np.array(u))))
            con = {}
            for n in names:
                if n == "belief_dp" or not B:
                    continue
                d = np.array(boot["belief_dp"]) - np.array(boot[n])
                con[n] = {"diff": block["belief_dp"]["u_auc"] - block[n]["u_auc"], "ci_lo": float(np.percentile(d, 2.5)),
                          "ci_hi": float(np.percentile(d, 97.5))}
            res["m"][str(m)] = {"n_tuples": int(len(tp)), "rules": block, "belief_dp_minus": con}
        out[sig] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", required=True, help="Iteration-1 results.json (for the frozen layer / C of the hidden-state probe)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--boot-partitions", type=int, nargs="+", default=[0])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--calib", choices=["platt", "isotonic"], default=None, help="force a calibrator (default: gate-selected)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tree = TreeData.load(Path(args.tree))
    ins = generate_instances(cfg["pool"]["n"], cfg["pool"]["seed"], cfg["pool"]["prefix"])
    assert [i.instance_id for i in ins] == list(tree.instance_ids)
    J = json.loads(Path(args.results).read_text())
    lc = tree.leaf_correct.astype(int)
    pos = neg = 0
    for i, j in ((4, 2), (4, 1), (2, 1)):
        for base in (0, 7 - i - j):
            d = lc[:, base | i | j] - lc[:, base | i] - lc[:, base | j] + lc[:, base]
            pos, neg = pos + int((d > 0).sum()), neg + int((d < 0).sum())
    res = {"label": "EXPLORATORY; seen pools; predictions in docs/09 written first", "model": cfg["model"]["id"], "dataset_hash": dataset_hash(ins),
           "lambdas": list(LAMBDAS), "ms": list(MS), "bootstrap": args.boot, "reps": args.reps,
           "second_differences": {"complements": pos, "substitutes": neg, "total_pairs": 6 * tree.n},
           "pattern_accuracy": lc.mean(0).tolist()}
    t0 = time.time()
    for r, s in enumerate(cfg["partition_seeds"]):
        prior = J[f"partition_{r}"]["fit_info"]["mup"]
        res[f"partition_{r}"] = partition(tree, ins, cfg, prior, s, args.boot if r in args.boot_partitions else 0, args.reps, args.calib)
        print(f"partition {r} done {time.time() - t0:.0f}s", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
