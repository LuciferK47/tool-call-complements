"""Composite-horizon experiment on exact outcome trees: same signal, different decision rule.

  python scripts/run_horizon.py --tree <path/to/*_tree.npz> --config configs/q35_0p8b.yaml --results <results.json> --out results/horizon_q35_0p8b.json
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
from toolcomp.horizon import evaluate_signals, surface_nobin
from toolcomp.splits import make_splits
from toolcomp.tree import LEVEL_OF, N_NODES, TreeData

MS = (1, 2, 4, 8)
_RULES_UNUSED = ("never", "always", "all_or_nothing", "local", "optimistic", "pessimistic", "thr_transfer", "thr_retuned", "belief_dp", "hindsight")


def partition(tree, ins, cfg, prior, seed, B, reps, calib=None):
    S0, S1, S4 = surface_nobin(tree, ins), hidden_features(tree, prior["layer"]), draft_features(tree)
    feats = {"S0": S0, "S1": S1, "S4": S4, "S0+S4": np.concatenate([S0, S4], -1)}
    Cs = {"S0": 1.0, "S1": prior["C"], "S4": 1.0, "S0+S4": 1.0}
    return evaluate_signals(feats, tree.local_no_correct, tree.leaf_correct, tree.instance_ids, cfg["splits"], seed, Cs, B, reps, calib, MS)


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
    res = {"label": "EXPLORATORY; seen pools; predictions in configs/predictions.yaml", "model": cfg["model"]["id"], "dataset_hash": dataset_hash(ins),
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
