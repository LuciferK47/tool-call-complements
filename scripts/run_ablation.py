"""Training-size x head ablation on partition 0. EXPLORATORY.

Frozen evaluation: cal, gate and test are exactly partition 0's splits of the main pool. Training subsets are NESTED and identical for every
head: n=|train| (the original train split), then train + the first m extra-pool instances (fixed seeded order). Every variant is refitted on each
subset (hyper-parameters by grouped CV on that subset only), calibrated on cal, evaluated on gate and test. Any 'best' cell is chosen on GATE.
Usage: python scripts/run_ablation.py --config configs/q35_2b.yaml [--sizes 450,1000,2000,4450] [--heads H1,...] [--out name] [--max-extra N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import ROOT, append_budget, manifest, write_json
from toolcomp.ablation import HEADS, HEAD_DESC, fit_head, plan_subsets
from toolcomp.config import load_config
from toolcomp.envs.arith import generate_instances
from toolcomp.eval_harness import paired_bootstrap, u_auc_instances, utility_matrix
from toolcomp.probes import PolicyConsistentMUP
from toolcomp.probes import TwoHeadMUP, select_layer_and_C
from toolcomp.splits import make_splits
from toolcomp.tree import N_NODES, TreeData, node_labels, walk


def load_extra(cfg, croot: Path, chunks=None):
    ex = cfg["extra_pool"]
    parts = []
    for c in range(ex["n"] // ex["chunk"]):
        f = croot / (cfg["name"] + "_extra") / f"{cfg['name']}_xchunk{c}_tree.npz"
        if f.exists():
            parts.append(TreeData.load(f))
        elif chunks is None:
            raise FileNotFoundError(f"missing extra chunk {c}: {f}")
    return TreeData.concat(parts)


def evaluate(tree: TreeData, act_fn, lams, B, seed):
    C, K = [], []
    for lam in lams:
        c, k, _ = walk(tree, act_fn(lam))
        C.append(c)
        K.append(k)
    U = utility_matrix(np.stack(C, 1), np.stack(K, 1), lams)
    return u_auc_instances(U, lams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--heads", default=",".join(HEADS))
    ap.add_argument("--max-extra", type=int, default=None, help="smoke only: cap the number of extra instances used")
    ap.add_argument("--cache-root", default=str(ROOT / "cache"))
    ap.add_argument("--no-mup-pc", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = args.out or cfg["name"]
    croot = Path(args.cache_root)
    lams = list(cfg["lambdas"])
    B = cfg["bootstrap"]["n_resamples"]
    seed = cfg["partition_seeds"][0]
    main_tree = TreeData.load(croot / cfg["name"] / f"{cfg['name']}_tree.npz")
    extra = load_extra(cfg, croot, chunks=True if args.max_extra else None)
    if args.max_extra:
        extra = extra.subset(np.arange(min(args.max_extra, extra.n)))
    # ---- leakage checks (abort on any violation)
    main_ids = set(main_tree.instance_ids)
    assert not (main_ids & set(extra.instance_ids)), "extra pool shares instance ids with the main pool"
    ex_cfg = cfg["extra_pool"]
    xt = {x.problem_text(): x.instance_id for x in generate_instances(ex_cfg["n"], ex_cfg["seed"], ex_cfg["prefix"])}
    mt = {x.problem_text() for x in generate_instances(cfg["pool"]["n"], cfg["pool"]["seed"], cfg["pool"]["prefix"])}
    assert not (set(xt) & mt), "extra pool shares problem texts with the main pool"
    sizes = [int(s) for s in (args.sizes.split(",") if args.sizes else cfg.get("ablation_sizes", [450, 1000, 2000, 4450]))]
    sp, subsets, order = plan_subsets(main_tree.instance_ids, extra.instance_ids, cfg["splits"], seed, sizes)
    tr0, cal, gate, test = (main_tree.subset(sp[k]) for k in ("train", "cal", "gate", "test"))
    resdir = ROOT / "results" / out
    resdir.mkdir(parents=True, exist_ok=True)
    write_json(resdir / "train_subsets.json", {"sizes": sizes, "seed": seed, "subsets": subsets,
                                              "cal": sp["cal"].tolist(), "gate": sp["gate"].tolist(), "test": sp["test"].tolist()})
    heads = [h for h in args.heads.split(",") if h]
    cells, per_inst = {}, {}
    pc_cells = {}
    t_all = time.perf_counter()
    for s in sizes:
        tr = TreeData.concat([main_tree.subset(np.array(subsets[s]["main_train_indices"])),
                              extra.subset(np.array(subsets[s]["extra_indices"], dtype=int))]) if subsets[s]["extra_indices"] else tr0
        assert tr.n == s or args.max_extra, (tr.n, s)
        for h in heads:
            t0 = time.perf_counter()
            key = f"{h}@{s}"
            try:
                fitted = fit_head(h, tr, cal, cfg["layers"], args.device, seed)
                ug = evaluate(gate, lambda l: fitted.actions(gate, l), lams, B, seed)
                ut = evaluate(test, lambda l: fitted.actions(test, l), lams, B, seed)
                per_inst[key] = ut
                cells[key] = {"head": h, "desc": HEAD_DESC[h], "n_train": int(tr.n), "gate_u_auc": float(ug.mean()), "test_u_auc": float(ut.mean()),
                              "fit": fitted.info, "seconds": time.perf_counter() - t0}
            except Exception as e:                                # an explicit failed cell, never a silent gap
                cells[key] = {"head": h, "n_train": int(tr.n), "failed": f"{type(e).__name__}: {e}"}
        if not args.no_mup_pc:
            try:
                t0 = time.perf_counter()
                lay, C, _, _ = select_layer_and_C(tr, dict(zip(("yc", "yn"), node_labels(tr, 1))), cfg["layers"], seed)
                pc = PolicyConsistentMUP(lay, C, seed=seed).fit(tr, cal, lams)
                ug = evaluate(gate, lambda l: pc.actions(gate, l), lams, B, seed)
                ut = evaluate(test, lambda l: pc.actions(test, l), lams, B, seed)
                per_inst[f"mup_pc@{s}"] = ut
                pc_cells[f"mup_pc@{s}"] = {"n_train": int(tr.n), "gate_u_auc": float(ug.mean()), "test_u_auc": float(ut.mean()), "layer": lay, "C": C,
                                           "seconds": time.perf_counter() - t0}
            except Exception as e:
                pc_cells[f"mup_pc@{s}"] = {"n_train": int(tr.n), "failed": f"{type(e).__name__}: {e}"}
        print(f"size {s} done", flush=True)
    # paired differences vs the smallest size, per head (test split), bootstrap CI
    diffs = {}
    s0 = sizes[0]
    for h in heads + ([] if args.no_mup_pc else ["mup_pc"]):
        for s in sizes[1:]:
            a, b = per_inst.get(f"{h}@{s}"), per_inst.get(f"{h}@{s0}")
            if a is not None and b is not None:
                diffs[f"{h}@{s}_minus_{h}@{s0}"] = paired_bootstrap(a, b, B, seed=seed)
    ok = [(k, v["gate_u_auc"]) for k, v in cells.items() if "failed" not in v]
    best_gate = max(ok, key=lambda kv: kv[1])[0] if ok else None
    res = {"frozen_eval": {"n_cal": int(cal.n), "n_gate": int(gate.n), "n_test": int(test.n), "partition_seed": seed},
           "sizes": sizes, "heads": heads, "cells": cells, "mup_pc_by_size": pc_cells, "paired_diff_vs_smallest_size_test": diffs,
           "best_cell_chosen_on_gate": best_gate,
           "best_cell_test_u_auc": (cells[best_gate]["test_u_auc"] if best_gate else None),
           "note": "exploratory; every cell is reported whether or not it improves; selection on GATE only",
           "manifest": manifest(cfg, {"stage": "ablate"})}
    write_json(resdir / "ablations.json", res)
    append_budget(resdir / "budget.json", {"stage": "ablate", "run": cfg["name"], "wall_seconds": time.perf_counter() - t_all,
                                           "gpu_seconds": 0.0, "peak_gpu_mem_gb": None})
    print(json.dumps({k: (round(v.get("test_u_auc", float("nan")), 2) if "failed" not in v else v["failed"]) for k, v in cells.items()}, indent=1))


if __name__ == "__main__":
    main()
