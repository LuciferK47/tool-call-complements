#!/usr/bin/env python3
"""Horizon study on every available run -> results/scaling/<tag>.json (rule-by-rule U-AUC for each signal, partition, horizon m).

    python scripts/run_scaling.py --kind text  --tag q35_2b --model Qwen3.5-2B --sidecar data/trees_text/q35_2b.json --seed 20261101
    python scripts/run_scaling.py --kind tree  --tag q35_0p8b --model Qwen3.5-0.8B --tree <tree.npz> --config configs/q35_0p8b.yaml --results <results.json>
    python scripts/run_scaling.py --kind light --tag ... --light out/light/light_x.npz --seed ...
    python scripts/run_scaling.py --kind deep  --tag arith6_q35_2b --model Qwen3.5-2B --deep out/deep/arith6_q35_2b.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolcomp import scaling
from toolcomp.config import load_config
from toolcomp.envs.arith import generate_instances
from toolcomp.tree import TreeData

SEEDS = [20261111, 20261112, 20261113]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("text", "tree", "light", "deep"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--sidecar"), ap.add_argument("--seed", type=int), ap.add_argument("--tree"), ap.add_argument("--config")
    ap.add_argument("--results"), ap.add_argument("--light"), ap.add_argument("--deep")
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--price-scale", type=float, default=1.0, help="multiply every per-call price (0.5 gives K=6 parts the same total price range as K=3)")
    ap.add_argument("--noisy-tool", action="store_true", help="also run Belief-DP with the measured tool accuracy per level (for tools that are not exact, e.g. retrieval)")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--signals", nargs="+", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "scaling"))
    a = ap.parse_args()
    t0 = time.time()
    if a.kind == "text":
        run = scaling.run_from_text(a.tag, a.model, a.sidecar, a.seed)
    elif a.kind == "tree":
        cfg = load_config(a.config)
        tree = TreeData.load(Path(a.tree))
        ins = generate_instances(cfg["pool"]["n"], cfg["pool"]["seed"], cfg["pool"]["prefix"])
        J = json.loads(Path(a.results).read_text())
        run = scaling.run_from_tree(a.tag, a.model, tree, ins, [J[f"partition_{r}"]["fit_info"]["mup"] for r in range(len(SEEDS))])
    elif a.kind == "light":
        run = scaling.run_from_light(a.tag, a.model, a.light, Path(a.light).with_suffix(".json"), a.seed)
    else:
        run = scaling.run_from_deep(a.tag, a.model, a.deep)
    res = {"tag": run.tag, "model": run.model, "env": run.env, "K": run.K, "n": run.n, "note": run.note, "price_scale": a.price_scale, "ms": a.ms, "reps": a.reps,
           "step_accuracy": scaling.step_accuracy(run), "x_by_m": {str(m): scaling.x_of(run, m) for m in a.ms},
           "never_accuracy": float(run.leaf[:, 0].mean()), "all_call_accuracy": float(run.leaf[:, -1].mean())}
    if a.noisy_tool:
        res["noisy_tool"] = True
    if run.K == 3:
        c, s = scaling.complement_counts(run)
        res["second_differences"] = {"complements": c, "substitutes": s}
    res["runs"] = scaling.evaluate_run(run, SEEDS, tuple(a.ms), a.reps, a.boot, signals=a.signals, price_scale=a.price_scale, noisy=a.noisy_tool)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run.tag}.json").write_text(json.dumps(res))
    print(f"{run.tag}: n={run.n} K={run.K} p_j={np.round(res['step_accuracy'], 3).tolist()} done in {time.time() - t0:.0f}s -> {out / (run.tag + '.json')}")


if __name__ == "__main__":
    main()
