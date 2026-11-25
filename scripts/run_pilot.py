"""Dev pilot on a SEPARATE seed: real model, real tool, exhaustive outcome tree on a small pool.

Measures (all from real runs): per-step no-call accuracy by difficulty bin, never/always accuracy, share of decision
nodes where calling helps (Delta=1 under always-call continuation), runtime per node, and paired-difference SDs
for the power calculation. Writes results/pilot/pilot.json. Nothing here touches the main pool or any test data.
Usage: python scripts/run_pilot.py [--config configs/pilot.yaml]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import ROOT, make_runner, manifest, write_json
from toolcomp.config import load_config
from toolcomp.envs.arith import generate_instances
from toolcomp.eval_harness import LAMBDAS, mde_paired, u_auc_instances, utility_matrix
from toolcomp.probes import best_global_pattern, hindsight_utility, rollout_reference_actions
from toolcomp.tree import LEVEL_OF, N_NODES, TreeBuilder, node_labels, pattern_actions, walk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--out", default=None, help="results subfolder (default: config name -> results/<name>/pilot.json)")
    ap.add_argument("--n", type=int, default=None, help="smoke test: first N instances only")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ins = generate_instances(cfg["pool"]["n"], cfg["pool"]["seed"], cfg["pool"]["prefix"])
    if args.n:
        ins = ins[:args.n]
    out_name = args.out or cfg["name"]
    cache_dir = ROOT / "cache" / (cfg["name"] + (f"_n{args.n}" if args.n else ""))      # keyed on the config: never resumes another model's nodes
    runner = make_runner(cfg)
    t0 = time.perf_counter()
    tb = TreeBuilder(runner, cfg["layers"], batch_size=cfg["batch_size"], max_new_tokens=cfg["max_new_tokens"],
                     parser=cfg.get("answer_parser", "first_int"))
    tree = tb.build(ins, cache_dir, cfg["name"])
    wall = time.perf_counter() - t0
    tree.save(cache_dir / f"{cfg['name']}_tree.npz")

    n = tree.n
    bins = ["easy", "medium", "hard"]
    # per-step no-call accuracy by difficulty bin (local correctness of the model's own step answer given its input)
    acc_bin = {}
    for b, name in enumerate(bins):
        vals = []
        for k in range(N_NODES):
            lv = int(LEVEL_OF[k])
            m = (tree.step_bin[:, lv] == b) & tree.v_in_ok[:, k]
            vals.append(tree.local_no_correct[m, k])
        v = np.concatenate(vals)
        acc_bin[name] = {"n_nodes": int(len(v)), "step_accuracy": float(v.mean()) if len(v) else None}
    # parse failures
    parse_fail = float(1 - tree.v_no_ok.mean())

    yc, yn = node_labels(tree, 1)
    delta = yc - yn
    reachable_on_always = np.array([0, 2, 6])             # nodes visited by always-call (prefix all ones)
    prev_all = float((delta > 0).mean())
    never_acc = float(tree.leaf_correct[:, 0].mean())
    always_acc = float(tree.leaf_correct[:, 7].mean())

    lam = list(cfg["lambdas"])
    def per_inst_uauc(correct_fn):
        U = np.zeros((n, len(lam)))
        for j, l in enumerate(lam):
            c, k = correct_fn(l)
            U[:, j] = c - l * k
        return u_auc_instances(U, lam)
    never = per_inst_uauc(lambda l: walk(tree, np.zeros((n, N_NODES), int))[:2])
    always = per_inst_uauc(lambda l: walk(tree, np.ones((n, N_NODES), int))[:2])
    hind = per_inst_uauc(lambda l: hindsight_utility(tree, l))
    roll = per_inst_uauc(lambda l: walk(tree, rollout_reference_actions(tree, l))[:2])
    glob = per_inst_uauc(lambda l: walk(tree, np.tile(pattern_actions(best_global_pattern(tree, l)), (n, 1)))[:2])
    sds = {"sd_rollout_minus_global": float(np.std(roll - glob, ddof=1)), "sd_hindsight_minus_global": float(np.std(hind - glob, ddof=1)),
           "sd_global_minus_never": float(np.std(glob - never, ddof=1)), "sd_always_minus_never": float(np.std(always - never, ddof=1))}
    rep_sd = float(max(sds["sd_rollout_minus_global"], sds["sd_global_minus_never"]))
    sec_per_node = float(sum(tree.meta["level_seconds_this_run"]) / max(1, n * N_NODES))
    out = {
        "manifest": manifest(cfg, {"pilot_wall_seconds": wall}),
        "n_instances": n, "step_accuracy_by_bin": acc_bin, "parse_failure_rate": parse_fail,
        "never_call_accuracy": never_acc, "always_call_accuracy": always_acc,
        "share_nodes_delta_positive": prev_all,
        "u_auc_points": {"never": float(never.mean()), "always": float(always.mean()), "best_global_pattern": float(glob.mean()),
                         "rollout_reference": float(roll.mean()), "hindsight": float(hind.mean())},
        "paired_sds_points": sds, "representative_sd_points": rep_sd,
        "mde_points_by_n": {str(k): float(mde_paired(rep_sd, k)) for k in (100, 200, 300, 400, 600, 800, 1000)},
        "seconds_per_node_incl_generation": sec_per_node, "avg_gen_tokens": float(tree.n_gen_tokens.mean()),
        "avg_prompt_tokens": float(tree.n_prompt_tokens.mean()),
        "answer_parser": cfg.get("answer_parser", "first_int"), "parse_categories": tree.meta.get("parse_categories"),
        "tool_failure_rate": float((tree.v_in_ok & ~tree.v_call_ok).sum() / max(1, tree.v_in_ok.sum())),
        "thinking_text_nodes": int((tree.meta.get("parse_categories") or {}).get("thinking_text", 0)),
        "env": tree.meta.get("env"), "resolved_revision": tree.meta.get("resolved_revision"),
    }
    write_json(ROOT / "results" / out_name / "pilot.json", out)
    # parser report: every distinct reply that is not a plain integer, with its category (attached by the remote agent)
    from toolcomp.tree import parse_category
    rows = {}
    for r_ in tree.texts:
        for tx in r_:
            c = parse_category(tx)
            if c != "integer":
                rows.setdefault((c, tx[:60]), 0)
                rows[(c, tx[:60])] += 1
    lines = [f"{n:5d}  {c:24s}  {tx!r}" for (c, tx), n in sorted(rows.items(), key=lambda kv: -kv[1])]
    (ROOT / "results" / out_name).mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / out_name / "parser_report.txt").write_text("\n".join(lines) + "\n")
    print(__import__("json").dumps({k: v for k, v in out.items() if k != "manifest"}, indent=1, default=float))


if __name__ == "__main__":
    main()
