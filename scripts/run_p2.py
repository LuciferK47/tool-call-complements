"""P2-I1 black-box track with REAL temperature samples from the model (text only). Pre-registered addendum applies.

Usage: python scripts/run_p2.py [--config configs/p2.yaml]
Sampling is cached in cache/<primary name>/p2_samples_chunk*.json and is resumable.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import ROOT, make_runner, manifest, write_json
from toolcomp.config import load_config
from toolcomp.envs.arith import generate_instances, node_prompt
from toolcomp.eval_harness import holm, paired_bootstrap, u_auc_instances
from toolcomp.p2 import build_table, mean_samples, run_all
from toolcomp.splits import make_splits
from toolcomp.tree import TreeData
from toolcomp.tree import PARSERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p2.yaml"))
    ap.add_argument("--out", default="main", help="results subfolder (default main = the frozen primary run)")
    ap.add_argument("--max-queries", type=int, default=None, help="smoke test only: use the first N cal and N test queries")
    args = ap.parse_args()
    c2 = load_config(args.config)
    cp = load_config(ROOT / c2["primary_config"])
    lams, rhos, m_max = list(c2["lambdas"]), list(c2["rhos"]), c2["sampling"]["m_max"]
    cache = ROOT / "cache" / cp["name"]
    tree = TreeData.load(cache / f"{cp['name']}_tree.npz")
    parse = PARSERS[cp.get("answer_parser", "first_int")]
    sfx = f"_q{args.max_queries}" if args.max_queries else ""
    ins = generate_instances(cp["pool"]["n"], cp["pool"]["seed"], cp["pool"]["prefix"])
    assert [x.instance_id for x in ins] == tree.instance_ids
    sp = make_splits(tree.instance_ids, cp["splits"], cp["partition_seeds"][0])
    cal_idx = np.sort(np.concatenate([sp["cal"], sp["gate"]]))
    test_idx = sp["test"]
    if args.max_queries:
        cal_idx, test_idx = cal_idx[:args.max_queries], test_idx[:args.max_queries]
    q_idx = np.concatenate([cal_idx, test_idx])
    prompts = [node_prompt(ins[i], 0, ins[i].first, []) for i in q_idx]

    runner = make_runner(cp)
    t0 = time.perf_counter()
    texts = []
    ch = c2["sampling"]["chunk"]
    for ci, s in enumerate(range(0, len(prompts), ch)):
        f = cache / f"p2_samples{sfx}_chunk{ci:03d}.json"
        if f.exists():
            texts.extend(json.loads(f.read_text()))
            continue
        out = runner.sample(prompts[s:s + ch], m_max, c2["sampling"]["temperature"], cp["max_new_tokens"], c2["sampling"]["batch_size"],
                            c2["sampling"]["seed"] + ci * 1000)
        f.write_text(json.dumps(out))
        texts.extend(out)
        print(f"sampled chunk {ci}: {len(texts)}/{len(prompts)} queries", flush=True)
    sample_seconds = time.perf_counter() - t0

    parsed = [[parse(t) for t in row] for row in texts]
    valid = np.array([[v is not None for v in row] for row in parsed])
    samp = np.array([[v if v is not None else 0 for v in row] for row in parsed], dtype=np.int64)
    gold = np.array([ins[i].gold_values[0] for i in q_idx], dtype=np.int64)
    _, s, ok = build_table(samp, valid, gold, m_max)
    tool_ok = (tree.v_call_ok[q_idx, 0] & (tree.v_call[q_idx, 0] == gold))
    data = dict(s=s, ok=ok, tool_ok=tool_ok, greedy_ok=tree.local_no_correct[q_idx, 0], greedy_lp=tree.mean_logprob[q_idx, 0])
    nc = len(cal_idx)
    cal = {k: v[:nc] for k, v in data.items()}
    tst = {k: v[nc:] for k, v in data.items()}

    out, chosen, calib = run_all(cal, tst, lams, rhos, m_max)
    B = c2["bootstrap"]["n_resamples"]
    p_tool = float(cal["tool_ok"].mean())
    res = {"n_cal_queries": int(nc), "n_test_queries": int(len(test_idx)), "m_max": m_max, "p_tool_cal": p_tool,
           "parse_failure_rate_samples": float(1 - valid.mean()), "answer_parser": cp.get("answer_parser", "first_int"),
           "thinking_text_samples": int(sum(("<think>" in x) or ("Thinking Process" in x) for row in texts for x in row)), "sample_seconds_this_invocation": sample_seconds,
           "direct_accuracy_single_sample_test": float(np.mean(tst["ok"][:, 0])),
           "greedy_accuracy_test": float(np.mean(tst["greedy_ok"])), "by_rho": {}}
    for rho in rhos:
        per = {}
        for meth in out:
            U = np.stack([out[meth][l][rho] for l in lams], 1)
            per[meth] = u_auc_instances(U, lams)
        r = {"net_u_auc_points": {m: float(v.mean()) for m, v in per.items()}}
        prim = {"sequential_minus_fixed_k": paired_bootstrap(per["sequential"], per["fixed_k"], B, seed=1),
                "sequential_minus_logprob_thr": paired_bootstrap(per["sequential"], per["logprob_thr"], B, seed=2)}
        adj = holm({k: v["p"] for k, v in prim.items()})
        for k in prim:
            prim[k]["p_holm"] = adj[k]
        r["contrasts"] = prim
        r["exploratory"] = {f"sequential_minus_{m}": paired_bootstrap(per["sequential"], per[m], B, seed=3) for m in ("never", "always")}
        r["mean_samples_sequential_by_lambda"] = {f"{l:.2f}": mean_samples(tst, calib, p_tool, l, chosen["z"][f"{l:.2f}_{rho:.2f}"], m_max) for l in lams}
        r["chosen_z"] = {f"{l:.2f}": chosen["z"][f"{l:.2f}_{rho:.2f}"] for l in lams}
        r["chosen_k"] = {f"{l:.2f}": chosen["k"][f"{l:.2f}_{rho:.2f}"] for l in lams}
        res["by_rho"][f"{rho:.2f}"] = r
    # break-even rho between sequential and the best fixed k (linear interpolation across the rho grid)
    diffs = [(rho, res["by_rho"][f"{rho:.2f}"]["contrasts"]["sequential_minus_fixed_k"]["diff"]) for rho in rhos]
    be = None
    for (r0, d0), (r1, d1) in zip(diffs, diffs[1:]):
        if d0 == 0 or d0 * d1 < 0:
            be = r0 + (r1 - r0) * (0 - d0) / (d1 - d0) if d1 != d0 else r0
            break
    res["break_even_rho_vs_fixed_k"] = be
    res["manifest"] = manifest(cp, {"p2_config_hash": c2["_config_hash"], "sampling": c2["sampling"]})
    write_json(ROOT / "results" / args.out / "p2_results.json", res)
    for k, v in res["by_rho"].items():
        print("rho", k, {m: round(x, 2) for m, x in v["net_u_auc_points"].items()})


if __name__ == "__main__":
    main()
