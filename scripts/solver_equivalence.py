"""Do the scipy (frozen, float64) and torch (float32 GPU L-BFGS) probe solvers give the same MUP on a REAL tree?
Fits TwoHeadMUP on partition one with each solver at the layer/C of results/<out>/results.json (or selects on train), then reports
max |delta p| of both heads, decision agreement per lambda, and the test U-AUC difference. Writes results/<out>/solver_equivalence.json.
Without CUDA the torch arm is recorded as skipped (never silently replaced)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import ROOT, write_json
from scripts.run_posthoc import run_actions
from toolcomp.config import load_config
from toolcomp.eval_harness import u_auc_instances, utility_matrix
from toolcomp.probes import SoftLogReg, TwoHeadMUP, select_layer_and_C
from toolcomp.splits import make_splits
from toolcomp.tree import TreeData, node_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cache = Path(args.cache_dir) if args.cache_dir else ROOT / "cache" / cfg["name"]
    pool = TreeData.load(cache / f"{cfg['name']}_tree.npz")
    seed, lams = cfg["partition_seeds"][0], list(cfg["lambdas"])
    sp = make_splits(pool.instance_ids, cfg["splits"], seed)
    tr, cal, test = (pool.subset(sp[k]) for k in ("train", "cal", "test"))
    rj = ROOT / "results" / args.out / "results.json"
    if rj.exists():
        fi = json.loads(rj.read_text())["partition_0"]["fit_info"]["mup"]
        layer, C = fi["layer"], fi["C"]
    else:
        yc, yn = node_labels(tr, 1)
        layer, C, _, _ = select_layer_and_C(tr, {"yc": yc, "yn": yn}, cfg["layers"], seed)
    res = {"config": cfg["name"], "layer": layer, "C": C, "arms": {}}
    heads, ua = {}, {}
    import torch
    for solver in ("scipy", "torch"):
        if solver == "torch" and not torch.cuda.is_available():
            res["arms"]["torch"] = "skipped: CUDA not available"
            continue
        SoftLogReg.SOLVER = solver
        m = TwoHeadMUP(layer, C).fit(tr, cal)
        heads[solver] = m.heads(test)
        Cm, Km = zip(*[run_actions(test, lambda l, m=m: m.actions(test, l), [lam]) for lam in lams])
        Cm, Km = np.concatenate(Cm, 1), np.concatenate(Km, 1)
        ua[solver] = (u_auc_instances(utility_matrix(Cm, Km, lams), lams), (m.delta(test)))
        res["arms"][solver] = {"u_auc_points": float(ua[solver][0].mean())}
    SoftLogReg.SOLVER = "scipy"
    if "torch" in heads:
        (pc_s, pn_s), (pc_t, pn_t) = heads["scipy"], heads["torch"]
        res["max_abs_diff_p_call"] = float(np.abs(pc_s - pc_t).max())
        res["max_abs_diff_p_no"] = float(np.abs(pn_s - pn_t).max())
        d_s, d_t = ua["scipy"][1], ua["torch"][1]
        res["decision_agreement_by_lambda"] = {f"{l:.2f}": float(((d_s > l) == (d_t > l)).mean()) for l in lams}
        res["u_auc_difference_points_torch_minus_scipy"] = res["arms"]["torch"]["u_auc_points"] - res["arms"]["scipy"]["u_auc_points"]
    write_json(ROOT / "results" / args.out / "solver_equivalence.json", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
