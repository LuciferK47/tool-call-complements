"""POST-HOC exploratory analysis: does a hidden-state head add anything beyond the model's own answer confidence?
Reads a cached outcome tree, reuses the MUP layer/C from results/<out>/results.json when present (else selects on train), fits
the two confidence variants (src/posthoc.py) and compares them with MUP and the answer-confidence threshold on the frozen test split.
Writes results/<out>/posthoc.json. Labelled post hoc: no registered verdict changes. Light (CPU)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import ROOT, manifest, write_json
from toolcomp.config import load_config
from toolcomp.eval_harness import holm, paired_bootstrap, u_auc_instances, utility_matrix
from toolcomp.posthoc import ConfidenceMUP
from toolcomp.probes import ThresholdPolicy, TwoHeadMUP, select_layer_and_C
from toolcomp.splits import make_splits
from toolcomp.tree import N_NODES, TreeData, node_labels, walk

CONTRASTS = [("mup_conf_stack", "mup"), ("mup_conf_stack", "conf_thr"), ("mup_conf_only", "conf_thr"), ("mup_conf_stack", "mup_conf_only")]


def run_actions(tree, fn, lams):
    C, K = [], []
    for lam in lams:
        c, k, _ = walk(tree, fn(lam))
        C.append(c)
        K.append(k)
    return np.stack(C, 1), np.stack(K, 1)


def posthoc_partition(pool, cfg, r, seed, prior=None):
    lams = list(cfg["lambdas"])
    B = cfg["bootstrap"]["n_resamples"]
    sp = make_splits(pool.instance_ids, cfg["splits"], seed)
    tr, cal, test = (pool.subset(sp[k]) for k in ("train", "cal", "test"))
    yc, yn = node_labels(tr, 1)
    if prior and f"partition_{r}" in prior:
        fi = prior[f"partition_{r}"]["fit_info"]["mup"]
        layer, C, src = fi["layer"], fi["C"], "reused from results.json"
    else:
        layer, C, _, _ = select_layer_and_C(tr, {"yc": yc, "yn": yn}, cfg["layers"], seed)
        src = "selected on train"
    mup = TwoHeadMUP(layer, C).fit(tr, cal)
    q = np.unique(np.quantile(cal.mean_logprob.reshape(-1), np.linspace(0.02, 0.98, 49)))
    conf = ThresholdPolicy("conf_thr", lambda t: t.mean_logprob, "low", q).tune(cal, lams)
    pols = {"mup": mup, "conf_thr": conf,
            "mup_conf_only": ConfidenceMUP(layer, C, "conf_only").fit(tr, cal),
            "mup_conf_stack": ConfidenceMUP(layer, C, "stack").fit(tr, cal)}
    U = {}
    summ = {}
    for n, pol in pols.items():
        Cm, Km = run_actions(test, lambda l, pol=pol: pol.actions(test, l), lams)
        ua = u_auc_instances(utility_matrix(Cm, Km, lams), lams)
        U[n] = ua
        summ[n] = {"u_auc_points": float(ua.mean()), "calls_per_instance_by_lambda": Km.mean(0).tolist(), "accuracy_by_lambda": Cm.mean(0).tolist()}
    con = {f"{a}_vs_{b}": paired_bootstrap(U[a], U[b], B, seed=seed) for a, b in CONTRASTS}
    adj = holm({k: v["p"] for k, v in con.items()})
    for k in con:
        con[k]["p_holm_guidance_only"] = adj[k]
    out = {"seed": seed, "mup_layer": layer, "mup_C": C, "layer_C_source": src, "summary_test": summ, "contrasts_test": con}
    if prior and f"partition_{r}" in prior:
        ref = prior[f"partition_{r}"]["summary"]["test"]
        out["consistency_with_results_json"] = {n: abs(summ[n]["u_auc_points"] - ref[n]["u_auc_points"]) for n in ("mup", "conf_thr") if n in ref}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="results/<out>/ (reads results.json there if present)")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cache = Path(args.cache_dir) if args.cache_dir else ROOT / "cache" / cfg["name"]
    pool = TreeData.load(cache / f"{cfg['name']}_tree.npz")
    rj = ROOT / "results" / args.out / "results.json"
    prior = json.loads(rj.read_text()) if rj.exists() else None
    parts = {f"partition_{r}": posthoc_partition(pool, cfg, r, s, prior) for r, s in enumerate(cfg["partition_seeds"])}
    top = {"label": "POST HOC, exploratory: not part of any pre-registered family; Holm over the four listed contrasts is for guidance only",
           "manifest": manifest(cfg, {"n_pool": pool.n}), **parts}
    write_json(ROOT / "results" / args.out / "posthoc.json", top)
    for pk, p in parts.items():
        print(pk, {n: round(v["u_auc_points"], 2) for n, v in p["summary_test"].items()})


if __name__ == "__main__":
    main()
