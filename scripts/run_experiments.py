"""Main experiment: gate, policies, primary contrasts (Holm), rank reversal, ablations, resampled partitions.

Pre-registered analysis. Everything is computed from the cached
outcome tree (real model generations, real tool results, real hidden states). Run AFTER freezing the config.
Usage: python scripts/run_experiments.py --config configs/primary.yaml
"""

import os
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import ROOT, manifest, write_json
from toolcomp.config import load_config
from toolcomp.eval_harness import (LAMBDAS, any_call_utility_matrix, holm, kendall_tau_boot, mde_paired,
                              paired_bootstrap, u_auc_instances, utility_matrix)
from toolcomp.probes import (PlanProbe, PolicyConsistentMUP, select_root_layer_and_C, DeltaRegression, RidgeReg, Standardizer, ThresholdPolicy, TwoHeadMUP, best_global_pattern,
                        fit_need_probe, grouped_folds, hindsight_utility, node_features,
                        rollout_reference_actions, select_layer_and_C)
from toolcomp.splits import make_splits
from toolcomp.tree import N_NODES, TreeData, node_labels, pattern_actions, walk

PROB_GRID = np.linspace(0.02, 0.98, 49)


def mean_ci(x: np.ndarray, n_resamples: int, seed: int = 0):
    """Percentile bootstrap 95% CI for the mean of per-instance values."""
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_resamples)
    for s0 in range(0, n_resamples, 500):
        e = min(s0 + 500, n_resamples)
        means[s0:e] = x[rng.integers(0, n, size=(e - s0, n))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def taxonomy(tree: TreeData, actions: np.ndarray):
    """Counts over visited decision nodes (labels under always-call continuation):
    over-call  = CALL where the direct answer would already have led to a correct final answer;
    under-call = NOCALL where calling would have fixed an otherwise wrong final answer (missed necessary call)."""
    yc, yn = node_labels(tree, 1)
    n = tree.n
    ar = np.arange(n)
    nid = np.zeros(n, int)
    visited = over = under = 0
    for _ in range(3):
        a = actions[ar, nid]
        visited += n
        over += int(((a == 1) & (yn[ar, nid] == 1)).sum())
        under += int(((a == 0) & (yc[ar, nid] == 1) & (yn[ar, nid] == 0)).sum())
        nid = 2 * nid + 1 + a
    return {"visited_nodes": visited, "over_call_nodes": over, "under_call_nodes": under,
            "over_call_rate": over / visited, "under_call_rate": under / visited}


def head_diagnostics(mup, tree: TreeData):
    """Quality and spread of the two MUP heads on held-out nodes (watchlist W-11: is Delta-hat degenerate?)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    pc, pn = mup.heads(tree)
    yc, yn = node_labels(tree, 1)
    d, dl = pc - pn, yc - yn

    def auc(y, sc):
        y = np.asarray(y).ravel()
        return float(roc_auc_score(y, np.asarray(sc).ravel())) if len(np.unique(y)) > 1 else None

    return {"auroc_p_call": auc(yc, pc), "auroc_p_no": auc(yn, pn), "auroc_delta_vs_benefit": auc((dl > 0).astype(int), d),
            "brier_p_call": float(np.mean((pc - yc) ** 2)), "brier_p_no": float(np.mean((pn - yn) ** 2)),
            "spearman_delta_vs_label": float(spearmanr(d.ravel(), dl.ravel()).statistic),
            "mean_p_call": float(pc.mean()), "std_p_call": float(pc.std()), "mean_p_no": float(pn.mean()),
            "std_p_no": float(pn.std()), "std_delta_hat": float(d.std()), "benefit_prevalence": float((dl > 0).mean())}


def pool_stats(tree: TreeData):
    from toolcomp.tree import LEVEL_OF
    names = ["easy", "medium", "hard"]
    acc = {}
    for b, nm in enumerate(names):
        vals = []
        for k in range(N_NODES):
            m = (tree.step_bin[:, int(LEVEL_OF[k])] == b) & tree.v_in_ok[:, k]
            vals.append(tree.local_no_correct[m, k])
        v = np.concatenate(vals)
        acc[nm] = {"n_nodes": int(len(v)), "step_accuracy": float(v.mean())}
    yc, yn = node_labels(tree, 1)
    return {"n_instances": tree.n, "step_accuracy_by_bin": acc, "parse_failure_rate": float(1 - tree.v_no_ok.mean()),
            "tool_failure_rate": float((tree.v_in_ok & ~tree.v_call_ok).sum() / max(1, tree.v_in_ok.sum())), "never_call_accuracy": float(tree.leaf_correct[:, 0].mean()),
            "always_call_accuracy": float(tree.leaf_correct[:, 7].mean()),
            "share_nodes_delta_positive": float(((yc - yn) > 0).mean()),
            "mean_gen_tokens_per_node": float(tree.n_gen_tokens.mean()), "mean_prompt_tokens_per_node": float(tree.n_prompt_tokens.mean()),
            "label_generation_seconds": tree.meta.get("build_seconds_this_run"), "device": tree.meta.get("device"),
            "parser": tree.meta.get("parser"), "parse_categories": tree.meta.get("parse_categories"),
            "resolved_revision": tree.meta.get("resolved_revision"), "tree_env": tree.meta.get("env"),
            "nodes_reused_from_cache": tree.meta.get("nodes_reused"), "nodes_generated": tree.meta.get("nodes_generated")}


def run_actions(tree: TreeData, act_fn, lambdas):
    """act_fn(lam) -> (N,7) decision matrix. Returns correct[N,L], calls[N,L] from the real outcome tree."""
    C, K = [], []
    for lam in lambdas:
        c, k, _ = walk(tree, act_fn(lam))
        C.append(c)
        K.append(k)
    return np.stack(C, 1), np.stack(K, 1)


def gen_tokens_wasted(tree: TreeData, act_fn, lambdas, policy_generates_everywhere: bool):
    """Online generation tokens: NOCALL answers always count; policies that must generate BEFORE deciding
    (confidence thresholding) also pay for answers they then discard by calling the tool."""
    out = []
    for lam in lambdas:
        a = act_fn(lam)
        n = tree.n
        nid = np.zeros(n, int)
        tok = np.zeros(n)
        for _ in range(3):
            act = a[np.arange(n), nid]
            gen = tree.n_gen_tokens[np.arange(n), nid]
            tok += np.where(act == 0, gen, gen if policy_generates_everywhere else 0)
            nid = 2 * nid + 1 + act
        out.append(tok)
    return np.stack(out, 1)


def random_matched(tree: TreeData, p_by_lam, lams=LAMBDAS):
    """Exact expectation of a Bernoulli(p) call rule at every step (no sampling noise)."""
    C, K = [], []
    for lam, p in zip(lams, p_by_lam):
        w = np.array([(p ** bin(m).count("1")) * ((1 - p) ** (3 - bin(m).count("1"))) for m in range(8)])
        C.append(tree.leaf_correct.astype(float) @ w)
        K.append(np.full(tree.n, 3 * p))
    return np.stack(C, 1), np.stack(K, 1)


def cv_ridge_alpha(tr: TreeData, layer: int, grid=(10.0, 100.0, 1e3, 1e4), seed=0):
    yc, yn = node_labels(tr, 1)
    Y = (yc - yn)
    X = node_features(tr, layer)
    best, best_s = grid[0], np.inf
    for a in grid:
        errs = []
        for f in grouped_folds(tr.n, 5, seed):
            va = np.zeros(tr.n, bool)
            va[f] = True
            Xtr = X[~va].reshape(-1, X.shape[-1])
            st = Standardizer.fit(Xtr)
            m = RidgeReg(a).fit(st(Xtr), Y[~va].reshape(-1))
            errs.append(np.mean((m.predict(st(X[va].reshape(-1, X.shape[-1]))) - Y[va].reshape(-1)) ** 2))
        if np.mean(errs) < best_s:
            best, best_s = a, float(np.mean(errs))
    return best


def fit_all(tr: TreeData, cal: TreeData, layers, lams, seed, extras=()):
    """Fit every learned policy on TRAIN, calibrate/tune on CAL only. Returns dict name -> action function(tree, lam)."""
    yc, yn = node_labels(tr, 1)
    info = {}
    # --- MUP (two heads)
    lay, C, sc, tab = select_layer_and_C(tr, {"yc": yc, "yn": yn}, layers, seed)
    mup = TwoHeadMUP(lay, C, continuation=1).fit(tr, cal)
    info["mup"] = {"layer": lay, "C": C, "cv_brier": sc}
    # --- MUP variants for ablations
    mup_nocal = TwoHeadMUP(lay, C, continuation=1, calibrate=False).fit(tr, None)
    lay0, C0, _, _ = select_layer_and_C(tr, {"yc": node_labels(tr, 0)[0], "yn": node_labels(tr, 0)[1]}, layers, seed)
    mup_never = TwoHeadMUP(lay0, C0, continuation=0).fit(tr, cal)
    info["mup_never_continuation"] = {"layer": lay0, "C": C0}
    # --- delta regression
    a_best = cv_ridge_alpha(tr, lay, seed=seed)
    dreg = DeltaRegression(lay, alpha=a_best).fit(tr)
    info["delta_reg"] = {"layer": lay, "alpha": a_best}
    # --- pp_step_direct: probe predicts y_no (direct correctness), call iff p_no < tau(lambda)
    layd, Cd, _, _ = select_layer_and_C(tr, {"yn": yn}, layers, seed)
    f_direct = fit_need_probe(tr, layd, Cd, yn)
    pp_direct = ThresholdPolicy("pp_step_direct", f_direct, "low", PROB_GRID).tune(cal, lams)
    info["pp_step_direct"] = {"layer": layd, "C": Cd}
    # --- pp_step_benefit: probe predicts 1[Delta>0], call iff p > tau(lambda)
    ben = ((yc - yn) > 0).astype(float)
    layb, Cb, _, _ = select_layer_and_C(tr, {"ben": ben}, layers, seed)
    f_ben = fit_need_probe(tr, layb, Cb, ben)
    pp_ben = ThresholdPolicy("pp_step_benefit", f_ben, "high", PROB_GRID).tune(cal, lams)
    info["pp_step_benefit"] = {"layer": layb, "C": Cb, "benefit_prevalence_train": float(ben.mean())}
    # --- pp_first: single decision from the t=0 state; need = never-call chain is wrong
    need = 1.0 - tr.leaf_correct[:, 0].astype(float)
    # layer/C for the root probe: reuse the benefit layer search on root features via simple CV over the same grid
    from toolcomp.probes import root_features, SoftLogReg, C_GRID
    best, bs = (layers[0], C_GRID[0]), np.inf
    for L in layers:
        Xr = root_features(tr, L)
        for Cc in C_GRID:
            errs = []
            for f in grouped_folds(tr.n, 5, seed):
                va = np.zeros(tr.n, bool)
                va[f] = True
                st = Standardizer.fit(Xr[~va])
                m = SoftLogReg(Cc, 100).fit(st(Xr[~va]), need[~va])
                errs.append(np.mean((m.predict_proba(st(Xr[va])) - need[va]) ** 2))
            if np.mean(errs) < bs:
                best, bs = (L, Cc), float(np.mean(errs))
    f_root = fit_need_probe(tr, best[0], best[1], need, per_instance_root=True)
    pp_first = ThresholdPolicy("pp_first", f_root, "high", PROB_GRID, per_instance=True).tune(cal, lams)
    info["pp_first"] = {"layer": best[0], "C": best[1], "need_prevalence_train": float(need.mean())}
    # --- confidence threshold (Anchor-lite on the model's own answer log-prob)
    q = np.unique(np.quantile(cal.mean_logprob.reshape(-1), np.linspace(0.02, 0.98, 49)))
    conf = ThresholdPolicy("conf_thr", lambda t: t.mean_logprob, "low", q).tune(cal, lams)
    pols = {"mup": lambda t, l: mup.actions(t, l), "mup_no_calibration": lambda t, l: mup_nocal.actions(t, l),
            "mup_never_continuation": lambda t, l: mup_never.actions(t, l), "delta_reg": lambda t, l: dreg.actions(t, l),
            "pp_step_direct": lambda t, l: pp_direct.actions(t, l), "pp_step_benefit": lambda t, l: pp_ben.actions(t, l),
            "pp_first": lambda t, l: pp_first.actions(t, l), "conf_thr": lambda t, l: conf.actions(t, l)}
    if "pp_plan" in extras:
        lp, Cp, _ = select_root_layer_and_C(tr, layers, seed)
        plan = PlanProbe(lp, Cp).fit(tr, cal)
        pols["pp_plan"] = lambda t, l: plan.actions(t, l)
        info["pp_plan"] = {"layer": lp, "C": Cp}
    if "mup_pc" in extras:
        pc = PolicyConsistentMUP(lay, C, seed=seed).fit(tr, cal, lams)
        pols["mup_pc"] = lambda t, l: pc.actions(t, l)
        info["mup_pc"] = {"layer": lay, "C": C, "note": "layer/C reused from MUP's train-only CV"}
    return pols, info, mup


def evaluate_split(tree: TreeData, pols, cal: TreeData, lams):
    """Per-instance correct/calls matrices (N,L) for every policy on `tree`."""
    res = {}
    res["never"] = run_actions(tree, lambda l: np.zeros((tree.n, N_NODES), int), lams)
    res["always"] = run_actions(tree, lambda l: np.ones((tree.n, N_NODES), int), lams)
    gp = {l: best_global_pattern(cal, l) for l in lams}
    res["best_global_pattern"] = run_actions(tree, lambda l: np.tile(pattern_actions(gp[l]), (tree.n, 1)), lams)
    res["rollout_reference"] = run_actions(tree, lambda l: rollout_reference_actions(tree, l), lams)
    hc, hk = zip(*[hindsight_utility(tree, l) for l in lams])
    res["hindsight_nondeployable"] = (np.stack(hc, 1), np.stack(hk, 1))
    for name, fn in pols.items():
        res[name] = run_actions(tree, lambda l, fn=fn: fn(tree, l), lams)
    p_by = (res["mup"][1].mean(0) / 3.0)
    res["random_matched_to_mup"] = random_matched(tree, p_by, lams)
    res["_global_patterns"] = {f"{l:.2f}": int(gp[l]) for l in lams}
    return res


def summarise(res, lams, tree, n_boot=10000):
    out, per = {}, {}
    for name, v in res.items():
        if name.startswith("_"):
            continue
        C, K = v
        U = utility_matrix(C, K, lams)
        ua = u_auc_instances(U, lams)
        per[name] = ua
        lo, hi = mean_ci(ua, n_boot)
        out[name] = {"u_auc_points": float(ua.mean()), "u_auc_ci95": [lo, hi], "accuracy_by_lambda": C.mean(0).tolist(),
                     "calls_per_instance_by_lambda": K.mean(0).tolist(), "utility_by_lambda": U.mean(0).tolist()}
    return out, per


_POOL = None


def run_partition(cfg, r, seed, out_dir):
    """One resample of the pool: fit on train, tune on cal, evaluate on gate/test. Returns the partition's result dict."""
    pool = _POOL
    lams = list(cfg["lambdas"])
    B = cfg["bootstrap"]["n_resamples"]
    extras = tuple(cfg.get("extra_policies", []))
    sp = make_splits(pool.instance_ids, cfg["splits"], seed)
    tr, cal, gate, test = (pool.subset(sp[k]) for k in ("train", "cal", "gate", "test"))
    pols, info, mup = fit_all(tr, cal, cfg["layers"], lams, seed, extras)
    R = {s: evaluate_split(t, pols, cal, lams) for s, t in (("gate", gate), ("test", test))}
    S, P = {}, {}
    for s, t in (("gate", gate), ("test", test)):
        S[s], P[s] = summarise(R[s], lams, t, B)
    # --- gate (Task 3 rule restored): H = U-AUC(rollout reference) - U-AUC(best global pattern), on GATE split
    g = paired_bootstrap(P["gate"]["rollout_reference"], P["gate"]["best_global_pattern"], B, seed=seed)
    g["threshold_points"] = cfg["gate"]["threshold_points"]
    g["passed"] = bool(g["diff"] >= cfg["gate"]["threshold_points"] and g["ci_lo"] > 0)
    gh = paired_bootstrap(P["gate"]["hindsight_nondeployable"], P["gate"]["best_global_pattern"], B, seed=seed)
    # --- primary contrasts on TEST with Holm
    prim = {}
    for name, ref in (("mup_vs_pp_step_direct", "pp_step_direct"), ("mup_vs_best_global_pattern", "best_global_pattern"),
                      ("mup_vs_pp_first", "pp_first")):
        prim[name] = paired_bootstrap(P["test"]["mup"], P["test"][ref], B, seed=seed)
    adj = holm({k: v["p"] for k, v in prim.items()})
    for k in prim:
        prim[k]["p_holm"] = adj[k]
    expl = {f"mup_vs_{ref}": paired_bootstrap(P["test"]["mup"], P["test"][ref], B, seed=seed)
            for ref in ("pp_step_benefit", "delta_reg", "conf_thr", "never", "always", "rollout_reference",
                        "mup_no_calibration", "mup_never_continuation")}
    # --- registered SECONDARY family (pre-registration addendum 2): policy-consistent MUP, Holm over four contrasts
    sec, extra_expl = {}, {}
    if "mup_pc" in extras:
        for name, ref in (("mup_pc_vs_mup", "mup"), ("mup_pc_vs_pp_step_direct", "pp_step_direct"),
                          ("mup_pc_vs_best_global_pattern", "best_global_pattern"), ("mup_pc_vs_pp_plan", "pp_plan")):
            if ref in P["test"]:
                sec[name] = paired_bootstrap(P["test"]["mup_pc"], P["test"][ref], B, seed=seed)
        adj2 = holm({k: v["p"] for k, v in sec.items()})
        for k in sec:
            sec[k]["p_holm"] = adj2[k]
    if "pp_plan" in extras:
        for name, ref in (("pp_plan_vs_pp_first", "pp_first"), ("pp_plan_vs_best_global_pattern", "best_global_pattern"), ("pp_plan_vs_mup", "mup")):
            extra_expl[name] = paired_bootstrap(P["test"]["pp_plan"], P["test"][ref], B, seed=seed)
    # --- rank reversal (registered comparator): cumulative cost vs any-call indicator, deployable methods only
    meths = ["never", "always", "random_matched_to_mup", "best_global_pattern", "pp_first", "pp_step_direct",
             "pp_step_benefit", "conf_thr", "delta_reg", "mup"]
    A_, Bn = {}, {}
    for m in meths:
        C_, K_ = R["test"][m]
        A_[m] = P["test"][m]
        if m == "random_matched_to_mup":      # K_ is an EXPECTED call count: use the exact P(any call) = 1-(1-p)^T
            any_ind = 1.0 - (1.0 - K_ / 3.0) ** 3
            a_ = np.asarray(lams)[None, :]
            Bn[m] = u_auc_instances(C_ - a_ * any_ind, lams)
        else:
            Bn[m] = u_auc_instances(any_call_utility_matrix(C_, K_, lams), lams)
    rr = kendall_tau_boot(A_, Bn, 2000, seed)
    rr["any_call_u_auc_points"] = {m: float(Bn[m].mean()) for m in meths}
    rr["cumulative_u_auc_points"] = {m: float(A_[m].mean()) for m in meths}
    # --- cost ledger: online tokens
    ledger = {}
    for name, fn in pols.items():
        ledger[name] = {"gen_tokens_per_instance_by_lambda": gen_tokens_wasted(test, lambda l, fn=fn: fn(test, l), lams,
                        policy_generates_everywhere=(name == "conf_thr")).mean(0).tolist()}
    ref_i = lams.index(cfg["reference_lambda"])
    tax = {}
    for name, fn in pols.items():
        tax[name] = taxonomy(test, fn(test, cfg["reference_lambda"]))
    tax["best_global_pattern"] = taxonomy(test, np.tile(pattern_actions(R["test"]["_global_patterns"][f"{cfg['reference_lambda']:.2f}"]), (test.n, 1)))
    # --- power / sd
    sd = {k: float(np.std(P["test"]["mup"] - P["test"][k], ddof=1)) for k in ("pp_step_direct", "best_global_pattern", "pp_first")}
    power = {"n_test": test.n, "alpha": 0.05, "power_target": 0.8, "sd_paired_points": sd,
             "mde_points": {k: float(mde_paired(v, test.n)) for k, v in sd.items()}}
    # --- ablation: per-layer MUP at the selected C (test U-AUC), reporting layer sensitivity
    layer_abl = {}
    for L in cfg["layers"]:
        m = TwoHeadMUP(L, info["mup"]["C"], continuation=1).fit(tr, cal)
        C_, K_ = run_actions(test, lambda l, m=m: m.actions(test, l), lams)
        layer_abl[str(L)] = float(u_auc_instances(utility_matrix(C_, K_, lams), lams).mean())
    # --- ablation A9: normalised cost U_norm (cost / T) ranking stability
    norm_rank = {m: float(u_auc_instances(utility_matrix(R["test"][m][0], R["test"][m][1] / 3.0, lams), lams).mean()) for m in meths}
    part = {
        "seed": seed, "split_sizes": {k: int(len(v)) for k, v in sp.items()},
        "split_instance_id_hash": __import__("hashlib").sha256(
            "".join(pool.instance_ids[i] for k in sp for i in sp[k]).encode()).hexdigest()[:16],
        "fit_info": info, "global_patterns_chosen_on_cal": R["test"]["_global_patterns"],
        "gate": {"headroom_rollout_minus_global": g, "hindsight_minus_global": gh},
        "summary": S, "primary_contrasts_test_holm": prim, "exploratory_contrasts_test": expl,
        "secondary_contrasts_test_holm": sec, "extra_exploratory_contrasts_test": extra_expl,
        "rank_reversal_registered": rr, "ledger_online_tokens": ledger, "error_taxonomy_at_reference_lambda": tax, "power": power,
        "head_diagnostics_test": head_diagnostics(mup, test), "ablation_layer_mup_test_u_auc": layer_abl, "ablation_u_norm_u_auc": norm_rank}
    np.savez_compressed(out_dir / f"per_instance_partition_{r}.npz",
                        **{f"{s}__{k}": v for s in P for k, v in P[s].items()})
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "primary.yaml"))
    ap.add_argument("--out", default="main")
    ap.add_argument("--cache-dir", default=None, help="override cache/<name> (used by tests)")
    ap.add_argument("--workers", type=int, default=1, help="run partitions in parallel processes (same results)")
    ap.add_argument("--probe-solver", choices=["scipy", "torch"], default=None,
                    help="logistic-head solver: scipy (frozen default, float64) or torch (float32 L-BFGS on the GPU; no silent fallback; forces --workers 1)")
    args = ap.parse_args()
    if args.probe_solver:
        from toolcomp.probes import SoftLogReg
        SoftLogReg.SOLVER = args.probe_solver
        os.environ["T4_PROBE_SOLVER"] = args.probe_solver
    from toolcomp.probes import SoftLogReg as _S
    if _S.SOLVER == "torch":
        args.workers = 1
    cfg = load_config(args.config)
    (ROOT / "results" / args.out).mkdir(parents=True, exist_ok=True)
    lams = list(cfg["lambdas"])
    cache = Path(args.cache_dir) if args.cache_dir else ROOT / "cache" / cfg["name"]
    npz_cands = list(cache.glob("*_tree.npz"))
    tree_path = npz_cands[0] if npz_cands else (cache / f"{cfg['name']}_tree.npz")
    pool = TreeData.load(tree_path)
    B = cfg["bootstrap"]["n_resamples"]
    parts = {}
    t_all = time.perf_counter()
    global _POOL
    _POOL = pool
    out_dir = ROOT / "results" / args.out
    seeds = list(enumerate(cfg["partition_seeds"]))
    if args.workers > 1 and len(seeds) > 1 and "fork" in __import__("multiprocessing").get_all_start_methods():
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds)), mp_context=mp.get_context("fork")) as ex:
            futs = {r: ex.submit(run_partition, cfg, r, s, out_dir) for r, s in seeds}
            for r, f in futs.items():
                parts[f"partition_{r}"] = f.result()
    else:
        for r, s in seeds:
            parts[f"partition_{r}"] = run_partition(cfg, r, s, out_dir)
    top = {"config": {k: v for k, v in cfg.items() if not k.startswith("_")},
           "manifest": manifest(cfg, {"dataset_hash": pool.meta.get("dataset_hash"), "n_pool": pool.n,
                                      "analysis_seconds": time.perf_counter() - t_all,
                                      "probe_solver": _S.SOLVER}),
           "pool_stats": pool_stats(pool), "splits_config": cfg["splits"], "partition_seeds": cfg["partition_seeds"],
           "design": {"primary_partition": "partition_0", "lambdas": lams, "gate": cfg["gate"], "reference_lambda": cfg["reference_lambda"],
                      "note": "partitions are resamples of ONE pool (overlapping test sets); they measure split sensitivity, "
                              "not independent replication."}, **parts}
    write_json(ROOT / "results" / args.out / "results.json", top)
    p0 = parts["partition_0"]
    print("GATE:", {k: round(v, 3) if isinstance(v, float) else v for k, v in p0["gate"]["headroom_rollout_minus_global"].items()})
    for k, v in p0["primary_contrasts_test_holm"].items():
        print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})


if __name__ == "__main__":
    main()
