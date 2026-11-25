"""Iteration-2 zero-GPU matrix on a cached outcome tree: signal family x estimator x {untuned, tuned threshold}.

EXPLORATORY (the 0.8B tree and its test results were already seen in Iteration 1). Nothing here changes a registered verdict.
Signal families (src/features.py): S0 surface | S1 hidden | S2 prefill entropy | S4 draft confidence | S1' distilled student |
fused S0+S4 and S1+S4 (late fusion, identical protocol). Estimators (src/estimators.py, src/probes.py): two-head plug-in,
single-head delta, policy-consistent two-head. Selection of the best controller per family uses the GATE split only.
Token ledger: draft tokens are charged when the policy needs the drafted answer (post-draft policies at every visited node,
pre-generation policies only where they do not call) plus an emission charge per tool call; net U-AUC = U-AUC - kappa * tokens.

  python scripts/run_iter2_matrix.py --config configs/q35_0p8b.yaml --out q35_0p8b [--boot 10000]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import ROOT, write_json
from toolcomp.config import load_config
from toolcomp.envs.arith import generate_instances
from toolcomp.estimators import (DistilledTwoHead, FusedTwoHead, SingleHeadDelta, pick_alpha, pick_C, tune_thresholds)
from toolcomp.eval_harness import LAMBDAS, holm, paired_bootstrap, u_auc_instances
from toolcomp.features import (combine, draft_features, hidden_features, prefill_features, surface_features, with_features)
from toolcomp.probes import PolicyConsistentMUP, ThresholdPolicy, TwoHeadMUP, best_global_pattern, pattern_actions
from toolcomp.splits import make_splits
from toolcomp.tree import N_NODES, TreeData, node_labels, walk

EMISSION_TOKENS = 15          # tokens a real agent would emit to write one tool call (the scaffold charges 0)
KAPPAS = (0.0, 0.25, 0.5, 1.0, 2.0)   # utility points per generated token
PRE = {"S0", "S1", "S2", "S1p"}   # families that decide before any answer token is generated


def run_policy(tree: TreeData, act_fn, lams, needs_draft: bool):
    """Walk a policy over the tree. Returns correct (N,L), calls (N,L), generated tokens (N,L) incl. emission charge."""
    n = tree.n
    ar = np.arange(n)
    C, K, T = [], [], []
    for lam in lams:
        A = act_fn(lam)
        nid = np.zeros(n, dtype=np.int64)
        bits = np.zeros(n, dtype=np.int64)
        calls = np.zeros(n)
        toks = np.zeros(n)
        for _ in range(3):
            a = A[ar, nid].astype(np.int64)
            gen = tree.n_gen_tokens[ar, nid].astype(np.float64)
            toks += gen if needs_draft else gen * (1 - a)
            calls += a
            bits = bits * 2 + a
            nid = 2 * nid + 1 + a
        C.append(tree.leaf_correct[ar, bits].astype(np.float64))
        K.append(calls)
        T.append(toks + EMISSION_TOKENS * calls)
    return np.stack(C, 1), np.stack(K, 1), np.stack(T, 1)


def score(C, K, T, lams):
    ua = u_auc_instances(C - np.asarray(lams)[None, :] * K, lams)
    lam = np.asarray(lams, float)
    tok = trapezoid(T, lam, axis=1) / (lam[-1] - lam[0])
    return ua, tok, K, C


def partition(pool, instances, cfg, prior, r, seed, B, stored):
    lams = list(cfg["lambdas"])
    sp = make_splits(pool.instance_ids, cfg["splits"], seed)
    tr, ca, ga, te = (pool.subset(sp[k]) for k in ("train", "cal", "gate", "test"))
    layer, C1 = (prior[f"partition_{r}"]["fit_info"]["mup"]["layer"], prior[f"partition_{r}"]["fit_info"]["mup"]["C"])
    F = {"S0": surface_features(pool, instances), "S1": hidden_features(pool, layer),
         "S2": prefill_features(pool), "S4": draft_features(pool)}
    ctrl = {}                                    # name -> dict(needs_draft, act_fn per split)
    shim = {k: with_features(pool, v) for k, v in F.items()}
    sub = {k: tuple(s.subset(sp[n]) for n in ("train", "cal", "gate", "test")) for k, s in shim.items()}
    conf = {n: F["S4"][sp[n]] for n in ("train", "cal", "gate", "test")}
    info = {"S1_layer": layer, "S1_C": C1}

    def reg(name, needs_draft, delta_by_split, with_pc=None):
        ctrl[name + "_u"] = {"draft": needs_draft, "act": {s: (lambda lam, d=d: (d > lam).astype(np.int64)) for s, d in delta_by_split.items()}}
        tau = tune_thresholds(delta_by_split["cal"], ca, lams)
        ctrl[name + "_t"] = {"draft": needs_draft, "act": {s: (lambda lam, d=d, tau=tau: (d > tau[lam]).astype(np.int64)) for s, d in delta_by_split.items()}, "tau": tau}

    for fam in ("S0", "S1", "S2", "S4"):
        t_tr, t_ca, t_ga, t_te = sub[fam]
        C = C1 if fam == "S1" else pick_C(t_tr, seed)
        info[f"{fam}_C"] = C
        m = TwoHeadMUP(0, C).fit(t_tr, t_ca)
        reg(f"two_{fam}", fam == "S4", {"cal": m.delta(t_ca), "gate": m.delta(t_ga), "test": m.delta(t_te)})
        a = pick_alpha(t_tr, seed)
        info[f"{fam}_alpha"] = a
        h = SingleHeadDelta(a).fit(t_tr, t_ca)
        reg(f"dlt_{fam}", fam == "S4", {"cal": h.delta(t_ca), "gate": h.delta(t_ga), "test": h.delta(t_te)})
        pc = PolicyConsistentMUP(0, C).fit(t_tr, t_ca, lams)
        ctrl[f"pc_{fam}"] = {"draft": fam == "S4", "act": {"cal": lambda lam, t=t_ca, p=pc: p.actions(t, lam),
                                                              "gate": lambda lam, t=t_ga, p=pc: p.actions(t, lam),
                                                              "test": lambda lam, t=t_te, p=pc: p.actions(t, lam)}}
    for fam in ("S0", "S1"):                      # late fusion with draft confidence, same protocol for both bases
        t_tr, t_ca, t_ga, t_te = sub[fam]
        fm = FusedTwoHead(info[f"{fam}_C"]).fit(t_tr, t_ca, conf["train"], conf["cal"], seed)
        reg(f"two_{fam}+S4", True, {"cal": fm.delta_with(t_ca, conf["cal"]), "gate": fm.delta_with(t_ga, conf["gate"]),
                                    "test": fm.delta_with(t_te, conf["test"])})
    t_tr, t_ca, t_ga, t_te = sub["S1"]            # distilled pre-generation student (teacher = draft confidence)
    ds = DistilledTwoHead(C1).fit(t_tr, t_ca, conf["train"], seed)
    reg("two_S1p", False, {"cal": ds.delta(t_ca), "gate": ds.delta(t_ga), "test": ds.delta(t_te)})

    # reference policies (same definitions as Iteration 1)
    q = np.unique(np.quantile(ca.mean_logprob.reshape(-1), np.linspace(0.02, 0.98, 49)))
    conf_thr = ThresholdPolicy("conf_thr", lambda t: t.mean_logprob, "low", q).tune(ca, lams)
    qe = np.unique(np.quantile(ca.entropy0.reshape(-1), np.linspace(0.02, 0.98, 49)))
    ent_thr = ThresholdPolicy("ent_thr", lambda t: t.entropy0, "high", qe).tune(ca, lams)
    trees = {"cal": ca, "gate": ga, "test": te}
    ctrl["conf_thr"] = {"draft": True, "act": {s: (lambda lam, t=t: conf_thr.actions(t, lam)) for s, t in trees.items()}}
    ctrl["ent_thr_prefill"] = {"draft": False, "act": {s: (lambda lam, t=t: ent_thr.actions(t, lam)) for s, t in trees.items()}}
    pats = {lam: best_global_pattern(ca, lam) for lam in lams}
    ctrl["best_global"] = {"draft": False, "act": {s: (lambda lam, t=t: np.tile(pattern_actions(pats[lam]), (t.n, 1))) for s, t in trees.items()}}
    ctrl["never"] = {"draft": False, "act": {s: (lambda lam, t=t: np.zeros((t.n, N_NODES), np.int64)) for s, t in trees.items()}}

    res, per = {}, {}
    for name, c in ctrl.items():
        out = {}
        for s in ("gate", "test"):
            C_, K_, T_ = run_policy(trees[s], c["act"][s], lams, c["draft"])
            out[s] = score(C_, K_, T_, lams)
        per[name] = out
        i6 = lams.index(0.3)
        res[name] = {"needs_draft": c["draft"], "gate_u_auc": float(out["gate"][0].mean()), "test_u_auc": float(out["test"][0].mean()),
                     "test_tokens_per_instance": float(out["test"][1].mean()),
                     "test_calls_by_lambda": out["test"][2].mean(0).tolist(),
                     "test_net_u_auc_by_kappa": {str(k): float((out["test"][0] - k * out["test"][1]).mean()) for k in KAPPAS}}
    # integrity vs Iteration-1 stored per-instance arrays (partition order == sorted test ids)
    chk = {}
    if stored is not None:
        chk = {k: float(np.max(np.abs(per[n]["test"][0] - stored["test__" + sk]))) for k, n, sk in
               (("two_S1_u_vs_stored_mup", "two_S1_u", "mup"), ("conf_thr_vs_stored", "conf_thr", "conf_thr"))}

    def con(a, b, name=None):
        d = paired_bootstrap(per[a]["test"][0], per[b]["test"][0], B, seed=seed)
        d["a"], d["b"] = a, b
        return d

    fam_pre = {f"{e}_S1_{t}-{e}_S0_{t}": con(f"{e}_S1_{t}", f"{e}_S0_{t}") for e in ("two", "dlt") for t in ("u", "t")}
    fam_pre["two_S1p_t-two_S0_t"] = con("two_S1p_t", "two_S0_t")
    fam_post = {f"two_S1+S4_{t}-two_S0+S4_{t}": con(f"two_S1+S4_{t}", f"two_S0+S4_{t}") for t in ("u", "t")}
    for fam in (fam_pre, fam_post):
        adj = holm({k: v["p"] for k, v in fam.items()})
        for k in fam:
            fam[k]["p_holm_guidance_only"] = adj[k]
    vs_conf = {n: con(n, "conf_thr") for n in per if n not in ("conf_thr",)}
    vs_s0 = {n: con(n, "two_S0_t") for n in per if n.startswith(("two_S1", "pc_S1", "dlt_S1")) or n in ("conf_thr", "ent_thr_prefill")}
    # gate-selected best controller per timing class, and kappa* between them (points per token)
    cand = {n: v for n, v in res.items() if n not in ("never", "best_global", "ent_thr_prefill")}
    best_pre = max((n for n in cand if not cand[n]["needs_draft"]), key=lambda n: cand[n]["gate_u_auc"])
    best_post = max((n for n in cand if cand[n]["needs_draft"]), key=lambda n: cand[n]["gate_u_auc"])
    du = res[best_post]["test_u_auc"] - res[best_pre]["test_u_auc"]
    dt = res[best_post]["test_tokens_per_instance"] - res[best_pre]["test_tokens_per_instance"]
    sel = {"best_pre_by_gate": best_pre, "best_post_by_gate": best_post, "test_u_auc_gap_post_minus_pre": du,
           "test_token_gap_post_minus_pre": dt, "kappa_star_points_per_token": (du / dt if dt > 0 and du > 0 else None),
           "contrast_post_minus_pre": con(best_post, best_pre)}
    return {"seed": seed, "info": info, "controllers": res, "integrity_max_abs_diff_vs_stored": chk,
            "family_pregen_hidden_value": fam_pre, "family_postdraft_hidden_value": fam_post,
            "vs_conf_thr": vs_conf, "vs_two_S0_tuned": vs_s0, "selection": sel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    B = args.boot or cfg["bootstrap"]["n_resamples"]
    cache = Path(args.cache_dir) if args.cache_dir else ROOT / "cache" / cfg["name"]
    npz_candidates = list(cache.glob("*_tree.npz"))
    tree_path = npz_candidates[0] if npz_candidates else (cache / f"{cfg['name']}_tree.npz")
    pool = TreeData.load(tree_path)
    seed = args.seed or pool.meta.get("seed") or cfg["pool"]["seed"]
    ins = generate_instances(cfg["pool"]["n"], seed, cfg["pool"]["prefix"])
    prior_path = ROOT / "results" / args.out / "results.json"
    if not prior_path.exists():
        prior_path = ROOT / "results" / cfg["name"] / "results.json"
    prior = json.loads(prior_path.read_text())
    t0 = time.time()
    parts = {}
    for r, s in enumerate(cfg["partition_seeds"]):
        sp = ROOT / "results" / args.out / f"per_instance_partition_{r}.npz"
        stored = dict(np.load(sp)) if sp.exists() else None
        parts[f"partition_{r}"] = partition(pool, ins, cfg, prior, r, s, B, stored)
        print(f"partition_{r} done {time.time() - t0:.0f}s", flush=True)
    write_json(ROOT / "results" / args.out / "iter2_matrix.json",
               {"label": "ITERATION 2, EXPLORATORY: zero-GPU signal x estimator matrix on cached trees; no registered verdict changes",
                "config": args.config, "config_hash": cfg["_config_hash"], "bootstrap": B, "emission_tokens_per_call": EMISSION_TOKENS,
                "kappas_points_per_token": KAPPAS, **parts})


if __name__ == "__main__":
    main()
