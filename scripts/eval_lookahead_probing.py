#!/usr/bin/env python3
"""B1 + I2: Real S3 Early-Token Look-Ahead Probing & Two-Stage Selective Cascade.

Extracts cached NodeOutput.gen_token_logprobs (tokens 1-2) from *_level{0,1,2}.pkl
for q35_0p8b_seed20261120 and q35_0p8b_incremental, evaluates k=1 and k=2 early-exit
controllers with exact token ledger, and evaluates Two-Stage Selective Cascade (S0 -> S0+S4).
"""

import json
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toolcomp.config import load_config
from toolcomp.envs.arith import generate_instances
from toolcomp.estimators import FusedTwoHead, pick_C, tune_thresholds
from toolcomp.eval_harness import LAMBDAS, paired_bootstrap, u_auc_instances, utility_matrix
from toolcomp.features import combine, draft_features, hidden_features, prefill_features, surface_features, with_features
from toolcomp.probes import ThresholdPolicy, TwoHeadMUP
from toolcomp.splits import make_splits
from toolcomp.tree import N_NODES, T_STEPS, TreeData, node_labels

EMISSION_TOKENS = 15
KAPPAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def load_s3_from_pickles(cache_dir: Path, tag: str, n: int):
    """Extract (N, 7) early-token logprob features from level0..2 checkpoints."""
    lp1 = np.zeros((n, N_NODES), dtype=np.float64)
    lp2 = np.zeros((n, N_NODES), dtype=np.float64)
    mlp2 = np.zeros((n, N_NODES), dtype=np.float64)
    has2 = np.zeros((n, N_NODES), dtype=np.float64)
    for lv in range(T_STEPS):
        ck = cache_dir / f"{tag}_level{lv}.pkl"
        saved = pickle.load(open(ck, "rb"))
        ids = [(i, 2 ** lv - 1 + p) for i in range(n) for p in range(2 ** lv)]
        for (i, k), o in zip(ids, saved["outs"]):
            lps = getattr(o, "gen_token_logprobs", None) or []
            if len(lps) >= 1:
                lp1[i, k] = float(lps[0])
            if len(lps) >= 2:
                lp2[i, k] = float(lps[1])
                mlp2[i, k] = 0.5 * (float(lps[0]) + float(lps[1]))
                has2[i, k] = 1.0
            elif len(lps) == 1:
                lp2[i, k] = float(lps[0])
                mlp2[i, k] = float(lps[0])
    return lp1, lp2, mlp2, has2


def eval_policy_ledger(tree: TreeData, act_fn, lams, draft_budget: str, k_exit: int = 0):
    """Walk policy over tree.
    draft_budget: 'none' (prefill only), 'full' (always full draft), 'early_k' (k tokens if call else full).
    """
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
            calls += a
            ng = tree.n_gen_tokens[ar, nid].astype(np.float64)
            if draft_budget == "full":
                toks += ng + a * EMISSION_TOKENS
            elif draft_budget == "early_k":
                toks += np.where(a == 1, np.minimum(k_exit, ng) + EMISSION_TOKENS, ng)
            else:
                toks += np.where(a == 0, ng, EMISSION_TOKENS)
            bits = 2 * bits + a
            nid = 2 * nid + 1 + a
        C.append(tree.leaf_correct[ar, bits].astype(np.float64))
        K.append(calls)
        T.append(toks)
    C, K, T = np.stack(C, 1), np.stack(K, 1), np.stack(T, 1)
    u = u_auc_instances(utility_matrix(C, K, lams), lams)
    tok = T.mean(1)
    return u, tok


def eval_cascade_ledger(tree: TreeData, delta_pre, delta_post, th_post, lams, tau_lo: float, tau_hi: float):
    """Two-stage cascade: prefill extremes at 0 draft tokens, middle band drafts for post."""
    n = tree.n
    ar = np.arange(n)
    C, K, T = [], [], []
    for li, lam in enumerate(lams):
        A = np.zeros((n, N_NODES), dtype=np.int64)
        drafted = np.zeros((n, N_NODES), dtype=bool)
        hi_mask = delta_pre > (lam + tau_hi)
        lo_mask = delta_pre < (lam - tau_lo)
        mid_mask = ~(hi_mask | lo_mask)
        A[hi_mask] = 1
        A[lo_mask] = 0
        A[mid_mask] = (delta_post[mid_mask] > th_post[lam]).astype(np.int64)
        drafted[mid_mask] = True

        nid = np.zeros(n, dtype=np.int64)
        bits = np.zeros(n, dtype=np.int64)
        calls = np.zeros(n)
        toks = np.zeros(n)
        for _ in range(3):
            a = A[ar, nid]
            d = drafted[ar, nid]
            ng = tree.n_gen_tokens[ar, nid].astype(np.float64)
            calls += a
            toks += np.where(d | (a == 0), ng, 0.0) + a * EMISSION_TOKENS
            bits = 2 * bits + a
            nid = 2 * nid + 1 + a
        C.append(tree.leaf_correct[ar, bits].astype(np.float64))
        K.append(calls)
        T.append(toks)
    C, K, T = np.stack(C, 1), np.stack(K, 1), np.stack(T, 1)
    return u_auc_instances(utility_matrix(C, K, lams), lams), T.mean(1)


def run_tag(tag: str, seed: int):
    cfg = load_config(ROOT / "configs" / "q35_0p8b.yaml")
    cache_dir = ROOT / "cache" / tag
    tree_path = cache_dir / f"{tag}_tree.npz"
    pool = TreeData.load(tree_path)
    ins = generate_instances(cfg["pool"]["n"], seed, cfg["pool"]["prefix"])
    lams = list(cfg["lambdas"])
    sp = make_splits(pool.instance_ids, cfg["splits"], cfg["partition_seeds"][0])

    lp1, lp2, mlp2, has2 = load_s3_from_pickles(cache_dir, tag, pool.n)
    S0 = surface_features(pool, ins)
    S1 = hidden_features(pool, 12)
    S2 = prefill_features(pool)
    S4 = draft_features(pool)
    S3_k1 = np.stack([lp1, pool.entropy0], -1)
    S3_k2 = np.stack([lp1, lp2, mlp2, pool.entropy0, has2], -1)

    # Fit TwoHeadMUP on S0, S1, S0+S2, S3_k1, S3_k2, S4, S0+S4
    def fit_two(F):
        ft = with_features(pool, F)
        tr_ft, cal_ft = ft.subset(sp["train"]), ft.subset(sp["cal"])
        C = pick_C(tr_ft, seed=0)
        m = TwoHeadMUP(0, C).fit(tr_ft, cal_ft)
        th = tune_thresholds(m.delta(cal_ft), cal_ft, lams)
        return ft, m, th

    ft_s0, m_s0, th_s0 = fit_two(S0)
    ft_s1, m_s1, th_s1 = fit_two(S1)
    ft_s02, m_s02, th_s02 = fit_two(combine(S0, S2))
    ft_k1, m_k1, th_k1 = fit_two(combine(S0, S3_k1))
    ft_k2, m_k2, th_k2 = fit_two(combine(S0, S3_k2))
    ft_s4, m_s4, th_s4 = fit_two(S4)
    ft_s04, m_s04, th_s04 = fit_two(combine(S0, S4))

    # Confidence threshold baseline
    cal_t = pool.subset(sp["cal"])
    test_t = pool.subset(sp["test"])
    conf_grid = np.unique(np.quantile(cal_t.mean_logprob.reshape(-1), np.linspace(0.02, 0.98, 49)))
    conf_m = ThresholdPolicy("conf_thr", lambda t: t.mean_logprob, "low", conf_grid).tune(cal_t, lams)

    res = {}
    configs = [
        ("S0_t", ft_s0, m_s0, th_s0, "none", 0),
        ("S1_t", ft_s1, m_s1, th_s1, "none", 0),
        ("S0+S2_t", ft_s02, m_s02, th_s02, "none", 0),
        ("S0+S3_k1_t", ft_k1, m_k1, th_k1, "early_k", 1),
        ("S0+S3_k2_t", ft_k2, m_k2, th_k2, "early_k", 2),
        ("S4_t", ft_s4, m_s4, th_s4, "full", 0),
        ("S0+S4_t", ft_s04, m_s04, th_s04, "full", 0),
    ]
    u_test = {}
    tok_test = {}
    for name, ft, m, th, bud, k_ex in configs:
        sub_te = ft.subset(sp["test"])
        u, tk = eval_policy_ledger(sub_te, lambda lam, m=m, th=th, sub=sub_te: (m.delta(sub) > th[lam]).astype(np.int64), lams, bud, k_ex)
        u_test[name], tok_test[name] = u, tk
        res[name] = {
            "u_auc": float(u.mean()),
            "tokens": float(tk.mean()),
            "net_u_auc": {str(kp): float((u - kp * tk).mean()) for kp in KAPPAS},
        }

    u_conf, tk_conf = eval_policy_ledger(test_t, lambda lam: conf_m.actions(test_t, lam), lams, "full", 0)
    u_test["conf_thr"], tok_test["conf_thr"] = u_conf, tk_conf
    res["conf_thr"] = {
        "u_auc": float(u_conf.mean()),
        "tokens": float(tk_conf.mean()),
        "net_u_auc": {str(kp): float((u_conf - kp * tk_conf).mean()) for kp in KAPPAS},
    }

    # Two-stage cascade: pre = S0_t delta, post = S0+S4_t delta
    d_pre_cal = m_s0.delta(ft_s0.subset(sp["cal"]))
    d_post_cal = m_s04.delta(ft_s04.subset(sp["cal"]))
    d_pre_te = m_s0.delta(ft_s0.subset(sp["test"]))
    d_post_te = m_s04.delta(ft_s04.subset(sp["test"]))

    cascade_res = {}
    for kp in KAPPAS:
        best_net, best_pair = -1e9, (0.1, 0.1)
        for t_lo in [0.05, 0.1, 0.2, 0.3]:
            for t_hi in [0.05, 0.1, 0.2, 0.3]:
                u_c, tk_c = eval_cascade_ledger(cal_t, d_pre_cal, d_post_cal, th_s04, lams, t_lo, t_hi)
                net = float((u_c - kp * tk_c).mean())
                if net > best_net:
                    best_net, best_pair = net, (t_lo, t_hi)
        u_te_c, tk_te_c = eval_cascade_ledger(test_t, d_pre_te, d_post_te, th_s04, lams, best_pair[0], best_pair[1])
        cascade_res[f"kappa_{kp}"] = {
            "tau_lo": best_pair[0], "tau_hi": best_pair[1],
            "u_auc": float(u_te_c.mean()), "tokens": float(tk_te_c.mean()),
            "net_u_auc": float((u_te_c - kp * tk_te_c).mean()),
            "vs_S0_net": paired_bootstrap(u_te_c - kp * tk_te_c, u_test["S0_t"] - kp * tok_test["S0_t"], n_resamples=1000, seed=0),
            "vs_S0+S4_net": paired_bootstrap(u_te_c - kp * tk_te_c, u_test["S0+S4_t"] - kp * tok_test["S0+S4_t"], n_resamples=1000, seed=0),
        }

    # Key paired contrasts
    contrasts = {
        "S0+S3_k1_vs_S0": paired_bootstrap(u_test["S0+S3_k1_t"], u_test["S0_t"], n_resamples=1000, seed=0),
        "S0+S3_k2_vs_S0": paired_bootstrap(u_test["S0+S3_k2_t"], u_test["S0_t"], n_resamples=1000, seed=0),
        "S0+S3_k2_vs_conf_thr_k0": paired_bootstrap(u_test["S0+S3_k2_t"], u_test["conf_thr"], n_resamples=1000, seed=0),
        "S0+S3_k1_vs_conf_thr_k05": paired_bootstrap(u_test["S0+S3_k1_t"] - 0.5 * tok_test["S0+S3_k1_t"],
                                                     u_test["conf_thr"] - 0.5 * tok_test["conf_thr"], n_resamples=1000, seed=0),
        "S0+S2_vs_S1": paired_bootstrap(u_test["S0+S2_t"], u_test["S1_t"], n_resamples=1000, seed=0),
    }
    return {"controllers": res, "cascade_by_kappa": cascade_res, "contrasts": contrasts}


def main():
    out = {
        "q35_0p8b_seed20261120": run_tag("q35_0p8b_seed20261120", 20261120),
        "q35_0p8b_incremental": run_tag("q35_0p8b_incremental", 20261101),
    }
    out_json = ROOT / "results" / "s3_and_cascade_results.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[+] Wrote {out_json}")

    docs_dir = ROOT.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 10 - Real S3 Early-Token Look-Ahead & Two-Stage Selective Cascade",
        "",
        "Evaluated on cached `gen_token_logprobs` (tokens 1-2) from `level{0,1,2}.pkl` (partition 0, 1000 bootstrap).",
        "",
    ]
    for tag, d in out.items():
        lines.append(f"## {tag}")
        lines.append("")
        lines.append("| Controller | U-AUC (k=0) | Tokens | Net k=0.25 | Net k=0.5 | Net k=1.0 |")
        lines.append("|---|---|---|---|---|---|")
        for name, v in d["controllers"].items():
            nu = v["net_u_auc"]
            lines.append(f"| {name} | {v['u_auc']:.2f} | {v['tokens']:.2f} | {nu['0.25']:.2f} | {nu['0.5']:.2f} | {nu['1.0']:.2f} |")
        lines.append("")
        lines.append("### Two-Stage Selective Cascade (S0 -> S0+S4)")
        lines.append("| Target kappa | (tau_lo, tau_hi) | U-AUC | Tokens | Net U-AUC | vs S0 net [CI] | vs S0+S4 net [CI] |")
        lines.append("|---|---|---|---|---|---|---|")
        for kstr, cv in d["cascade_by_kappa"].items():
            c1 = cv["vs_S0_net"]
            c2 = cv["vs_S0+S4_net"]
            lines.append(f"| {kstr} | ({cv['tau_lo']}, {cv['tau_hi']}) | {cv['u_auc']:.2f} | {cv['tokens']:.2f} | {cv['net_u_auc']:.2f} | {c1['diff']:+.2f} [{c1['ci_lo']:+.2f}, {c1['ci_hi']:+.2f}] | {c2['diff']:+.2f} [{c2['ci_lo']:+.2f}, {c2['ci_hi']:+.2f}] |")
        lines.append("")
        lines.append("### Key Paired Contrasts")
        for cname, c in d["contrasts"].items():
            lines.append(f"- `{cname}`: {c['diff']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}], p={c['p']:.4f}")
        lines.append("")
    md_path = docs_dir / "10_s3_lookahead_and_cascade_results.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote {md_path}")


if __name__ == "__main__":
    main()
