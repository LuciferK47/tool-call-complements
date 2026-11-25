#!/usr/bin/env python3
"""I3 + I5: Step x Operand Error Decomposition, Calibration ECE/Brier, and Feature-Complete S0+S2 / S0+S1+S4+n_gen."""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toolcomp.config import load_config
from toolcomp.envs.arith import generate_instances
from toolcomp.estimators import pick_C, tune_thresholds
from toolcomp.eval_harness import LAMBDAS, paired_bootstrap, u_auc_instances, utility_matrix
from toolcomp.features import combine, draft_features, hidden_features, prefill_features, surface_features, with_features
from toolcomp.probes import ThresholdPolicy, TwoHeadMUP
from toolcomp.splits import make_splits
from toolcomp.tree import LEVEL_OF, N_NODES, TreeData, node_labels

EMISSION_TOKENS = 15


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (probs >= b0) & (probs < b1 if b1 < 1.0 else probs <= b1)
        if m.any():
            ece += (m.sum() / n) * abs(float(y[m].mean()) - float(probs[m].mean()))
    return float(ece)


def error_decomposition(tree: TreeData, actions: np.ndarray):
    """False-call and missed-call rates stratified by step (0,1,2) x operand bin (easy=0,med=1,hard=2)."""
    yc, yn = node_labels(tree, 1)
    n = tree.n
    ar = np.arange(n)
    nid = np.zeros(n, dtype=np.int64)
    grid = {f"step{s}_bin{b}": {"visited": 0, "false_call": 0, "missed_call": 0} for s in range(3) for b in range(3)}
    for s in range(3):
        a = actions[ar, nid]
        fc = (a == 1) & (yn[ar, nid] == 1)
        mc = (a == 0) & (yc[ar, nid] == 1) & (yn[ar, nid] == 0)
        bins = tree.step_bin[:, s]
        for b in range(3):
            m = bins == b
            g = grid[f"step{s}_bin{b}"]
            g["visited"] += int(m.sum())
            g["false_call"] += int((fc & m).sum())
            g["missed_call"] += int((mc & m).sum())
        nid = 2 * nid + 1 + a
    return grid


def eval_model(tag: str, cfg_name: str, seed: int):
    cfg = load_config(ROOT / "configs" / f"{cfg_name}.yaml")
    cache_dir = ROOT / "cache" / tag
    npz_cands = list(cache_dir.glob("*_tree.npz"))
    pool = TreeData.load(npz_cands[0])
    ins = generate_instances(cfg["pool"]["n"], seed, cfg["pool"]["prefix"])
    lams = list(cfg["lambdas"])
    sp = make_splits(pool.instance_ids, cfg["splits"], cfg["partition_seeds"][0])

    S0 = surface_features(pool, ins)
    layer = cfg["layers"][0]
    S1 = hidden_features(pool, layer)
    S2 = prefill_features(pool)
    S4 = draft_features(pool)
    ngen_feat = pool.n_gen_tokens[:, :, None].astype(np.float64)

    def fit_eval(F, pregen: bool):
        ft = with_features(pool, F)
        tr_ft, cal_ft = ft.subset(sp["train"]), ft.subset(sp["cal"])
        C = pick_C(tr_ft, seed=0)
        m = TwoHeadMUP(0, C).fit(tr_ft, cal_ft)
        th = tune_thresholds(m.delta(cal_ft), cal_ft, lams)
        sub_te = ft.subset(sp["test"])
        d_te = m.delta(sub_te)
        ar = np.arange(sub_te.n)
        C_arr, K_arr, T_arr = [], [], []
        for lam in lams:
            A = (d_te > th[lam]).astype(np.int64)
            nid = np.zeros(sub_te.n, dtype=np.int64)
            bits = np.zeros(sub_te.n, dtype=np.int64)
            calls = np.zeros(sub_te.n)
            toks = np.zeros(sub_te.n)
            for _ in range(3):
                a = A[ar, nid]
                calls += a
                ng = sub_te.n_gen_tokens[ar, nid].astype(np.float64)
                toks += (np.where(a == 0, ng, EMISSION_TOKENS) if pregen else ng + a * EMISSION_TOKENS)
                bits = 2 * bits + a
                nid = 2 * nid + 1 + a
            C_arr.append(sub_te.leaf_correct[ar, bits].astype(np.float64))
            K_arr.append(calls)
            T_arr.append(toks)
        u = u_auc_instances(utility_matrix(np.stack(C_arr, 1), np.stack(K_arr, 1), lams), lams)
        return m, th, ft, u, np.stack(T_arr, 1).mean(1)

    m_s1, th_s1, ft_s1, u_s1, tk_s1 = fit_eval(S1, True)
    m_s0, th_s0, ft_s0, u_s0, tk_s0 = fit_eval(S0, True)
    m_s02, th_s02, ft_s02, u_s02, tk_s02 = fit_eval(combine(S0, S2), True)
    m_all, th_all, ft_all, u_all, tk_all = fit_eval(combine(S0, S1, S4, ngen_feat), False)
    _, _, _, u_s14, tk_s14 = fit_eval(combine(S1, S4), False)

    # Calibration ECE & Brier on test split
    sub_te_s1 = ft_s1.subset(sp["test"])
    yc_te, yn_te = node_labels(sub_te_s1, 1)
    pc, pn = m_s1.heads(sub_te_s1)
    calib = {
        "S1_p_call_brier": float(np.mean((pc - yc_te) ** 2)),
        "S1_p_call_ece": ece_score(pc.ravel(), yc_te.ravel()),
        "S1_p_no_brier": float(np.mean((pn - yn_te) ** 2)),
        "S1_p_no_ece": ece_score(pn.ravel(), yn_te.ravel()),
    }

    # Error decomposition at lambda=0.3
    err_s1 = error_decomposition(sub_te_s1, (m_s1.delta(sub_te_s1) > th_s1[0.3]).astype(np.int64))
    sub_te_s0 = ft_s0.subset(sp["test"])
    err_s0 = error_decomposition(sub_te_s0, (m_s0.delta(sub_te_s0) > th_s0[0.3]).astype(np.int64))

    return {
        "u_auc": {
            "S0_t": float(u_s0.mean()), "S1_t": float(u_s1.mean()), "S0+S2_t": float(u_s02.mean()),
            "S1+S4_t": float(u_s14.mean()), "S0+S1+S4+ngen_t": float(u_all.mean()),
        },
        "contrasts": {
            "S0+S2_vs_S1": paired_bootstrap(u_s02, u_s1, n_resamples=1000, seed=0),
            "S0+S2_vs_S0": paired_bootstrap(u_s02, u_s0, n_resamples=1000, seed=0),
            "all_vs_S1+S4": paired_bootstrap(u_all, u_s14, n_resamples=1000, seed=0),
        },
        "calibration": calib,
        "error_decomposition_s1_ref_lam": err_s1,
        "error_decomposition_s0_ref_lam": err_s0,
    }


def main():
    targets = [
        ("q35_0p8b", "q35_0p8b", 20261101),
        ("q35_2b", "q35_2b", 20261101),
        ("q35_4b", "q35_4b", 20261101),
        ("q35_9b", "q35_9b", 20261101),
        ("q35_0p8b_seed20261120", "q35_0p8b", 20261120),
        ("q35_0p8b_incremental", "q35_0p8b", 20261101),
    ]
    out = {tag: eval_model(tag, cfg, sd) for tag, cfg, sd in targets}
    out_json = ROOT / "results" / "error_calibration_and_features.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[+] Wrote {out_json}")

    docs_dir = ROOT.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 11 - Feature-Complete Pre/Post Combinations, Calibration (ECE/Brier) & Error Audit",
        "",
        "## 1. S0+S2 (Zero-Token) & S0+S1+S4+n_gen_tokens Across All 6 Caches (Partition 0)",
        "",
        "| Run | S0_t | S1_t | S0+S2_t | S0+S2 vs S1 [CI] | S1+S4_t | S0+S1+S4+ngen_t | All vs S1+S4 [CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, d in out.items():
        u = d["u_auc"]
        c1 = d["contrasts"]["S0+S2_vs_S1"]
        c2 = d["contrasts"]["all_vs_S1+S4"]
        lines.append(
            f"| {tag} | {u['S0_t']:.2f} | {u['S1_t']:.2f} | {u['S0+S2_t']:.2f} | "
            f"{c1['diff']:+.2f} [{c1['ci_lo']:+.2f}, {c1['ci_hi']:+.2f}] | "
            f"{u['S1+S4_t']:.2f} | {u['S0+S1+S4+ngen_t']:.2f} | "
            f"{c2['diff']:+.2f} [{c2['ci_lo']:+.2f}, {c2['ci_hi']:+.2f}] |"
        )
    lines.extend(["", "## 2. S1 Head Calibration Diagnostics (Test Split)", "",
                  "| Run | p_call Brier | p_call ECE | p_no Brier | p_no ECE |", "|---|---|---|---|---|"])
    for tag, d in out.items():
        cb = d["calibration"]
        lines.append(f"| {tag} | {cb['S1_p_call_brier']:.4f} | {cb['S1_p_call_ece']:.4f} | {cb['S1_p_no_brier']:.4f} | {cb['S1_p_no_ece']:.4f} |")

    lines.extend(["", "## 3. Step x Operand Error Audit (`S1_t` at lambda=0.3, Test Split)", ""])
    for tag in ["q35_0p8b", "q35_2b", "q35_0p8b_incremental"]:
        lines.append(f"### {tag}")
        lines.append("| Step x Bin | Visited | False Calls | Missed Calls |")
        lines.append("|---|---|---|---|")
        for k, v in out[tag]["error_decomposition_s1_ref_lam"].items():
            lines.append(f"| {k} | {v['visited']} | {v['false_call']} | {v['missed_call']} |")
        lines.append("")

    md_path = docs_dir / "11_error_audit_and_calibration.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote {md_path}")


if __name__ == "__main__":
    main()
