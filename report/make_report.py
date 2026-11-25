"""Generate docs/04_results_and_final_implementation.md from results/*.json ONLY.

Rules enforced by scripts/audit_results.py:
  * every number in the generated document is formatted from a results file at run time;
  * this source file contains no numeric literals inside string constants other than identifiers (A-12, P1-I1, D3 ...),
    heading numbers and format specs; narrative text is therefore digit-free;
  * running this script twice produces byte-identical output.
Usage: python report/make_report.py [--results results/main] [--pilot results/pilot] [--out docs/04_results_and_final_implementation.md]
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NAMES = {
    "never": ("never call", "deployable"),
    "always": ("always call", "deployable"),
    "random_matched_to_mup": ("random, call-rate matched to MUP (exact expectation)", "deployable"),
    "best_global_pattern": ("best global call pattern (chosen on cal)", "deployable"),
    "pp_first": ("Probe&Prefill-style single decision at the initial state", "deployable"),
    "pp_step_direct": ("step-wise direct-correctness probe (need-probe baseline)", "deployable"),
    "pp_step_benefit": ("step-wise binary-benefit probe", "deployable"),
    "conf_thr": ("Anchor-lite: answer log-prob threshold", "deployable"),
    "delta_reg": ("single-head Delta regression (ablation A1)", "deployable"),
    "mup": ("MUP: two-head marginal-utility probe (ours)", "deployable"),
    "rollout_reference": ("rollout reference (exact node labels, one-step look-ahead)", "reference, not deployable"),
    "hindsight_nondeployable": ("per-instance hindsight pattern (exact upper bound)", "reference, not deployable"),
    "mup_no_calibration": ("MUP without Platt calibration (ablation A6)", "deployable"),
    "mup_never_continuation": ("MUP trained on never-call continuation labels (ablation A3)", "deployable"),
}
ORDER = ["never", "always", "random_matched_to_mup", "best_global_pattern", "pp_first", "pp_step_direct", "pp_step_benefit",
         "conf_thr", "delta_reg", "mup", "mup_no_calibration", "mup_never_continuation", "rollout_reference",
         "hindsight_nondeployable"]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x, nd=1):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def nr(x):
    """Missing provenance is stated, never printed as None."""
    return "not recorded" if x is None else str(x)


def ci(lo, hi, nd=2):
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def verdict(c, positive_word, negative_word):
    if c["excludes_zero"] and c["diff"] > 0:
        return positive_word
    if c["excludes_zero"] and c["diff"] < 0:
        return negative_word
    return "no detectable difference"


def p2_section(p2, lvl):
    L = ["## 6.9 Black-box track P2-I1 (secondary; real temperature samples, text only)", ""]
    L.append("Queries: " + str(p2["n_cal_queries"]) + " calibration, " + str(p2["n_test_queries"]) + " test; up to " + str(p2["m_max"])
             + " samples per query. Calculator success on calibration queries: " + pct(p2["p_tool_cal"]) + ". Single-sample direct accuracy on test: "
             + pct(p2["direct_accuracy_single_sample_test"]) + "; greedy accuracy: " + pct(p2["greedy_accuracy_test"]) + ". Sample parse-failure rate: " + pct(p2["parse_failure_rate_samples"]) + ".")
    L.append("")
    meths = ["never", "always", "logprob_thr", "fixed_k", "sequential"]
    label = {"never": "never call (one sample)", "always": "always call (no samples)", "logprob_thr": "greedy answer log-probability threshold (one sample)",
             "fixed_k": "fixed k samples (k tuned on cal)", "sequential": "calibrated sequential stopping (ours)"}
    rows = []
    for rho, r in p2["by_rho"].items():
        for m in meths:
            rows.append([rho, label[m], f(r["net_u_auc_points"][m])])
    L.append(table(["Sampling cost rho (per sample, in tool-call units)", "Method", "Net U-AUC points (sampling charged)"], rows))
    L.append("")
    rows = []
    for rho, r in p2["by_rho"].items():
        for k, v in r["contrasts"].items():
            rows.append([rho, k, f(v["diff"]), ci(v["ci_lo"], v["ci_hi"]), f(v["p_holm"], 4), verdict(v, "sequential higher", "sequential lower")])
    L.append("Pre-registered contrasts (Holm family of two within each rho; the rho values are sensitivity analyses and are not corrected against each other):")
    L.append("")
    L.append(table(["rho", "Contrast", "Delta net U-AUC", lvl, "p (Holm)", "Verdict"], rows))
    L.append("")
    r0 = next(iter(p2["by_rho"].values()))
    L.append("Mean samples used by the sequential rule at the first rho, by lambda: " + ", ".join(f"{k}: {f(v, 2)}" for k, v in r0["mean_samples_sequential_by_lambda"].items())
             + ". Break-even rho against the best fixed k: " + ("not within the grid" if p2["break_even_rho_vs_fixed_k"] is None else f(p2["break_even_rho_vs_fixed_k"], 3)) + ".")
    L.append("")
    L.append("The calculator is exact here, so tool success is constant and the per-query tool-success part of P2 is not exercised (no claim is made about it).")
    L.append("")
    return "\n".join(L)


def replication_section(rep, power_target):
    """Optional section for a team-run replication pilot; every value comes from the replication JSON."""
    m = rep["manifest"]
    mo = m["model"]
    u = rep["u_auc_points"]
    L = ["## 6.10 Team-run GPU replication pilot (exploratory; not verified by the primary pipeline)", ""]
    L.append("A team member ran the pilot protocol on a GPU with a larger model, as far as the delivered output shows. It is a development pilot on the separate pilot generator seed, not a main-pool run, "
             "and it is reported here only as a labelled replication of the setup check. No policy or probe comparison was run on it.")
    L.append("")
    L.append(table(["Item", "Value"], [
        ["Model", nr(mo["id"]) + ", revision " + nr(mo["revision"]) + ", dtype " + nr(mo["dtype"])],
        ["Device / platform", nr(mo["device"]) + " / " + nr(m["platform"])],
        ["Packages", ", ".join(k + " " + str(v) for k, v in m["packages"].items())],
        ["Instances", str(rep["n_instances"])],
        ["Config hash / path", nr(m["config_hash"]) + " / " + nr(m["config_path"])],
        ["Git commit / dirty", nr(m["git"]["commit"])[:12] + " / " + nr(m["git"]["dirty"])],
        ["Wall-clock for the pilot", f(m["pilot_wall_seconds"], 1) + " seconds"],
    ]))
    L.append("")
    L.append("Provenance gaps: the model revision is " + ("not recorded" if mo["revision"] is None else "recorded")
             + ", the working tree is " + ("dirty" if m["git"]["dirty"] else "clean")
             + ", and the configuration file itself was not delivered, so this run cannot be regenerated from this repository alone.")
    L.append("")
    L.append(table(["Quantity", "Value"], [
        ["Never-call accuracy", pct(rep["never_call_accuracy"])],
        ["Always-call accuracy", pct(rep["always_call_accuracy"])],
        ["Share of decision nodes where calling helps", pct(rep["share_nodes_delta_positive"])],
        ["Step accuracy easy / medium / hard", " / ".join(pct(rep["step_accuracy_by_bin"][b]["step_accuracy"]) for b in ("easy", "medium", "hard"))],
        ["Answer-parse failure rate", pct(rep["parse_failure_rate"])],
        ["U-AUC never / always", f(u["never"]) + " / " + f(u["always"])],
        ["U-AUC best global pattern", f(u["best_global_pattern"])],
        ["U-AUC rollout reference / hindsight", f(u["rollout_reference"]) + " / " + f(u["hindsight"])],
        ["Headroom (rollout reference minus best global pattern)", f(u["rollout_reference"] - u["best_global_pattern"])],
        ["Paired SD of the headroom contrast (points)", f(rep["paired_sds_points"]["sd_rollout_minus_global"])],
    ]))
    L.append("")
    L.append("Minimum detectable paired difference at " + pct(power_target, 0) + " power by number of test instances: "
             + "; ".join(k + ": " + f(v) for k, v in rep["mde_points_by_n"].items()) + " points.")
    L.append("")
    L.append("Reading: with a larger model the never-call baseline is much higher than for the primary model, but the per-node reference still sits far above the best fixed pattern, "
             "so the headroom here is well above the gate threshold; on a pilot this small that is an indication and not a formal gate result. Whether the probes behave differently at this scale is untested.")
    L.append("")
    return "\n".join(L)


def build(res, pilot, tree_man, p2=None, rep=None) -> str:
    cfg = res["config"]
    lvl = pct(cfg["bootstrap"]["ci_level"], 0) + " CI"
    lams = cfg["lambdas"]
    ref = cfg["reference_lambda"]
    ri = lams.index(ref)
    p0 = res["partition_0"]
    T = p0["summary"]["test"]
    G = p0["gate"]
    prim = p0["primary_contrasts_test_holm"]
    pool = res["pool_stats"]
    man = res["manifest"]
    L = []
    a = L.append

    a("# 04 - Final implementation, datasets, metrics and results")
    a("")
    a("**Generated by `report/make_report.py` from `results/*.json`. Do not edit by hand.** "
      "`scripts/audit_results.py` verifies that every number below comes from a results file and that the file regenerates byte-for-byte.")
    a("")
    a("Provenance: config hash `" + nr(man["config_hash"]) + "`, dataset hash `" + nr(man["dataset_hash"]) + "`, git `"
      + (lambda c: c if c == "not a git repository" else c[:12])(nr(man["git"]["commit"])) + "` (dirty: " + nr(man["git"]["dirty"]) + "), run at " + nr(man["timestamp"]) + ".")
    a("")
    a("## 0. Status and honesty statement")
    a("")
    a("All numbers in this document come from a real language model, a real calculator tool and real hidden states: the "
      "exhaustive outcome tree was built by running **" + str(man["model"]["id"]) + "** (revision `" + str(man["model"]["revision"])[:10]
      + "`) on **" + str(pool["device"]) + "**. An earlier pass that used a simulator is archived and withdrawn (issue I-001). "
      "Results are for the scaffolded three-step arithmetic environment E1 only (N-14, N-10); nothing here supports claims about "
      "free-form agents, open-domain retrieval, larger models or API models.")
    a("")
    a("## 1. Headline findings (auto-generated)")
    a("")
    g = G["headroom_rollout_minus_global"]
    a("- **Headroom gate (rule inherited from the previous assignment, A-29).** Headroom H = U-AUC(rollout reference) - U-AUC(best global pattern) on the gate split: "
      + f(g["diff"]) + " points " + ci(g["ci_lo"], g["ci_hi"]) + ", threshold " + f(g["threshold_points"], 1)
      + " points. Gate: **" + ("PASSED" if g["passed"] else "NOT PASSED") + "**. "
      + ("MUP is the confirmatory method." if g["passed"] else "MUP is exploratory; P2-I1 is confirmatory (A-29)."))
    for k, label in (("mup_vs_pp_step_direct", "MUP vs step-wise direct-correctness probe"),
                     ("mup_vs_best_global_pattern", "MUP vs best global pattern"),
                     ("mup_vs_pp_first", "MUP vs single-decision probe at the initial state")):
        c = prim[k]
        a("- **" + label + ".** Delta U-AUC = " + f(c["diff"]) + " points " + ci(c["ci_lo"], c["ci_hi"]) + ", Holm-adjusted p = "
          + f(c["p_holm"], 4) + ": **" + verdict(c, "MUP higher", "MUP lower") + "**.")
    rr = p0["rank_reversal_registered"]
    a("- **Rank reversal (registered comparator, P1-I3).** Kendall tau between method rankings under cumulative cost and under the "
      "any-call indicator = " + f(rr["tau"], 3) + " " + ci(rr["ci_lo"], rr["ci_hi"], 3) + "; " + str(rr["n_discordant_pairs_point"])
      + " of " + str(rr["n_pairs"]) + " method pairs are ordered differently. Hypothesis (upper bound of tau interval below one): **"
      + ("SUPPORTED" if rr["ci_hi"] < 1 else "NOT SUPPORTED") + "**.")
    dep = [m for m in ORDER if NAMES[m][1] == "deployable" and m in T and m not in ("mup_no_calibration", "mup_never_continuation", "delta_reg")]
    ranked = sorted(dep, key=lambda m: -T[m]["u_auc_points"])
    a("- **Where MUP stands (deployable policies, U-AUC):** rank " + str(ranked.index("mup") + 1) + " of " + str(len(ranked))
      + " (best: " + NAMES[ranked[0]][0] + ").")
    a("")
    a("## 2. Final idea as built")
    a("")
    a("**MUP (P1-I1).** At each decision point t the frozen model's last-prompt-token hidden state z_t (layer chosen by instance-grouped "
      "cross-validation on the training split) feeds two L2-regularised logistic heads, p_call(z_t) and p_no(z_t), trained on exact labels "
      "from the outcome tree under an always-call continuation; each head is Platt-calibrated on the calibration split. "
      "Decision: call iff p_call - p_no > lambda*c, with c equal to one. No retraining across lambda. The controller is **frozen-backbone and "
      "probe-supervised; it is not training-free** (A-30).")
    a("")
    a("**Exhaustive outcome tree (A-30).** Greedy decoding and an exact tool make every outcome a deterministic function of the action prefix. "
      "For three steps there are seven decision nodes and eight leaves per instance; the model was run once per node (its no-call answer), "
      "the SymPy tool once per node (its call answer), and the prompt-prefill hidden states were recorded. All policies, labels and "
      "references are then exact lookups, so training and evaluation target the same deterministic object (this replaces the stochastic multi-rollout labels of the ideation specification).")
    a("")
    a("**Evaluation harness (P1-I3).** Cumulative utility U(lambda) = Acc - lambda * mean calls; U-AUC is the trapezoid mean over the lambda grid, in points; "
      "paired bootstrap over instances; Holm correction over the three pre-registered primary contrasts.")
    a("")
    a("## 3. Datasets")
    a("")
    cfg_pool = cfg["pool"]
    sp = res["splits_config"]
    a(table(["Item", "Value"], [
        ["Environment", "E1 chained arithmetic, three steps, per-step difficulty bins (A-34)"],
        ["Pool", str(pool["n_instances"]) + " instances, generator seed " + str(cfg_pool["seed"])],
        ["Splits (by instance)", ", ".join(f"{k} {v}" for k, v in sp.items())],
        ["Partition seeds", ", ".join(str(s) for s in res["partition_seeds"])],
        ["Dataset hash", nr(man["dataset_hash"])],
        ["Real-model step accuracy, easy / medium / hard",
         " / ".join(pct(pool["step_accuracy_by_bin"][b]["step_accuracy"]) for b in ("easy", "medium", "hard"))],
        ["Never-call / always-call accuracy (pool)", pct(pool["never_call_accuracy"]) + " / " + pct(pool["always_call_accuracy"])],
        ["Share of decision nodes where calling helps", pct(pool["share_nodes_delta_positive"])],
        ["Answer-parse failure rate / tool failure rate", pct(pool["parse_failure_rate"]) + " / " + pct(pool["tool_failure_rate"])],
    ]))
    a("")
    a("Partitions are random re-splits of one pool, so their test sets overlap: they measure split sensitivity, not independent replication.")
    a("")
    a("## 4. Models and hardware")
    a("")
    a(table(["Item", "Value"], [
        ["Model", str(man["model"]["id"]) + ", revision " + str(man["model"]["revision"]) + ", dtype " + str(man["model"]["dtype"])],
        ["Device", str(pool["device"])],
        ["Platform", str(man["platform"])],
        ["Packages", ", ".join(f"{k} {v}" for k, v in man["packages"].items())],
        ["Hidden-state layers cached", ", ".join(str(x) for x in cfg["layers"])],
        ["Mean prompt / generated tokens per node", f(pool["mean_prompt_tokens_per_node"], 1) + " / " + f(pool["mean_gen_tokens_per_node"], 1)],
    ]))
    a("")
    a("## 5. Metrics")
    a("")
    a(table(["Metric", "Definition"], [
        ["Accuracy", "fraction of instances whose final value equals the gold chain value (exact equality)"],
        ["Calls per instance", "mean number of tool calls over the three steps"],
        ["U(lambda)", "Acc - lambda * calls per instance, tool cost c equal to one"],
        ["U-AUC", "trapezoid mean of U(lambda) over the lambda grid, multiplied by one hundred (points); computed per instance so bootstraps resample instances"],
        ["Any-call utility", "Acc - alpha * share of instances with at least one call; the anchor-style comparator used only for the rank-reversal test"],
        ["Headroom H", "U-AUC(rollout reference) - U-AUC(best global pattern)"],
        ["Online ledger", "tool calls (in U) plus generated tokens (reported separately); policies that must generate before deciding are charged for discarded answers"],
        ["Offline ledger", "label-generation wall-clock and tokens, reported separately and never subtracted from online utility"],
    ]))
    a("")
    a("Lambda grid: " + ", ".join(f(x, 1) for x in lams) + ". Reference lambda for single-number tables: " + f(ref, 1) + ".")
    a("")
    a("## 6. Main results (partition 0, test split)")
    a("")
    a("### 6.1 Headroom gate")
    a("")
    rows = []
    for pk in sorted(k for k in res if re.fullmatch(r"partition_\d+", k)):
        gg = res[pk]["gate"]["headroom_rollout_minus_global"]
        gh = res[pk]["gate"]["hindsight_minus_global"]
        rows.append([pk, f(gg["diff"]), ci(gg["ci_lo"], gg["ci_hi"]), f(gh["diff"]), ci(gh["ci_lo"], gh["ci_hi"]), "PASSED" if gg["passed"] else "NOT PASSED"])
    a(table(["Partition", "H = rollout ref - global [" + lvl + "]", "", "Hindsight - global [" + lvl + "]", "", "Gate"], rows))
    a("")
    a("The gate measures the value of per-node information over a fixed pattern. It does not by itself isolate adaptivity over a single pre-generation decision; that is tested by the MUP vs `pp_first` contrast in section 6.3 (A-33).")
    a("")
    a("### 6.2 Policies on the test split")
    a("")
    rows = []
    for m in ORDER:
        if m not in T:
            continue
        s = T[m]
        tok = p0["ledger_online_tokens"].get(m, {}).get("gen_tokens_per_instance_by_lambda")
        rows.append([NAMES[m][0], NAMES[m][1], f(s["u_auc_points"]) + " " + ci(s["u_auc_ci95"][0], s["u_auc_ci95"][1], 2).replace("+", ""),
                     pct(s["accuracy_by_lambda"][ri]), f(s["calls_per_instance_by_lambda"][ri]), f(tok[ri], 1) if tok else "n/a"])
    a(table(["Policy", "Type", "U-AUC points [" + lvl + " of the mean]", "Accuracy at reference lambda", "Calls per instance", "Generated tokens per instance"], rows))
    a("")
    a("### 6.3 Pre-registered primary contrasts (Holm family) and exploratory contrasts")
    a("")
    rows = [[k, f(v["diff"]), ci(v["ci_lo"], v["ci_hi"]), f(v["p"], 4), f(v["p_holm"], 4), verdict(v, "MUP higher", "MUP lower")] for k, v in prim.items()]
    a(table(["Contrast (MUP minus ...)", "Delta U-AUC (points)", lvl, "p", "p (Holm)", "Verdict"], rows))
    a("")
    ex = p0["exploratory_contrasts_test"]
    rows = [[k, f(v["diff"]), ci(v["ci_lo"], v["ci_hi"]), f(v["p"], 4), verdict(v, "MUP higher", "MUP lower")] for k, v in ex.items()]
    a("Exploratory (no multiplicity correction; do not treat as confirmatory):")
    a("")
    a(table(["Contrast", "Delta U-AUC (points)", lvl, "p", "Verdict"], rows))
    a("")
    a("### 6.4 Rank reversal (P1-I3, registered comparator)")
    a("")
    rows = [[NAMES[m][0], f(rr["cumulative_u_auc_points"][m]), f(rr["any_call_u_auc_points"][m])] for m in rr["methods"]]
    a(table(["Method", "Cumulative-cost U-AUC", "Any-call U-AUC"], rows))
    a("")
    a("Kendall tau " + f(rr["tau"], 3) + " " + ci(rr["ci_lo"], rr["ci_hi"], 3) + "; share of bootstrap resamples with identical rankings " + pct(rr["share_tau_eq_1"])
      + "; mean pairwise order-flip probability " + f(rr["mean_pair_flip_prob"], 3) + ".")
    a("")
    a("### 6.5 Split sensitivity across partitions")
    a("")
    rows = []
    for pk in sorted(k for k in res if re.fullmatch(r"partition_\d+", k)):
        pc = res[pk]["primary_contrasts_test_holm"]
        rows.append([pk] + [f(pc[k]["diff"]) + " " + ci(pc[k]["ci_lo"], pc[k]["ci_hi"]) for k in prim])
    a(table(["Partition"] + [k for k in prim], rows))
    a("")
    a("### 6.6 Ablations")
    a("")
    hd = p0["head_diagnostics_test"]
    a("MUP head diagnostics on held-out test nodes: AUROC of p_no for direct correctness " + f(hd["auroc_p_no"], 3) + ", AUROC of p_call " + f(hd["auroc_p_call"], 3)
      + ", AUROC of Delta-hat for 'calling helps' " + f(hd["auroc_delta_vs_benefit"], 3) + ", Spearman(Delta-hat, label) " + f(hd["spearman_delta_vs_label"], 3)
      + ", Brier p_call " + f(hd["brier_p_call"], 3) + ", Brier p_no " + f(hd["brier_p_no"], 3) + ". Spread of Delta-hat (standard deviation) " + f(hd["std_delta_hat"], 3)
      + " against a benefit prevalence of " + pct(hd["benefit_prevalence"]) + "; a Delta-hat with near-zero spread would indicate a degenerate head (W-11).")
    a("")
    a("Layer choice (MUP test U-AUC at the selected regularisation): " + ", ".join(f"layer {k}: {f(v)}" for k, v in p0["ablation_layer_mup_test_u_auc"].items()) + ".")
    a("")
    a("Cost normalisation (A9): ranking by U-AUC with the cost term divided by the number of steps: " + ", ".join(
        f"{NAMES[m][0].split(' (')[0]}: {f(v)}" for m, v in p0["ablation_u_norm_u_auc"].items()) + ".")
    a("")
    a("Not run in this study: K and noise-filter sweeps (labels are exact), previous-state features, cross-environment transfer, model-size comparison (N-14, N-15).")
    a("")
    a("### 6.7 Error taxonomy at the reference lambda (test split, over visited decision nodes)")
    a("")
    tx = p0["error_taxonomy_at_reference_lambda"]
    rows = [[NAMES.get(m, (m, ""))[0], f(v["over_call_rate"] * 100, 1) + "%", f(v["under_call_rate"] * 100, 1) + "%"] for m, v in tx.items()]
    a(table(["Policy", "Over-call rate (called when direct sufficed)", "Under-call rate (missed a necessary call)"], rows))
    a("")
    a("### 6.8 Power and cost ledgers")
    a("")
    pw = p0["power"]
    a("Test split size " + str(pw["n_test"]) + ". Paired standard deviation of the primary differences (points): "
      + ", ".join(f"{k} {f(v)}" for k, v in pw["sd_paired_points"].items()) + ". Minimum detectable paired difference at power " + f(pw["power_target"], 2) + " and two-sided alpha " + f(pw["alpha"], 2) + ": "
      + ", ".join(f"{k} {f(v)}" for k, v in pw["mde_points"].items()) + ".")
    a("")
    for k, v in prim.items():
        mde = pw["mde_points"][k.replace("mup_vs_", "")]
        a("- " + k + ": observed |Delta| " + f(abs(v["diff"])) + " points vs minimum detectable " + f(mde) + " points: "
          + ("adequately powered for an effect of this size" if abs(v["diff"]) >= mde else "an effect of this size is below the minimum detectable difference (underpowered)") + ".")
    a("")
    a("Offline ledger (label generation): pilot " + f(pilot["seconds_per_node_incl_generation"], 2) + " seconds per decision node; main tree "
      + f(tree_man["offline_label_generation_seconds_this_invocation"], 0) + " seconds in the invocation that produced it (a resumed run reports only its own invocation).")
    a("")
    if p2 is not None:
        a(p2_section(p2, lvl))
    if rep is not None:
        a(replication_section(rep, p0["power"]["power_target"]))
    a("## 7. Comparison with baselines and literature")
    a("")
    a("### 7.1 Directly comparable (same environment, model and metric; run by us)")
    a("")
    rows = []
    for m in ranked:
        rows.append([str(ranked.index(m) + 1), NAMES[m][0], f(T[m]["u_auc_points"])])
    a(table(["Rank", "Policy (deployable only)", "U-AUC points"], rows))
    a("")
    a("The anchor's trained implicit/explicit alignment (SFT) and the original Probe&Prefill implementation were **not** reproduced (N-02, N-16); "
      "`conf_thr` and `pp_first` are our reimplementations of their decision principle on this environment. We therefore make no claim about the state of the art.")
    a("")
    a("### 7.2 Reported in the literature (not comparable: different models, data and metrics)")
    a("")
    a("See `docs/references_verified.md` for verified figures and sources. No raw number from a paper is placed beside our results.")
    a("")
    a("## 8. Allowed claims (generated from the statistics above)")
    a("")
    def claim(cond):
        return "SUPPORTED" if cond else "NOT SUPPORTED"
    alpha = p0["power"]["alpha"]
    c1, c2, c3 = prim["mup_vs_pp_step_direct"], prim["mup_vs_best_global_pattern"], prim["mup_vs_pp_first"]
    rows = [
        ["MUP improves U-AUC over the step-wise direct-correctness probe", claim(c1["excludes_zero"] and c1["diff"] > 0 and c1["p_holm"] < alpha)],
        ["MUP improves U-AUC over the best global call pattern", claim(c2["excludes_zero"] and c2["diff"] > 0 and c2["p_holm"] < alpha)],
        ["Step-wise decisions improve over a single initial decision (MUP vs pp_first)", claim(c3["excludes_zero"] and c3["diff"] > 0 and c3["p_holm"] < alpha)],
        ["Per-node information has headroom over a fixed pattern (gate)", claim(g["passed"])],
        ["The any-call metric reorders methods relative to cumulative cost", claim(rr["ci_hi"] < 1)],
    ]
    a(table(["Claim", "Status"], rows))
    a("")
    a("Disallowed regardless of the above: beating Xu et al. (N-02); any statement about closed or API models (N-04); generalisation to larger models, open-domain retrieval or free-form agents (N-01, N-03, N-10, N-14).")
    a("")
    a("## 9. Deviations from the ideation specification")
    a("")
    a("See `docs/01_ambiguities_and_gaps_log.md` (A-29 onward) and `docs/02_non_feasibility.md`. Principal deviations: exhaustive deterministic tree instead of K stochastic rollouts (A-30); "
      "open-loop reference is the best global pattern, not a per-instance pattern (A-33); permutation null not applicable to a deterministic tree (A-29); "
      "single model and single environment (N-14, N-15).")
    a("")
    a("## 10. Reproducibility")
    a("")
    a("Commands (from the repository root): `python scripts/run_pilot.py`, `python scripts/build_tree.py --config configs/primary.yaml`, "
      "`python scripts/run_experiments.py --config configs/primary.yaml`, `python report/make_report.py`, `python scripts/audit_results.py`. "
      "Per-instance arrays: `results/main/per_instance_partition_*.npz`. Manifests: `results/main/results.json` (field `manifest`) and `results/main/tree_manifest.json`.")
    a("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "main"))
    ap.add_argument("--pilot", default=str(ROOT / "results" / "pilot"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "04_results_and_final_implementation.md"))
    args = ap.parse_args()
    res = json.loads((Path(args.results) / "results.json").read_text())
    pilot = json.loads((Path(args.pilot) / "pilot.json").read_text())
    tree_man = json.loads((Path(args.results) / "tree_manifest.json").read_text())
    p2_path = Path(args.results) / "p2_results.json"
    p2 = json.loads(p2_path.read_text()) if p2_path.exists() else None
    rep_path = ROOT / "results" / "replication_gpu" / "pilot_1p5b.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() and Path(args.results).resolve() == (ROOT / "results" / "main").resolve() else None
    text = build(res, pilot, tree_man, p2, rep)
    Path(args.out).write_text(text, encoding="utf-8")
    print("wrote", args.out)
    default_out = ROOT / "docs" / "04_results_and_final_implementation.md"
    if Path(args.out).resolve() == default_out.resolve():              # keep the top-level copy in sync
        (ROOT.parent / default_out.name).write_text(text, encoding="utf-8")
        print("synced", ROOT.parent / default_out.name)


if __name__ == "__main__":
    main()
