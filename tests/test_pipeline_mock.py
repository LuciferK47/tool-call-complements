"""End-to-end test of scripts/run_experiments.py on a MOCK tree: checks plumbing and invariants, not scientific claims."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from mock_runner import MockRunner
from toolcomp.envs.arith import generate_instances
from toolcomp.tree import TreeBuilder

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_runs_and_invariants_hold(tmp_path):
    ins = generate_instances(240, seed=21, prefix="mock")
    tree = TreeBuilder(MockRunner(), [1, 2], log=lambda *_: None).build(ins, tmp_path, "mock")
    tree.save(tmp_path / "mock_tree.npz")
    cfg = {"name": "mock", "layers": [1, 2], "lambdas": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
           "splits": {"train": 120, "cal": 40, "gate": 30, "test": 50}, "partition_seeds": [1, 2],
           "gate": {"threshold_points": 2.0}, "reference_lambda": 0.3, "bootstrap": {"n_resamples": 300, "seed": 0, "ci_level": 0.95},
           "model": {"id": "mock", "revision": "mock-revision", "dtype": "float32"}, "pool": {"n": 240, "seed": 21}}
    cfg_path = tmp_path / "mock.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_name = "_pytest_mock"
    out_dir = ROOT / "results" / out_name
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir).mkdir(parents=True)
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiments.py"), "--config", str(cfg_path),
                            "--cache-dir", str(tmp_path), "--out", out_name], capture_output=True, text=True, cwd=ROOT, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        res = json.loads((out_dir / "results.json").read_text())
        for pk in ("partition_0", "partition_1"):
            p = res[pk]
            summ = p["summary"]["test"]
            # closed-form facts about reference policies
            assert summ["hindsight_nondeployable"]["u_auc_points"] >= max(v["u_auc_points"] for v in summ.values()) - 1e-9
            for lam_i, lam in enumerate(cfg["lambdas"]):
                assert abs(summ["never"]["calls_per_instance_by_lambda"][lam_i]) < 1e-12
                assert abs(summ["always"]["calls_per_instance_by_lambda"][lam_i] - 3.0) < 1e-12
                assert abs(summ["never"]["utility_by_lambda"][lam_i] - summ["never"]["accuracy_by_lambda"][lam_i]) < 1e-12
            # Holm-adjusted p-values never smaller than raw ones
            for v in p["primary_contrasts_test_holm"].values():
                assert v["p_holm"] >= v["p"] - 1e-12
            assert p["gate"]["headroom_rollout_minus_global"]["passed"] in (True, False)
            assert p["split_sizes"] == {"train": 120, "cal": 40, "gate": 30, "test": 50}
        # the two partitions use different instance assignments
        assert res["partition_0"]["split_instance_id_hash"] != res["partition_1"]["split_instance_id_hash"]
        # no policy other than the two oracle-style references may beat hindsight; matched random must be matched
        # report generation from these results: complete, deterministic, no leaked None/nan
        (out_dir / "tree_manifest.json").write_text(json.dumps({"offline_label_generation_seconds_this_invocation": 1.0}))
        pilot_dir = tmp_path / "pilot"
        pilot_dir.mkdir()
        (pilot_dir / "pilot.json").write_text(json.dumps({"seconds_per_node_incl_generation": 0.5}))
        docs = []
        for k in range(2):
            o = tmp_path / f"doc{k}.md"
            rr = subprocess.run([sys.executable, str(ROOT / "report" / "make_report.py"), "--results", str(out_dir),
                                 "--pilot", str(pilot_dir), "--out", str(o)], capture_output=True, text=True, cwd=ROOT)
            assert rr.returncode == 0, rr.stderr[-1500:]
            docs.append(o.read_text())
        assert docs[0] == docs[1]
        for needle in ("Headroom gate", "Pre-registered primary contrasts", "Rank reversal", "Allowed claims"):
            assert needle in docs[0]
        bad_lines = [ln[:160] for ln in docs[0].splitlines() if " nan" in ln.lower() or "None" in ln.replace("None of", "")]
        assert not bad_lines, bad_lines[:5]
        s = res["partition_0"]["summary"]["test"]
        assert abs(sum(s["random_matched_to_mup"]["calls_per_instance_by_lambda"]) - sum(s["mup"]["calls_per_instance_by_lambda"])) < 1e-9
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_p2_report_section_renders():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_report", ROOT / "report" / "make_report.py")
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    contrast = {"diff": 1.0, "ci_lo": 0.2, "ci_hi": 1.8, "p": 0.01, "p_holm": 0.02, "excludes_zero": True}
    r = {"net_u_auc_points": {m: 1.0 for m in ("never", "always", "logprob_thr", "fixed_k", "sequential")},
         "contrasts": {"sequential_minus_fixed_k": contrast, "sequential_minus_logprob_thr": contrast},
         "mean_samples_sequential_by_lambda": {"0.10": 3.2}}
    p2 = {"n_cal_queries": 10, "n_test_queries": 20, "m_max": 8, "p_tool_cal": 1.0, "direct_accuracy_single_sample_test": 0.5,
          "greedy_accuracy_test": 0.5, "parse_failure_rate_samples": 0.0, "by_rho": {"0.02": r}, "break_even_rho_vs_fixed_k": None}
    text = mr.p2_section(p2, "95% CI")
    assert "Black-box track" in text and "not within the grid" in text and "None" not in text


def test_replication_section_renders_and_states_provenance_gaps():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_report", ROOT / "report" / "make_report.py")
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    rep = {"manifest": {"model": {"id": "Qwen/test", "revision": None, "device": "cuda", "dtype": "float16"}, "platform": "Windows-test",
                        "packages": {"torch": "x"}, "config_hash": "abc", "config_path": "cfg.yaml", "git": {"commit": "0123456789abcdef", "dirty": True},
                        "pilot_wall_seconds": 12.0},
           "n_instances": 5, "never_call_accuracy": 0.5, "always_call_accuracy": 1.0, "share_nodes_delta_positive": 0.2, "parse_failure_rate": 0.0,
           "step_accuracy_by_bin": {b: {"step_accuracy": 0.8} for b in ("easy", "medium", "hard")},
           "u_auc_points": {"never": 50.0, "always": -5.0, "best_global_pattern": 55.0, "rollout_reference": 80.0, "hindsight": 81.0},
           "paired_sds_points": {"sd_rollout_minus_global": 20.0}, "mde_points_by_n": {"100": 8.0}}
    text = mr.replication_section(rep, 0.8)
    assert "not verified by the primary pipeline" in text and "revision is not recorded" in text and "dirty" in text and "None" not in text
    assert "25.00" in text            # headroom computed from the JSON, not typed
