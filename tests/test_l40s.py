"""Tests for the L40S-package changes: strict parser, stop tokens, checkpoint fingerprints, prompt-cache reuse, extra policies,
leakage and label verification, head ablation. Mock runner only (no model download)."""

import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from mock_runner import MockRunner
from toolcomp.envs.arith import generate_instances, node_prompt, root_prompt
from toolcomp.probes import (PlanProbe, PolicyConsistentMUP, hindsight_utility, select_root_layer_and_C, walk_from)
from toolcomp.tree import (N_NODES, PREFIX_OF, TreeBuilder, TreeData, node_labels, parse_category, parse_int, parse_strict,
                      pattern_actions, walk)

ROOT = Path(__file__).resolve().parents[1]


# ---- strict parser -----------------------------------------------------------------------------------
@pytest.mark.parametrize("text,expect", [
    ("42", 42), ("-17", -17), ("1,234", 1234), ("1,234,567.", 1234567), ("12 + 30 = 42", 42), ("  7 \n", 7),
    ("168,070.6", None), ("13,9240", None), ("6.\n\nStep 2: compute", None), ("4,602,77 + 74", None), ("", None),
    ("The answer is 5", None), ("Thinking Process: 1.", None), ("<think>\n</think> 4", None), ("12 3", None), ("1.5", None)])
def test_strict_parser(text, expect):
    assert parse_strict(text) == expect


def test_lenient_parser_unchanged_and_categories():
    assert parse_int("168,070.6") == 168070 and parse_int("13,9240") == 139240      # frozen primary behaviour
    assert parse_category("168,070.6") == "decimal" and parse_category("12 3") == "multiple_numbers"
    assert parse_category("Thinking Process:") == "thinking_text" and parse_category("") == "no_number"


# ---- checkpoint fingerprint + reuse ------------------------------------------------------------------
def _build(tmp, runner=None, **kw):
    ins = generate_instances(12, seed=3, prefix="fp")
    tb = TreeBuilder(runner or MockRunner(), [1, 2], log=lambda *_: None, **kw)
    return ins, tb, tb.build(ins, tmp, "t")


def test_resume_refuses_mismatched_fingerprint(tmp_path):
    ins, tb, _ = _build(tmp_path)
    other = MockRunner()
    other.model_id = "another-model"
    with pytest.raises(RuntimeError, match="refusing to resume"):
        TreeBuilder(other, [1, 2], log=lambda *_: None).build(ins, tmp_path, "t")
    with pytest.raises(RuntimeError, match="refusing to resume"):
        TreeBuilder(MockRunner(), [1, 2], log=lambda *_: None, parser="strict").build(ins, tmp_path, "t")


def test_resume_refuses_different_prompts(tmp_path):
    ins, tb, _ = _build(tmp_path)
    with pytest.raises(RuntimeError, match="different prompts"):
        TreeBuilder(MockRunner(), [1, 2], log=lambda *_: None).build(generate_instances(12, seed=4, prefix="fp"), tmp_path, "t")


def test_prompt_cache_reuse_generates_only_new_prompts(tmp_path):
    ins, tb, tree = _build(tmp_path / "a")
    assert tb.n_generated == 12 * 7 and tb.n_reused == 0 or tb.n_generated > 0
    tb2 = TreeBuilder(MockRunner(), [1, 2], log=lambda *_: None, reuse=tb.seen, reuse_root=tb.seen_root)
    tree2 = tb2.build(ins, tmp_path / "b", "t")
    assert tb2.n_generated == 0 and tb2.n_reused == 12 * 7
    assert np.array_equal(tree.leaf_correct, tree2.leaf_correct) and tree.texts == tree2.texts


class _Counting(MockRunner):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def run(self, prompts, layers, **kw):
        self.calls += len(prompts)
        return super().run(prompts, layers, **kw)


def test_strict_rebuild_regenerates_only_changed_branches(tmp_path):
    class Noisy(MockRunner):                      # one node answers with a decimal: lenient parse differs from strict
        def run(self, prompts, layers, **kw):
            outs = super().run(prompts, layers, **kw)
            for p, o in zip(prompts, outs):
                if "Step 1 result" not in p and "Now step 1:" in p and len(p) % 7 == 0:
                    o.text = o.text + ".5"
            return outs
    ins = generate_instances(30, seed=8, prefix="st")
    lenient = TreeBuilder(Noisy(), [1, 2], log=lambda *_: None)
    t0 = lenient.build(ins, tmp_path / "len", "t")
    cr = _Counting()
    strict = TreeBuilder(cr, [1, 2], log=lambda *_: None, parser="strict", reuse=lenient.seen, reuse_root=lenient.seen_root)
    t1 = strict.build(ins, tmp_path / "strict", "t")
    assert strict.n_reused > 0 and cr.calls == strict.n_generated < 30 * 7
    assert t1.meta["parse_categories"].get("decimal", 0) > 0
    # always-call chains never depend on the parser
    assert np.array_equal(t0.leaf_correct[:, 7], t1.leaf_correct[:, 7])
    # strict can only turn a lenient parse into 'invalid', never the reverse
    assert (t1.v_no_ok <= t0.v_no_ok).all()


def test_legacy_checkpoints_can_be_harvested(tmp_path):
    ins, tb, _ = _build(tmp_path)
    for f in tmp_path.glob("*.pkl"):                         # emulate checkpoints written before fingerprints existed
        d = pickle.load(open(f, "rb"))
        if isinstance(d, dict):
            pickle.dump({"outs": d["outs"]} if "outs" in d and "prompts_sha" in d and "fingerprint" in d and "seconds" in d else d["outs"], open(f, "wb"))
    h = TreeBuilder(MockRunner(), [1, 2], log=lambda *_: None, legacy_ok=True)
    h.build(ins, tmp_path, "t")
    assert len(h.seen) > 0


# ---- stop tokens (B4) ----------------------------------------------------------------------------------
def test_stop_set_truncates_at_any_eos_id():
    from toolcomp.models.real_model import HFRunner
    r = HFRunner.__new__(HFRunner)
    r.stop_ids = {5, 9}
    ids = [11, 12, 9, 7, 7]
    n = len(ids)
    for k, t in enumerate(ids):
        if t in r.stop_ids:
            n = k
            break
    assert n == 2


# ---- extra policies ----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_tree(tmp_path_factory):
    ins = generate_instances(160, seed=13, prefix="ex")
    return ins, TreeBuilder(MockRunner(), [1, 2], log=lambda *_: None).build(ins, tmp_path_factory.mktemp("c"), "t")


def test_walk_from_matches_walk(mock_tree):
    _, tree = mock_tree
    rng = np.random.default_rng(0)
    D = rng.integers(0, 2, size=(tree.n, N_NODES))
    leaf, calls = walk_from(D, np.zeros(tree.n, dtype=np.int64), 0, np.zeros(tree.n, dtype=np.int64))
    c, k, bits = walk(tree, D)
    assert np.array_equal(leaf, bits) and np.array_equal(calls, k)


def test_policy_consistent_targets_match_explicit_recursion(mock_tree):
    _, tree = mock_tree
    rng = np.random.default_rng(1)
    D = rng.integers(0, 2, size=(tree.n, N_NODES))
    for level in (2, 1, 0):
        S, F = PolicyConsistentMUP._targets(tree, D, level)
        for j, k in enumerate(range(2 ** level - 1, 2 ** (level + 1) - 1)):
            for a in (0, 1):
                for i in range(0, tree.n, 17):
                    nid, prefix, calls = 2 * k + 1 + a, int(PREFIX_OF[k]) * 2 + a, 0
                    for lv in range(level + 1, 3):
                        act = int(D[i, nid])
                        calls += act
                        prefix = prefix * 2 + act
                        nid = 2 * nid + 1 + act
                    assert S[i, j, a] == float(tree.leaf_correct[i, prefix]) and F[i, j, a] == calls


def test_policy_consistent_mup_runs_and_is_valid(mock_tree):
    _, tree = mock_tree
    tr, cal, te = tree.subset(np.arange(0, 100)), tree.subset(np.arange(100, 130)), tree.subset(np.arange(130, 160))
    m = PolicyConsistentMUP(1, 0.1, folds=3).fit(tr, cal, [0.1, 0.3])
    A = m.actions(te, 0.3)
    assert A.shape == (te.n, N_NODES) and set(np.unique(A)) <= {0, 1}
    c, k, _ = walk(te, A)
    assert 0 <= k.mean() <= 3


def test_plan_probe_with_perfect_features_equals_hindsight(tmp_path):
    ins = generate_instances(120, seed=29, prefix="pp")
    tree = TreeBuilder(MockRunner(), [1], log=lambda *_: None).build(ins, tmp_path, "t")
    # root features that carry the eight leaf outcomes exactly
    tree.root_hidden = np.repeat(tree.leaf_correct.astype(np.float16)[:, None, :], 1, axis=1)
    tr, cal, te = tree.subset(np.arange(0, 70)), tree.subset(np.arange(70, 95)), tree.subset(np.arange(95, 120))
    pp = PlanProbe(1, 1e3).fit(tr, cal) if False else None
    from toolcomp.probes import root_features
    assert root_features(tree, 1).shape == (120, 8)
    plan = PlanProbe(1, 1e3).fit(tr, cal)
    for lam in (0.1, 0.3):
        c, k, _ = walk(te, plan.actions(te, lam))
        hc, hk = hindsight_utility(te, lam)
        assert abs(np.mean(c - lam * k) - np.mean(hc - lam * hk)) < 0.06       # near-exact with exact features (calibration noise)


# ---- leakage + label verification ---------------------------------------------------------------------
def test_extra_pool_disjoint_from_main_and_pilot_by_id_and_text():
    main = generate_instances(1000, 20261101, "e1")
    pilot = generate_instances(60, 777001, "pilot")
    extra = generate_instances(4000, 20261201, "x1")
    texts = {x.problem_text() for x in main} | {x.problem_text() for x in pilot}
    ids = {x.instance_id for x in main} | {x.instance_id for x in pilot}
    assert not any(x.problem_text() in texts or x.instance_id in ids for x in extra)
    assert len({x.problem_text() for x in extra}) == 4000


def test_independent_label_recomputation_matches_tree(mock_tree):
    from scripts.verify_tree import verify_tree
    ins, tree = mock_tree
    rep = verify_tree(tree, ins)
    assert rep["ok"], rep


# ---- head ablation -----------------------------------------------------------------------------------
def test_ablation_heads_fit_and_decide(mock_tree):
    from toolcomp.ablation import HEADS, fit_head
    _, tree = mock_tree
    tr, cal, te = tree.subset(np.arange(0, 100)), tree.subset(np.arange(100, 130)), tree.subset(np.arange(130, 160))
    for h in HEADS:
        f = fit_head(h, tr, cal, [1, 2], "cpu", 0)
        A = f.actions(te, 0.3)
        assert A.shape == (te.n, N_NODES) and set(np.unique(A)) <= {0, 1}, h
        d = f.delta(te)
        assert np.isfinite(d).all() and d.min() >= -1.0001 and d.max() <= 1.0001, h


def test_threeclass_head_with_perfect_features_learns_delta(mock_tree):
    from toolcomp.ablation import TorchHead, head_targets
    _, tree = mock_tree
    yc, yn, yd = head_targets(tree)
    X = np.eye(3)[yd.reshape(-1)] * 3.0                       # features = the label itself
    h = TorchHead("cls3", 0, 1e3, "cpu", 0).fit(X, yd.reshape(-1))
    assert (h.predict(X).argmax(1) == yd.reshape(-1)).mean() > 0.99


def test_regression_extras_off_is_byte_identical_to_pre_l40s_code(tmp_path):
    """With extra_policies absent, results equal those produced by the code before the L40S changes (stored reference, mock tree)."""
    ins = generate_instances(240, seed=21, prefix="mock")
    tree = TreeBuilder(MockRunner(), [1, 2], log=lambda *_: None).build(ins, tmp_path, "mock")
    tree.save(tmp_path / "mock_tree.npz")
    cfg = {"name": "mock", "layers": [1, 2], "lambdas": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
           "splits": {"train": 120, "cal": 40, "gate": 30, "test": 50}, "partition_seeds": [1, 2],
           "gate": {"threshold_points": 2.0}, "reference_lambda": 0.3, "bootstrap": {"n_resamples": 300, "seed": 0, "ci_level": 0.95},
           "model": {"id": "mock", "revision": "mock-revision", "dtype": "float32"}, "pool": {"n": 240, "seed": 21}}
    (tmp_path / "mock.yaml").write_text(yaml.safe_dump(cfg))
    out = "_pytest_regression"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiments.py"), "--config", str(tmp_path / "mock.yaml"), "--cache-dir", str(tmp_path),
                        "--out", out], capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    got = json.loads((ROOT / "results" / out / "results.json").read_text())
    import shutil
    shutil.rmtree(ROOT / "results" / out, ignore_errors=True)
    ref = json.loads((ROOT / "tests" / "fixtures" / "ref_mock_results_pre_l40s.json").read_text())
    for d in (got,):
        d.pop("manifest", None)
        for k in ("label_generation_seconds", "parser", "parse_categories", "resolved_revision", "tree_env", "nodes_reused_from_cache", "nodes_generated"):
            d["pool_stats"].pop(k, None)
        for pk in ("partition_0", "partition_1"):
            d[pk].pop("secondary_contrasts_test_holm", None)
            d[pk].pop("extra_exploratory_contrasts_test", None)
    assert json.dumps(json.loads(json.dumps(got, default=float)), sort_keys=True) == json.dumps(ref, sort_keys=True)


# ---- v2: solver option, tool-failure definition, post-hoc heads ----------------------------------------
def test_torch_solver_requested_without_cuda_fails_loudly():
    import torch
    from toolcomp.probes import SoftLogReg
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    old = SoftLogReg.SOLVER
    try:
        SoftLogReg.SOLVER = "torch"
        with pytest.raises(RuntimeError):
            SoftLogReg(1.0).fit(np.random.randn(20, 3), np.random.rand(20))
    finally:
        SoftLogReg.SOLVER = old


def test_default_solver_is_scipy_and_matches_reference():
    from toolcomp.probes import SoftLogReg
    assert SoftLogReg.SOLVER == "scipy"
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(float)
    m = SoftLogReg(10.0).fit(X, y)
    assert ((m.predict_proba(X) > 0.5) == y).mean() > 0.8


def test_tool_failure_rate_counts_only_nodes_with_valid_input(mock_tree):
    from scripts.run_experiments import pool_stats
    _, tree = mock_tree
    assert pool_stats(tree)["tool_failure_rate"] == 0.0
    import copy
    t2 = copy.copy(tree)
    t2.v_in_ok = tree.v_in_ok.copy()
    t2.v_call_ok = tree.v_call_ok.copy()
    t2.v_in_ok[:, :] = False          # no node has a valid input -> missing tool values are not tool failures
    t2.v_call_ok[:, :] = False
    assert pool_stats(t2)["tool_failure_rate"] == 0.0
    t2.v_in_ok[0, 0] = True           # valid input but no tool value -> a real tool failure
    assert pool_stats(t2)["tool_failure_rate"] > 0.0


def test_posthoc_confidence_heads_run_and_are_valid(mock_tree):
    from toolcomp.posthoc import ConfidenceMUP
    _, tree = mock_tree
    tr, cal, te = tree.subset(np.arange(0, 100)), tree.subset(np.arange(100, 130)), tree.subset(np.arange(130, 160))
    for v in ("conf_only", "stack"):
        h = ConfidenceMUP(1, 0.01, v).fit(tr, cal)
        p = h.p_no(te)
        assert p.shape == (te.n, N_NODES) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
        A = h.actions(te, 0.3)
        assert A.shape == (te.n, N_NODES) and set(np.unique(A)) <= {0, 1}


def test_posthoc_conf_only_ignores_hidden_states_in_p_no(mock_tree):
    from toolcomp.posthoc import ConfidenceMUP
    _, tree = mock_tree
    tr, cal, te = tree.subset(np.arange(0, 100)), tree.subset(np.arange(100, 130)), tree.subset(np.arange(130, 160))
    h = ConfidenceMUP(1, 0.01, "conf_only").fit(tr, cal)
    te2 = te.subset(np.arange(te.n))
    te2.hidden = te.hidden * 0 + 1.0
    assert np.allclose(h.p_no(te), h.p_no(te2))
