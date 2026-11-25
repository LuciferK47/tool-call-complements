"""Iteration-2 checks. They need the cached 0.8B outcome tree (cache/q35_0p8b/) and skip cleanly without it."""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "cache" / "q35_0p8b" / "q35_0p8b_tree.npz"
RESULTS = ROOT / "results" / "q35_0p8b" / "results.json"
pytestmark = pytest.mark.skipif(not (CACHE.exists() and RESULTS.exists()), reason="0.8B tree cache / results not present")

from toolcomp.envs.arith import generate_instances
from toolcomp.estimators import DistilledTwoHead, FusedTwoHead, tune_thresholds
from toolcomp.eval_harness import LAMBDAS, u_auc_instances
from toolcomp.features import (draft_features, hidden_features, prefill_features, surface_features, with_features)
from toolcomp.probes import ThresholdPolicy, TwoHeadMUP
from toolcomp.splits import make_splits
from toolcomp.tree import N_NODES, TreeData, node_labels, walk

CFG_SPLITS = {"train": 450, "cal": 100, "gate": 100, "test": 350}
SEED0 = 20261111


@pytest.fixture(scope="module")
def pool():
    return TreeData.load(CACHE)


@pytest.fixture(scope="module")
def ins():
    return generate_instances(1000, 20261101, "e1")


@pytest.fixture(scope="module")
def sp(pool):
    return make_splits(pool.instance_ids, CFG_SPLITS, SEED0)


def _u(tree, A_fn):
    us = []
    for lam in LAMBDAS:
        c, k, _ = walk(tree, A_fn(lam))
        us.append(c - lam * k)
    return u_auc_instances(np.stack(us, 1)).mean()


def test_dataset_hash_matches_config(ins):
    from toolcomp.envs.arith import dataset_hash
    assert dataset_hash(ins) == "6517fd246393057f"


def test_features_are_deterministic_and_label_free(pool, ins):
    a = surface_features(pool, ins)
    blank = dataclasses.replace(pool, leaf_correct=np.zeros_like(pool.leaf_correct))
    b = surface_features(blank, ins)
    assert a.shape[:2] == (pool.n, N_NODES) and np.array_equal(a, b)          # no label information inside S0
    assert np.array_equal(a, surface_features(pool, ins))
    assert prefill_features(pool).shape == (pool.n, N_NODES, 1)
    assert draft_features(pool).shape == (pool.n, N_NODES, 2)


def test_shim_tree_runs_existing_estimators(pool, ins, sp):
    sh = with_features(pool, surface_features(pool, ins))
    tr, ca = sh.subset(sp["train"]), sh.subset(sp["cal"])
    m = TwoHeadMUP(0, 0.01).fit(tr, ca)
    d = m.delta(sh.subset(sp["test"]))
    assert d.shape == (350, N_NODES) and np.isfinite(d).all()


def test_reproduces_iteration1_mup_and_conf_thr(pool, sp):
    """S1 two-head, untuned, with the frozen layer/C == MUP 41.23; answer-confidence threshold == 45.66 (partition 0)."""
    prior = json.loads(RESULTS.read_text())["partition_0"]["fit_info"]["mup"]
    sh = with_features(pool, hidden_features(pool, prior["layer"]))
    tr, ca, te = sh.subset(sp["train"]), sh.subset(sp["cal"]), sh.subset(sp["test"])
    m = TwoHeadMUP(0, prior["C"]).fit(tr, ca)
    assert abs(_u(te, lambda lam: m.actions(te, lam)) - 41.23) < 0.02
    caT, teT = pool.subset(sp["cal"]), pool.subset(sp["test"])
    q = np.unique(np.quantile(caT.mean_logprob.reshape(-1), np.linspace(0.02, 0.98, 49)))
    thr = ThresholdPolicy("conf_thr", lambda t: t.mean_logprob, "low", q).tune(caT, list(LAMBDAS))
    assert abs(_u(teT, lambda lam: thr.actions(teT, lam)) - 45.66) < 0.02


def test_tuned_threshold_never_worse_than_untuned_on_cal(pool, ins, sp):
    sh = with_features(pool, surface_features(pool, ins))
    tr, ca = sh.subset(sp["train"]), sh.subset(sp["cal"])
    d = TwoHeadMUP(0, 0.01).fit(tr, ca).delta(ca)
    tau = tune_thresholds(d, ca)
    for lam in LAMBDAS:
        cu, ku, _ = walk(ca, (d > lam).astype(np.int64))
        ct, kt, _ = walk(ca, (d > tau[lam]).astype(np.int64))
        assert np.mean(ct - lam * kt) >= np.mean(cu - lam * ku) - 1e-12     # the untuned rule is always a candidate


def test_distilled_and_fused_fit_without_touching_gate_or_test(pool, ins, sp):
    prior = json.loads(RESULTS.read_text())["partition_0"]["fit_info"]["mup"]
    sh = with_features(pool, hidden_features(pool, prior["layer"]))
    tr, ca = sh.subset(sp["train"]), sh.subset(sp["cal"])
    conf = draft_features(pool)
    ds = DistilledTwoHead(prior["C"]).fit(tr, ca, conf[sp["train"]])
    d = ds.delta(sh.subset(sp["test"]))
    assert d.shape == (350, N_NODES) and np.isfinite(d).all()
    fm = FusedTwoHead(prior["C"]).fit(tr, ca, conf[sp["train"]], conf[sp["cal"]])
    d2 = fm.delta_with(sh.subset(sp["test"]), conf[sp["test"]])
    assert d2.shape == (350, N_NODES) and np.isfinite(d2).all()
