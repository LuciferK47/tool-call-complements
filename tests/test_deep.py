"""Plumbing and correctness checks for the deep-tree builder and the scaling loaders (mock runner, no model)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from mock_runner import MockRunner

from toolcomp import scaling
from toolcomp.deep_tree import BM25, ArithEnv, MusiqueEnv, build_deep, child, load_deep, match, save_deep


def _build(tmp_path, K=4, n=60):
    env = ArithEnv(K, n, 7)
    return env, build_deep(env, MockRunner(), tmp_path / "c", "t", max_new=8, batch=16, log=lambda *_: None)


def test_arith_tree_invariants(tmp_path):
    env, res = _build(tmp_path)
    a = res["arrays"]
    assert a["leaf_correct"].shape == (60, 16) and a["local_no_correct"].shape == (60, 15)
    assert a["leaf_correct"][:, -1].all(), "calling every step with an exact tool must always be right"
    # a pattern with no call is right iff every unaided step along it is right
    for i in range(60):
        ok, nid = True, 0
        for lv in range(4):
            ok &= bool(a["local_no_correct"][i, nid])
            nid = child(nid, 0)
        assert ok == bool(a["leaf_correct"][i, 0])


def test_deep_roundtrip_and_scaling(tmp_path):
    env, res = _build(tmp_path, K=4, n=120)
    save_deep(res, tmp_path / "x.npz")
    arrays, side = load_deep(tmp_path / "x.npz")
    assert side["K"] == 4 and arrays["leaf_correct"].shape[1] == 16
    run = scaling.run_from_deep("mock", "mock", tmp_path / "x.npz")
    assert run.feats["S0"].shape[:2] == (120, 15)
    p = scaling.step_accuracy(run)
    assert len(p) == 4 and all(0 <= q <= 1 for q in p)
    out = scaling.evaluate_run(run, [11, 12, 13], (1, 2), reps=5, signals=["S0", "S4"])
    r = out["partition_0"]["S0"]["m"]["2"]["rules"]
    assert set(r) >= {"belief_dp", "optimistic", "pessimistic", "thr_retuned", "hindsight"}
    assert r["hindsight"]["u_auc"] >= max(v["u_auc"] for k, v in r.items() if k != "hindsight") - 1e-9


def test_musique_substitution_and_retrieval():
    env = object.__new__(MusiqueEnv)
    env.K, env._bm = 2, {}
    it = {"id": "q", "steps": ["Green >> performer", "#1 >> spouse"], "gold": ["Steve Hillage", "Miquette Giraudy"], "aliases": [],
          "paragraphs": [("Green", "Green is an album by Steve Hillage."), ("Hillage", "Steve Hillage is married to Miquette Giraudy."), ("Other", "Unrelated text about cats.")]}
    hist = [{"value": "Steve Hillage", "how": "model"}]
    assert "Steve Hillage >> spouse" in env.prompt(it, 1, hist)
    cp = env.call_prompt(it, 1, hist)
    assert "Miquette Giraudy" in cp and cp.index("Hillage") < cp.index("cats")
    assert env.local_ok(it, 1, hist, "Miquette Giraudy") and not env.local_ok(it, 1, hist, "Someone Else")
    assert match("the Steve Hillage", ["Steve Hillage"]) and not match("Steve", ["Steve Hillage"])
    assert BM25(["a b c", "x y z"]).top("b", 1) == [0]
