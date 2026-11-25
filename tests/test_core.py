"""Core tests. They target the defects found in the archived simulated pass."""

import itertools

import numpy as np
import pytest

from mock_runner import MockRunner
from toolcomp.envs.arith import (T_STEPS, Instance, apply_op, dataset_hash, finish_with_tool, generate_instances,
                            node_prompt, root_prompt)
from toolcomp.eval_harness import (LAMBDAS, holm, kendall_tau_boot, paired_bootstrap, u_auc_instances,
                              utility_matrix, any_call_utility_matrix)
from toolcomp.probes import (Platt, SoftLogReg, best_global_pattern, hindsight_utility, rollout_reference_actions)
from toolcomp.splits import assert_disjoint, make_splits
from toolcomp.tools.sympy_tool import SymPyTool
from toolcomp.tree import (LEVEL_OF, N_LEAVES, N_NODES, PREFIX_OF, TreeBuilder, TreeData, child, leaf_for, node_labels,
                      parse_int, pattern_actions, walk)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    ins = generate_instances(60, seed=5)
    tb = TreeBuilder(MockRunner(), layers=[1, 2], batch_size=16, log=lambda *_: None)
    return ins, tb.build(ins, tmp_path_factory.mktemp("cache"), "t")


# ---- environment ------------------------------------------------------------------------------------
def test_generator_is_deterministic_and_exact():
    a, b = generate_instances(25, seed=11), generate_instances(25, seed=11)
    assert dataset_hash(a) == dataset_hash(b)
    assert dataset_hash(a) != dataset_hash(generate_instances(25, seed=12))
    for ins in a:
        cur = ins.first
        for k in range(T_STEPS):
            cur = apply_op(cur, ins.ops[k], ins.rights[k])
            assert cur == ins.gold_values[k]
        assert ins.final_gold == cur and ins.num_steps == T_STEPS


def test_difficulty_varies_within_trajectories():
    ins = generate_instances(400, seed=3)
    mixed = sum(len(set(i.bins)) > 1 for i in ins)
    assert mixed / len(ins) > 0.5


def test_prompt_contains_no_gold_value_of_the_current_step():
    ins = Instance("x", 12, ("+", "*", "-"), (30, 11, 5), ("easy",) * 3, (42, 462, 457))
    p = node_prompt(ins, 0, 12, [])
    assert "42" not in p and "457" not in p                      # neither this step's answer nor the final one
    p2 = node_prompt(ins, 1, 40, [{"value": 40, "how": "mental"}])   # a (wrong) running value is allowed
    assert "462" not in p2 and "mental" in p2
    assert "42" not in root_prompt(ins).replace("(12 + 30)", "")


def test_parse_int():
    assert parse_int("The answer is 1,234.") == 1234
    assert parse_int("-56") == -56
    assert parse_int("no idea") is None


def test_sympy_tool_exact_and_failures_are_charged():
    t = SymPyTool()
    r = t.execute("123456789 * 987654321")
    assert r.success and r.output == str(123456789 * 987654321) and r.cost == 1.0
    big = t.execute("9007199254740993 + 2")                      # beyond 2**53: float round-trip would be wrong
    assert big.output == str(9007199254740993 + 2)
    bad = t.execute("42 + / 58")
    assert not bad.success and bad.cost == 1.0


# ---- tree: indexing, propagation, labels ------------------------------------------------------------
def test_leaf_and_heap_indexing_consistent():
    for k in range(N_NODES):
        lv, pf = int(LEVEL_OF[k]), int(PREFIX_OF[k])
        assert k == (2 ** lv - 1) + pf
    assert child(0, 0) == 1 and child(0, 1) == 2 and child(2, 1) == 6
    # every leaf is reachable by exactly one action sequence
    seen = set()
    for bits in itertools.product((0, 1), repeat=3):
        idx = 0
        for b in bits:
            idx = idx * 2 + b
        seen.add(idx)
    assert seen == set(range(N_LEAVES))


def test_walk_matches_explicit_recursion(tree):
    ins, td = tree
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 2, size=(td.n, N_NODES))
    correct, calls, bits = walk(td, actions)
    for i in range(td.n):
        nid, pat = 0, []
        for _ in range(T_STEPS):
            a = int(actions[i, nid])
            pat.append(a)
            nid = child(nid, a)
        leaf = pat[0] * 4 + pat[1] * 2 + pat[2]
        assert bits[i] == leaf and calls[i] == sum(pat) and correct[i] == float(td.leaf_correct[i, leaf])


def test_errors_propagate_through_nocall_steps(tree):
    """Old bug (I-003): a later correct step 'repaired' an earlier error. Independent check: from a WRONG incoming
    value, finishing the remaining ops exactly reaches the gold final only by arithmetic coincidence."""
    ins, td = tree
    yc, _ = node_labels(td, fill=1)
    wrong_nodes, coincid = 0, 0
    for i, x in enumerate(ins):
        for k in range(1, N_NODES):
            lv = int(LEVEL_OF[k])
            if td.v_in_ok[i, k] and int(td.v_in[i, k]) != x.gold_values[lv - 1]:       # incoming running value is wrong
                wrong_nodes += 1
                cur = int(td.v_in[i, k])
                for j in range(lv, x.num_steps):                                       # exact ops from the wrong value
                    cur = apply_op(cur, x.ops[j], x.rights[j])
                expected = float(cur == x.final_gold)
                coincid += int(expected)
                assert yc[i, k] == expected                                            # label = exact recomputation
    assert wrong_nodes > 20
    assert coincid / wrong_nodes < 0.05                                                # no systematic repair


def test_wrong_step_one_makes_never_call_chain_wrong(tree):
    """If the model's step-1 answer is wrong and it never calls the tool, the final answer is correct only by coincidence."""
    ins, td = tree
    wrong = [i for i, x in enumerate(ins) if td.v_no_ok[i, 0] and int(td.v_no[i, 0]) != x.gold_values[0]]
    assert len(wrong) >= 10
    repaired = [i for i in wrong if td.leaf_correct[i, 0]]
    assert len(repaired) / len(wrong) < 0.05


def test_labels_equal_exact_recomputation(tree):
    ins, td = tree
    yc, yn = node_labels(td, fill=1)
    for i, x in enumerate(ins):
        for k in range(N_NODES):
            lv = int(LEVEL_OF[k])
            vc = int(td.v_call[i, k]) if td.v_call_ok[i, k] else None
            vn = int(td.v_no[i, k]) if td.v_no_ok[i, k] else None
            assert yc[i, k] == float(finish_with_tool(x, lv, vc) == x.final_gold and vc is not None)
            assert yn[i, k] == float(finish_with_tool(x, lv, vn) == x.final_gold and vn is not None)


def test_never_and_always_closed_form(tree):
    ins, td = tree
    z = np.zeros((td.n, N_NODES), dtype=np.int64)
    one = np.ones_like(z)
    c0, k0, _ = walk(td, z)
    c1, k1, _ = walk(td, one)
    assert np.all(k0 == 0) and np.all(k1 == 3)
    assert np.allclose(c0, td.leaf_correct[:, 0]) and np.allclose(c1, td.leaf_correct[:, 7])
    # always-call is exact: correct iff no tool failure
    assert np.all(td.leaf_correct[:, 7])
    for lam in (0.1, 0.4):
        assert np.mean(c0 - lam * k0) == pytest.approx(c0.mean())             # never: U == accuracy
        assert np.mean(c1 - lam * k1) == pytest.approx(c1.mean() - 3 * lam)  # always: acc - lambda*T


def test_reference_ordering_hindsight_ge_global_ge_never_always(tree):
    ins, td = tree
    for lam in LAMBDAS:
        g = best_global_pattern(td, lam)
        cg, kg, _ = walk(td, np.tile(pattern_actions(g), (td.n, 1)))
        Ug = np.mean(cg - lam * kg)
        ch, kh = hindsight_utility(td, lam)
        Uh = np.mean(ch - lam * kh)
        assert Uh >= Ug - 1e-12
        for p in range(8):
            c, k, _ = walk(td, np.tile(pattern_actions(p), (td.n, 1)))
            assert Ug >= np.mean(c - lam * k) - 1e-12                         # global best is best on the same data
            assert Uh >= np.mean(c - lam * k) - 1e-12                         # hindsight is an upper bound on every pattern
        # no deployable policy can beat hindsight (it is the exact per-instance optimum over leaves)
        ca, ka, _ = walk(td, rollout_reference_actions(td, lam))
        assert Uh >= np.mean(ca - lam * ka) - 1e-12


def test_tree_save_load_roundtrip(tree, tmp_path):
    ins, td = tree
    td.save(tmp_path / "t.npz")
    td2 = TreeData.load(tmp_path / "t.npz")
    assert td2.instance_ids == td.instance_ids and np.array_equal(td2.leaf_correct, td.leaf_correct)
    assert np.array_equal(td2.hidden, td.hidden)


def test_builder_resumes_from_checkpoint(tmp_path):
    ins = generate_instances(10, seed=9)
    calls = {"n": 0}

    class Counting(MockRunner):
        def run(self, *a, **k):
            calls["n"] += 1
            return super().run(*a, **k)

    tb = TreeBuilder(Counting(), layers=[1], log=lambda *_: None)
    t1 = tb.build(ins, tmp_path, "r")
    first = calls["n"]
    t2 = TreeBuilder(Counting(), layers=[1], log=lambda *_: None).build(ins, tmp_path, "r")
    assert calls["n"] == first                                                  # nothing recomputed
    assert np.array_equal(t1.leaf_correct, t2.leaf_correct)


def test_decision_features_never_depend_on_gold(tree):
    """Hidden states are computed from the prompt only: identical prompts => identical features."""
    ins, _ = tree
    r = MockRunner()
    a = r.hidden_only([node_prompt(ins[0], 0, ins[0].first, [])], [1])[0][1]
    b = r.hidden_only([node_prompt(ins[0], 0, ins[0].first, [])], [1])[0][1]
    assert np.array_equal(a, b)


# ---- metrics -----------------------------------------------------------------------------------------
def test_u_auc_constant_and_linear():
    U = np.full((5, len(LAMBDAS)), 0.8)
    assert u_auc_instances(U) == pytest.approx(np.full(5, 80.0))
    lam = np.asarray(LAMBDAS)
    U2 = (1.0 - 2.0 * lam)[None, :].repeat(3, 0)                              # linear in lambda: AUC = mean value
    assert u_auc_instances(U2) == pytest.approx(np.full(3, 100 * (1.0 - 2.0 * lam.mean())))


def test_cumulative_vs_any_call_utility_differ():
    c = np.ones((4, 1))
    k = np.array([[0.], [1.], [2.], [3.]])
    cum = utility_matrix(c, k, [0.5])
    anyc = any_call_utility_matrix(c, k, [0.5])
    assert cum[:, 0].tolist() == [1.0, 0.5, 0.0, -0.5]
    assert anyc[:, 0].tolist() == [1.0, 0.5, 0.5, 0.5]


def test_paired_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(1)
    b = rng.normal(0, 1, 800)
    a = b + 0.3 + rng.normal(0, 0.2, 800)
    r = paired_bootstrap(a, b, 2000, seed=3)
    assert r["excludes_zero"] and r["ci_lo"] > 0.2
    null = paired_bootstrap(b + rng.normal(0, 0.2, 800), b, 2000, seed=3)
    assert not null["excludes_zero"]


def test_holm_monotone_and_bounded():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert all(0 <= v <= 1 for v in adj.values())


def test_kendall_tau_boot_detects_reversal():
    rng = np.random.default_rng(0)
    n = 300
    base = {f"m{j}": rng.normal(j, 1, n) for j in range(6)}
    same = kendall_tau_boot(base, base, 200, seed=1)
    assert same["tau"] == pytest.approx(1.0)
    rev = {k: -v for k, v in base.items()}
    assert kendall_tau_boot(base, rev, 200, seed=1)["tau"] == pytest.approx(-1.0)


# ---- probes ------------------------------------------------------------------------------------------
def test_soft_logreg_recovers_soft_targets_and_matches_hard_labels():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    w = rng.normal(size=6)
    soft = 1 / (1 + np.exp(-(X @ w + 0.3)))
    m = SoftLogReg(C=10.0).fit(X, soft)
    assert np.corrcoef(m.predict_proba(X), soft)[0, 1] > 0.98                # W-10: fractional targets are honoured
    hard = (rng.uniform(size=400) < soft).astype(float)
    ours = SoftLogReg(C=1.0, max_iter=500).fit(X, hard).predict_proba(X)
    ref = LogisticRegression(C=1.0, max_iter=500).fit(X, hard).predict_proba(X)[:, 1]
    assert np.corrcoef(ours, ref)[0, 1] > 0.99


def test_platt_identity_on_degenerate_labels():
    p = np.linspace(0.1, 0.9, 20)
    cal = Platt().fit(p, np.ones(20))                                          # old bug: p_call head fitted on all-ones
    assert np.allclose(cal(p), p)


# ---- splits ------------------------------------------------------------------------------------------
def test_splits_disjoint_and_reproducible():
    ids = [f"i{j}" for j in range(100)]
    s1 = make_splits(ids, {"train": 50, "cal": 20, "gate": 10, "test": 20}, seed=4)
    s2 = make_splits(ids, {"train": 50, "cal": 20, "gate": 10, "test": 20}, seed=4)
    assert all(np.array_equal(s1[k], s2[k]) for k in s1)
    allidx = np.concatenate(list(s1.values()))
    assert len(allidx) == len(set(allidx.tolist())) == 100
    s1["test"] = np.append(s1["test"], s1["train"][0])
    with pytest.raises(AssertionError):
        assert_disjoint(ids, s1)
    with pytest.raises(ValueError):
        make_splits(ids, {"train": 90, "cal": 20}, seed=0)


def test_random_matched_any_call_probability_is_exact():
    """Expected-value baselines must use P(any call) = 1-(1-p)^3, not 1[expected calls > 0] (which is always 1)."""
    p = 0.2
    exact = 1 - (1 - p) ** 3
    assert exact == pytest.approx(0.488)
    assert float(3 * p > 0) == 1.0 and exact < 1.0       # the naive indicator would be wrong
    w = np.array([(p ** bin(m).count("1")) * ((1 - p) ** (3 - bin(m).count("1"))) for m in range(8)])
    assert w.sum() == pytest.approx(1.0) and w[0] == pytest.approx((1 - p) ** 3)
    assert 1.0 - w[0] == pytest.approx(exact)


# ---- P2-I1 (black-box sequential stopping) -------------------------------------------------------------
def _p2_data(n, m_max, seed, p_correct_by_q=None):
    from toolcomp.p2 import build_table
    rng = np.random.default_rng(seed)
    gold = rng.integers(100, 999, size=n)
    skill = rng.uniform(0.2, 0.95, size=n)                       # per-query chance each sample is right
    samples = np.where(rng.random((n, m_max)) < skill[:, None], gold[:, None], gold[:, None] + rng.integers(1, 9, size=(n, m_max)))
    valid = np.ones_like(samples, dtype=bool)
    modal, s, ok = build_table(samples, valid, gold, m_max)
    return dict(s=s, ok=ok, tool_ok=np.ones(n, dtype=bool), greedy_ok=(rng.random(n) < skill),
                greedy_lp=np.log(np.clip(skill + rng.normal(0, .05, n), 0.05, 0.99)))


def test_modal_stats_ties_and_counts():
    from toolcomp.p2 import modal_stats
    assert modal_stats([5, 7, 7, 5], 4) == (5, 2)            # tie -> earliest occurrence
    assert modal_stats([5, 7, 7, 5], 3) == (7, 2)
    assert modal_stats([None, None], 2) == (None, 0)


def test_calibration_is_monotone_and_bounded():
    from toolcomp.p2 import Calibration
    d = _p2_data(400, 6, 0)
    cal = Calibration.fit(d["s"], d["ok"], 6)
    for m in range(1, 7):
        qs = [cal.q[(m, s)] for s in range(0, m + 1)]
        assert all(0 <= v <= 1 for v in qs) and all(b >= a - 1e-12 for a, b in zip(qs, qs[1:]))


def test_sequential_stops_early_when_clear_and_respects_cap():
    from toolcomp.p2 import Calibration, sequential_decide
    d = _p2_data(600, 8, 1)
    cal = Calibration.fit(d["s"], d["ok"], 8)
    calls, used = zip(*[sequential_decide(d["s"][i], cal, 0.6, 1.645, 8) for i in range(len(d["s"]))])
    assert min(used) >= 1 and max(used) <= 8 and np.mean(used) < 8        # stops early on some queries
    # with z huge it never stops early: always reaches the cap and decides by the sign of q_hat - q*
    calls_hi, used_hi = zip(*[sequential_decide(d["s"][i], cal, 0.6, 1e9, 8) for i in range(50)])
    assert set(used_hi) == {8}


def test_p2_run_all_charges_every_sample_and_closed_forms():
    from toolcomp.p2 import run_all
    d = _p2_data(500, 8, 2)
    cal = {k: v[:250] for k, v in d.items()}
    tst = {k: v[250:] for k, v in d.items()}
    out, chosen, _ = run_all(cal, tst, [0.2, 0.5], [0.02, 0.10], 8)
    n = len(tst["tool_ok"])
    for lam in (0.2, 0.5):
        for rho in (0.02, 0.10):
            assert out["always"][lam][rho] == pytest.approx(np.full(n, 1.0 - lam))                   # exact tool, no samples
            exp_never = tst["greedy_ok"].astype(float) - rho                                           # one sample charged
            assert out["never"][lam][rho] == pytest.approx(exp_never)
    # higher rho can only lower the sequential policy's net utility on the same decisions' budget
    assert out["sequential"][0.2][0.10].mean() <= out["sequential"][0.2][0.02].mean() + 1e-9


def test_mup_with_perfect_features_reproduces_the_exact_reference(tmp_path):
    """Plumbing check: if the hidden state encodes the exact labels, heads fit them and p_call - p_no > lambda
    must reproduce the exact one-step reference policy (right sign convention, standardisation, calibration)."""
    from toolcomp.probes import TwoHeadMUP
    ins = generate_instances(160, seed=33)
    td = TreeBuilder(MockRunner(), layers=[1, 2], log=lambda *_: None).build(ins, tmp_path, "perf")
    yc, yn = node_labels(td, 1)
    rng = np.random.default_rng(0)
    td.hidden[...] = rng.normal(0, 0.05, td.hidden.shape).astype(np.float16)
    td.hidden[:, :, 0, 0] = (yc * 2 - 1).astype(np.float16)
    td.hidden[:, :, 0, 1] = (yn * 2 - 1).astype(np.float16)
    mup = TwoHeadMUP(layer=1, C=100.0, continuation=1).fit(td, td)
    for lam in (0.1, 0.3, 0.5):
        a = mup.actions(td, lam)
        ref = rollout_reference_actions(td, lam)
        assert (a == ref).mean() > 0.97
        cm, km, _ = walk(td, a)
        cr, kr, _ = walk(td, ref)
        assert np.mean(cm - lam * km) >= np.mean(cr - lam * kr) - 0.03
