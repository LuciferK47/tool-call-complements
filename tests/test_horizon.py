"""Theory and simulator checks for toolcomp/horizon.py (no model, no cache needed except the last test)."""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcomp.horizon import GRID, belief_dp, best_set, rule_dp, set_value, simulate


def test_complement_second_difference():
    """Prop 1: second difference of V equals prod(rest)(1-p_i)(1-p_j) >= 0."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(0.05, 0.99, 6)
        S = list(rng.choice(6, 2, replace=False))
        i, j = [k for k in range(6) if k not in S][:2]
        d = set_value(p, S + [i, j], .1) - set_value(p, S + [i], .1) - set_value(p, S + [j], .1) + set_value(p, S, .1)
        rest = np.prod([p[k] for k in range(6) if k not in S + [i, j]])
        assert abs(d - rest * (1 - p[i]) * (1 - p[j])) < 1e-12 and d >= -1e-12


def test_best_set_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(60):
        T = int(rng.integers(2, 8))
        p = rng.uniform(0.05, 0.99, T)
        c = float(rng.uniform(0.01, 0.5))
        brute = max(set_value(p, S, c) for k in range(T + 1) for S in itertools.combinations(range(T), k))
        assert abs(best_set(p, c)[1] - brute) < 1e-12


def test_homogeneous_chain_is_all_or_nothing():
    """V(k) is convex in k for equal p, so the optimum is k in {0, T}; threshold (1-p^T)/T."""
    for p in (0.3, 0.6, 0.9):
        for T in (3, 6, 12):
            c_star = (1 - p ** T) / T
            for c in (0.5 * c_star, 1.5 * c_star):
                S, _ = best_set(np.full(T, p), c)
                assert len(S) == (T if c < c_star else 0)


def test_belief_dp_matches_expectimax():
    """Finite-support signal laws: the grid DP equals exact expectimax within interpolation error."""
    laws = [np.array([0.3, 0.9]), np.array([0.5, 0.95]), np.array([0.6, 0.99])]
    for price in (0.05, 0.15, 0.3):
        for T in (3, 6):
            def ev(j, b):
                if j == T:
                    return b
                p = laws[j % 3]
                return float(np.mean([max(ev(j + 1, b) - price, ev(j + 1, b * q)) for q in p]))
            W = belief_dp(laws, T, price)
            for b in (0.0, 0.2, 0.5, 1.0):
                assert abs(np.interp(b, GRID, W[0]) - ev(0, b)) < 4e-3


def test_belief_dp_shape():
    laws = [np.linspace(0.2, 0.99, 20)] * 3
    for T, price in ((3, 0.2), (12, 0.05)):
        W = belief_dp(laws, T, price)
        for j in range(T + 1):
            assert abs(W[j, 0]) < 1e-12
            assert (np.diff(W[j]) >= -1e-12).all()                                   # non-decreasing
            assert (np.diff(W[j], 2) >= -1e-6).all()                                  # convex
            assert (np.diff(W[j]) <= (GRID[1] - GRID[0]) + 1e-12).all()               # slope <= 1


def test_call_requires_continuation_value_above_price():
    """Abandonment: where W_{j+1}(b) <= price the DP never calls."""
    laws = [np.linspace(0.2, 0.99, 20)] * 3
    W = belief_dp(laws, 12, 0.2)
    f = rule_dp(W, 0.2)
    for j in range(12):
        for b in (0.01, 0.05, 0.1):
            if np.interp(b, GRID, W[j + 1]) <= 0.2:
                assert not f(j, np.array([b]), np.array([0.5]), None)[0]


def test_simulator_counts_calls_and_multiplies_parts():
    leaf = np.zeros((2, 8), bool)
    leaf[0, 7] = True            # instance 0 right only if everything is called
    leaf[1, :] = True            # instance 1 always right
    P = np.full((2, 7), 0.5)
    ok, c = simulate(P, leaf, np.array([[0, 1]]), lambda j, b, p, s: np.ones(len(p), bool))
    assert ok[0] == 1 and c[0] == 6
    ok, c = simulate(P, leaf, np.array([[0, 1]]), lambda j, b, p, s: np.zeros(len(p), bool))
    assert ok[0] == 0 and c[0] == 0


def test_bracket_and_abandonment():
    """Prop 3: pessimistic gain <= W(b)-W(bp) <= optimistic gain; and b <= c implies no call at any later step."""
    laws = [np.linspace(0.2, 0.99, 25), np.linspace(0.4, 0.97, 25), np.linspace(0.3, 0.999, 25)]
    for T, c in ((6, 0.1), (12, 0.05), (24, 0.0375)):
        W = belief_dp(laws, T, c)
        for j in range(T):
            tail = np.prod([laws[k % 3].mean() for k in range(j + 1, T)])
            for b in (0.05, 0.3, 0.7, 1.0):
                for p in (0.25, 0.6, 0.95):
                    g = np.interp(b, GRID, W[j + 1]) - np.interp(b * p, GRID, W[j + 1])
                    assert tail * b * (1 - p) - 2e-3 <= g <= b * (1 - p) + 2e-3
            f = rule_dp(W, c)
            for b in (c, 0.5 * c):
                for p in (0.01, 0.5):
                    assert not f(j, np.array([b]), np.array([p]), None)[0]


# ---------------------------------------------------------------- noisy tool and scaling-limit statements used in the paper
def _v_noisy(p, q, S, c):
    return float(np.prod([q[i] if i in S else p[i] for i in range(len(p))]) - c * len(S))


def test_noisy_tool_second_difference():
    """Tool accuracy q >= p: second difference of V is R (q_i - p_i)(q_j - p_j) >= 0, R = product over the other steps of q (called) or p (not called)."""
    rng = np.random.default_rng(3)
    for _ in range(60):
        p = rng.uniform(0.05, 0.9, 6)
        q = np.minimum(1.0, p + rng.uniform(0, 0.5, 6))
        S = set(int(k) for k in rng.choice(6, 2, replace=False))
        i, j = [k for k in range(6) if k not in S][:2]
        d = _v_noisy(p, q, S | {i, j}, .1) - _v_noisy(p, q, S | {i}, .1) - _v_noisy(p, q, S | {j}, .1) + _v_noisy(p, q, S, .1)
        R = np.prod([q[k] if k in S else p[k] for k in range(6) if k not in {i, j}])
        assert abs(d - R * (q[i] - p[i]) * (q[j] - p[j])) < 1e-12 and d >= -1e-12


def test_noisy_homogeneous_threshold():
    from toolcomp.horizon import best_set
    for p, q in ((0.6, 0.9), (0.8, 0.95)):
        for T in (3, 6):
            cstar = (q ** T - p ** T) / T
            for c in (0.7 * cstar, 1.3 * cstar):
                v = [(q ** k) * (p ** (T - k)) - c * k for k in range(T + 1)]
                assert int(np.argmax(v)) == (T if c < cstar else 0)


def test_scaling_limit_of_per_step_regret():
    """With x = T(1-p) and total price C = Tc fixed, sup regret of the optimistic / pessimistic rule -> x-1+e^-x and 1-(1+x)e^-x."""
    from toolcomp.horizon import rule_expected_utility
    for x in (0.2, 0.5, 1.0, 2.0):
        T = 4000
        p = 1 - x / T
        best_o = best_p = 0.0
        for C in np.linspace(0.01, 3.0, 600):
            c = C / T
            opt = rule_expected_utility("optimal_open_loop", [p], T, c)
            best_o = max(best_o, opt - rule_expected_utility("optimistic", [p], T, c))
            best_p = max(best_p, opt - rule_expected_utility("pessimistic", [p], T, c))
        assert abs(best_o - (x - 1 + np.exp(-x))) < 0.01, (x, best_o)
        assert abs(best_p - (1 - (1 + x) * np.exp(-x))) < 0.01, (x, best_p)


def test_signal_aware_equals_closed_form_for_constant_signal():
    """With a constant (uninformative) signal the signal-aware prediction must reduce to the closed-form regret of the best per-step rule."""
    from toolcomp.horizon import predicted_regret, predicted_regret_signal
    for p, m, scale in ((0.7, 2, 1.0), (0.9, 4, 1.0), (0.6, 1, 0.5)):
        samples = [np.full(60, p)] * 3
        closed = min(predicted_regret([p] * 3, m, price_scale=scale).values())
        assert abs(predicted_regret_signal(samples, m, scale, G=50) - closed) < 0.2, (p, m, scale)


def test_signal_aware_is_zero_for_a_perfect_signal():
    from toolcomp.horizon import predicted_regret_signal
    samples = [np.concatenate([np.full(30, 0.02), np.full(30, 0.999)])] * 3
    assert predicted_regret_signal(samples, 1, G=2000) < 2.0


# ---------------------------------------------------------------- noisy-tool Belief-DP (call makes the step right with probability q, not 1)
def test_noisy_belief_dp_matches_expectimax():
    """Finite-support laws and per-level tool accuracies: the grid DP equals exact expectimax with the call branch W(b q)."""
    laws = [np.array([0.3, 0.9]), np.array([0.5, 0.95]), np.array([0.6, 0.99])]
    q = [0.8, 0.6, 0.9]
    for price in (0.02, 0.1, 0.25):
        for T in (3, 6):
            def ev(j, b):
                if j == T:
                    return b
                return float(np.mean([max(ev(j + 1, b * q[j % 3]) - price, ev(j + 1, b * p)) for p in laws[j % 3]]))
            W = belief_dp(laws, T, price, q)
            for b in (0.0, 0.3, 0.7, 1.0):
                assert abs(np.interp(b, GRID, W[0]) - ev(0, b)) < 4e-3


def test_noisy_belief_dp_with_exact_tool_is_the_exact_dp():
    laws = [np.array([0.3, 0.9]), np.array([0.5, 0.95]), np.array([0.6, 0.99])]
    for price in (0.05, 0.2):
        assert np.array_equal(belief_dp(laws, 9, price, [1.0, 1.0, 1.0]), belief_dp(laws, 9, price))
        assert np.array_equal(belief_dp(laws, 9, price, None), belief_dp(laws, 9, price))


def test_simulate_multiplies_belief_by_tool_accuracy_on_a_call():
    """A rule that records the belief it is shown: after a call the belief is b q_j, after an unaided step b p."""
    P = np.full((4, 3), 0.5)
    leaf = np.ones((4, 4), bool)
    tuples = np.arange(4).reshape(4, 1)
    seen = []

    def rule(j, b, p, s):
        seen.append(b.copy())
        return np.full(len(p), j == 0)               # call at step 0 only

    simulate(P, leaf, tuples, rule, q=[0.7, 0.9])
    assert np.allclose(seen[0], 1.0) and np.allclose(seen[1], 0.7) and np.allclose(seen[1] * 0.5, 0.35)
    seen.clear()
    simulate(P, leaf, tuples, rule)                  # exact tool: a call leaves the belief unchanged
    assert np.allclose(seen[1], 1.0)
