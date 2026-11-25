"""P2-I1: calibrated sequential stopping for black-box tool-call decisions (pure array logic, no model calls).

Black-box interface: the controller sees only sampled answers (text) and, for one baseline, the greedy answer's mean
log-probability. Per query it draws samples one at a time (up to m_max), maps (m, agreement count s) to a calibrated
probability that the modal answer is correct, and stops when that probability is clearly above or below the break-even
q* = p_tool - lambda*c. Every sample costs rho (in units of one tool call). Replaces the earlier Beta-posterior version
whose posterior treated agreement as Bernoulli correctness evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

Z_GRID = (0.0, 0.5, 1.0, 1.645, 2.5)
K_GRID = (1, 2, 3, 5, 8)


def modal_stats(samples: Sequence[Optional[int]], m: int) -> Tuple[Optional[int], int]:
    """Modal answer among the first m samples (ties broken by earliest occurrence) and its count."""
    c = Counter(x for x in samples[:m] if x is not None)
    if not c:
        return None, 0
    best = max(c.values())
    for x in samples[:m]:
        if x is not None and c[x] == best:
            return x, best
    return None, 0


def build_table(samples: np.ndarray, valid: np.ndarray, gold: np.ndarray, m_max: int):
    """For every query and every m: modal answer, agreement count s, correctness of the modal answer.
    samples (N, m_max) int; valid (N, m_max) bool; gold (N,)."""
    n = len(gold)
    modal = np.zeros((n, m_max), dtype=np.int64)
    ok = np.zeros((n, m_max), dtype=bool)
    s = np.zeros((n, m_max), dtype=np.int64)
    for i in range(n):
        row = [int(samples[i, j]) if valid[i, j] else None for j in range(m_max)]
        for m in range(1, m_max + 1):
            a, cnt = modal_stats(row, m)
            s[i, m - 1] = cnt
            ok[i, m - 1] = a is not None and a == gold[i]
            modal[i, m - 1] = a if a is not None else -10 ** 12
    return modal, s, ok


@dataclass
class Calibration:
    q: Dict[Tuple[int, int], float]
    se: Dict[Tuple[int, int], float]

    @staticmethod
    def fit(s: np.ndarray, ok: np.ndarray, m_max: int) -> "Calibration":
        q, se = {}, {}
        for m in range(1, m_max + 1):
            xs = np.arange(0, m + 1)
            k = np.array([ok[(s[:, m - 1] == a), m - 1].sum() for a in xs], dtype=float)
            n = np.array([(s[:, m - 1] == a).sum() for a in xs], dtype=float)
            raw = (k + 1.0) / (n + 2.0)                       # Laplace smoothing; empty cells fall back to 0.5
            iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0).fit(xs, raw, sample_weight=n + 2.0)
            sm = iso.predict(xs)
            for a, v, nn in zip(xs, sm, n):
                q[(m, int(a))] = float(v)
                se[(m, int(a))] = float(np.sqrt(max(v * (1 - v), 1e-4) / (nn + 2.0)))
        return Calibration(q, se)


def sequential_decide(s_row: np.ndarray, cal: Calibration, q_star: float, z: float, m_max: int) -> Tuple[int, int]:
    """Return (call(1)/nocall(0), samples used). At the cap decide by the sign of q_hat - q*."""
    for m in range(1, m_max + 1):
        key = (m, int(s_row[m - 1]))
        qh, se = cal.q[key], cal.se[key]
        if abs(qh - q_star) > z * se or m == m_max:
            return int(qh < q_star), m
    raise AssertionError("unreachable")


def evaluate(calls: np.ndarray, used: np.ndarray, direct_ok: np.ndarray, tool_ok: np.ndarray, lam: float, rho: float):
    """Per-query accuracy and net utility. NOCALL answers with the modal answer of the samples used."""
    correct = np.where(calls == 1, tool_ok, direct_ok).astype(float)
    return correct, correct - lam * calls - rho * used


def run_all(cal: dict, test: dict, lambdas: Sequence[float], rhos: Sequence[float], m_max: int):
    """cal/test dicts: s (N,m_max), ok (N,m_max) modal-correct flags, tool_ok (N,), greedy_ok (N,), greedy_lp (N,).

    Tunes z, k and the log-probability threshold on CAL per (lambda, rho) by net utility; evaluates on TEST.
    Returns per-query net utilities out[method][lambda][rho] -> (N_test,), the tuned parameters, and the calibration.
    """
    calib = Calibration.fit(cal["s"], cal["ok"], m_max)
    p_tool = float(cal["tool_ok"].mean())
    lp_grid = np.unique(np.quantile(cal["greedy_lp"], np.linspace(0.02, 0.98, 49)))
    methods = ("sequential", "fixed_k", "logprob_thr", "never", "always")
    out: Dict[str, Dict[float, Dict[float, np.ndarray]]] = {m: {} for m in methods}
    chosen: Dict[str, dict] = {"z": {}, "k": {}, "tau": {}}

    def seq_run(d, z, q_star):
        res = [sequential_decide(d["s"][i], calib, q_star, z, m_max) for i in range(len(d["tool_ok"]))]
        calls = np.array([r[0] for r in res])
        used = np.array([r[1] for r in res])
        direct = np.array([d["ok"][i, used[i] - 1] for i in range(len(used))])
        return calls, used, direct

    def fixed_run(d, k, q_star):
        qh = np.array([calib.q[(k, int(d["s"][i, k - 1]))] for i in range(len(d["tool_ok"]))])
        calls = (qh < q_star).astype(int)
        return calls, np.full(len(calls), k), np.array([d["ok"][i, k - 1] for i in range(len(calls))])

    def lp_run(d, tau, q_star):
        calls = (d["greedy_lp"] < tau).astype(int)
        return calls, np.ones(len(calls), dtype=int), d["greedy_ok"]

    def net(d, runner, param, q_star, lam, rho):
        calls, used, direct = runner(d, param, q_star)
        return evaluate(calls, used, direct, d["tool_ok"], lam, rho)[1]

    for lam in lambdas:
        q_star = p_tool - lam
        for rho in rhos:
            key = f"{lam:.2f}_{rho:.2f}"
            for name, runner, grid, slot in (("sequential", seq_run, Z_GRID, "z"), ("fixed_k", fixed_run, K_GRID, "k"),
                                              ("logprob_thr", lp_run, lp_grid, "tau")):
                best = max(grid, key=lambda g: net(cal, runner, g, q_star, lam, rho).mean())
                chosen[slot][key] = float(best) if slot == "tau" else best
                out[name].setdefault(lam, {})[rho] = net(test, runner, best, q_star, lam, rho)
            n = len(test["tool_ok"])
            out["never"].setdefault(lam, {})[rho] = evaluate(np.zeros(n, int), np.ones(n, int), test["greedy_ok"], test["tool_ok"], lam, rho)[1]
            out["always"].setdefault(lam, {})[rho] = evaluate(np.ones(n, int), np.zeros(n, int), test["greedy_ok"], test["tool_ok"], lam, rho)[1]
    return out, chosen, calib


def mean_samples(test: dict, calib: Calibration, p_tool: float, lam: float, z: float, m_max: int) -> float:
    q_star = p_tool - lam
    return float(np.mean([sequential_decide(test["s"][i], calib, q_star, z, m_max)[1] for i in range(len(test["tool_ok"]))]))
