"""Evaluation harness: utilities, U-AUC, paired bootstrap, Holm correction, rank-reversal statistics.

Definitions:
  U(lambda)      = Acc - (lambda/N) * sum_i calls_i * c          (cumulative per-trajectory cost, c = 1)
  U_any(alpha)   = Acc - alpha * mean_i 1[calls_i > 0]           (anchor-style single indicator per response)
  U-AUC          = trapezoid mean of U(lambda) over the grid, x100 ("points")
Everything is computed per instance so paired bootstraps resample instances, the unit of independence.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import kendalltau

LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def utility_matrix(correct_by_lam: np.ndarray, calls_by_lam: np.ndarray, lambdas: Sequence[float],
                   cost: float = 1.0) -> np.ndarray:
    """Per-instance cumulative utility, shape (N, len(lambdas)); inputs are (N, len(lambdas))."""
    lam = np.asarray(lambdas)[None, :]
    return correct_by_lam - lam * cost * calls_by_lam


def any_call_utility_matrix(correct_by_lam: np.ndarray, calls_by_lam: np.ndarray,
                            alphas: Sequence[float]) -> np.ndarray:
    a = np.asarray(alphas)[None, :]
    return correct_by_lam - a * (calls_by_lam > 0).astype(float)


def u_auc_instances(U: np.ndarray, lambdas: Sequence[float] = LAMBDAS) -> np.ndarray:
    """Per-instance U-AUC in points: trapezoid mean over the lambda grid, times 100."""
    lam = np.asarray(lambdas, dtype=float)
    return trapezoid(U, lam, axis=1) / (lam[-1] - lam[0]) * 100.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000, ci: float = 0.95,
                     seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap over instances for mean(a - b). p-value is the two-sided bootstrap sign probability."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    block = 500
    for s in range(0, n_resamples, block):
        e = min(s + block, n_resamples)
        idx = rng.integers(0, n, size=(e - s, n))
        means[s:e] = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    obs = float(d.mean())
    p = 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0)))
    return {"diff": obs, "ci_lo": float(lo), "ci_hi": float(hi), "p": min(1.0, max(p, 1.0 / n_resamples)),
            "n": n, "excludes_zero": bool(lo > 0 or hi < 0)}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family = the dict)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adj[k] = running
    return adj


def kendall_tau_boot(score_a: Dict[str, np.ndarray], score_b: Dict[str, np.ndarray], n_resamples: int = 2000,
                     seed: int = 0) -> Dict[str, object]:
    """Rank agreement between two metrics over methods; bootstrap over instances.

    score_a[method], score_b[method] are per-instance metric values (same instance order).
    Returns point tau, 95% interval, share of resamples with tau == 1, and the pairwise order-flip probability.
    """
    methods = sorted(score_a)
    n = len(next(iter(score_a.values())))
    A = np.stack([score_a[m] for m in methods])       # (M, N)
    B = np.stack([score_b[m] for m in methods])
    tau0 = float(kendalltau(A.mean(1), B.mean(1)).statistic)
    rng = np.random.default_rng(seed)
    taus = np.empty(n_resamples)
    flips = np.zeros((len(methods), len(methods)))
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a, b = A[:, idx].mean(1), B[:, idx].mean(1)
        t = kendalltau(a, b).statistic
        taus[r] = 1.0 if np.isnan(t) else t
        flips += ((a[:, None] - a[None, :]) * (b[:, None] - b[None, :]) < 0)
    flips /= n_resamples
    iu = np.triu_indices(len(methods), 1)
    return {"methods": methods, "tau": tau0, "ci_lo": float(np.percentile(taus, 2.5)),
            "ci_hi": float(np.percentile(taus, 97.5)), "share_tau_eq_1": float(np.mean(taus >= 1 - 1e-12)),
            "mean_pair_flip_prob": float(flips[iu].mean()),
            "n_discordant_pairs_point": int(((A.mean(1)[:, None] - A.mean(1)[None, :]) *
                                              (B.mean(1)[:, None] - B.mean(1)[None, :]) < 0)[iu].sum()),
            "n_pairs": int(len(iu[0]))}


def mde_paired(sd: float, n: int, z_alpha: float = 1.959964, z_beta: float = 0.841621) -> float:
    """Minimum detectable paired difference at 80% power, two-sided alpha=0.05."""
    return (z_alpha + z_beta) * sd / np.sqrt(n)
