"""Complementarity of tool calls in conjunctive (all-steps-must-be-right) tasks, and the belief-DP rule that accounts for it.

Task: T steps. Step j answered without the tool is right with probability p_j; with the tool it is right (exact tool). The task
pays 1 iff every step is right; each call costs `price`. b = belief that everything so far is right.

  W_T(b) = b
  W_j(b) = E_p[ max( W_{j+1}(b) - price ,  W_{j+1}(b p) ) ]          p ~ empirical law of the calibrated signal at step j
  call at step j iff  W_{j+1}(b) - price > W_{j+1}(b p_j)

W_j is convex, non-decreasing, W_j(0) = 0 and slope <= 1. Rules that fix the continuation are special cases with a LINEAR W:
always-call afterwards: W(b) = b - const (slope 1); never-call afterwards: W(b) = b * prod(future p).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from .features import _digits
from .probes import Platt, SoftLogReg, Standardizer, grouped_folds
from .tree import LEVEL_OF, N_NODES, T_STEPS, TreeData

GRID = np.linspace(0.0, 1.0, 401)


# ---------------------------------------------------------------- theory helpers (homogeneous chain, calibrated p)
def set_value(p: np.ndarray, S: Sequence[int], price: float) -> float:
    """V(S) = prod_{i not in S} p_i - price |S| (exact tool)."""
    mask = np.ones(len(p), bool)
    mask[list(S)] = False
    return float(np.prod(p[mask]) - price * len(S))


def best_set(p: np.ndarray, price: float):
    """Optimal open-loop call set: for a given size, call the steps with the smallest p; scan sizes. O(T log T)."""
    order = np.argsort(p)
    best, bv = (), float(np.prod(p))
    for k in range(1, len(p) + 1):
        v = float(np.prod(p[order[k:]]) - price * k)
        if v > bv + 1e-12:
            best, bv = tuple(int(i) for i in order[:k]), v
    return best, bv


# ---------------------------------------------------------------- calibrated step-correctness signals
def surface_nobin(tree: TreeData, instances) -> np.ndarray:
    """Operator, operand digit counts, node id. (Iteration-2 S0 without the generator's bin label, which is not printed in the prompt.)"""
    out = []
    for k in range(N_NODES):
        lv = int(LEVEL_OF[k])
        op = np.array([i.ops[lv] for i in instances])
        right = np.array([i.rights[lv] for i in instances], float)
        left = np.array([i.first for i in instances], float) if lv == 0 else tree.v_in[:, k].astype(float)
        dl, dr, mul = _digits(left), _digits(right), (op == "*").astype(float)
        one = np.zeros((tree.n, N_NODES))
        one[:, k] = 1.0
        out.append(np.column_stack([op == "+", op == "-", op == "*", dl, dr, dl * dr * mul,
                                    np.log10(np.maximum(np.abs(left), 1.0)) * mul, one]).astype(float))
    return np.stack(out, 1)


class StepSignal:
    """Calibrated P(step answered correctly without the tool | features at the node). Fit on TRAIN, Platt on CAL."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, Xtr, ytr, Xca, yca):
        d = Xtr.shape[-1]
        self.st = Standardizer.fit(Xtr.reshape(-1, d))
        self.h = SoftLogReg(self.C).fit(self.st(Xtr.reshape(-1, d)), ytr.reshape(-1).astype(float))
        self.platt = Platt().fit(self.h.predict_proba(self.st(Xca.reshape(-1, d))), yca.reshape(-1).astype(float))
        return self

    def __call__(self, X):
        d = X.shape[-1]
        return np.clip(self.platt(self.h.predict_proba(self.st(X.reshape(-1, d)))).reshape(X.shape[:2]), 1e-4, 1 - 1e-4)


def pava(y: np.ndarray, w: np.ndarray):
    """Pool-adjacent-violators: non-decreasing fit. Returns fitted values and block weights per point."""
    vals, wts, lens = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
            wts[-1] += w2
            lens[-1] += l2
    return np.repeat(vals, lens), np.repeat(wts, lens)


class IsoSignal:
    """Calibrated P(step right) with isotonic calibration of out-of-fold scores (grouped by instance) on TRAIN u CAL.

    Block means are shrunk as (k+1)/(n+2) so the top block is never exactly 1 (a p of 1 would mean 'never verify')."""

    def __init__(self, C: float = 1.0, folds: int = 5, seed: int = 0):
        self.C, self.k, self.seed = C, folds, seed

    def fit(self, X, y):
        n, K, d = X.shape
        y = y.astype(float)
        oof = np.zeros((n, K))
        for f in grouped_folds(n, self.k, self.seed):
            va = np.zeros(n, bool)
            va[f] = True
            st = Standardizer.fit(X[~va].reshape(-1, d))
            h = SoftLogReg(self.C).fit(st(X[~va].reshape(-1, d)), y[~va].reshape(-1))
            oof[va] = (st(X[va].reshape(-1, d)) @ h.w + h.b).reshape(-1, K)
        self.oof = oof
        s, yy = oof.reshape(-1), y.reshape(-1)
        o = np.argsort(s, kind="stable")
        fit, wt = pava(yy[o], np.ones(len(o)))
        fit = (fit * wt + 1.0) / (wt + 2.0)
        fit = np.maximum.accumulate(fit)
        self.xs, self.ys = s[o], fit
        self.st = Standardizer.fit(X.reshape(-1, d))
        self.h = SoftLogReg(self.C).fit(self.st(X.reshape(-1, d)), yy)
        return self

    def raw(self, X):
        d = X.shape[-1]
        return (self.st(X.reshape(-1, d)) @ self.h.w + self.h.b).reshape(X.shape[:2])

    def cal(self, s):
        return np.clip(np.interp(s, self.xs, self.ys), 1e-4, 1 - 1e-4)

    def __call__(self, X):
        return self.cal(self.raw(X))


# ---------------------------------------------------------------- belief DP
def level_samples(P: np.ndarray, n_q: int = 60) -> List[np.ndarray]:
    """Empirical law of the signal at each within-segment level (quantile-thinned)."""
    out = []
    for lv in range(T_STEPS):
        s = P[:, [k for k in range(N_NODES) if LEVEL_OF[k] == lv]].reshape(-1)
        out.append(np.quantile(s, (np.arange(n_q) + 0.5) / n_q))
    return out


def belief_dp(samples: List[np.ndarray], T: int, price: float) -> np.ndarray:
    """W[j, g] on GRID for j = 0..T. samples[l] is the law of p at within-segment level l."""
    W = np.zeros((T + 1, len(GRID)))
    W[T] = GRID
    for j in range(T - 1, -1, -1):
        p = samples[j % T_STEPS]
        nxt = W[j + 1]
        call = nxt - price
        no = np.interp(GRID[:, None] * p[None, :], GRID, nxt)
        W[j] = np.maximum(call[:, None], no).mean(1)
    return W


# ---------------------------------------------------------------- simulation on exact trees
def simulate(P: np.ndarray, leaf: np.ndarray, tuples: np.ndarray, rule: Callable, score: np.ndarray = None):
    """Run a sequential rule on composite tasks. tuples (G, m) index instances. rule(j, b, p, s) -> bool array (call?).
    Returns success (G,), calls (G,)."""
    G, m = tuples.shape
    b = np.ones(G)
    calls = np.zeros(G)
    ok = np.ones(G, bool)
    j = 0
    for s in range(m):
        inst = tuples[:, s]
        nid = np.zeros(G, np.int64)
        bits = np.zeros(G, np.int64)
        for _ in range(T_STEPS):
            p = P[inst, nid]
            sc = p if score is None else score[inst, nid]
            a = rule(j, b, p, sc).astype(np.int64)
            calls += a
            b = np.where(a == 1, b, b * p)
            bits = bits * 2 + a
            nid = 2 * nid + 1 + a
            j += 1
        ok &= leaf[inst, bits]
    return ok.astype(float), calls


def rule_local(price):            # per-step expected utility (MICE / R2V style)
    return lambda j, b, p, s: (1 - p) > price


def rule_optimistic(price):       # marginal utility, always-call continuation (what MUP's fill=1 labels encode)
    return lambda j, b, p, s: b * (1 - p) > price


def rule_pessimistic(price, means, T):   # marginal utility, never-call continuation
    tail = np.ones(T + 1)
    for j in range(T - 1, -1, -1):
        tail[j] = tail[j + 1] * means[j % T_STEPS]
    return lambda j, b, p, s: b * (1 - p) * tail[j + 1] > price


def rule_threshold(tau):          # tuned score threshold (Anchor-lite)
    return lambda j, b, p, s: s < tau


def rule_dp(W, price):
    def f(j, b, p, s):
        nx = W[j + 1]
        return np.interp(b, GRID, nx) - price > np.interp(b * p, GRID, nx) + 1e-12
    return f


def rule_all_or_nothing(price, T):      # trivial reference: call everything if it pays, otherwise never call
    return lambda j, b, p, s: np.full(len(p), 1.0 - T * price > 0)


def make_tuples(idx: np.ndarray, m: int, rng: np.random.Generator, reps: int = 1) -> np.ndarray:
    out = []
    for _ in range(reps):
        q = rng.permutation(idx)
        g = len(q) // m
        out.append(q[:g * m].reshape(g, m))
    return np.concatenate(out)


def min_calls(leaf: np.ndarray) -> np.ndarray:
    pops = np.array([bin(q).count("1") for q in range(leaf.shape[1])])
    return np.where(leaf, pops[None, :], 99).min(1)


def hindsight(leaf: np.ndarray, tuples: np.ndarray, price: float):
    k = min_calls(leaf)[tuples].sum(1)
    take = 1 - price * k > 0
    return take.astype(float), np.where(take, k, 0).astype(float)
