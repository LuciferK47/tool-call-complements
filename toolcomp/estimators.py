"""Iteration-2 estimators on arbitrary feature families (additive; reuses the frozen heads in src.probes).

All classes expose `delta(tree) -> (N,7)` (estimated marginal utility of calling) so one rule serves them all:
  untuned : call iff delta > lambda            (the principled plug-in rule MUP uses)
  tuned   : call iff delta > tau_lambda        (tau_lambda chosen on CAL to maximise true tree-walk utility, exactly as the
                                                baselines' ThresholdPolicy is tuned; removes the tuning asymmetry of Iteration 1)
Trees passed in are feature shims built with `src.features.with_features` (single layer id 0).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .eval_harness import LAMBDAS
from .probes import (C_GRID, Platt, RidgeReg, SoftLogReg, Standardizer, grouped_folds, node_features,
                     select_layer_and_C)
from .tree import N_NODES, TreeData, node_labels, walk


def pick_C(tr: TreeData, seed: int = 0) -> float:
    """C by instance-grouped 5-fold CV on TRAIN only (same criterion as Iteration 1: mean Brier over both heads)."""
    yc, yn = node_labels(tr, 1)
    return select_layer_and_C(tr, {"yc": yc, "yn": yn}, [0], seed)[1]


def pick_alpha(tr: TreeData, seed: int = 0, grid: Sequence[float] = (1.0, 10.0, 100.0, 1e3, 1e4)) -> float:
    yc, yn = node_labels(tr, 1)
    D = (yc - yn)
    X = node_features(tr, 0)
    best, bs = grid[0], np.inf
    for a in grid:
        err = []
        for f in grouped_folds(tr.n, 5, seed):
            va = np.zeros(tr.n, bool)
            va[f] = True
            st = Standardizer.fit(X[~va].reshape(-1, X.shape[-1]))
            m = RidgeReg(a).fit(st(X[~va].reshape(-1, X.shape[-1])), D[~va].reshape(-1))
            err.append(np.mean((m.predict(st(X[va].reshape(-1, X.shape[-1]))) - D[va].reshape(-1)) ** 2))
        if np.mean(err) < bs:
            best, bs = a, float(np.mean(err))
    return best


class SingleHeadDelta:
    """One ridge head regressing Delta = y_call - y_no directly (same labels as MUP, no probability differencing)."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, tr: TreeData, cal: Optional[TreeData] = None) -> "SingleHeadDelta":
        yc, yn = node_labels(tr, 1)
        X = node_features(tr, 0)
        self.st = Standardizer.fit(X.reshape(-1, X.shape[-1]))
        self.reg = RidgeReg(self.alpha).fit(self.st(X.reshape(-1, X.shape[-1])), (yc - yn).reshape(-1))
        return self

    def delta(self, tree: TreeData) -> np.ndarray:
        X = node_features(tree, 0)
        return self.reg.predict(self.st(X.reshape(-1, X.shape[-1]))).reshape(tree.n, N_NODES)


def _oof_logits(tr: TreeData, y: np.ndarray, C: float, seed: int) -> np.ndarray:
    """Out-of-fold (instance-grouped) head logits on TRAIN for target y (N,7)."""
    X = node_features(tr, 0)
    out = np.zeros((tr.n, N_NODES))
    for f in grouped_folds(tr.n, 5, seed):
        va = np.zeros(tr.n, bool)
        va[f] = True
        st = Standardizer.fit(X[~va].reshape(-1, X.shape[-1]))
        h = SoftLogReg(C).fit(st(X[~va].reshape(-1, X.shape[-1])), y[~va].reshape(-1))
        out[va] = (st(X[va].reshape(-1, X.shape[-1])) @ h.w + h.b).reshape(-1, N_NODES)
    return out


class FusedTwoHead:
    """Late fusion of a base feature family with draft confidence, identical protocol for any base (S0 or S1).

    Level 1: base two-head logits (out-of-fold on train, plain fit at test time).
    Level 2: per head, a logistic on [base logit, mean_logprob, entropy0] fitted on TRAIN out-of-fold logits; Platt on CAL.
    Needs the drafted answer (post-draft information)."""

    def __init__(self, C: float):
        self.C = C

    def fit(self, tr: TreeData, cal: TreeData, conf_tr: np.ndarray, conf_cal: np.ndarray, seed: int = 0) -> "FusedTwoHead":
        yc, yn = node_labels(tr, 1)
        X = node_features(tr, 0)
        self.st1 = Standardizer.fit(X.reshape(-1, X.shape[-1]))
        Xs = self.st1(X.reshape(-1, X.shape[-1]))
        self.base = [SoftLogReg(self.C).fit(Xs, y.reshape(-1)) for y in (yc, yn)]
        oof = [_oof_logits(tr, y, self.C, seed) for y in (yc, yn)]
        self.st2, self.l2, self.platt = [], [], []
        ycc, ycn = node_labels(cal, 1)
        for j, (y, yl) in enumerate(((yc, ycc), (yn, ycn))):
            Z = np.column_stack([oof[j].reshape(-1), conf_tr.reshape(-1, conf_tr.shape[-1])])
            s2 = Standardizer.fit(Z)
            h2 = SoftLogReg(1.0).fit(s2(Z), y.reshape(-1))
            self.st2.append(s2)
            self.l2.append(h2)
            Zc = self._z(cal, conf_cal, j)
            self.platt.append(Platt().fit(h2.predict_proba(s2(Zc)), yl.reshape(-1)))
        return self

    def _z(self, tree: TreeData, conf: np.ndarray, j: int) -> np.ndarray:
        X = node_features(tree, 0)
        lg = self.st1(X.reshape(-1, X.shape[-1])) @ self.base[j].w + self.base[j].b
        return np.column_stack([lg, conf.reshape(-1, conf.shape[-1])])

    def delta_with(self, tree: TreeData, conf: np.ndarray) -> np.ndarray:
        p = [self.platt[j](self.l2[j].predict_proba(self.st2[j](self._z(tree, conf, j)))).reshape(tree.n, N_NODES)
             for j in (0, 1)]
        return p[0] - p[1]


class DistilledTwoHead:
    """Pre-generation student distilled from a post-draft teacher (Future-Confidence-Distillation style).

    Teacher: two-head logistic on draft confidence [mean_logprob, entropy0], out-of-fold on train (so it is not trained on the
    node it labels). Student: two-head logistic on the hidden state predicting the teacher's SOFT probabilities. Platt on CAL with
    the true labels. At inference only the hidden state is read: no draft is generated."""

    def __init__(self, C: float):
        self.C = C

    def fit(self, tr: TreeData, cal: TreeData, conf_tr: np.ndarray, seed: int = 0) -> "DistilledTwoHead":
        yc, yn = node_labels(tr, 1)
        T = []
        for y in (yc, yn):
            t = np.zeros((tr.n, N_NODES))
            for f in grouped_folds(tr.n, 5, seed):
                va = np.zeros(tr.n, bool)
                va[f] = True
                s = Standardizer.fit(conf_tr[~va].reshape(-1, conf_tr.shape[-1]))
                h = SoftLogReg(1.0).fit(s(conf_tr[~va].reshape(-1, conf_tr.shape[-1])), y[~va].reshape(-1))
                t[va] = h.predict_proba(s(conf_tr[va].reshape(-1, conf_tr.shape[-1]))).reshape(-1, N_NODES)
            T.append(t)
        X = node_features(tr, 0)
        self.st = Standardizer.fit(X.reshape(-1, X.shape[-1]))
        Xs = self.st(X.reshape(-1, X.shape[-1]))
        self.heads = [SoftLogReg(self.C).fit(Xs, t.reshape(-1)) for t in T]
        Xc = self.st(node_features(cal, 0).reshape(-1, X.shape[-1]))
        ycc, ycn = node_labels(cal, 1)
        self.platt = [Platt().fit(h.predict_proba(Xc), yl.reshape(-1)) for h, yl in zip(self.heads, (ycc, ycn))]
        return self

    def delta(self, tree: TreeData) -> np.ndarray:
        X = self.st(node_features(tree, 0).reshape(-1, node_features(tree, 0).shape[-1]))
        p = [self.platt[j](self.heads[j].predict_proba(X)).reshape(tree.n, N_NODES) for j in (0, 1)]
        return p[0] - p[1]


def tune_thresholds(delta_cal: np.ndarray, cal: TreeData, lams: Sequence[float] = LAMBDAS,
                    cost: float = 1.0) -> Dict[float, float]:
    """tau_lambda = argmax over a grid of the true tree-walk utility on CAL (grid: 49 quantiles of delta_cal plus a fixed ladder)."""
    grid = np.unique(np.concatenate([np.quantile(delta_cal.reshape(-1), np.linspace(0.02, 0.98, 49)),
                                     np.linspace(-0.3, 1.0, 27)]))
    tau = {}
    for lam in lams:
        best, bu = float(lam), -np.inf
        for t in [float(lam)] + [float(g) for g in grid]:        # the untuned rule is always a candidate
            c, k, _ = walk(cal, (delta_cal > t).astype(np.int64))
            u = float(np.mean(c - lam * cost * k))
            if u > bu + 1e-12:
                best, bu = t, u
        tau[lam] = best
    return tau
