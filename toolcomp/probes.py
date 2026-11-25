"""Probes and probe-based policies on cached hidden states (A-07, A-11, A-12, A-28, A-34).

All heads are L2-regularised logistic regressions fit with L-BFGS on standardised features and accept soft targets
(with our deterministic tree the labels are exactly 0/1; the soft-target capability is kept and unit-tested).
Hyper-parameters (layer, C) are chosen by instance-grouped 5-fold CV on the TRAIN split only. Calibration (Platt)
and thresholds are fitted on the CAL split only. Nothing here ever sees the gate or test splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import os
import numpy as np
from scipy.optimize import minimize

from .tree import N_NODES, PREFIX_OF, TreeData, node_labels, pattern_actions, walk
from .eval_harness import LAMBDAS


# ---- building blocks -------------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class SoftLogReg:
    """min_w  sum BCE(sigmoid(Xw+b), y) + (1/(2C)) ||w||^2 ;  y may be fractional."""

    def __init__(self, C: float = 1.0, max_iter: int = 200):
        self.C, self.max_iter = C, max_iter
        self.w: Optional[np.ndarray] = None
        self.b = 0.0

    def _fg(self, theta, X, y):
        w, b = theta[:-1], theta[-1]
        p = sigmoid(X @ w + b)
        eps = 1e-12
        loss = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + 0.5 / self.C * np.dot(w, w)
        err = p - y
        return float(loss), np.append(X.T @ err + w / self.C, err.sum())

    SOLVER = os.environ.get("T4_PROBE_SOLVER", "scipy")     # "scipy" (frozen, float64, CPU) or "torch" (float32 L-BFGS on the GPU)

    def _fit_torch(self, X: np.ndarray, y: np.ndarray) -> "SoftLogReg":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("probe solver 'torch' requested but CUDA is not available (no silent fallback)")
        Xt = torch.as_tensor(X, dtype=torch.float32, device="cuda")
        yt = torch.as_tensor(y, dtype=torch.float32, device="cuda")
        w = torch.zeros(X.shape[1], dtype=torch.float32, device="cuda", requires_grad=True)
        b = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=self.max_iter, tolerance_grad=1e-5, line_search_fn="strong_wolfe")
        inv_C = 0.5 / float(self.C)

        def closure():
            opt.zero_grad()
            pr = torch.sigmoid(Xt @ w + b)
            eps = 1e-12
            loss = -torch.sum(yt * torch.log(pr + eps) + (1.0 - yt) * torch.log(1.0 - pr + eps)) + inv_C * torch.sum(w * w)
            loss.backward()
            return loss
        opt.step(closure)
        self.w, self.b = w.detach().cpu().numpy().astype(np.float64), float(b.detach().cpu().item())
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftLogReg":
        if SoftLogReg.SOLVER == "torch":
            return self._fit_torch(X, y)
        res = minimize(self._fg, np.zeros(X.shape[1] + 1), args=(X, y), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.w, self.b = res.x[:-1], float(res.x[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)


class Platt:
    """1-D logistic calibration of probabilities. Degenerate (constant) labels fall back to identity."""

    def __init__(self):
        self.a, self.b, self.identity = 1.0, 0.0, True

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Platt":
        if len(np.unique(np.round(y))) < 2:
            self.identity = True
            return self
        eps = 1e-6
        z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        f = lambda t: -np.sum(y * np.log(sigmoid(t[0] * z + t[1]) + 1e-12) + (1 - y) * np.log(1 - sigmoid(t[0] * z + t[1]) + 1e-12))
        r = minimize(f, [1.0, 0.0], method="Nelder-Mead")
        self.a, self.b, self.identity = float(r.x[0]), float(r.x[1]), False
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if self.identity:
            return p
        eps = 1e-6
        z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        return sigmoid(self.a * z + self.b)


class RidgeReg:
    """Closed-form ridge for a single-head Delta regression (ablation A1)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha, self.w, self.b = alpha, None, 0.0

    def fit(self, X, y):
        Xc = np.hstack([X, np.ones((len(X), 1))])
        reg = self.alpha * np.eye(Xc.shape[1])
        reg[-1, -1] = 0.0
        sol = np.linalg.solve(Xc.T @ Xc + reg, Xc.T @ y)
        self.w, self.b = sol[:-1], float(sol[-1])
        return self

    def predict(self, X):
        return X @ self.w + self.b


@dataclass
class Standardizer:
    mu: np.ndarray
    sd: np.ndarray

    @staticmethod
    def fit(X: np.ndarray) -> "Standardizer":
        return Standardizer(X.mean(0), X.std(0) + 1e-6)

    def __call__(self, X):
        return (X - self.mu) / self.sd


# ---- feature access --------------------------------------------------------------------------------
def node_features(tree: TreeData, layer: int) -> np.ndarray:
    """(N,7,d) float32 features at decision nodes for hidden-state layer `layer`."""
    li = tree.layers.index(layer)
    return tree.hidden[:, :, li, :].astype(np.float32)


def root_features(tree: TreeData, layer: int) -> np.ndarray:
    li = tree.layers.index(layer)
    return tree.root_hidden[:, li, :].astype(np.float32)


def grouped_folds(n_instances: int, k: int = 5, seed: int = 0) -> List[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n_instances)
    return [perm[i::k] for i in range(k)]


C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def select_layer_and_C(tree: TreeData, targets: Dict[str, np.ndarray], layers: Sequence[int], seed: int = 0):
    """Pick (layer, C) minimising mean validation Brier over all `targets` (name -> (N,7) labels), grouped CV by instance.
    Done on TRAIN only."""
    folds = grouped_folds(tree.n, 5, seed)
    best = (None, None, np.inf, {})
    table = {}
    for layer in layers:
        X = node_features(tree, layer)
        for C in C_GRID:
            briers = []
            for f in folds:
                va = np.zeros(tree.n, bool)
                va[f] = True
                tr = ~va
                Xtr = X[tr].reshape(-1, X.shape[-1])
                st = Standardizer.fit(Xtr)
                Xtr_s, Xva_s = st(Xtr), st(X[va].reshape(-1, X.shape[-1]))
                for name, Y in targets.items():
                    clf = SoftLogReg(C=C, max_iter=100).fit(Xtr_s, Y[tr].reshape(-1))
                    briers.append(np.mean((clf.predict_proba(Xva_s) - Y[va].reshape(-1)) ** 2))
            score = float(np.mean(briers))
            table[f"L{layer}_C{C:g}"] = score
            if score < best[2]:
                best = (layer, C, score, {})
    return best[0], best[1], best[2], table


# ---- fitted policies -------------------------------------------------------------------------------
class TwoHeadMUP:
    """Two logistic heads p_call(z), p_no(z) (labels from an exact continuation); call iff p_call - p_no > lambda*c."""

    name = "mup"

    def __init__(self, layer: int, C: float, continuation: int = 1, calibrate: bool = True):
        self.layer, self.C, self.cont, self.calibrate = layer, C, continuation, calibrate

    def fit(self, tr: TreeData, cal: Optional[TreeData] = None) -> "TwoHeadMUP":
        yc, yn = node_labels(tr, self.cont)
        X = node_features(tr, self.layer).reshape(-1, tr.hidden.shape[-1])
        self.st = Standardizer.fit(X)
        Xs = self.st(X)
        self.hc = SoftLogReg(self.C).fit(Xs, yc.reshape(-1))
        self.hn = SoftLogReg(self.C).fit(Xs, yn.reshape(-1))
        self.pc, self.pn = Platt(), Platt()
        if cal is not None and self.calibrate:
            ycc, ycn = node_labels(cal, self.cont)
            Xc = self.st(node_features(cal, self.layer).reshape(-1, X.shape[-1]))
            self.pc.fit(self.hc.predict_proba(Xc), ycc.reshape(-1))
            self.pn.fit(self.hn.predict_proba(Xc), ycn.reshape(-1))
        return self

    def heads(self, tree: TreeData) -> Tuple[np.ndarray, np.ndarray]:
        X = self.st(node_features(tree, self.layer).reshape(-1, tree.hidden.shape[-1]))
        pc = self.pc(self.hc.predict_proba(X)).reshape(tree.n, N_NODES)
        pn = self.pn(self.hn.predict_proba(X)).reshape(tree.n, N_NODES)
        return pc, pn

    def delta(self, tree: TreeData) -> np.ndarray:
        pc, pn = self.heads(tree)
        return pc - pn

    def actions(self, tree: TreeData, lam: float, cost: float = 1.0) -> np.ndarray:
        return (self.delta(tree) > lam * cost).astype(np.int64)


class DeltaRegression:
    """Single head regressing Delta = y_call - y_no directly (ablation A1)."""

    name = "delta_reg"

    def __init__(self, layer: int, alpha: float = 100.0, continuation: int = 1):
        self.layer, self.alpha, self.cont = layer, alpha, continuation

    def fit(self, tr: TreeData, cal=None):
        yc, yn = node_labels(tr, self.cont)
        X = node_features(tr, self.layer).reshape(-1, tr.hidden.shape[-1])
        self.st = Standardizer.fit(X)
        self.reg = RidgeReg(self.alpha).fit(self.st(X), (yc - yn).reshape(-1))
        return self

    def actions(self, tree: TreeData, lam: float, cost: float = 1.0):
        X = self.st(node_features(tree, self.layer).reshape(-1, tree.hidden.shape[-1]))
        return (self.reg.predict(X).reshape(tree.n, N_NODES) > lam * cost).astype(np.int64)


class ThresholdPolicy:
    """Generic 'score vs per-lambda threshold' policy; thresholds are tuned on CAL by true tree-walk utility.

    score_fn(tree) -> (N,7) score; call iff score < tau (direction='low') or score > tau (direction='high').
    """

    def __init__(self, name: str, score_fn, direction: str, grid: np.ndarray, per_instance: bool = False):
        self.name, self.score_fn, self.direction, self.grid, self.per_instance = name, score_fn, direction, grid, per_instance
        self.tau: Dict[float, float] = {}

    def _actions(self, tree: TreeData, tau: float) -> np.ndarray:
        s = self.score_fn(tree)
        a = (s < tau) if self.direction == "low" else (s > tau)
        if self.per_instance:                         # decision made once, applied to every step
            a = np.repeat(a[:, :1], N_NODES, axis=1) if a.ndim == 2 else np.repeat(a[:, None], N_NODES, axis=1)
        return a.astype(np.int64)

    def tune(self, cal: TreeData, lambdas: Sequence[float] = LAMBDAS, cost: float = 1.0) -> "ThresholdPolicy":
        for lam in lambdas:
            best, best_u = self.grid[0], -np.inf
            for tau in self.grid:
                c, k, _ = walk(cal, self._actions(cal, tau))
                u = float(np.mean(c - lam * cost * k))
                if u > best_u + 1e-12:
                    best, best_u = float(tau), u
            self.tau[lam] = best
        return self

    def actions(self, tree: TreeData, lam: float, cost: float = 1.0) -> np.ndarray:
        return self._actions(tree, self.tau[lam])


def fit_need_probe(tr: TreeData, layer: int, C: float, target: np.ndarray, per_instance_root: bool = False):
    """Logistic probe predicting `target` ((N,7) node labels, or (N,) trajectory label when per_instance_root)."""
    if per_instance_root:
        X = root_features(tr, layer)
        st = Standardizer.fit(X)
        clf = SoftLogReg(C).fit(st(X), target.astype(float))
        return lambda tree: clf.predict_proba(st(root_features(tree, layer)))
    X = node_features(tr, layer).reshape(-1, tr.hidden.shape[-1])
    st = Standardizer.fit(X)
    clf = SoftLogReg(C).fit(st(X), target.reshape(-1))
    return lambda tree: clf.predict_proba(st(node_features(tree, layer).reshape(-1, tree.hidden.shape[-1]))).reshape(tree.n, N_NODES)


# ---- reference policies (exact, label-based) ---------------------------------------------------------
def best_global_pattern(cal: TreeData, lam: float, cost: float = 1.0) -> int:
    best, best_u = 0, -np.inf
    for p in range(8):
        c, k, _ = walk(cal, np.tile(pattern_actions(p), (cal.n, 1)))
        u = float(np.mean(c - lam * cost * k))
        if u > best_u + 1e-12:
            best, best_u = p, u
    return best


def hindsight_utility(tree: TreeData, lam: float, cost: float = 1.0):
    """Non-deployable per-instance best fixed pattern (sees this instance's own outcomes). Returns (correct, calls)."""
    n = tree.n
    pops = np.array([bin(p).count("1") for p in range(8)], dtype=float)
    U = tree.leaf_correct.astype(float) - lam * cost * pops[None, :]
    # tie-break: fewer calls
    U_tb = U - 1e-9 * pops[None, :]
    best = U_tb.argmax(1)
    return tree.leaf_correct[np.arange(n), best].astype(float), pops[best]


def rollout_reference_actions(tree: TreeData, lam: float, cost: float = 1.0, continuation: int = 1) -> np.ndarray:
    """One-step look-ahead with exact node labels: call iff (y_call - y_no) > lambda*c. This is what MUP estimates."""
    yc, yn = node_labels(tree, continuation)
    return ((yc - yn) > lam * cost).astype(np.int64)


# =====================================================================================================
# Extra policies. Switched on by config `extra_policies`.
# =====================================================================================================
def walk_from(D: np.ndarray, nid: np.ndarray, level: int, prefix: np.ndarray):
    """Follow decision matrix D (N,7) from node `nid` at `level` (whose action prefix is `prefix`) to a leaf.
    Returns (leaf index, calls made from this node onward). Vectorised over instances."""
    from .envs.arith import T_STEPS
    n = len(nid)
    ar = np.arange(n)
    nid, prefix = nid.copy(), prefix.copy()
    calls = np.zeros(n, dtype=np.int64)
    for _ in range(level, T_STEPS):
        a = D[ar, nid].astype(np.int64)
        calls += a
        prefix = prefix * 2 + a
        nid = 2 * nid + 1 + a
    return prefix, calls


def _level_nodes(level: int) -> np.ndarray:
    return np.arange(2 ** level - 1, 2 ** (level + 1) - 1)


class PolicyConsistentMUP:
    """Policy-consistent two-head marginal-utility probe (A-45).

    Fixes the target mismatch of MUP: MUP is trained on labels under an always-call (or never-call) continuation but deployed
    with the learned policy. Here, for every lambda, levels are fitted backwards (t=2,1,0). The target of action a at a node is the
    exact final correctness (and the exact number of later calls) when the LEARNED policy is followed afterwards, read from the tree;
    the continuation actions on training instances are OUT-OF-FOLD predictions (grouped 5-fold by instance) so targets are not
    optimistic. Per level: two L2-logistic success heads p_a(z) (Platt-calibrated on cal with the deployed continuation) and ridge
    heads for the expected future calls f_a(z). Decision: call iff p1 - p0 > lambda * (1 + f1 - f0).
    """

    name = "mup_pc"

    def __init__(self, layer: int, C: float, alpha: float = 100.0, folds: int = 5, seed: int = 0):
        self.layer, self.C, self.alpha, self.k, self.seed = layer, C, alpha, folds, seed

    # --- targets from the tree under a decision matrix D (levels > t already filled)
    @staticmethod
    def _targets(tree: TreeData, D: np.ndarray, level: int):
        n = tree.n
        nodes = _level_nodes(level)
        S = np.zeros((n, len(nodes), 2))
        F = np.zeros((n, len(nodes), 2))
        for j, k in enumerate(nodes):
            pf = int(PREFIX_OF[k])
            for a in (0, 1):
                prefix = np.full(n, pf * 2 + a, dtype=np.int64)
                if level == 2:
                    leaf, calls = prefix, np.zeros(n, dtype=np.int64)
                else:
                    leaf, calls = walk_from(D, np.full(n, 2 * k + 1 + a, dtype=np.int64), level + 1, prefix)
                S[:, j, a] = tree.leaf_correct[np.arange(n), leaf]
                F[:, j, a] = calls
        return S, F

    def _fit_level(self, X, S, F):
        """X (n,m,d) standardised features; S,F (n,m,2). Returns heads fitted on all rows."""
        n, m, d = X.shape
        Xr = X.reshape(-1, d)
        h = {"p": [SoftLogReg(self.C).fit(Xr, S[:, :, a].reshape(-1)) for a in (0, 1)],
             "f": [RidgeReg(self.alpha).fit(Xr, F[:, :, a].reshape(-1)) for a in (0, 1)]}
        return h

    @staticmethod
    def _predict(h, X):
        n, m, d = X.shape
        Xr = X.reshape(-1, d)
        p = [h["p"][a].predict_proba(Xr).reshape(n, m) for a in (0, 1)]
        f = [np.clip(h["f"][a].predict(Xr).reshape(n, m), 0, 3) for a in (0, 1)]
        return p, f

    def _decide(self, p, f, lam, platt=None):
        if platt is not None:
            p = [platt[a](p[a].reshape(-1)).reshape(p[a].shape) for a in (0, 1)]
        return (p[1] - p[0] > lam * (1.0 + f[1] - f[0])).astype(np.int64)

    def fit(self, tr: TreeData, cal: TreeData, lams: Sequence[float]) -> "PolicyConsistentMUP":
        feats = node_features(tr, self.layer)
        self.st = Standardizer.fit(feats.reshape(-1, feats.shape[-1]))
        Xtr = self.st(feats.reshape(-1, feats.shape[-1])).reshape(feats.shape)
        Xcal = self.st(node_features(cal, self.layer).reshape(-1, feats.shape[-1])).reshape(cal.n, N_NODES, -1)
        folds = grouped_folds(tr.n, self.k, self.seed)
        self.heads: Dict[Tuple[float, int], dict] = {}
        self.platt: Dict[Tuple[float, int], list] = {}
        self.lams = list(lams)
        for lam in self.lams:
            Dtr = np.zeros((tr.n, N_NODES), dtype=np.int64)       # out-of-fold continuation decisions (training instances)
            Dcal = np.zeros((cal.n, N_NODES), dtype=np.int64)     # deployed continuation decisions (cal instances)
            for level in (2, 1, 0):
                if level == 2 and lam != self.lams[0]:           # level 2 does not depend on lambda: reuse the fit, new decisions
                    key0 = (self.lams[0], 2)
                    self.heads[(lam, 2)], self.platt[(lam, 2)] = self.heads[key0], self.platt[key0]
                else:
                    S, F = self._targets(tr, Dtr, level)
                    nodes = _level_nodes(level)
                    X = Xtr[:, nodes, :]
                    self.heads[(lam, level)] = self._fit_level(X, S, F)
                    # platt on cal with deployed continuation
                    Sc, _ = self._targets(cal, Dcal, level)
                    pc_, _ = self._predict(self.heads[(lam, level)], Xcal[:, nodes, :])
                    self.platt[(lam, level)] = [Platt().fit(pc_[a].reshape(-1), Sc[:, :, a].reshape(-1)) for a in (0, 1)]
                nodes = _level_nodes(level)
                # out-of-fold decisions for training instances at this level
                for f_idx in folds:
                    va = np.zeros(tr.n, bool)
                    va[f_idx] = True
                    S, F = self._targets(tr, Dtr, level)
                    h = self._fit_level(Xtr[~va][:, nodes, :], S[~va], F[~va])
                    p, fq = self._predict(h, Xtr[va][:, nodes, :])
                    Dtr[np.ix_(np.where(va)[0], nodes)] = self._decide(p, fq, lam)
                # deployed decisions for cal instances (used as continuation when calibrating the level above)
                h = self.heads[(lam, level)]
                p, fq = self._predict(h, Xcal[:, nodes, :])
                Dcal[:, nodes] = self._decide(p, fq, lam, self.platt[(lam, level)])
        return self

    def actions(self, tree: TreeData, lam: float, cost: float = 1.0) -> np.ndarray:
        assert abs(cost - 1.0) < 1e-12
        feats = node_features(tree, self.layer)
        X = self.st(feats.reshape(-1, feats.shape[-1])).reshape(feats.shape)
        D = np.zeros((tree.n, N_NODES), dtype=np.int64)
        for level in (2, 1, 0):
            nodes = _level_nodes(level)
            p, f = self._predict(self.heads[(lam, level)], X[:, nodes, :])
            D[:, nodes] = self._decide(p, f, lam, self.platt[(lam, level)])
        return D


class PlanProbe:
    """Stronger initial-state baseline (A-46): ONE decision before generation. Eight logistic heads on the root (t=0) state predict
    the success of each fixed call pattern p in {0..7}; for each lambda execute argmax_p  p_hat(p) - lambda * |p| open-loop."""

    name = "pp_plan"

    def __init__(self, layer: int, C: float):
        self.layer, self.C = layer, C

    def fit(self, tr: TreeData, cal: TreeData) -> "PlanProbe":
        X = root_features(tr, self.layer)
        self.st = Standardizer.fit(X)
        Xs = self.st(X)
        self.heads = [SoftLogReg(self.C).fit(Xs, tr.leaf_correct[:, p].astype(float)) for p in range(8)]
        Xc = self.st(root_features(cal, self.layer))
        self.platt = [Platt().fit(self.heads[p].predict_proba(Xc), cal.leaf_correct[:, p].astype(float)) for p in range(8)]
        self.pops = np.array([bin(p).count("1") for p in range(8)], dtype=float)
        return self

    def probs(self, tree: TreeData) -> np.ndarray:
        X = self.st(root_features(tree, self.layer))
        return np.stack([self.platt[p](self.heads[p].predict_proba(X)) for p in range(8)], 1)

    def patterns(self, tree: TreeData, lam: float, cost: float = 1.0) -> np.ndarray:
        score = self.probs(tree) - lam * cost * self.pops[None, :]
        return (score - 1e-9 * self.pops[None, :]).argmax(1)           # tie-break: fewer calls

    def actions(self, tree: TreeData, lam: float, cost: float = 1.0) -> np.ndarray:
        pats = self.patterns(tree, lam, cost)
        table = np.stack([pattern_actions(p) for p in range(8)])
        return table[pats]


def select_root_layer_and_C(tr: TreeData, layers: Sequence[int], seed: int = 0):
    """Layer/C for the root-state heads by grouped CV on TRAIN (mean Brier over the eight pattern targets)."""
    best, bs = (layers[0], C_GRID[0]), np.inf
    for L in layers:
        Xr = root_features(tr, L)
        for Cc in C_GRID:
            errs = []
            for f in grouped_folds(tr.n, 5, seed):
                va = np.zeros(tr.n, bool)
                va[f] = True
                st = Standardizer.fit(Xr[~va])
                for pat in range(8):
                    y = tr.leaf_correct[:, pat].astype(float)
                    m = SoftLogReg(Cc, 100).fit(st(Xr[~va]), y[~va])
                    errs.append(np.mean((m.predict_proba(st(Xr[va])) - y[va]) ** 2))
            if np.mean(errs) < bs:
                best, bs = (L, Cc), float(np.mean(errs))
    return best[0], best[1], bs
