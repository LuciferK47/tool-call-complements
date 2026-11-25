"""Matched head / training-size ablation. Exploratory.

Heads (all estimate the marginal utility Delta = y_call - y_no under the always-call continuation and call iff estimate > lambda):
  H1 two-head logistic (the MUP of the main run, scipy)        H2 3-class Delta head (softmax over {-1,0,+1}, E[Delta])
  H3 two-head MLP                                              H4 3-class MLP
  H5 = H1 on all cached layers concatenated                    H6 = H2 on all cached layers concatenated
Matching: every head sees the same train subset, the same instance-grouped 5-fold split, and searches the same regularisation grid
(C_GRID; plus the layer for single-layer heads). Calibration (Platt per head / temperature for the 3-class head) is on the frozen CAL split.
Nothing here ever looks at gate or test while fitting; any 'best' variant is chosen on GATE by the caller.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .probes import (C_GRID, Platt, SoftLogReg, Standardizer, TwoHeadMUP, grouped_folds, node_features,
                     select_layer_and_C)
from .tree import N_NODES, TreeData, node_labels

HEADS = ("H1", "H2", "H3", "H4", "H5", "H6")
HEAD_DESC = {"H1": "two-head logistic (MUP)", "H2": "3-class Delta head (logistic)", "H3": "two-head MLP", "H4": "3-class MLP",
             "H5": "two-head logistic, all layers concatenated", "H6": "3-class Delta head, all layers concatenated"}


def _torch():
    import torch
    return torch


def _dev(device: str):
    torch = _torch()
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))


class TorchHead:
    """kind 'bin2': two sigmoid outputs; 'cls3': 3-way softmax. hidden=0 -> linear. L2 strength = 1/(C*n_rows) on the weights
    (the same convention as SoftLogReg's  sum-loss + ||w||^2/(2C))."""

    def __init__(self, kind: str, hidden: int, C: float, device: str = "auto", seed: int = 0, steps: int = 300):
        import os
        if os.environ.get("T4_MOCK") == "1":          # package self-test only: keep the mock end-to-end fast
            steps = min(steps, 25)
        self.kind, self.hidden, self.C, self.seed, self.steps = kind, hidden, C, seed, steps
        self.dev = _dev(device)
        self.net = None

    def _build(self, d: int):
        torch = _torch()
        torch.manual_seed(self.seed)
        out = 2 if self.kind == "bin2" else 3
        if self.hidden:
            self.net = torch.nn.Sequential(torch.nn.Linear(d, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, out))
        else:
            self.net = torch.nn.Linear(d, out)
        self.net.to(self.dev)

    def _loss(self, logits, Y):
        torch = _torch()
        if self.kind == "bin2":
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, Y)
        return torch.nn.functional.cross_entropy(logits, Y)

    def _reg(self, n):
        torch = _torch()
        w = [p for nme, p in self.net.named_parameters() if nme.endswith("weight")]
        return 0.5 / (self.C * n) * sum((x ** 2).sum() for x in w)

    def fit(self, X: np.ndarray, Y: np.ndarray, groups: Optional[np.ndarray] = None) -> "TorchHead":
        torch = _torch()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        Yt = torch.as_tensor(Y, dtype=torch.float32 if self.kind == "bin2" else torch.long, device=self.dev)
        n, d = Xt.shape
        self._build(d)
        if not self.hidden:                                   # convex: full-batch L-BFGS
            opt = torch.optim.LBFGS(self.net.parameters(), lr=1.0, max_iter=self.steps, history_size=20, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = self._loss(self.net(Xt), Yt) + self._reg(n)
                loss.backward()
                return loss
            opt.step(closure)
            return self
        # MLP: AdamW full-batch, early stopping on a 10% instance-grouped holdout inside the training rows
        rng = np.random.default_rng(self.seed)
        g = groups if groups is not None else np.arange(n)
        ug = np.unique(g)
        hold = set(rng.choice(ug, size=max(1, len(ug) // 10), replace=False).tolist())
        hm = np.array([x in hold for x in g])
        tr_i = torch.as_tensor(np.where(~hm)[0], device=self.dev)
        ho_i = torch.as_tensor(np.where(hm)[0], device=self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=0.0)
        best, best_state, bad = np.inf, None, 0
        for step in range(self.steps):
            opt.zero_grad()
            loss = self._loss(self.net(Xt[tr_i]), Yt[tr_i]) + self._reg(len(tr_i))
            loss.backward()
            opt.step()
            if step % 5 == 4:
                with torch.no_grad():
                    v = float(self._loss(self.net(Xt[ho_i]), Yt[ho_i]))
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = {k: x.detach().clone() for k, x in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 8:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            lo = self.net(torch.as_tensor(X, dtype=torch.float32, device=self.dev))
            p = torch.sigmoid(lo) if self.kind == "bin2" else torch.softmax(lo, -1)
        return p.cpu().numpy().astype(np.float64)

    def logits(self, X: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            return self.net(torch.as_tensor(X, dtype=torch.float32, device=self.dev)).cpu().numpy().astype(np.float64)


def temperature_fit(logits: np.ndarray, y: np.ndarray) -> float:
    """Single temperature T minimising NLL of softmax(logits / T) on the calibration rows (grid + refinement; degenerate -> 1)."""
    if len(np.unique(y)) < 2:
        return 1.0
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 60))

    def nll(T):
        z = logits / T
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    return float(grid[int(np.argmin([nll(t) for t in grid]))])


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def head_targets(tr: TreeData):
    yc, yn = node_labels(tr, 1)
    return yc, yn, (yc - yn).astype(int) + 1           # classes 0,1,2 = Delta -1,0,+1


def _features(tree: TreeData, layers: Sequence[int], concat: bool, layer: Optional[int]) -> np.ndarray:
    if concat:
        return np.concatenate([node_features(tree, l) for l in layers], axis=-1)
    return node_features(tree, layer)


class AblationHead:
    """One fitted variant. delta(tree) -> (N,7) estimate of Delta; actions(tree, lam) -> call iff delta > lam."""

    def __init__(self, name: str):
        self.name = name
        self.info: Dict = {}

    def delta(self, tree: TreeData) -> np.ndarray:
        X = self.st(_features(tree, self.layers, self.concat, self.layer).reshape(-1, self.dim))
        if self.kind == "bin2":
            p = self.head.predict(X)
            pc, pn = self.platt[0](p[:, 0]), self.platt[1](p[:, 1])
            return (pc - pn).reshape(tree.n, N_NODES)
        pr = _softmax(self.head.logits(X) / self.T)
        return (pr[:, 2] - pr[:, 0]).reshape(tree.n, N_NODES)

    def actions(self, tree: TreeData, lam: float, cost: float = 1.0) -> np.ndarray:
        return (self.delta(tree) > lam * cost).astype(np.int64)


def _cv_score(kind: str, hidden: int, C: float, X: np.ndarray, groups_inst: np.ndarray, tr: TreeData, folds, device: str, seed: int):
    yc, yn, yd = head_targets(tr)
    errs = []
    n_inst = tr.n
    for f in folds:
        va_i = np.zeros(n_inst, bool)
        va_i[f] = True
        rows_va = np.repeat(va_i, N_NODES)
        Xtr_raw, Xva_raw = X[~rows_va], X[rows_va]
        st = Standardizer.fit(Xtr_raw)
        Xtr, Xva = st(Xtr_raw), st(Xva_raw)
        if kind == "bin2":
            Y = np.stack([yc.reshape(-1), yn.reshape(-1)], 1)
            h = TorchHead("bin2", hidden, C, device, seed).fit(Xtr, Y[~rows_va], groups_inst[~rows_va])
            errs.append(np.mean((h.predict(Xva) - Y[rows_va]) ** 2))
        else:
            y = yd.reshape(-1)
            h = TorchHead("cls3", hidden, C, device, seed).fit(Xtr, y[~rows_va], groups_inst[~rows_va])
            P = h.predict(Xva)
            oh = np.eye(3)[y[rows_va]]
            errs.append(np.mean(((P - oh) ** 2).sum(1)))
    return float(np.mean(errs))


def fit_head(name: str, tr: TreeData, cal: TreeData, layers: Sequence[int], device: str = "auto", seed: int = 0,
             fixed_layer_C: Optional[Tuple[int, float]] = None) -> AblationHead:
    """Fit variant `name` on `tr` (TRAIN ONLY), calibrate on `cal`. Hyper-parameters by grouped CV on `tr` with the shared C grid."""
    t0 = time.perf_counter()
    h = AblationHead(name)
    h.layers = list(layers)
    h.concat = name in ("H5", "H6")
    h.kind = "bin2" if name in ("H1", "H3", "H5") else "cls3"
    hidden = 256 if name in ("H3", "H4") else 0
    yc, yn, yd = head_targets(tr)
    d_one = tr.hidden.shape[-1]
    if name == "H1":                                           # exactly the MUP of the main run
        lay, C, sc, _ = select_layer_and_C(tr, {"yc": yc, "yn": yn}, layers, seed)
        m = TwoHeadMUP(lay, C, continuation=1).fit(tr, cal)
        h.mup = m
        h.layer, h.C = lay, C
        h.delta = m.delta                                      # type: ignore[assignment]
        h.info = {"layer": lay, "C": C, "cv_score": sc, "fit_seconds": time.perf_counter() - t0}
        return h
    folds = grouped_folds(tr.n, 5, seed)
    groups = np.repeat(np.arange(tr.n), N_NODES)
    cand_layers = [None] if h.concat else list(layers)
    best = (None, None, np.inf)
    for lay in cand_layers:
        X = _features(tr, layers, h.concat, lay).reshape(tr.n * N_NODES, -1)
        for C in C_GRID:
            s = _cv_score(h.kind, hidden, C, X, groups, tr, folds, device, seed)
            if s < best[2]:
                best = (lay, C, s)
    h.layer, h.C = best[0], best[1]
    h.dim = (len(layers) * d_one) if h.concat else d_one
    X = _features(tr, layers, h.concat, h.layer).reshape(tr.n * N_NODES, -1)
    h.st = Standardizer.fit(X)
    Xs = h.st(X)
    if h.kind == "bin2":
        h.head = TorchHead("bin2", hidden, h.C, device, seed).fit(Xs, np.stack([yc.reshape(-1), yn.reshape(-1)], 1), groups)
        h.platt = [Platt(), Platt()]
        ccy, ccn = node_labels(cal, 1)
        Xc = h.st(_features(cal, layers, h.concat, h.layer).reshape(cal.n * N_NODES, -1))
        p = h.head.predict(Xc)
        h.platt[0].fit(p[:, 0], ccy.reshape(-1))
        h.platt[1].fit(p[:, 1], ccn.reshape(-1))
    else:
        h.head = TorchHead("cls3", hidden, h.C, device, seed).fit(Xs, yd.reshape(-1), groups)
        _, _, ydc = head_targets(cal)
        Xc = h.st(_features(cal, layers, h.concat, h.layer).reshape(cal.n * N_NODES, -1))
        h.T = temperature_fit(h.head.logits(Xc), ydc.reshape(-1))
    h.info = {"layer": h.layer, "C": h.C, "cv_score": best[2], "fit_seconds": time.perf_counter() - t0, "hidden": hidden}
    return h


def plan_subsets(main_ids: Sequence[str], extra_ids: Sequence[str], split_sizes: Dict[str, int], seed: int, sizes: Sequence[int]):
    """Deterministic plan used by BOTH the pre-flight leakage check and the ablation runner: partition-0 splits of the main pool,
    nested training subsets (original train split + the first m extra instances of a seeded order). Raises on any overlap."""
    from .splits import make_splits
    sp = make_splits(list(main_ids), split_sizes, seed)
    order = np.random.default_rng(seed + 7).permutation(len(extra_ids))
    base = len(sp["train"])
    subsets = {}
    for s in sizes:
        m = max(0, s - base)
        if m > len(extra_ids):
            raise ValueError(f"size {s} needs {m} extra instances, have {len(extra_ids)}")
        subsets[s] = {"main_train_indices": sp["train"].tolist(), "extra_indices": order[:m].tolist()}
    prev = None
    for s in sizes:
        cur = set(subsets[s]["extra_indices"])
        if prev is not None and not prev <= cur:
            raise AssertionError("training subsets are not nested")
        prev = cur
    ev = {main_ids[i] for k in ("cal", "gate", "test") for i in sp[k]}
    for s in sizes:
        tr = {main_ids[i] for i in subsets[s]["main_train_indices"]} | {extra_ids[i] for i in subsets[s]["extra_indices"]}
        if tr & ev:
            raise AssertionError(f"training subset {s} overlaps cal/gate/test")
    return sp, subsets, order
