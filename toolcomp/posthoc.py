"""POST-HOC (exploratory) probe-plus-answer-confidence heads. Not part of any pre-registered family.

Question: do hidden states add information beyond the log-probability / entropy of the model's own drafted answer?
  variant "conf_only": p_no from [mean_logprob, entropy0] alone (train), Platt on cal;  p_call = MUP's calibrated hidden-state head.
  variant "stack":     p_no = logistic([logit(MUP raw hidden-state p_no), mean_logprob, entropy0]) fitted on cal (3 inputs, standardised).
Both need the drafted answer before deciding (same information advantage as the answer-confidence threshold)."""

from __future__ import annotations

import numpy as np

from .probes import Platt, SoftLogReg, Standardizer, TwoHeadMUP, node_features
from .tree import N_NODES, TreeData, node_labels


def conf_feats(tree: TreeData) -> np.ndarray:
    return np.stack([tree.mean_logprob, tree.entropy0], -1).reshape(-1, 2).astype(np.float64)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class ConfidenceMUP:
    def __init__(self, layer: int, C: float, variant: str):
        assert variant in ("conf_only", "stack")
        self.layer, self.C, self.variant = layer, C, variant

    def _raw_no(self, tree: TreeData) -> np.ndarray:
        X = self.base.st(node_features(tree, self.layer).reshape(-1, tree.hidden.shape[-1]))
        return self.base.hn.predict_proba(X)

    def _stack_x(self, tree: TreeData) -> np.ndarray:
        return np.column_stack([_logit(self._raw_no(tree)), conf_feats(tree)])

    def fit(self, tr: TreeData, cal: TreeData) -> "ConfidenceMUP":
        self.base = TwoHeadMUP(self.layer, self.C).fit(tr, cal)
        if self.variant == "conf_only":
            _, yn = node_labels(tr, 1)
            self.st = Standardizer.fit(conf_feats(tr))
            self.head = SoftLogReg(1.0).fit(self.st(conf_feats(tr)), yn.reshape(-1))
            _, ycn = node_labels(cal, 1)
            self.platt = Platt().fit(self.head.predict_proba(self.st(conf_feats(cal))), ycn.reshape(-1))
        else:
            _, ycn = node_labels(cal, 1)
            Z = self._stack_x(cal)
            self.st = Standardizer.fit(Z)
            self.head = SoftLogReg(1.0).fit(self.st(Z), ycn.reshape(-1))
        return self

    def p_no(self, tree: TreeData) -> np.ndarray:
        if self.variant == "conf_only":
            p = self.platt(self.head.predict_proba(self.st(conf_feats(tree))))
        else:
            p = self.head.predict_proba(self.st(self._stack_x(tree)))
        return p.reshape(tree.n, N_NODES)

    def delta(self, tree: TreeData) -> np.ndarray:
        pc, _ = self.base.heads(tree)
        return pc - self.p_no(tree)

    def actions(self, tree: TreeData, lam: float, cost: float = 1.0) -> np.ndarray:
        return (self.delta(tree) > lam * cost).astype(np.int64)
