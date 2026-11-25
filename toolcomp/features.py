"""Iteration-2 feature families per decision node (additive module; no frozen file is touched).

Signal families, ordered by when the information becomes available:
  S0  surface   : operator, digit counts, difficulty bin, node id  (readable from the prompt; no LLM at all)
  S1  hidden    : frozen-LLM hidden state at the last prompt token  (pre-generation)           [src.probes.node_features]
  S2  prefill   : entropy of the first answer token                 (free, pre-generation; already cached as `entropy0`)
  S4  draft     : mean answer-token log-probability + entropy0       (needs the drafted answer)

Every function returns (N, 7, d) float64 arrays. `with_features` wraps any of them as a TreeData whose single "layer" 0 is that
array, so every existing estimator (TwoHeadMUP, DeltaRegression, PolicyConsistentMUP, select_layer_and_C) runs on it unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from .envs.arith import Instance
from .probes import node_features
from .tree import LEVEL_OF, N_NODES, TreeData


def _digits(x: np.ndarray) -> np.ndarray:
    return np.floor(np.log10(np.maximum(np.abs(x).astype(np.float64), 1.0))) + 1.0


def surface_features(tree: TreeData, instances: Sequence[Instance]) -> np.ndarray:
    """S0. Only information printed in the node prompt (operator, operand sizes, running value size) plus the node id."""
    assert [i.instance_id for i in instances] == list(tree.instance_ids), "instances must match the tree order"
    N = tree.n
    out = []
    for k in range(N_NODES):
        lv = int(LEVEL_OF[k])
        op = np.array([i.ops[lv] for i in instances])
        right = np.array([i.rights[lv] for i in instances], dtype=np.float64)
        left = (np.array([i.first for i in instances], dtype=np.float64) if lv == 0
                else tree.v_in[:, k].astype(np.float64))
        v_ok = (np.ones(N, dtype=np.float64) if lv == 0
                else (tree.v_in[:, k] != -1).astype(np.float64))
        dl, dr = _digits(left), _digits(right)
        mul = (op == "*").astype(np.float64)
        node1h = np.zeros((N, N_NODES))
        node1h[:, k] = 1.0
        cols = [(op == "+"), (op == "-"), (op == "*"), dl, dr, dl * dr * mul,
                np.log10(np.maximum(np.abs(left), 1.0)) * mul, v_ok]
        cols += [tree.step_bin[:, lv] == b for b in range(3)]
        out.append(np.column_stack([np.column_stack(cols).astype(np.float64), node1h]))
    return np.stack(out, 1)


def hidden_features(tree: TreeData, layer: int) -> np.ndarray:
    """S1."""
    return node_features(tree, layer).astype(np.float64)


def prefill_features(tree: TreeData) -> np.ndarray:
    """S2: first-token entropy of the answer distribution = computed from the prefill logits, before any token is emitted."""
    return tree.entropy0[:, :, None].astype(np.float64)


def draft_features(tree: TreeData) -> np.ndarray:
    """S4: confidence of the drafted answer (available only after the draft is generated)."""
    return np.stack([tree.mean_logprob, tree.entropy0], -1).astype(np.float64)


def combine(*blocks: np.ndarray) -> np.ndarray:
    return np.concatenate(blocks, axis=-1)


def with_features(tree: TreeData, F: np.ndarray) -> TreeData:
    """A TreeData identical to `tree` except that its only hidden-state 'layer' (id 0) is F, shape (N, 7, d)."""
    assert F.shape[:2] == (tree.n, N_NODES)
    root = np.zeros((tree.n, 1, F.shape[-1]), dtype=np.float32)
    return dataclasses.replace(tree, hidden=F[:, :, None, :].astype(np.float32), root_hidden=root, layers=[0])
