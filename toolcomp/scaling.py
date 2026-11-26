"""Standardised runs for the scaling study: loaders for every kind of stored outcome tree, the surface signal for each environment,
and the regret-versus-x analysis. Everything downstream (horizon.evaluate_signals) sees only (features, step labels, leaf outcomes).

Run kinds
  tree    a full TreeData (.npz): signals S0 surface, S1 hidden state, S4 draft confidence, S0+S4
  text    3-step arithmetic runs whose .json sidecar holds every model reply: the exact tree is rebuilt, signal S0 only
  light   scripts/export_light.py output: S0, S1 (64-d PCA), S4, S0+S4 (+ S3 when present)
  deep    toolcomp.deep_tree output (6-step arithmetic, MuSiQue): S0, S4
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from .envs.arith import apply_op, generate_instances
from .horizon import default_sizes, evaluate_signals, surface_nobin
from .tree import LEVEL_OF, N_NODES, child, parse_strict


@dataclass
class Run:
    tag: str
    model: str
    env: str
    K: int
    ids: List[str]
    leaf: np.ndarray                      # (N, 2^K) final correctness for each action pattern
    Y: np.ndarray                         # (N, 2^K - 1) unaided step correctness at every node
    call_ok: np.ndarray                   # (N, 2^K - 1) step correctness when the tool is called
    feats: Dict[str, np.ndarray] = field(default_factory=dict)
    C: Dict[str, float] = field(default_factory=dict)
    per_partition: Dict[int, dict] = field(default_factory=dict)   # partition index -> {"feats": {...}, "C": {...}} overrides (hidden-state layer chosen per partition)
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.ids)


# ---------------------------------------------------------------------------------- 3-step arithmetic rebuilt from reply texts
def rebuild_arith3(sidecar: Path, seed: int, prefix: str = "e1"):
    """Exact outcome tree from the stored NOCALL replies: unaided answer = parsed reply, call answer = exact tool value."""
    side = json.loads(Path(sidecar).read_text())
    texts = side["texts"]
    n = len(side["instance_ids"])
    ins = generate_instances(n, seed, prefix)
    assert [i.instance_id for i in ins] == side["instance_ids"], "sidecar does not belong to this generator seed"
    leaf = np.zeros((n, 8), bool)
    loc = np.zeros((n, N_NODES), bool)
    vin = np.full((n, N_NODES), -1, np.int64)
    for i, x in enumerate(ins):
        v = [None] * N_NODES
        v[0] = x.first
        for k in range(N_NODES):
            lv = int(LEVEL_OF[k])
            vin[i, k] = -1 if v[k] is None else v[k]
            p = parse_strict(texts[i][k])
            t = None if v[k] is None else apply_op(v[k], x.ops[lv], x.rights[lv])
            loc[i, k] = p is not None and t is not None and p == t
            if lv < 2:
                v[child(k, 0)], v[child(k, 1)] = p, t
            else:
                pf = k - 3
                leaf[i, pf * 2] = p is not None and p == x.final_gold
                leaf[i, pf * 2 + 1] = t is not None and t == x.final_gold
    return ins, SimpleNamespace(n=n, v_in=vin, v_in_ok=vin != -1, leaf_correct=leaf, local_no_correct=loc, instance_ids=side["instance_ids"])


def run_from_text(tag, model, sidecar, seed):
    ins, t = rebuild_arith3(sidecar, seed)
    S0 = surface_nobin(t, ins)
    return Run(tag, model, "arith3", 3, t.instance_ids, t.leaf_correct, t.local_no_correct, t.v_in_ok, {"S0": S0}, note="rebuilt from reply texts; S0 only")


# ---------------------------------------------------------------------------------- full trees and light exports
def run_from_tree(tag, model, tree, ins, priors):
    """priors: list (one per partition) of the frozen {"layer", "C"} of the hidden-state probe, as chosen in Iteration 1."""
    from .features import draft_features, hidden_features
    S0, S4 = surface_nobin(tree, ins), draft_features(tree)
    per = {}
    for r, pr in enumerate(priors):
        per[r] = {"feats": {"S0": S0, "S1": hidden_features(tree, pr["layer"]), "S4": S4, "S0+S4": np.concatenate([S0, S4], -1)}, "C": {"S1": pr["C"]}}
    return Run(tag, model, "arith3", 3, list(tree.instance_ids), tree.leaf_correct, tree.local_no_correct, tree.v_in_ok,
               per[0]["feats"], per[0]["C"], per_partition=per)


def run_from_light(tag, model, npz, sidecar, seed, prefix="e1"):
    z = np.load(npz)
    side = json.loads(Path(sidecar).read_text())
    ins = generate_instances(len(side["instance_ids"]), seed, prefix)
    assert [i.instance_id for i in ins] == side["instance_ids"]
    t = SimpleNamespace(n=len(ins), v_in=z["v_in"])
    S0 = surface_nobin(t, ins)
    S4 = np.stack([z["mean_logprob"], z["entropy0"]], -1).astype(np.float64)
    feats = {"S0": S0, "S1": z["hidden_pca"].astype(np.float64), "S4": S4, "S0+S4": np.concatenate([S0, S4], -1)}
    return Run(tag, model, "arith3", 3, side["instance_ids"], z["leaf_correct"], z["local_no_correct"], z["v_in_ok"], feats, {"S1": 0.1})


# ---------------------------------------------------------------------------------- deep trees (K = 6 arithmetic, MuSiQue)
def _popcount_prefix(K: int):
    nid = np.arange(2 ** K - 1)
    lvl = np.floor(np.log2(nid + 1)).astype(int)
    pre = nid - (2 ** lvl - 1)
    calls = np.array([bin(int(p)).count("1") for p in pre])
    last = pre & 1
    return lvl, calls, last


def _node_block(K, N):
    lvl, calls, last = _popcount_prefix(K)
    oh = np.zeros((2 ** K - 1, K))
    oh[np.arange(2 ** K - 1), lvl] = 1
    blk = np.column_stack([oh, calls, last * (lvl > 0)])
    return np.broadcast_to(blk, (N,) + blk.shape)


def deep_surface(env: str, K: int, arrays, side) -> np.ndarray:
    N = arrays["leaf_correct"].shape[0]
    NN = 2 ** K - 1
    lvl, _, _ = _popcount_prefix(K)
    cols = []
    if env == "arith":
        items = side["items"]
        ops = np.array([[it["ops"][l] for l in lvl] for it in items])
        right = np.array([[it["rights"][l] for l in lvl] for it in items], float)
        left = np.array([[it["first"] if l == 0 else 0 for l in lvl] for it in items], float)
        vin = arrays["v_in_feat"].astype(float)
        left = np.where(lvl[None, :] == 0, left, vin)
        dig = lambda x: np.floor(np.log10(np.maximum(np.abs(x), 1.0))) + 1.0
        dl, dr, mul = dig(left), dig(right), (ops == "*").astype(float)
        # no "input missing" flag: -1 marks a missing input in the stored arrays but is also a real intermediate result (7 - 8), so a flag "vin != -1" fired on
        # real inputs and the probe extrapolated to near-certainty on them (see results/scaling_before_input_flag_fix/). Same columns as horizon.surface_nobin.
        cols = [ops == "+", ops == "-", ops == "*", dl, dr, dl * dr * mul, np.log10(np.maximum(np.abs(left), 1.0)) * mul]
    else:
        q = side["questions"]
        nw = np.array([[len(s.split()) for s in row] for row in q], float)
        nc = np.array([[len(s) for s in row] for row in q], float) / 10.0
        rel = np.array([[">>" in s for s in row] for row in q], float)
        cap = np.array([[sum(w[:1].isupper() for w in s.split()[1:]) for s in row] for row in q], float)
        prev = arrays["v_in_feat"].astype(float)
        cols = [nw, nc, rel, cap, np.maximum(prev, 0.0), prev < 0]
    base = np.stack([np.asarray(c, float) for c in cols], -1)
    return np.concatenate([base, _node_block(K, N)], -1)


def run_from_deep(tag, model, npz, kind=None) -> Run:
    from .deep_tree import load_deep
    a, side = load_deep(npz)
    K, env = side["K"], side["env"]
    S0 = deep_surface(env, K, a, side)
    S4 = np.stack([a["mean_logprob"], a["entropy0"]], -1).astype(np.float64)
    return Run(tag, model, f"{env}{K}", K, side["ids"], a["leaf_correct"], a["local_no_correct"], a["local_call_correct"],
               {"S0": S0, "S4": S4, "S0+S4": np.concatenate([S0, S4], -1)})


# ---------------------------------------------------------------------------------- analysis
def prefix_correct(run: Run) -> np.ndarray:
    """(N, 2^K-1) True where every earlier step's chosen answer (unaided or called) was correct."""
    N, NN = run.Y.shape
    ok = np.zeros((N, NN), bool)
    ok[:, 0] = True
    for k in range(NN):
        if 2 * k + 2 < NN:
            ok[:, child(k, 0)] = ok[:, k] & run.Y[:, k]
            ok[:, child(k, 1)] = ok[:, k] & run.call_ok[:, k]
    return ok


def step_accuracy(run: Run) -> List[float]:
    """p_j: probability an unaided step is right given a correct prefix, by level j."""
    ok = prefix_correct(run)
    out = []
    for lv in range(run.K):
        sl = slice(2 ** lv - 1, 2 ** (lv + 1) - 1)
        m = ok[:, sl]
        out.append(float(run.Y[:, sl][m].mean()) if m.any() else float("nan"))
    return out


def x_of(run: Run, m: int) -> float:
    p = step_accuracy(run)
    return float(m * sum(1 - q for q in p))


def evaluate_run(run: Run, seeds, MS, reps=40, B=0, sizes=None, signals=None, price_scale=1.0, noisy=False):
    sizes = sizes or default_sizes(run.n)
    out = {}
    extra = {"call_ok": run.call_ok, "prefix_ok": prefix_correct(run)} if noisy else {}      # noisy: also run Belief-DP with the measured tool accuracy
    for r, s in enumerate(seeds):
        ov = run.per_partition.get(r, {})
        feats = {k: v for k, v in ov.get("feats", run.feats).items() if signals is None or k in signals}
        out[f"partition_{r}"] = evaluate_signals(feats, run.Y, run.leaf, run.ids, sizes, s, ov.get("C", run.C), B if r == 0 else 0, reps, None, MS, price_scale, **extra)
    return out


def complement_counts(run: Run):
    """Pairwise second differences of leaf correctness (only for K = 3, matching the paper's convention)."""
    assert run.K == 3
    lc = run.leaf.astype(int)
    pos = neg = 0
    for i, j in ((4, 2), (4, 1), (2, 1)):
        for base in (0, 7 - i - j):
            d = lc[:, base | i | j] - lc[:, base | i] - lc[:, base | j] + lc[:, base]
            pos, neg = pos + int((d > 0).sum()), neg + int((d < 0).sum())
    return pos, neg
