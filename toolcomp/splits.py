"""Instance-level splits (A-16). Splitting is by instance, never by node, and the id sets are asserted disjoint."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def make_splits(instance_ids: Sequence[str], sizes: Dict[str, int], seed: int) -> Dict[str, np.ndarray]:
    n = len(instance_ids)
    need = sum(sizes.values())
    if need > n:
        raise ValueError(f"split sizes {sizes} need {need} instances but the pool has {n}")
    perm = np.random.default_rng(seed).permutation(n)
    out, start = {}, 0
    for name, k in sizes.items():
        out[name] = np.sort(perm[start:start + k])
        start += k
    assert_disjoint(instance_ids, out)
    return out


def assert_disjoint(instance_ids: Sequence[str], splits: Dict[str, np.ndarray]) -> None:
    seen: Dict[str, str] = {}
    for name, idx in splits.items():
        for i in idx:
            iid = instance_ids[int(i)]
            if iid in seen:
                raise AssertionError(f"instance {iid} appears in both '{seen[iid]}' and '{name}'")
            seen[iid] = name
