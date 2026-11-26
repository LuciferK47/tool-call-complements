"""The 6-step surface features must not treat a real intermediate result of -1 as special (-1 also marks a missing input in the stored arrays)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcomp.scaling import deep_surface


def test_input_minus_one_is_not_flagged():
    K, nn = 3, 7
    side = {"items": [{"first": 7, "ops": ["-", "*", "+"], "rights": [8, 5, 2]}]}
    feats = [deep_surface("arith", K, {"leaf_correct": np.zeros((1, 8)), "v_in_feat": np.full((1, nn), v)}, side) for v in (-1, 1)]
    assert np.array_equal(feats[0], feats[1])               # same magnitude, so the same features: nothing singles out the value -1
