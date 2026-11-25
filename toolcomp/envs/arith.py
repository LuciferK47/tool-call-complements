"""E1: chained arithmetic (3 steps), deterministic instance generator and exact reference computation.

Design:
  * every instance has exactly T=3 steps: v1 = a op1 b1, v2 = v1 op2 b2, v3 = v2 op3 b3;
  * each step draws its own difficulty bin, so difficulty varies *within* a trajectory;
  * the full problem text is shown at every decision point (information at step t is a superset of
    the information at t=0), plus how earlier results were obtained (agent knows its own actions);
  * correctness of an instance is exact equality of the final value with the gold chain value.
No model, tool or simulator is used here; this file is pure, deterministic Python.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Optional

T_STEPS = 3
OPS = ("+", "-", "*")

# (min_digits_left, max_digits_left, min_digits_right, max_digits_right, allowed ops) per difficulty bin.
# For steps > 1 the left operand is the running value, so only the right operand's digits are drawn.
BINS = {
    "easy":   dict(dl=(1, 2), dr=(1, 2), ops=("+", "-")),
    "medium": dict(dl=(2, 3), dr=(2, 2), ops=("+", "-", "*")),
    "hard":   dict(dl=(3, 4), dr=(3, 3), ops=("+", "-", "*")),
}
BIN_WEIGHTS = (("easy", 0.4), ("medium", 0.35), ("hard", 0.25))


@dataclass(frozen=True)
class Instance:
    instance_id: str
    first: int
    ops: tuple          # length T
    rights: tuple       # length T
    bins: tuple         # length T, difficulty bin of each step
    gold_values: tuple  # length T, exact running values v1..vT

    @property
    def final_gold(self) -> int:
        return self.gold_values[-1]

    @property
    def num_steps(self) -> int:
        return len(self.ops)

    def problem_text(self, reveal: str = "full", step: Optional[int] = None) -> str:
        """Text of the expression. If reveal == 'incremental' and step is provided,
        future operations beyond `step` are hidden (delayed information setting)."""
        if reveal == "incremental" and step is not None:
            expr = str(self.first)
            for i in range(step + 1):
                expr = f"({expr} {self.ops[i]} {self.rights[i]})"
            remaining = self.num_steps - (step + 1)
            if remaining > 0:
                expr = f"{expr} followed by {remaining} further undisclosed operation{'s' if remaining > 1 else ''}"
            return expr
        expr = str(self.first)
        for op, r in zip(self.ops, self.rights):
            expr = f"({expr} {op} {r})"
        return expr


def apply_op(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(op)


def _draw_int(rng: random.Random, digits: int) -> int:
    return rng.randint(10 ** (digits - 1), 10 ** digits - 1)


def generate_instances(n: int, seed: int, prefix: str = "e1") -> List[Instance]:
    """Deterministic: same (n, seed) -> identical instances. Instance ids encode the seed."""
    rng = random.Random(seed)
    names = [b for b, _ in BIN_WEIGHTS]
    weights = [w for _, w in BIN_WEIGHTS]
    out = []
    for i in range(n):
        bins = tuple(rng.choices(names, weights)[0] for _ in range(T_STEPS))
        spec0 = BINS[bins[0]]
        first = _draw_int(rng, rng.randint(*spec0["dl"]))
        ops, rights, vals = [], [], []
        cur = first
        for b in bins:
            spec = BINS[b]
            op = rng.choice(spec["ops"])
            right = _draw_int(rng, rng.randint(*spec["dr"]))
            cur = apply_op(cur, op, right)
            ops.append(op)
            rights.append(right)
            vals.append(cur)
        out.append(Instance(f"{prefix}_{seed}_{i:05d}", first, tuple(ops), tuple(rights), bins, tuple(vals)))
    return out


def dataset_hash(instances: List[Instance]) -> str:
    payload = json.dumps([asdict(i) for i in instances], sort_keys=True, default=list)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def finish_with_tool(instance: Instance, step: int, value: Optional[int]) -> Optional[int]:
    """Value after applying the remaining ops *exactly* from running value `value` that is the result of
    step index `step` (0-based). Returns None if `value` is None (an earlier step produced no usable number)."""
    if value is None:
        return None
    cur = value
    for k in range(step + 1, instance.num_steps):
        cur = apply_op(cur, instance.ops[k], instance.rights[k])
    return cur


def node_prompt(instance: Instance, step: int, v_in: Optional[int], history: List[dict],
                reveal: str = "full") -> str:
    """Text shown at the decision point of step `step` (0-based).

    `history` = [{"value": int|None, "how": "calculator"|"mental"}, ...] for steps already done.
    The prompt never contains gold values, and never contains this step's own answer.
    Under reveal='incremental', future operations beyond `step` remain hidden.
    """
    prob = instance.problem_text(reveal=reveal, step=step)
    lines = [f"Problem: compute {prob} one step at a time."]
    for k, h in enumerate(history):
        shown = "?" if h["value"] is None else str(h["value"])
        lines.append(f"Step {k + 1} result: {shown} (done by {h['how']}).")
    left = "?" if v_in is None else str(v_in)
    lines.append(f"Now step {step + 1}: compute {left} {instance.ops[step]} {instance.rights[step]}.")
    lines.append("Reply with only the integer result.")
    return "\n".join(lines)


def root_prompt(instance: Instance, reveal: str = "full") -> str:
    """Pre-generation (t=0) state used by the single-decision baseline: the whole problem, nothing else."""
    prob = instance.problem_text(reveal=reveal, step=0 if reveal == "incremental" else None)
    return (f"Problem: compute {prob} one step at a time.\n"
            "You may use a calculator or compute mentally. Reply with only the final integer result.")
