"""Deterministic MOCK runner used ONLY by unit tests (plumbing checks without downloading a model).

It answers arithmetic prompts with a digit-dependent, hash-seeded chance of being right and returns hidden
features that carry digit/operator information plus noise. It is the sole surviving use of the old simulator
(see archive/simulated_smoke_test/README.md): no number produced here is ever reported as a result.
"""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import numpy as np

_STEP = re.compile(r"Now step (\d+): compute (\S+) ([+\-*]) (-?\d+)\.")


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


class MockRunner:
    model_id = "mock"
    revision = "mock"
    device = "cpu"
    hidden_dim = 16
    system = "mock"
    num_layers = 4
    dtype_name = "float32"
    resolved_revision = "mock"
    stop_ids = set()

    def format(self, user_text):
        return user_text

    def env_info(self):
        return {"device": "cpu", "dtype": "float32", "resolved_revision": "mock", "torch": "mock"}

    def __init__(self, p_easy: float = 0.9, p_hard: float = 0.1):
        self.p_easy, self.p_hard = p_easy, p_hard

    def _features(self, prompt: str, layers):
        m = _STEP.search(prompt)
        rng = np.random.default_rng(_h(prompt) % (2 ** 32))
        v = rng.normal(0, 0.3, self.hidden_dim)
        if m:
            digits = len(m.group(2).lstrip("-")) + len(m.group(4).lstrip("-"))
            v[0] = digits / 6.0
            v[1] = 1.0 if m.group(3) == "*" else 0.0
            v[2] = int(m.group(1)) / 3.0
        return {l: v.astype(np.float16) for l in layers}

    def run(self, prompts, layers, max_new_tokens=12, batch_size=16):
        outs = []
        for p in prompts:
            m = _STEP.search(p)
            text, mlp = "0", -2.0
            if m and m.group(2) != "?":
                a, op, b = int(m.group(2)), m.group(3), int(m.group(4))
                gold = a + b if op == "+" else a - b if op == "-" else a * b
                digits = len(m.group(2).lstrip("-")) + len(m.group(4).lstrip("-"))
                hard = min(1.0, max(0.0, (digits - 2) / 6.0 + (0.3 if op == "*" else 0.0)))
                p_ok = self.p_easy * (1 - hard) + self.p_hard * hard
                ok = (_h(p) % 10000) / 10000.0 < p_ok
                text = str(gold if ok else gold + 7)
                mlp = -0.1 if ok else -1.5
            outs.append(SimpleNamespace(text=text, n_tokens=len(text), mean_logprob=mlp, first_token_entropy=0.5,
                                        hidden=self._features(p, layers), n_prompt_tokens=40))
        return outs

    def hidden_only(self, prompts, layers, batch_size=16):
        return [self._features(p, layers) for p in prompts]

    def sample(self, prompts, n_samples, temperature=0.7, max_new_tokens=12, batch_size=8, seed=0):
        import random
        out = []
        for p in prompts:
            greedy = self.run([p], [1])[0].text
            rng = random.Random(_h(p) % (2 ** 31) + seed)
            try:
                g = int(greedy)
            except ValueError:
                g = 0
            out.append([str(g if rng.random() < 0.7 else g + rng.choice([-10, -1, 1, 10])) for _ in range(n_samples)])
        return out
