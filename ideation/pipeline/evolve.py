"""Seeded operator planner for one HypoEvolve-style generation.

Rules implemented (from the HypoEvolve text we read, Sec. 3.2):
 - each parent is chosen by a size-2 tournament: sample two distinct members uniformly, higher fitness wins;
 - per offspring: crossover with p_c = 0.6 (two parents from separate tournaments, resample 2nd if equal);
   mutation with p_m = 0.15 after crossover, or with probability 1 if crossover was skipped;
 - each operation picks uniformly between its two variants.
Variants adapted to our domain (HypoEvolve's 'drug substitution' has no NLP analogue):
   crossover: combination | inspiration;   mutation: targeted-fix | out-of-box.
The LLM only *executes* the planned operators; it does not choose them.

Usage: python -I evolve.py FITNESS.json N_OFFSPRING SEED OUT.json
"""
import json
import random
import sys
from pathlib import Path

PC, PM = 0.6, 0.15
CROSS = ["CROSSOVER-COMBINATION", "CROSSOVER-INSPIRATION"]
MUT = ["MUTATION-TARGETED", "MUTATION-OUT-OF-BOX"]


def tournament(rng, fit, exclude=None):
    ids = [i for i in fit if i != exclude] if exclude else list(fit)
    a, b = rng.sample(ids, 2)
    return a if fit[a] >= fit[b] else b


def plan(fit, n, seed):
    rng = random.Random(seed)
    out = []
    for k in range(n):
        ops, parents = [], []
        if rng.random() < PC:
            p1 = tournament(rng, fit)
            p2 = tournament(rng, fit)
            while p2 == p1:
                p2 = tournament(rng, fit)
            parents = [p1, p2]
            ops.append(rng.choice(CROSS))
            if rng.random() < PM:
                ops.append(rng.choice(MUT))
        else:
            parents = [tournament(rng, fit)]
            ops.append(rng.choice(MUT))
        out.append({"offspring_slot": k + 1, "parents": parents, "operators": ops})
    return out


if __name__ == "__main__":
    fit = json.loads(Path(sys.argv[1]).read_text())["fitness"]
    n, seed = int(sys.argv[2]), int(sys.argv[3])
    p = plan(fit, n, seed)
    Path(sys.argv[4]).write_text(json.dumps({"seed": seed, "pc": PC, "pm": PM, "plan": p}, indent=1))
    for r in p:
        print(r)
