"""(mu+lambda) truncation: keep the top-mu candidates by current fitness.

Ties are broken by previous-generation fitness (higher first; new offspring count as -inf), then by id.
Usage: python -I truncate.py FITNESS.json MU OUT_POPULATION.json [PREV_FITNESS.json]
Output has the retained ids, their fitness (for the next tournament) and the eliminated ids.
"""
import json
import sys
from pathlib import Path

fit = json.loads(Path(sys.argv[1]).read_text())
mu = int(sys.argv[2])
prev = json.loads(Path(sys.argv[4]).read_text())["fitness"] if len(sys.argv) > 4 else {}
ids = sorted(fit["fitness"], key=lambda i: (-fit["fitness"][i], -prev.get(i, float("-inf")), i))
keep, drop = ids[:mu], ids[mu:]
out = {"mu": mu, "retained": keep, "eliminated": drop,
       "fitness": {i: fit["fitness"][i] for i in keep},
       "tie_break": "previous-generation fitness, then id"}
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
print("kept", keep, "| eliminated", drop)
