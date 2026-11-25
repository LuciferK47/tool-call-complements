"""Bradley-Terry fitness from pairwise judgments (HypoEvolve-style comparative fitness).

Input: JSON list of {"a": id, "b": id, "winner": "a"|"b"|"tie"}.  Ties count 0.5 for each side.
BT strengths are fitted by the MM algorithm with a weak symmetric prior (0.5 pseudo-win vs a virtual
average opponent) so strengths stay finite when an idea wins or loses every comparison.
fitness = A * log(pi) + B. On the first fit A is chosen so the spread (max-min) is 60 points and B so the mean is 50
(adaptation of HypoEvolve's anchoring; their exact constants for our setting are not specified in the text we read).
Later fits reuse A and shift B so surviving candidates keep their previous mean (pass --anchor FILE).

Usage: python -I fitness.py JUDGMENTS.json OUT.json [--anchor PREV_FITNESS.json]
"""
import json
import math
import sys
from pathlib import Path


def fit_bt(judgments, iters=500):
    ids = sorted({j["a"] for j in judgments} | {j["b"] for j in judgments})
    w = {i: 0.5 for i in ids}                       # prior pseudo-wins
    n = {i: {} for i in ids}
    for j in judgments:
        a, b, r = j["a"], j["b"], j["winner"]
        wa = 1.0 if r == "a" else 0.0 if r == "b" else 0.5
        w[a] += wa
        w[b] += 1.0 - wa
        n[a][b] = n[a].get(b, 0) + 1
        n[b][a] = n[b].get(a, 0) + 1
    pi = {i: 1.0 for i in ids}
    for _ in range(iters):
        new = {}
        for i in ids:
            denom = 1.0 / (pi[i] + 1.0)             # virtual opponent of strength 1, one comparison
            for k, c in n[i].items():
                denom += c / (pi[i] + pi[k])
            new[i] = w[i] / denom
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        pi = {i: v / g for i, v in new.items()}
    return pi


def main(argv):
    js = json.loads(Path(argv[1]).read_text())
    if isinstance(js, dict):          # accept the S6 wrapper object
        js = js["judgments"]
    pi = fit_bt(js)
    logs = {i: math.log(v) for i, v in pi.items()}
    if "--anchor" in argv:
        prev = json.loads(Path(argv[argv.index("--anchor") + 1]).read_text())
        A = prev["A"]
        common = [i for i in logs if i in prev["fitness"]]
        B = (sum(prev["fitness"][i] - A * logs[i] for i in common) / len(common)) if common else 50.0
    else:
        spread = max(logs.values()) - min(logs.values())
        A = 60.0 / spread if spread > 0 else 1.0
        B = 50.0 - A * sum(logs.values()) / len(logs)
    fit = {i: round(A * logs[i] + B, 2) for i in logs}
    wins = {}
    for j in js:
        for side in ("a", "b"):
            wins.setdefault(j[side], 0.0)
        r = j["winner"]
        wins[j["a"]] += 1.0 if r == "a" else 0.5 if r == "tie" else 0.0
        wins[j["b"]] += 1.0 if r == "b" else 0.5 if r == "tie" else 0.0
    out = {"A": A, "B": B, "n_judgments": len(js), "fitness": fit,
           "wins": wins, "rank": sorted(fit, key=lambda k: -fit[k])}
    Path(argv[2]).write_text(json.dumps(out, indent=1))
    print(len(js), "judgments ->", out["rank"])
    print({k: fit[k] for k in out["rank"]})


if __name__ == "__main__":
    main(sys.argv)
