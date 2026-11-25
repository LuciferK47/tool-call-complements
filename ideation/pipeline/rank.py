"""Final ranking from S7 rubric scores + pairwise (Bradley-Terry) cross-check + weight sensitivity.

Outputs outputs/ranking.csv and outputs/ranking_sensitivity.json. Deterministic (fixed seed).
Weighted score = sum_c w_c * mean_over_judges(score_c), scale 1-5.
Flags: skeptic novelty <= 1 or min-judge feasibility <= 2 (a high total must not hide these).
"""
import csv
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CRIT = ["novelty", "feasibility", "impact", "testability", "clarity", "distinctiveness"]


def wscore(sc, w, judge=None):
    tot = 0.0
    for c in CRIT:
        s = sc[c]["s"]
        v = s[judge] if judge is not None else sum(s) / len(s)
        tot += w[c] * v
    return tot


def kendall(a, b):
    ids = [i for i in a if i in b]
    conc = disc = 0
    for x, y in itertools.combinations(ids, 2):
        d = (a[x] - a[y]) * (b[x] - b[y])
        conc += d > 0
        disc += d < 0
    n = conc + disc
    return (conc - disc) / n if n else float("nan")


def main():
    d = json.loads((OUT / "S7_rubric_scores.json").read_text())
    W, S = d["weights"], d["scores"]
    fin = json.loads((OUT / "S7a_finalists.json").read_text())
    # pairwise fitness: latest pool containing the idea
    fit = {}
    for p in ("P1", "P2"):
        for g in (0, 1, 2):
            f = json.loads((OUT / f"fitness_gen{g}_{p}.json").read_text())["fitness"]
            fit.update(f)                       # later generations overwrite earlier ones
    rows, sens = [], {}
    rng = random.Random(20261023)
    for p in ("P1", "P2"):
        ids = fin[p]["finalists"]
        per = {i: {"all": wscore(S[i], W),
                   "optimist": wscore(S[i], W, 0), "methodologist": wscore(S[i], W, 1), "skeptic": wscore(S[i], W, 2)} for i in ids}
        order = sorted(ids, key=lambda i: -per[i]["all"])
        rubric_rank = {i: r + 1 for r, i in enumerate(order)}
        tau = kendall({i: per[i]["all"] for i in ids}, {i: fit[i] for i in ids})
        # sensitivity: Dirichlet-perturbed weights + named alternatives
        base = [W[c] for c in CRIT]
        top5_count = {i: 0 for i in ids}
        rank_sum = {i: 0 for i in ids}
        N = 2000
        for _ in range(N):
            g = [rng.gammavariate(60 * b, 1) for b in base]
            s = sum(g)
            w = {c: x / s for c, x in zip(CRIT, g)}
            o = sorted(ids, key=lambda i: -wscore(S[i], w))
            for r, i in enumerate(o):
                rank_sum[i] += r + 1
                top5_count[i] += r < 5
        alts = {"equal": {c: 1 / 6 for c in CRIT},
                "feasibility_heavy": {"novelty": .15, "feasibility": .40, "impact": .15, "testability": .15, "clarity": .10, "distinctiveness": .05},
                "novelty_heavy": {"novelty": .40, "feasibility": .15, "impact": .20, "testability": .10, "clarity": .10, "distinctiveness": .05}}
        alt_rank = {k: [i for i in sorted(ids, key=lambda i: -wscore(S[i], w))] for k, w in alts.items()}
        sens[p] = {"kendall_tau_rubric_vs_pairwise": round(tau, 3),
                   "top5_probability_dirichlet": {i: round(top5_count[i] / N, 3) for i in ids},
                   "mean_rank_dirichlet": {i: round(rank_sum[i] / N, 2) for i in ids},
                   "alternative_weight_rankings": alt_rank}
        for i in ids:
            sc = S[i]
            flags = []
            if sc["novelty"]["s"][2] <= 1:
                flags.append("skeptic novelty<=1")
            if min(sc["feasibility"]["s"]) <= 2:
                flags.append("feasibility<=2 by some judge")
            if sc["distinctiveness"]["s"][2] <= 2:
                flags.append("low distinctiveness")
            rows.append({"problem": p, "id": i, "rubric_rank": rubric_rank[i], "weighted_all": round(per[i]["all"], 3),
                         "optimist": round(per[i]["optimist"], 3), "methodologist": round(per[i]["methodologist"], 3),
                         "skeptic": round(per[i]["skeptic"], 3), "bt_fitness": fit.get(i, ""),
                         "top5_prob": sens[p]["top5_probability_dirichlet"][i], "flags": ";".join(flags)})
    rows.sort(key=lambda r: (r["problem"], r["rubric_rank"]))
    with (OUT / "ranking.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "ranking_sensitivity.json").write_text(json.dumps(sens, indent=1))
    for r in rows:
        print(r["problem"], r["rubric_rank"], r["id"], r["weighted_all"], "sk=", r["skeptic"], "bt=", r["bt_fitness"], "top5p=", r["top5_prob"], r["flags"])
    for p in sens:
        print(p, "tau", sens[p]["kendall_tau_rubric_vs_pairwise"])


if __name__ == "__main__":
    main()
