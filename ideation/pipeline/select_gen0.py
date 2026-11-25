"""Select the generation-0 population from S3 reviews.

Mean of the five reviewer scores; ties broken by (a) higher methodology score, (b) higher novelty, (c) id.
To keep the population diverse, at most 2 ideas per `family_group` (first word group of the S1 family field).
Usage: python -I select_gen0.py MU
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def family_group(f):
    f = f.lower()
    for key in ("probing", "decision-focused", "sequential", "adaptive", "conformal", "cascade", "calibration",
                "label", "evaluation", "signal", "steering", "credit", "robust", "multi-tool", "proxy", "verbal",
                "learned", "online", "reliab", "multi-model", "estimator", "two-sided", "cheap"):
        if key in f:
            return key
    return f.split()[0]


def main(mu):
    reviews = json.loads((OUT / "S3_reviews.json").read_text())["reviews"]
    fams = {}
    for p in ("P1", "P2"):
        for i in json.loads((OUT / f"S1_pool_{p}.json").read_text())["ideas"]:
            fams[i["id"]] = i["family"]
    result = {}
    for p in ("P1", "P2"):
        rows = []
        for iid, r in reviews.items():
            if not iid.startswith(p):
                continue
            sc = {k: v[0] for k, v in r.items()}
            rows.append({"id": iid, "mean": round(sum(sc.values()) / 5, 2), "scores": sc,
                         "family_group": family_group(fams[iid])})
        rows.sort(key=lambda r: (-r["mean"], -r["scores"]["METHODOLOGY"], -r["scores"]["NOVELTY"], r["id"]))
        chosen, count = [], {}
        for r in rows:
            if len(chosen) == mu:
                break
            if count.get(r["family_group"], 0) >= 2:
                r["skipped"] = "family cap"
                continue
            count[r["family_group"]] = count.get(r["family_group"], 0) + 1
            chosen.append(r["id"])
        result[p] = {"selected": chosen, "table": rows}
    (OUT / "S4a_gen0_selection.json").write_text(json.dumps(result, indent=1))
    for p, v in result.items():
        print(p, v["selected"])
        for r in v["table"]:
            print("  ", r["id"], r["mean"], r["family_group"], "SELECTED" if r["id"] in v["selected"] else r.get("skipped", ""))


if __name__ == "__main__":
    main(int(sys.argv[1]))
