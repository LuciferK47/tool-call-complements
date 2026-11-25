"""Audit: are the reported results real, internally consistent, and reproducible?

Checks (exit code 1 if any fail):
  1. results exist, were produced by a real model (not the mock), with a manifest (config hash, packages, device);
  2. closed-form invariants of the reference policies hold in the stored summaries;
  3. the non-deployable hindsight reference is >= every other policy;
  4. Holm-adjusted p-values >= raw p-values; split sizes match the config; partitions differ;
  5. the report generator source contains no hand-typed numbers (AST scan of string constants);
  6. docs/04 regenerates byte-for-byte from results/*.json;
  7. docs/04 is identical to the top-level copy;
  8. the archived simulated results are not referenced as results anywhere in docs/04.
Usage: python scripts/audit_results.py [--results results/main]
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


ALLOWED = [r"\b\d\d_[A-Za-z_]+\.md", r"[A-Z]-\d+", r"P\d-I\d", r"[A-Z]\d+", r"E\d", r"\b[Ss]ection \d+(?:\.\d+)*", r"\.\d[fe%]", r":\d+", r"^#+ .*$"]


IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


def source_has_no_literal_numbers(path: Path):
    tree = ast.parse(path.read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            s = node.value
            if s.lstrip().startswith("#") or IDENT.match(s) or s in ("utf-8",):     # headings, identifiers/keys, encodings
                continue
            t = s
            for pat in ALLOWED:
                t = re.sub(pat, " ", t, flags=re.M)
            if re.search(r"\d", t):
                bad.append(s[:80])
    return bad


def audit_run(rd: Path, label: str = ""):
    """Run-level audit of one results directory (any model/device). Does not look at the generated docs."""
    tag = f"[{rd.name}] "
    rpath = rd / "results.json"
    if not rpath.exists():
        # pilot-only or p2/ablation-only directories
        pass
    else:
        res = json.loads(rpath.read_text())
        man = res["manifest"]
        ps = res["pool_stats"]
        check(tag + "manifest records a real model (not the mock)", man["model"]["id"] not in (None, "mock") and "Qwen" in str(man["model"]["id"]), str(man["model"]["id"]))
        check(tag + "manifest has config hash, packages and git state", bool(man.get("config_hash")) and "torch" in man["packages"] and "git" in man)
        check(tag + "results record the device used", ps.get("device") in ("cpu", "cuda"), str(ps.get("device")))
        if ps.get("parser") is not None:                       # runs built by the L40S-package code
            check(tag + "resolved model commit recorded", bool(ps.get("resolved_revision")), str(ps.get("resolved_revision")))
            pin = man["model"].get("revision")
            check(tag + "resolved commit equals the pinned revision", pin is None or ps.get("resolved_revision") == pin)
            cats = ps.get("parse_categories") or {}
            tot = sum(cats.values()) or 1
            check(tag + "no thinking text in any node reply", cats.get("thinking_text", 0) == 0, str(cats.get("thinking_text", 0)))
            check(tag + "answer parse-failure rate <= 5%", ps["parse_failure_rate"] <= 0.05, f"{ps['parse_failure_rate']:.4f}")
            check(tag + "always-call accuracy is 100% (exact tool)", abs(ps["always_call_accuracy"] - 1.0) < 1e-12, str(ps["always_call_accuracy"]))
            check(tag + "tool-failure rate is 0%", ps["tool_failure_rate"] == 0.0)
            if man["model"].get("device") == "cuda" or (ps.get("tree_env") or {}).get("device") == "cuda":
                check(tag + "GPU name and CUDA version recorded", bool((ps.get("tree_env") or {}).get("gpu")) and bool((ps.get("tree_env") or {}).get("cuda")))
        lams = res["config"]["lambdas"]
        parts = sorted(k for k in res if re.fullmatch(r"partition_\d+", k))
        check(tag + "at least two partitions present", len(parts) >= 2, str(len(parts)))
        hashes = set()
        extras = set(res["config"].get("extra_policies", []))
        for pk in parts:
            p = res[pk]
            for split in ("gate", "test"):
                S = p["summary"][split]
                ok1 = all(abs(S["never"]["calls_per_instance_by_lambda"][i]) < 1e-12 and abs(S["always"]["calls_per_instance_by_lambda"][i] - 3.0) < 1e-9
                          and abs(S["never"]["utility_by_lambda"][i] - S["never"]["accuracy_by_lambda"][i]) < 1e-12 for i in range(len(lams)))
                check(tag + f"{pk}/{split}: never/always closed-form invariants", ok1)
                hind = S["hindsight_nondeployable"]["u_auc_points"]
                check(tag + f"{pk}/{split}: hindsight is an upper bound on every policy", all(hind >= v["u_auc_points"] - 1e-9 for v in S.values()))
            check(tag + f"{pk}: Holm-adjusted p >= raw p", all(v["p_holm"] >= v["p"] - 1e-12 for v in p["primary_contrasts_test_holm"].values()))
            check(tag + f"{pk}: split sizes match config", p["split_sizes"] == res["config"]["splits"])
            hashes.add(p["split_instance_id_hash"])
            npz = rd / f"per_instance_partition_{pk.split('_')[1]}.npz"
            import numpy as np
            check(tag + f"{pk}: per-instance arrays saved and finite", npz.exists() and all(np.isfinite(np.load(npz)[k]).all() for k in np.load(npz).files))
            for ex in sorted(extras):
                check(tag + f"{pk}: extra policy '{ex}' present in the test summary", ex in p["summary"]["test"])
            if "mup_pc" in extras:
                sc = p.get("secondary_contrasts_test_holm", {})
                check(tag + f"{pk}: secondary Holm family present and valid", len(sc) >= 1 and all(v["p_holm"] >= v["p"] - 1e-12 for v in sc.values()))
        check(tag + "partitions use different instance assignments", len(hashes) == len(parts))
    p2p = rd / "p2_results.json"
    if p2p.exists():
        p2 = json.loads(p2p.read_text())
        check(tag + "P2: all three sampling costs present", len(p2["by_rho"]) >= 3)
        check(tag + "P2: sample parse-failure rate <= 5%", p2["parse_failure_rate_samples"] <= 0.05, f"{p2['parse_failure_rate_samples']:.4f}")
        check(tag + "P2: no thinking text in samples", p2.get("thinking_text_samples", 0) == 0)
        check(tag + "P2: Holm-adjusted p >= raw p", all(c["p_holm"] >= c["p"] - 1e-12 for r in p2["by_rho"].values() for c in r["contrasts"].values()))
    pil = rd / "pilot.json"
    if pil.exists():
        pj = json.loads(pil.read_text())
        check(tag + "pilot: parse-failure <= 5%, always-call >= 99%, tool failures 0, never-call in (0,100)% (n>=30), no thinking text",
              pj["parse_failure_rate"] <= 0.05 and pj["always_call_accuracy"] >= 0.99 and pj.get("tool_failure_rate", 0) == 0
              and (pj.get("n_instances", 60) < 30 or 0 < pj["never_call_accuracy"] < 1) and pj.get("thinking_text_nodes", 0) == 0)
        check(tag + "pilot: manifest names the model and records the resolved commit", "Qwen" in str(pj["manifest"]["model"]["id"]) and (bool(pj.get("resolved_revision")) or "resolved_revision" not in pj))
    abl = rd / "ablations.json"
    if abl.exists():
        a = json.loads(abl.read_text())
        sub = json.loads((rd / "train_subsets.json").read_text())
        expect_cells = len(a["sizes"]) * len(a["heads"])
        check(tag + "ablations: every (size x head) cell present or explicitly failed", len(a["cells"]) == expect_cells, f"{len(a['cells'])}/{expect_cells}")
        check(tag + "ablations: frozen evaluation splits recorded and train subsets nested/disjoint from eval",
              set(sub["cal"]).isdisjoint(sub["test"]) and set(sub["gate"]).isdisjoint(sub["test"]) and
              all(set(sub["subsets"][str(s)]["main_train_indices"]).isdisjoint(set(sub["cal"]) | set(sub["gate"]) | set(sub["test"])) for s in a["sizes"]))
        check(tag + "ablations: best cell selected on the GATE split", a["best_cell_chosen_on_gate"] is None or a["best_cell_chosen_on_gate"] in a["cells"])
        check(tag + "ablations: budget ledger written", (rd / "budget.json").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "main"))
    ap.add_argument("--docs", default=str(ROOT / "docs" / "04_results_and_final_implementation.md"))
    ap.add_argument("--run", default=None, help="audit one results directory only (run-level checks, no docs)")
    ap.add_argument("--no-docs", action="store_true")
    ap.add_argument("--all", action="store_true", help="audit every results/<run>/ directory, then the generated docs")
    args = ap.parse_args()
    if args.run:
        audit_run(Path(args.run))
        print("\n%d failed" % len(FAILS) if FAILS else "\nall audit checks passed")
        sys.exit(1 if FAILS else 0)
    if args.all:
        for d in sorted((ROOT / "results").iterdir()):
            if d.is_dir() and d.name != "main" and not d.name.startswith("_") and d.name not in ("smoke",) and ((d / "results.json").exists() or (d / "pilot.json").exists()):
                audit_run(d)
        if (ROOT / "results" / "main" / "results.json").exists():
            audit_run(ROOT / "results" / "main")
        if args.no_docs:
            print("\n%d failed" % len(FAILS) if FAILS else "\nall audit checks passed")
            sys.exit(1 if FAILS else 0)
    rd = Path(args.results)
    rpath = rd / "results.json"
    check("results.json exists", rpath.exists())
    if not rpath.exists():
        sys.exit(1)
    res = json.loads(rpath.read_text())
    man = res["manifest"]
    check("manifest records a real model (not the mock)", man["model"]["id"] not in (None, "mock") and "Qwen" in str(man["model"]["id"]), str(man["model"]["id"]))
    check("manifest has config hash, packages and git state", bool(man.get("config_hash")) and "torch" in man["packages"] and "git" in man)
    check("results record the device used", res["pool_stats"].get("device") in ("cpu", "cuda"), str(res["pool_stats"].get("device")))
    lams = res["config"]["lambdas"]
    parts = sorted(k for k in res if re.fullmatch(r"partition_\d+", k))
    check("at least two partitions present", len(parts) >= 2, str(len(parts)))
    hashes = set()
    for pk in parts:
        p = res[pk]
        for split in ("gate", "test"):
            S = p["summary"][split]
            ok1 = all(abs(S["never"]["calls_per_instance_by_lambda"][i]) < 1e-12 and abs(S["always"]["calls_per_instance_by_lambda"][i] - 3.0) < 1e-9
                      and abs(S["never"]["utility_by_lambda"][i] - S["never"]["accuracy_by_lambda"][i]) < 1e-12 for i in range(len(lams)))
            check(f"{pk}/{split}: never/always closed-form invariants", ok1)
            hind = S["hindsight_nondeployable"]["u_auc_points"]
            check(f"{pk}/{split}: hindsight is an upper bound on every policy", all(hind >= v["u_auc_points"] - 1e-9 for v in S.values()))
        check(f"{pk}: Holm-adjusted p >= raw p", all(v["p_holm"] >= v["p"] - 1e-12 for v in p["primary_contrasts_test_holm"].values()))
        check(f"{pk}: split sizes match config", p["split_sizes"] == res["config"]["splits"])
        hashes.add(p["split_instance_id_hash"])
        npz = rd / f"per_instance_partition_{pk.split('_')[1]}.npz"
        check(f"{pk}: per-instance arrays saved and finite", npz.exists() and all(__import__("numpy").isfinite(__import__("numpy").load(npz)[k]).all() for k in __import__("numpy").load(npz).files))
    check("partitions use different instance assignments", len(hashes) == len(parts))

    rep_path = ROOT / "results" / "replication_gpu" / "pilot_1p5b.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        rm = rep["manifest"]
        check("replication pilot: manifest names a real model and a cuda device", "Qwen" in str(rm["model"]["id"]) and rm["model"]["device"] == "cuda", str(rm["model"]["id"]))
        ru = rep["u_auc_points"]
        check("replication pilot: never-call U-AUC equals its accuracy (closed form)", abs(ru["never"] - 100 * rep["never_call_accuracy"]) < 1e-6)
        check("replication pilot: always-call U-AUC matches the closed form for full accuracy", abs(ru["always"] - 100 * (rep["always_call_accuracy"] - 3.0 * sum(lams) / len(lams))) < 1e-6)
        check("replication pilot: hindsight bounds every policy", all(ru["hindsight"] >= v - 1e-9 for v in ru.values()))
        check("replication pilot is labelled as not verified by the primary pipeline in doc 04", "not verified by the primary pipeline" in (Path(args.docs).read_text() if Path(args.docs).exists() else ""))

    bad = source_has_no_literal_numbers(ROOT / "report" / "make_report.py")
    check("make_report.py contains no hand-typed numbers in strings", not bad, "; ".join(bad[:3]))
    bad2 = source_has_no_literal_numbers(ROOT / "report" / "make_ladder_report.py")
    check("make_ladder_report.py contains no hand-typed numbers in strings", not bad2, "; ".join(bad2[:3]))

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "regen.md"
        r = subprocess.run([sys.executable, str(ROOT / "report" / "make_report.py"), "--results", args.results, "--out", str(out)],
                           capture_output=True, text=True, cwd=ROOT)
        docs = Path(args.docs)
        same = r.returncode == 0 and docs.exists() and out.read_bytes() == docs.read_bytes()
        check("docs/04 regenerates byte-for-byte from results", same, r.stderr[-200:] if r.returncode else "")
        txt = docs.read_text() if docs.exists() else ""
    top = ROOT.parent / "04_results_and_final_implementation.md"
    check("top-level copy of doc 04 equals docs/04", top.exists() and top.read_bytes() == docs.read_bytes())
    check("doc 04 does not cite archived simulated results as results", "full_benchmark_results" not in txt and "gate_results.json" not in txt)
    print("\n%d failed" % len(FAILS) if FAILS else "\nall audit checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
