"""Orchestrator logic: failed validation blocks dependants, a failed pilot blocks only its own model, preflight gates every heavy stage,
the checklist lists every unit. Uses run_all.py with its stage implementations replaced by stubs (no model, no subprocesses)."""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAND = [Path(os.environ["T4_RUN_ALL"])] if os.environ.get("T4_RUN_ALL") else []
CAND += [ROOT.parent / "run_all.py"]
RUN_ALL = next((p for p in CAND if p.exists()), None)
pytestmark = pytest.mark.skipif(RUN_ALL is None, reason="run_all.py not next to tool_alloc (development tree)")


@pytest.fixture()
def ra(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_all_mod", RUN_ALL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ctx = m.Ctx(tmp_path, True, 1, ROOT)
    return m, ctx


def _stub(m, fail=()):
    calls = []
    for st in m.STAGES:
        def impl(ctx, run, st=st):
            calls.append((st, run))
            return ((st, run) not in fail), "stub"
        m.IMPL[st] = impl
    return calls


def test_checklist_lists_every_unit(ra):
    m, ctx = ra
    txt = m.checklist(ctx)
    n_units = sum(len(v) for v in m.UNITS.values())
    assert txt.count("\n") >= n_units
    for st in m.STAGES:
        assert st in txt
    assert "pending" in txt and f"0/{n_units} units validated" in txt


def test_failed_pilot_blocks_only_its_own_model(ra):
    m, ctx = ra
    calls = _stub(m, fail={("pilot", "q35_4b")})
    for st in ("check", "tests", "download", "loadtest", "smoke", "pilot", "preflight", "tree"):
        m.run_stage(ctx, st)
    assert ctx.get("pilot", "q35_4b")["status"] == "failed"
    assert ctx.get("tree", "q35_4b")["status"] == "blocked"
    for r in ("gpu_0p5b", "q35_0p8b", "q35_2b", "q35_9b", "primary_strict"):
        assert ctx.ok("tree", r), r
    assert ("tree", "q35_4b") not in calls


def test_failed_preflight_blocks_every_heavy_stage(ra):
    m, ctx = ra
    calls = _stub(m, fail={("preflight", None)})
    for st in m.STAGES:
        if st in ("verify-primary", "bundle"):
            continue
        m.run_stage(ctx, st)
    heavy = [(s, r) for s, r in calls if s in ("tree", "extra-tree", "analyse", "ablate", "p2", "report")]
    assert heavy == []


def test_failed_validation_blocks_next_stage_but_not_independent_units(ra):
    m, ctx = ra
    calls = _stub(m, fail={("tree", "q35_2b")})
    for st in ("check", "tests", "download", "loadtest", "smoke", "pilot", "preflight", "tree", "extra-tree", "analyse", "ablate", "p2", "report"):
        m.run_stage(ctx, st)
    assert ctx.get("extra-tree", "q35_2b")["status"] == "blocked" and ctx.get("ablate", "q35_2b")["status"] == "blocked"
    assert ctx.get("p2", "q35_2b")["status"] == "blocked"
    assert ctx.ok("ablate", "q35_9b") and ctx.ok("p2", "gpu_0p5b")
    assert ctx.get("report", None)["status"] == "blocked"        # the report needs every main run's tree and analysis


def test_resume_skips_validated_units(ra):
    m, ctx = ra
    calls = _stub(m)
    m.run_stage(ctx, "check")
    n = len(calls)
    m.run_stage(ctx, "check")
    assert len(calls) == n
    m.run_stage(ctx, "check", force=True)
    assert len(calls) == n + 1


def test_new_stages_present_and_ordered(ra):
    m, _ = ra
    for st in ("solver-eq", "posthoc"):
        assert st in m.STAGES and st in m.UNITS and st in m.IMPL
    assert m.STAGES.index("posthoc") < m.STAGES.index("report")
    assert ("analyse", "q35_9b") in m.deps("posthoc", "q35_9b")


def test_failed_analysis_blocks_posthoc_and_report(ra):
    m, ctx = ra
    _stub(m, fail={("analyse", "q35_2b")})
    for st in ("check", "tests", "download", "loadtest", "smoke", "pilot", "preflight", "tree", "analyse", "posthoc", "report"):
        m.run_stage(ctx, st)
    assert ctx.get("posthoc", "q35_2b")["status"] == "blocked"
    assert ctx.get("report", None)["status"] == "blocked"
    assert ctx.ok("posthoc", "q35_9b")


def test_adopted_tree_satisfies_dependants_without_rebuilding(ra):
    m, ctx = ra
    calls = _stub(m)
    ctx.set("tree", "q35_2b", "validated", "ADOPTED from earlier run, re-verified: stub")
    for st in ("check", "tests", "download", "loadtest", "smoke", "pilot", "preflight", "tree", "analyse"):
        m.run_stage(ctx, st)
    assert ("tree", "q35_2b") not in calls and ("analyse", "q35_2b") in calls
