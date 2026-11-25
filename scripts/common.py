"""Shared helpers for the run scripts (paths are relative to the repo root; works on Linux and Windows)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_runner(cfg: dict):
    import os
    if os.environ.get("T4_MOCK") == "1":            # package self-test only (run_all.py --mock): never used for results
        sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
        from mock_runner import MockRunner
        r = MockRunner()
        r.model_id = cfg["model"]["id"]
        r.revision = cfg["model"].get("revision")
        r.resolved_revision = cfg["model"].get("revision") or "mock"
        return r
    from toolcomp.models.real_model import HFRunner
    m = cfg["model"]
    return HFRunner(model_id=m["id"], device=m.get("device", "auto"), dtype=m.get("dtype", "float32"),
                    revision=m.get("revision"), threads=m.get("threads"))


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = run("git", "rev-parse", "HEAD")
    return {"commit": commit or "not a git repository", "dirty": bool(run("git", "status", "--porcelain")) if commit else None}


def manifest(cfg: dict, extra: dict | None = None) -> dict:
    import numpy, scipy, sklearn, torch, transformers
    m = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_hash": cfg.get("_config_hash"),
         "config_path": cfg.get("_config_path"), "git": git_state(),
         "python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": __import__("os").cpu_count(),
         "packages": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": numpy.__version__,
                      "scipy": scipy.__version__, "sklearn": sklearn.__version__},
         "model": cfg.get("model"), "answer_parser": cfg.get("answer_parser", "first_int")}
    try:
        import torch as _t
        if _t.cuda.is_available():
            m["gpu"] = {"name": _t.cuda.get_device_name(0), "cuda": _t.version.cuda, "cudnn": _t.backends.cudnn.version()}
    except Exception:
        pass
    m.update(extra or {})
    return m


def append_budget(path: Path, entry: dict) -> None:
    """Compute-budget ledger (wall-clock, GPU-seconds, peak memory per stage/run/chunk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(path.read_text()) if path.exists() else []
    entry = dict(entry, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    cur.append(entry)
    path.write_text(json.dumps(cur, indent=2, default=float))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
