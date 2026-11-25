"""Append one record per executed ideation stage to logs/stage_log.jsonl.

Usage: python -I log_stage.py STAGE ROLE PROBLEM INPUTS OUTPUT [NOTE]
INPUTS is a comma-separated list of files. Records are what the report's stage counts are computed from.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "stage_log.jsonl"


def main(argv):
    stage, role, problem, inputs, output = argv[1:6]
    note = argv[6] if len(argv) > 6 else ""
    out = ROOT / output
    if not out.exists():
        sys.exit(f"output file missing: {output}")
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": stage, "role": role, "problem": problem,
        "inputs": [i for i in inputs.split(",") if i],
        "output": output, "output_bytes": out.stat().st_size,
        "executor": "in-session, one Claude model playing the role; no external API",
        "model": os.environ.get("MODEL", "not recorded (session model setting changed during the run)"),
        "note": note,
    }
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec["stage"], rec["role"], rec["problem"], "->", output)


if __name__ == "__main__":
    main(sys.argv)
