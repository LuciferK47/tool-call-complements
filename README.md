# Tool calls are complements

Code for the paper *Tool Calls Are Complements: Why Per-Step Call Rules Fail on Multi-Step Tasks*.

Models decide, at each step of a multi-step task, whether to pay for an exact tool call. The task pays only if every step is right,
so one call raises the value of the others. This repository contains

* `toolcomp/` — exhaustive outcome trees built from real model generations (`tree.py`, `envs/`, `models/`, `tools/`), the
  probes and estimators of Iterations 1–2 (`probes.py`, `estimators.py`, `features.py`), and the horizon study with the belief-state
  dynamic program (`horizon.py`).
* `scripts/` — tree building, Iteration-1/2 analyses, horizon study (`run_horizon.py`), figure/table generation.
* `configs/` — frozen run configurations and `predictions.yaml` (committed before the scaling runs were analysed).
* `tests/` — unit tests (theory propositions are checked against brute force).
* `data/` — small result files used by the paper.

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest tests -q
python scripts/make_paper_assets.py      # regenerates every number, table and figure in the paper
```
Building new trees needs a GPU: see `scripts/run_gpu.py --help`.
