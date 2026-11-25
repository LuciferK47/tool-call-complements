# Tool calls are complements

Code for the paper *Tool Calls Are Complements: Why Per-Step Call Rules Fail on Multi-Step Tasks* (Shah, Singh, Rana, Gadkari).

A model decides at each step of a multi-step task whether to pay for an exact tool call. The task pays only if every step is right, so calls are
complements. The repository contains the evaluation pipeline (exhaustive outcome trees from real generations), the probes and baselines of the
earlier iterations, the horizon study with the belief-state dynamic program, and the scripts that regenerate every number, table and figure.

| Path | Content |
|---|---|
| `toolcomp/tree.py`, `envs/`, `models/`, `tools/` | outcome trees: every action pattern of a task replayed exactly (greedy decoding, exact SymPy tool) |
| `toolcomp/deep_tree.py` | trees for 6-step arithmetic and 2-4 hop MuSiQue questions (BM25 retrieval as a noisy tool) |
| `toolcomp/probes.py`, `estimators.py`, `features.py` | the two-head marginal-utility probe (MUP), confidence thresholds, matched signal x estimator matrix |
| `toolcomp/horizon.py` | composite tasks, the per-step rules, Belief-DP, closed-form predictions |
| `toolcomp/scaling.py` | loaders for every kind of stored tree; regret-versus-x analysis |
| `scripts/` | tree building, Iteration 1-2 analyses, `run_scaling.py`, `run_label_efficiency.py`, `check_predictions.py`, `make_paper_assets.py`, `run_gpu.py` |
| `configs/` | run configurations (copies of the frozen ones; comments refer to documents not included here) and `predictions.yaml` |
| `data/` | reply texts of five 3-step runs (`trees_text/`), the pre-registered 0.5B contrasts |
| `results/` | horizon results for seven runs, label efficiency, scored predictions |
| `ideation/` | the deterministic parts of the Task 3 ideation pipeline |
| `tests/` | unit tests; the theory propositions are checked against brute force |

## Reproduce

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest tests -q
python scripts/run_scaling.py --kind text --tag q35_2b --model Qwen3.5-2B --sidecar data/trees_text/q35_2b.json --seed 20261101   # surface signal, any 3-step run
python scripts/check_predictions.py && python scripts/make_paper_assets.py --out paper_assets
```
Full trees (hidden states, draft confidence) for the 0.5B and 0.8B runs are about 70 MB each and are not stored here; with them
`run_scaling.py --kind tree` reproduces the stored numbers exactly. New trees need a GPU: `python scripts/run_gpu.py --smoke`, then `--profile l40s` or `--profile rtx4050`.

## Predictions

`configs/predictions.yaml` and `configs/predictions_protocol.yaml` were committed before the 2B, 4B, 9B, fresh-pool and hidden-operand runs were analysed
(see the commit history); `scripts/check_predictions.py` scores them and writes `results/predictions_scored.json`.
