# Tool calls are complements

Code for the paper *Tool Calls Are Complements: Why Per-Step Call Rules Fail on Multi-Step Tasks* (Singh, Rana, Shah, Gadkari).

A model decides at each step of a multi-step task whether to pay for an exact tool call. The task pays only if every step is right, so calls are
complements. The repository contains the evaluation pipeline (exhaustive outcome trees from real generations), the probes and baselines of the
earlier iterations, the horizon study with the belief-state dynamic program, and the scripts that regenerate every number, table and figure.

| Path | Content |
|---|---|
| `toolcomp/tree.py`, `envs/`, `models/`, `tools/` | outcome trees: every action pattern of a task replayed exactly (greedy decoding, exact SymPy tool) |
| `toolcomp/deep_tree.py` | trees for 6-step arithmetic and 2-4 hop MuSiQue questions (BM25 retrieval as a noisy tool) |
| `toolcomp/probes.py`, `estimators.py`, `features.py` | the two-head marginal-utility probe (MUP), confidence thresholds, matched signal x estimator matrix |
| `toolcomp/horizon.py` | composite tasks, the per-step rules, Belief-DP (exact tool, and the noisy-tool form with measured tool accuracy), closed-form and signal-aware predictions |
| `toolcomp/scaling.py` | loaders for every kind of stored tree; regret-versus-x analysis |
| `scripts/` | tree building, Iteration 1-2 analyses, `run_scaling.py`, `run_label_efficiency.py`, `check_predictions.py`, `make_paper_assets.py`, `run_gpu.py` |
| `configs/` | run configurations (copies of the frozen ones; comments refer to documents not included here) and `predictions.yaml` |
| `data/` | reply texts of five 3-step runs (`trees_text/`), compact arrays of the RTX 4050 runs (`gpu_rtx4050/`) and the L40S runs (`gpu_l40s/`): 6-step chains and fresh pools (all signals), the pre-registered 0.5B contrasts |
| `results/` | horizon results for 37 runs (twelve 3-step; five 6-step chains at the committed and, `_matched`, at half the per-call price; fifteen MuSiQue), label efficiency, MuSiQue summary, scored predictions |
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

```
python scripts/run_scaling.py --kind deep --tag arith6_q35_9b --model Qwen3.5-9B --deep data/gpu_l40s/deep/arith6_q35_9b.npz                      # 6-step chain, committed price
python scripts/run_scaling.py --kind deep --tag arith6_q35_9b_matched --model Qwen3.5-9B --deep data/gpu_l40s/deep/arith6_q35_9b.npz --price-scale 0.5   # half price
python scripts/run_scaling.py --kind light --tag q35_9b_seed20261201 --model Qwen3.5-9B --light data/gpu_l40s/light/light_q35_9b_seed20261201.npz --seed 20261201
python scripts/run_scaling.py --kind deep --tag musique_k2_q35_9b --deep <musique_k2_q35_9b.npz> --ms 1 2 4 --reps 20 --noisy-tool                    # MuSiQue; needs the rebuilt tree
```
`--noisy-tool` adds the rule `belief_dp_noisy` (Belief-DP with the tool accuracy measured on the training split); the nine rules of the study and every stored number are unchanged.

## Predictions

`configs/predictions.yaml` and `configs/predictions_protocol.yaml` were committed before the 2B, 4B, 9B, fresh-pool and hidden-operand runs were analysed
(see the commit history); `scripts/check_predictions.py` scores them and writes `results/predictions_scored.json`.

`configs/predictions_v2.yaml` (signal-aware prediction; commit `28277f1`) was committed before that prediction was computed on the runs it names as unseen; it is post hoc with respect to the first set. MuSiQue trees are not stored (the dataset text is not redistributed); rebuild them with `scripts/run_gpu.py`, then `scripts/musique_summary.py --deep <dir> [<dir> ...]`. The results of all 15 MuSiQue runs are in `results/scaling/` and `results/musique_summary.json`.

**Fix after scoring (6-step features).** The 6-step surface features had a flag for a missing input (`-1` in the stored arrays) that also fired on real intermediate results of `-1`; the probe then rated some steps near-certain, and Belief-DP trailed the best label-free rule by up to 10.4 points. The flag is removed (`toolcomp/scaling.py::deep_surface`, `tests/test_deep_surface.py`) and the ten 6-step analyses were re-run and re-scored; no prediction changed outcome. The earlier results are in `results/scaling_before_input_flag_fix/`.

The second set was scored on eleven runs it had not been computed on (`results/predictions_scored.json`: `v2_as_written_all_unseen`, 192 cells; `v2_l40s_completely_unseen`, 126 cells, where the last criterion fails; and the readings at half price and with MuSiQue counted). The noisy-tool Belief-DP was added after the MuSiQue failure was seen and is not part of either prediction set.
