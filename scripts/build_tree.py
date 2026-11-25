"""Build (or resume) the exhaustive outcome tree with the real model and tool. OFFLINE cost ledger.

Usage
  python scripts/build_tree.py --config configs/primary.yaml                     main pool (resumable per level)
  python scripts/build_tree.py --config configs/gpu_0p5b.yaml --pool extra --chunk 0   one checkpointed chunk of the extra training pool
  (config key `legacy_cache: {dir, tag}` re-scores previously generated nodes with the configured parser and generates only changed prompts)
  --n N  smoke test: use only the first N instances (never used for results)
Checkpoints are fingerprinted (model, resolved revision, dtype, device, prompt, layers, parser): a mismatch aborts instead of resuming.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import ROOT, append_budget, make_runner, manifest, write_json
from toolcomp.config import load_config
from toolcomp.envs.arith import dataset_hash, generate_instances
from toolcomp.tree import TreeBuilder, TreeData


class _Stub:
    """Runner stand-in used only to replay prompts from legacy checkpoints (never generates)."""
    def __init__(self, cfg, hidden_dim):
        self.model_id, self.revision, self.device, self.system = cfg["model"]["id"], cfg["model"].get("revision"), "cpu", "Answer with a number."
        self.hidden_dim = hidden_dim
        self.dtype_name = cfg["model"].get("dtype")

    def run(self, *a, **k):
        raise RuntimeError("legacy replay must not generate")

    hidden_only = run


def pools(cfg):
    ins = generate_instances(cfg["pool"]["n"], cfg["pool"]["seed"], cfg["pool"]["prefix"])
    return ins


def assert_disjoint_from_pilot(cfg, ins):
    pilot_cfg = load_config(ROOT / "configs" / "pilot.yaml")
    pil = {x.problem_text() for x in generate_instances(pilot_cfg["pool"]["n"], pilot_cfg["pool"]["seed"], "pilot")}
    overlap = sum(x.problem_text() in pil for x in ins)
    assert overlap == 0, f"{overlap} instances coincide with pilot instances"
    return overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "primary.yaml"))
    ap.add_argument("--out", default=None, help="results subfolder (default: config name)")
    ap.add_argument("--pool", choices=("main", "extra"), default="main")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n", type=int, default=None, help="smoke test: first N instances only")
    ap.add_argument("--cache-root", default=str(ROOT / "cache"))
    ap.add_argument("--reveal", choices=("full", "incremental"), default="full", help="operand visibility mode")
    ap.add_argument("--seed", type=int, default=None, help="override instance generator seed")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = args.out or cfg["name"]
    croot = Path(args.cache_root)
    parser = cfg.get("answer_parser", "first_int")
    if args.pool == "main":
        pool_seed = args.seed if args.seed is not None else cfg["pool"]["seed"]
        ins = generate_instances(cfg["pool"]["n"], pool_seed, cfg["pool"]["prefix"])
        tag = cfg["name"]
        if args.seed is not None:
            tag += f"_seed{args.seed}"
        if args.reveal != "full":
            tag += f"_{args.reveal}"
        cache = croot / tag
        exp = cfg.get("expected_dataset_hash")
        if args.n is None and args.seed is None and exp:
            assert dataset_hash(ins) == exp, f"dataset hash {dataset_hash(ins)} != expected {exp}"
    else:
        ex = cfg["extra_pool"]
        allx = generate_instances(ex["n"], ex["seed"], ex["prefix"])
        main_ins = pools(cfg)
        mt = {x.problem_text() for x in main_ins} | {x.instance_id for x in main_ins}
        assert not any(x.problem_text() in mt or x.instance_id in mt for x in allx), "extra pool overlaps the main pool"
        ins = allx[args.chunk * ex["chunk"]:(args.chunk + 1) * ex["chunk"]]
        cache, tag = croot / (cfg["name"] + "_extra"), f"{cfg['name']}_xchunk{args.chunk}"
    if args.n:
        ins = ins[:args.n]
        tag += f"_n{args.n}"
    overlap = assert_disjoint_from_pilot(cfg, ins)
    runner = make_runner(cfg)
    reuse = reuse_root = None
    if cfg.get("legacy_cache") and args.pool == "main":
        lc = cfg["legacy_cache"]
        h = TreeBuilder(_Stub(cfg, runner.hidden_dim), cfg["layers"], batch_size=cfg["batch_size"], max_new_tokens=cfg["max_new_tokens"],
                        parser="first_int", legacy_ok=True, log=lambda *_: None)
        h.build(ins, ROOT / lc["dir"], lc["tag"])
        reuse, reuse_root = h.seen, h.seen_root
        print(f"harvested {len(reuse)} cached node outputs and {len(reuse_root)} root states from {lc['dir']}")
    t0 = time.perf_counter()
    if hasattr(__import__("torch").cuda, "reset_peak_memory_stats") and __import__("torch").cuda.is_available():
        __import__("torch").cuda.reset_peak_memory_stats()
    tb = TreeBuilder(runner, cfg["layers"], batch_size=cfg["batch_size"], max_new_tokens=cfg["max_new_tokens"], parser=parser,
                     reuse=reuse, reuse_root=reuse_root, reveal=args.reveal)
    tree = tb.build(ins, cache, tag)
    tree.meta["dataset_hash"] = dataset_hash(ins)
    wall = time.perf_counter() - t0
    tree.meta["total_seconds_this_invocation"] = wall
    tree.save(cache / f"{tag}_tree.npz")
    peak = None
    try:
        import torch
        if torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    except Exception:
        pass
    append_budget(ROOT / "results" / out / "budget.json",
                  {"stage": "tree" if args.pool == "main" else f"extra-tree chunk {args.chunk}", "run": cfg["name"], "n_instances": tree.n,
                   "wall_seconds": wall, "gpu_seconds": wall if runner.device == "cuda" else 0.0, "peak_gpu_mem_gb": peak,
                   "nodes_generated": tb.n_generated, "nodes_reused": tb.n_reused})
    write_json(ROOT / "results" / out / ("tree_manifest.json" if args.pool == "main" else f"tree_manifest_extra_chunk{args.chunk}.json"),
               manifest(cfg, {"dataset_hash": tree.meta["dataset_hash"], "n_instances": tree.n, "pilot_overlap": overlap,
                              "pool": args.pool, "chunk": args.chunk, "parser": parser, "env": tree.meta.get("env"),
                              "nodes_generated": tb.n_generated, "nodes_reused": tb.n_reused,
                              "parse_categories": tree.meta.get("parse_categories"),
                              "offline_label_generation_seconds_this_invocation": wall}))
    print("tree saved:", cache / f"{tag}_tree.npz", "| instances:", tree.n, "| generated:", tb.n_generated, "| reused:", tb.n_reused)


if __name__ == "__main__":
    main()
