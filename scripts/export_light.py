#!/usr/bin/env python3
"""Compact export of outcome trees (a few MB instead of ~100 MB): everything the horizon study needs except the full hidden states.

    python scripts/export_light.py --all --out out/light            # every cache/*/*_tree.npz
    python scripts/export_light.py --tree cache/q35_2b/q35_2b_tree.npz --out out/light/light_q35_2b.npz

Stored per node: leaf correctness, local correctness, running-value sizes, draft confidence (mean log-prob, first-token entropy), token counts,
the first two generated tokens' log-probs when the level checkpoints are present, and a 64-d PCA (unsupervised, fitted on all nodes of the run)
of the hidden state at the last stored layer.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolcomp.tree import N_NODES, T_STEPS, TreeData


def s3_logprobs(cache_dir: Path, tag: str, n: int):
    lp = np.zeros((n, N_NODES, 2), np.float32)
    have = np.zeros((n, N_NODES), np.float32)
    for lv in range(T_STEPS):
        ck = cache_dir / f"{tag}_level{lv}.pkl"
        if not ck.exists():
            return None
        saved = pickle.load(open(ck, "rb"))
        ids = [(i, 2 ** lv - 1 + p) for i in range(n) for p in range(2 ** lv)]
        for (i, k), o in zip(ids, saved["outs"]):
            g = getattr(o, "gen_token_logprobs", None) or []
            for j in range(min(2, len(g))):
                lp[i, k, j] = g[j]
            have[i, k] = len(g)
    return lp, have


def export(npz: Path, out: Path, n_pc: int = 64):
    tree = TreeData.load(npz)
    tag = npz.stem.replace("_tree", "")
    H = tree.hidden[:, :, -1, :].astype(np.float32).reshape(-1, tree.hidden.shape[-1])
    mu = H.mean(0)
    _, _, Vt = np.linalg.svd(H - mu, full_matrices=False)
    k = min(n_pc, Vt.shape[0])
    pcs = np.zeros((H.shape[0], n_pc), np.float32)
    pcs[:, :k] = (H - mu) @ Vt[:k].T
    pcs = pcs.reshape(tree.n, N_NODES, n_pc).astype(np.float16)
    extra = {}
    s3 = s3_logprobs(npz.parent, tag, tree.n)
    if s3 is not None:
        extra["s3_logprobs"], extra["s3_have"] = s3
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, hidden_pca=pcs, leaf_correct=tree.leaf_correct, local_no_correct=tree.local_no_correct, v_in=tree.v_in,
                        v_in_ok=tree.v_in_ok, mean_logprob=tree.mean_logprob.astype(np.float32), entropy0=tree.entropy0.astype(np.float32),
                        n_gen_tokens=tree.n_gen_tokens.astype(np.int32), step_bin=tree.step_bin, **extra)
    out.with_suffix(".json").write_text(json.dumps({"tag": tag, "instance_ids": tree.instance_ids, "layer_used": tree.layers[-1],
                                                    "meta": {k: tree.meta.get(k) for k in ("model_id", "resolved_revision", "dtype", "device", "dataset_hash", "parser", "n")}}))
    print(f"{tag}: n={tree.n} -> {out} ({out.stat().st_size / 2 ** 20:.1f} MiB){' [with S3 log-probs]' if s3 is not None else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-root", default=str(ROOT / "cache"))
    a = ap.parse_args()
    if a.all:
        found = sorted(Path(a.cache_root).glob("*/*_tree.npz"))
        if not found:
            print("no trees found under", a.cache_root)
        for p in found:
            export(p, Path(a.out) / f"light_{p.stem.replace('_tree', '')}.npz")
    else:
        export(Path(a.tree), Path(a.out))


if __name__ == "__main__":
    main()
