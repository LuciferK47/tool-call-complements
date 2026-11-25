"""Independent verification of a built outcome tree (pre-flight check 'training labels').

Recomputes, from the RAW reply texts and exact Python arithmetic ONLY (no SymPy, no tree code paths), every node's input value, tool value,
parsed answer and every leaf's correctness, and compares with the stored arrays. Also checks node labels against an explicit recursion.
Usage: python scripts/verify_tree.py --tree cache/<name>/<name>_tree.npz --config configs/<name>.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcomp.envs.arith import apply_op, generate_instances
from toolcomp.tree import PARSERS, PREFIX_OF, LEVEL_OF, N_NODES, TreeData, child, node_labels


def verify_tree(tree: TreeData, instances, parser: str = None, max_report: int = 5) -> dict:
    parse = PARSERS[parser or tree.meta.get("parser", "first_int")]
    assert tree.texts is not None, "tree has no raw texts (built by older code)"
    by_id = {x.instance_id: x for x in instances}
    bad = []
    for i, iid in enumerate(tree.instance_ids):
        ins = by_id[iid]
        v_in = {0: ins.first}
        v_call, v_no = {}, {}
        for k in range(N_NODES):
            lv = int(LEVEL_OF[k])
            vi = v_in.get(k)
            parsed = parse(tree.texts[i][k])
            v_no[k] = parsed
            v_call[k] = None if vi is None else apply_op(vi, ins.ops[lv], ins.rights[lv])
            if lv < 2:
                v_in[child(k, 0)] = parsed
                v_in[child(k, 1)] = v_call[k]
            # stored arrays
            if (vi is not None) != bool(tree.v_in_ok[i, k]) or (vi is not None and vi != int(tree.v_in[i, k])):
                bad.append(("v_in", iid, k))
            if (v_call[k] is not None) != bool(tree.v_call_ok[i, k]) or (v_call[k] is not None and v_call[k] != int(tree.v_call[i, k])):
                bad.append(("v_call", iid, k))
            if (parsed is not None) != bool(tree.v_no_ok[i, k]) or (parsed is not None and parsed != int(tree.v_no[i, k])):
                bad.append(("v_no", iid, k))
        for leaf in range(8):
            a = [(leaf >> 2) & 1, (leaf >> 1) & 1, leaf & 1]
            nid = 0
            for t in range(2):
                nid = child(nid, a[t])
            final = v_call[nid] if a[2] else v_no[nid]
            correct = final is not None and final == ins.final_gold
            if correct != bool(tree.leaf_correct[i, leaf]):
                bad.append(("leaf", iid, leaf))
    # labels vs explicit recursion (always-call continuation)
    yc, yn = node_labels(tree, 1)
    for i in range(0, tree.n, max(1, tree.n // 50)):
        for k in range(N_NODES):
            lv, pf = int(LEVEL_OF[k]), int(PREFIX_OF[k])
            for a, y in ((1, yc), (0, yn)):
                prefix = pf * 2 + a
                for _ in range(lv + 1, 3):
                    prefix = prefix * 2 + 1
                if float(tree.leaf_correct[i, prefix]) != y[i, k]:
                    bad.append(("label", tree.instance_ids[i], k))
    # always-call must be exact arithmetic
    always_ok = bool(tree.leaf_correct[:, 7].all())
    return {"ok": not bad and always_ok, "n_bad": len(bad), "examples": bad[:max_report], "always_call_all_correct": always_ok, "n_instances": tree.n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool", default="main")
    ap.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args()
    from toolcomp.config import load_config
    cfg = load_config(a.config)
    t = TreeData.load(a.tree)
    pool = cfg["pool"] if a.pool == "main" else cfg["extra_pool"]
    ins = generate_instances(pool["n"], pool["seed"], pool["prefix"])
    rep = verify_tree(t, ins)
    print(rep)
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
