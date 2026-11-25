"""Exhaustive outcome tree for E1 (T=3) built from REAL model generations and REAL tool calls.

Because decoding is greedy and the tool is exact, an instance's outcome is a deterministic function of the
action prefix. For T=3 there are 7 decision nodes (1 + 2 + 4) and 8 leaves. We run the model once per node
(its NOCALL answer) and the tool once per node (its CALL answer) and store everything; afterwards every
policy, label and reference is a lookup (no further model calls). See docs/01 A-30 (design) and A-31 (labels).

Layout
  node id (heap order): level t in {0,1,2}, prefix p in [0, 2^t)  ->  id = (2^t - 1) + p, where the prefix is the
  binary number a1 a2 .. (1 = CALL). The child of node `id` under action a is  2*id + 1 + a  (NOCALL first).
  leaf index (pattern) = a1*4 + a2*2 + a3.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .envs.arith import (Instance, T_STEPS, apply_op, finish_with_tool, node_prompt, root_prompt)
from .tools.sympy_tool import SymPyTool

N_NODES = 2 ** T_STEPS - 1      # 7
N_LEAVES = 2 ** T_STEPS         # 8
LEVEL_OF = np.array([0, 1, 1, 2, 2, 2, 2])
PREFIX_OF = np.array([0, 0, 1, 0, 1, 2, 3])


def node_id(level: int, prefix: int) -> int:
    return (2 ** level - 1) + prefix


def child(nid: int, action: int) -> int:
    """Heap child: action 0 (NOCALL) -> 2*nid+1, action 1 (CALL) -> 2*nid+2."""
    return 2 * nid + 1 + action


def parse_int(text: str) -> Optional[int]:
    """Lenient parser used by the frozen primary run: the FIRST integer-looking token (decimals truncated). Kept unchanged."""
    m = re.search(r"-?\d[\d,]*", text)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


_INT = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)"          # plain integer or correctly grouped thousands
_THINK = re.compile(r"<think>|</think>|thinking process", re.I)


def parse_category(text: str) -> str:
    """Why a reply does or does not parse under the strict rule (counted in pool stats and the parser report)."""
    s = text.strip()
    if _THINK.search(s):
        return "thinking_text"
    core = s.rstrip(".").strip()
    if re.fullmatch(_INT, core):
        return "integer"
    if "=" in core and re.fullmatch(_INT, core.rsplit("=", 1)[1].strip().rstrip(".").strip()):
        return "integer_after_equals"
    if re.search(r"\d\.\d", core):
        return "decimal"
    nums = re.findall(r"-?\d[\d,]*", core)
    if not nums:
        return "no_number"
    if len(nums) == 1:
        return "malformed_or_extra_text"
    return "multiple_numbers"


def parse_strict(text: str) -> Optional[int]:
    """Strict parser: the reply must be exactly one integer (optional trailing period, correct thousands
    grouping), or an expression whose part after the LAST '=' is exactly one integer. Decimals, malformed grouping,
    several numbers or thinking text are invalid (the step is then treated like a parse failure: wrong, shown as '?')."""
    cat = parse_category(text)
    s = text.strip().rstrip(".").strip()
    if cat == "integer":
        return int(s.replace(",", ""))
    if cat == "integer_after_equals":
        return int(s.rsplit("=", 1)[1].strip().rstrip(".").strip().replace(",", ""))
    return None


PARSERS = {"first_int": parse_int, "strict": parse_strict}


def pattern_index(bits: Sequence[int]) -> int:
    idx = 0
    for b in bits:
        idx = idx * 2 + int(b)
    return idx


def leaf_for(level: int, prefix: int, action: int, fill: int) -> int:
    """Leaf reached from node (level,prefix) by `action`, then `fill` (0=NOCALL,1=CALL) at all later steps."""
    rest = T_STEPS - level - 1
    return (prefix << (rest + 1)) | (action << rest) | (((1 << rest) - 1) if fill else 0)


@dataclass
class TreeData:
    instance_ids: List[str]
    layers: List[int]
    v_in: np.ndarray                 # (N,7) int64, -1 sentinel where invalid
    v_in_ok: np.ndarray              # (N,7) bool
    v_call: np.ndarray               # (N,7) tool result (int64, sentinel when failed)
    v_call_ok: np.ndarray            # (N,7) bool
    v_no: np.ndarray                 # (N,7) parsed model answer
    v_no_ok: np.ndarray              # (N,7) bool
    local_no_correct: np.ndarray     # (N,7) bool: model's answer equals exact(v_in op b)
    leaf_correct: np.ndarray         # (N,8) bool
    hidden: np.ndarray               # (N,7,len(layers),d) float16
    root_hidden: np.ndarray          # (N,len(layers),d) float16
    mean_logprob: np.ndarray         # (N,7)
    entropy0: np.ndarray             # (N,7)
    n_gen_tokens: np.ndarray         # (N,7) tokens of the node's NOCALL answer
    n_prompt_tokens: np.ndarray      # (N,7)
    step_bin: np.ndarray             # (N,3) difficulty bin index per step (0 easy,1 medium,2 hard)
    meta: dict = field(default_factory=dict)
    texts: Optional[list] = None     # (N,7) raw NOCALL reply text per node (None for trees built before 2026-11-01)

    @property
    def n(self) -> int:
        return len(self.instance_ids)

    def subset(self, idx: np.ndarray) -> "TreeData":
        kw = {}
        for k, v in self.__dict__.items():
            kw[k] = v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (self.n,) else v
        kw["instance_ids"] = [self.instance_ids[i] for i in idx]
        kw["texts"] = None if self.texts is None else [self.texts[i] for i in idx]
        return TreeData(**kw)

    @staticmethod
    def concat(parts: List["TreeData"]) -> "TreeData":
        """Stack independently built trees (e.g. checkpointed chunks of one pool) along the instance axis."""
        first = parts[0]
        for p in parts[1:]:
            assert p.layers == first.layers, "chunks built with different layers"
        kw = {}
        for k, v in first.__dict__.items():
            if isinstance(v, np.ndarray):
                kw[k] = np.concatenate([p.__dict__[k] for p in parts], axis=0)
        ids = [i for p in parts for i in p.instance_ids]
        assert len(set(ids)) == len(ids), "duplicate instance ids across chunks"
        texts = None if any(p.texts is None for p in parts) else [r for p in parts for r in p.texts]
        meta = dict(first.meta)
        meta["chunks"] = [p.meta for p in parts]
        return TreeData(instance_ids=ids, layers=first.layers, meta=meta, texts=texts, **kw)

    # ---- persistence ---------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path, **arrays)
        side = {"instance_ids": self.instance_ids, "layers": self.layers, "meta": self.meta}
        if self.texts is not None:
            side["texts"] = self.texts
        path.with_suffix(".json").write_text(json.dumps(side, indent=1))

    @staticmethod
    def load(path: Path) -> "TreeData":
        path = Path(path)
        z = np.load(path if path.suffix == ".npz" else path.with_suffix(".npz"))
        side = json.loads(path.with_suffix(".json").read_text())
        return TreeData(instance_ids=side["instance_ids"], layers=side["layers"], meta=side["meta"],
                        texts=side.get("texts"), **{k: z[k] for k in z.files})


# ---- labels ----------------------------------------------------------------------------------------
def node_labels(tree: TreeData, fill: int = 1):
    """Exact 0/1 labels for every node under continuation `fill` (1 = always-call, 0 = never-call).

    y_call[i,k] = final correctness if we CALL at node k then continue with `fill`;
    y_no[i,k]   = final correctness if we do NOT call at node k then continue with `fill`.
    """
    y_call = np.zeros((tree.n, N_NODES), dtype=np.float64)
    y_no = np.zeros((tree.n, N_NODES), dtype=np.float64)
    for k in range(N_NODES):
        lv, pf = int(LEVEL_OF[k]), int(PREFIX_OF[k])
        y_call[:, k] = tree.leaf_correct[:, leaf_for(lv, pf, 1, fill)]
        y_no[:, k] = tree.leaf_correct[:, leaf_for(lv, pf, 0, fill)]
    return y_call, y_no


def walk(tree: TreeData, actions: np.ndarray):
    """Execute a policy given as a decision matrix actions[N,7] (1=CALL). Returns (correct[N], calls[N], leaf[N])."""
    n = tree.n
    nid = np.zeros(n, dtype=np.int64)
    bits = np.zeros(n, dtype=np.int64)
    calls = np.zeros(n, dtype=np.int64)
    for _ in range(T_STEPS):
        a = actions[np.arange(n), nid].astype(np.int64)
        calls += a
        bits = bits * 2 + a
        nid = 2 * nid + 1 + a
    correct = tree.leaf_correct[np.arange(n), bits]
    return correct.astype(np.float64), calls.astype(np.float64), bits


def pattern_actions(pattern: int) -> np.ndarray:
    """Decision vector (len 7) that realises a fixed open-loop pattern (a1,a2,a3) regardless of prefix."""
    a = [(pattern >> (T_STEPS - 1 - t)) & 1 for t in range(T_STEPS)]
    return np.array([a[int(LEVEL_OF[k])] for k in range(N_NODES)], dtype=np.int64)


# ---- builder ---------------------------------------------------------------------------------------
def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class TreeBuilder:
    """Builds the outcome tree. Checkpoints carry a fingerprint (model, resolved revision, dtype, device, system prompt,
    layers, max_new_tokens, parser) and the hash of their prompt list; resuming with anything different raises.

    `reuse` (optional): {user_prompt_text: NodeOutput} from an earlier build with the SAME generation fingerprint; prompts found
    there are not regenerated (used by primary_strict to re-score the cached CPU generations with the strict parser and to
    generate only the nodes whose prompt changed)."""

    def __init__(self, runner, layers: Sequence[int], tool: Optional[SymPyTool] = None, batch_size: int = 16,
                 max_new_tokens: int = 12, log=print, parser: str = "first_int", reuse: Optional[dict] = None,
                 reuse_root: Optional[dict] = None, legacy_ok: bool = False, reveal: str = "full"):
        self.runner = runner
        self.layers = list(layers)
        self.tool = tool or SymPyTool()
        self.bs = batch_size
        self.max_new = max_new_tokens
        self.log = log
        self.parser_name = parser
        self.parse = PARSERS[parser]
        self.reuse = reuse or {}
        self.reuse_root = reuse_root or {}
        self.n_reused = 0
        self.n_generated = 0
        self.legacy_ok = legacy_ok
        self.reveal = reveal
        self.seen: Dict[str, object] = {}        # prompt -> NodeOutput of every node (levels)
        self.seen_root: Dict[str, object] = {}   # root prompt -> hidden dict

    def fingerprint(self) -> dict:
        r = self.runner
        return {"model_id": getattr(r, "model_id", None), "revision": getattr(r, "resolved_revision", getattr(r, "revision", None)),
                "dtype": getattr(r, "dtype_name", None), "device": getattr(r, "device", None), "system": getattr(r, "system", None),
                "layers": self.layers, "max_new_tokens": self.max_new, "parser": self.parser_name, "reveal": self.reveal}

    def _generate(self, prompts: List[str]):
        """Generate only prompts not available in `reuse`; return outputs in prompt order."""
        todo = [i for i, p in enumerate(prompts) if p not in self.reuse]
        outs = [self.reuse.get(p) for p in prompts]
        if todo:
            new = self.runner.run([prompts[i] for i in todo], self.layers, max_new_tokens=self.max_new, batch_size=self.bs)
            for i, o in zip(todo, new):
                outs[i] = o
        self.n_reused += len(prompts) - len(todo)
        self.n_generated += len(todo)
        return outs, len(todo)

    def _load_checkpoint(self, ck: Path, prompts: List[str]):
        saved = pickle.load(open(ck, "rb"))
        if self.legacy_ok:                       # checkpoints written before 2026-11-01 (no fingerprint); used only to harvest prompt->output pairs
            return saved if isinstance(saved, list) else saved["outs"]
        fp, ph = saved.get("fingerprint"), saved.get("prompts_sha")
        if fp is None or ph is None:
            raise RuntimeError(f"{ck} has no fingerprint (built by older code). Delete it or build into a new cache directory.")
        if fp != self.fingerprint():
            raise RuntimeError(f"{ck} was built with {fp}, current settings are {self.fingerprint()}; refusing to resume.")
        if ph != _sha(prompts):
            raise RuntimeError(f"{ck} was built from different prompts; refusing to resume.")
        return saved["outs"]

    def _tool_value(self, v_in: Optional[int], op: str, right: int):
        if v_in is None:
            return None, False
        res = self.tool.execute(f"{v_in} {op} {right}")
        if not res.success:
            return None, False
        try:
            return int(res.output), True
        except (TypeError, ValueError):
            return None, False

    def build(self, instances: List[Instance], cache_dir: Path, tag: str) -> TreeData:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        n = len(instances)
        L = len(self.layers)
        d = self.runner.hidden_dim
        t_start = time.perf_counter()

        # per instance, per node state
        v_in: List[List[Optional[int]]] = [[None] * N_NODES for _ in range(n)]
        hist: List[List[Optional[list]]] = [[None] * N_NODES for _ in range(n)]
        for i, ins in enumerate(instances):
            v_in[i][0] = ins.first
            hist[i][0] = []
        v_call = np.full((n, N_NODES), -1, dtype=np.int64)
        v_call_ok = np.zeros((n, N_NODES), dtype=bool)
        v_no = np.full((n, N_NODES), -1, dtype=np.int64)
        v_no_ok = np.zeros((n, N_NODES), dtype=bool)
        local_ok = np.zeros((n, N_NODES), dtype=bool)
        hidden = np.zeros((n, N_NODES, L, d), dtype=np.float16)
        texts = [[""] * N_NODES for _ in range(n)]
        mlp = np.zeros((n, N_NODES))
        ent = np.zeros((n, N_NODES))
        ngen = np.zeros((n, N_NODES), dtype=np.int64)
        nprm = np.zeros((n, N_NODES), dtype=np.int64)
        v_in_arr = np.full((n, N_NODES), -1, dtype=np.int64)
        v_in_ok = np.zeros((n, N_NODES), dtype=bool)
        level_seconds = []

        for lv in range(T_STEPS):
            ck = cache_dir / f"{tag}_level{lv}.pkl"
            ids = [(i, 2 ** lv - 1 + p) for i in range(n) for p in range(2 ** lv)]
            prompts = [node_prompt(instances[i], lv, v_in[i][k], hist[i][k], reveal=self.reveal) for i, k in ids]
            if ck.exists():
                outs = self._load_checkpoint(ck, prompts)
                self.log(f"[{tag}] level {lv}: resumed {len(outs)} node outputs from checkpoint")
            else:
                t0 = time.perf_counter()
                outs, n_new = self._generate(prompts)
                dt = time.perf_counter() - t0
                level_seconds.append(dt)
                pickle.dump({"outs": outs, "seconds": dt, "fingerprint": self.fingerprint(), "prompts_sha": _sha(prompts),
                             "n_generated": n_new}, open(ck, "wb"))
                self.log(f"[{tag}] level {lv}: {len(outs)} nodes ({n_new} generated) in {dt:.0f}s")
            for p_, o_ in zip(prompts, outs):
                self.seen.setdefault(p_, o_)
            for (i, k), o in zip(ids, outs):
                ins = instances[i]
                vi = v_in[i][k]
                if vi is not None:
                    v_in_arr[i, k], v_in_ok[i, k] = vi, True
                for li, l in enumerate(self.layers):
                    hidden[i, k, li] = o.hidden[l]
                mlp[i, k], ent[i, k] = o.mean_logprob, o.first_token_entropy
                ngen[i, k], nprm[i, k] = o.n_tokens, o.n_prompt_tokens
                texts[i][k] = o.text
                parsed = self.parse(o.text)
                if parsed is not None:
                    v_no[i, k], v_no_ok[i, k] = parsed, True
                tv, tok_ok = self._tool_value(vi, ins.ops[lv], ins.rights[lv])
                if tok_ok:
                    v_call[i, k], v_call_ok[i, k] = tv, True
                if vi is not None and parsed is not None:
                    local_ok[i, k] = parsed == apply_op(vi, ins.ops[lv], ins.rights[lv])
                if lv < T_STEPS - 1:      # create children
                    for a, (val, ok, how) in ((0, (parsed, parsed is not None, "mental")),
                                              (1, (tv, tok_ok, "calculator"))):
                        c = child(k, a)
                        v_in[i][c] = val if ok else None
                        hist[i][c] = (hist[i][k] or []) + [{"value": val if ok else None, "how": how}]

        # leaves
        leaf_correct = np.zeros((n, N_LEAVES), dtype=bool)
        for i, ins in enumerate(instances):
            for k in range(2 ** (T_STEPS - 1) - 1, N_NODES):      # level-2 nodes
                pf = int(PREFIX_OF[k])
                for a in (0, 1):
                    ok = (v_call_ok[i, k] if a else v_no_ok[i, k])
                    val = (v_call[i, k] if a else v_no[i, k])
                    leaf_correct[i, (pf << 1) | a] = bool(ok and int(val) == ins.final_gold)

        # t=0 states for the single-decision baseline
        ck = cache_dir / f"{tag}_root.pkl"
        rprompts = [root_prompt(x, reveal=self.reveal) for x in instances]
        if ck.exists():
            root_h = self._load_checkpoint(ck, rprompts)
        else:
            t0 = time.perf_counter()
            todo = [i for i, p in enumerate(rprompts) if p not in self.reuse_root]
            root_h = [self.reuse_root.get(p) for p in rprompts]
            if todo:
                new = self.runner.hidden_only([rprompts[i] for i in todo], self.layers, batch_size=self.bs)
                for i, h in zip(todo, new):
                    root_h[i] = h
            pickle.dump({"outs": root_h, "fingerprint": self.fingerprint(), "prompts_sha": _sha(rprompts)}, open(ck, "wb"))
            level_seconds.append(time.perf_counter() - t0)
        for p_, h_ in zip(rprompts, root_h):
            self.seen_root.setdefault(p_, h_)
        root_hidden = np.stack([np.stack([h[l] for l in self.layers]) for h in root_h]).astype(np.float16)

        bin_idx = {"easy": 0, "medium": 1, "hard": 2}
        step_bin = np.array([[bin_idx[b] for b in ins.bins] for ins in instances], dtype=np.int64)
        cats: Dict[str, int] = {}
        for row in texts:
            for tx in row:
                c = parse_category(tx)
                cats[c] = cats.get(c, 0) + 1
        meta = {"tag": tag, "n": n, "model_id": self.runner.model_id, "revision": self.runner.revision,
                "resolved_revision": getattr(self.runner, "resolved_revision", None),
                "layers": self.layers, "max_new_tokens": self.max_new, "system_prompt": self.runner.system,
                "build_seconds_this_run": time.perf_counter() - t_start, "level_seconds_this_run": level_seconds,
                "device": self.runner.device, "parser": self.parser_name, "parse_categories": cats,
                "env": (self.runner.env_info() if hasattr(self.runner, "env_info") else {}), "nodes_reused": self.n_reused, "nodes_generated": self.n_generated, "fingerprint": self.fingerprint()}
        return TreeData(instance_ids=[x.instance_id for x in instances], layers=self.layers, v_in=v_in_arr,
                        v_in_ok=v_in_ok, v_call=v_call, v_call_ok=v_call_ok, v_no=v_no, v_no_ok=v_no_ok,
                        local_no_correct=local_ok, leaf_correct=leaf_correct, hidden=hidden,
                        root_hidden=root_hidden, mean_logprob=mlp, entropy0=ent, n_gen_tokens=ngen,
                        n_prompt_tokens=nprm, step_bin=step_bin, meta=meta, texts=texts)
