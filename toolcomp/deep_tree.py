"""Exhaustive outcome trees for chains of K dependent steps, built from real model generations.

Two environments share one builder.

  arith    K-step arithmetic chain (K <= 6). Unaided: the model computes the step. Tool call: exact calculator (SymPy).
  musique  2-4 hop MuSiQue questions. Unaided: closed-book answer to the (substituted) sub-question.
           Tool call: BM25 over the question's own 20 paragraphs, then the same model reads the top 3 passages.
           This tool is noisy: it can retrieve the wrong passages or be misread.

Node layout is the one in tree.py: level lv, prefix p -> id = 2^lv - 1 + p (prefix bits are the earlier actions, 1 = call);
child(nid, a) = 2 nid + 1 + a; the leaf for action pattern (a_1..a_K) has index a_1 2^(K-1) + ... + a_K.
Decoding is greedy, so every outcome is a deterministic function of the action prefix and the tree stores all of them.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .tools.sympy_tool import SymPyTool

MUSIQUE_URL = "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/c8f4f8c9465fb69d31a8eae894c3fd509c4ca321/musique_ans_v1.0_dev.jsonl"
MUSIQUE_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"


def child(nid: int, a: int) -> int:
    return 2 * nid + 1 + a


def n_nodes(K: int) -> int:
    return 2 ** K - 1


def level_of(nid: int) -> int:
    return int(math.floor(math.log2(nid + 1)))


def prefix_of(nid: int) -> int:
    return nid - (2 ** level_of(nid) - 1)


# ------------------------------------------------------------------------------------------------ BM25 (self-contained)
class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tok = [self._t(d) for d in docs]
        self.avg = max(1e-9, sum(len(t) for t in self.tok) / max(1, len(self.tok)))
        df: Dict[str, int] = {}
        for t in self.tok:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        n = len(self.tok)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    @staticmethod
    def _t(s: str) -> List[str]:
        return re.findall(r"\w+", s.lower())

    def scores(self, query: str) -> List[float]:
        q = self._t(query)
        out = []
        for t in self.tok:
            tf: Dict[str, int] = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q:
                if w in tf:
                    s += self.idf.get(w, 0.0) * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(t) / self.avg))
            out.append(s)
        return out

    def top(self, query: str, k: int = 3) -> List[int]:
        sc = self.scores(query)
        return sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:k]


# ------------------------------------------------------------------------------------------------ answer matching
def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def match(pred: Optional[str], golds: Sequence[str]) -> bool:
    """Normalised exact match, or the gold phrase appears as whole words in a short prediction (<= gold length + 3 words)."""
    if pred is None:
        return False
    p = norm(pred)
    if not p:
        return False
    for g in golds:
        g = norm(g)
        if not g:
            continue
        if p == g:
            return True
        if len(p.split()) <= len(g.split()) + 3 and re.search(r"\b" + re.escape(g) + r"\b", p):
            return True
    return False


# ------------------------------------------------------------------------------------------------ environments
def _apply(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b if op == "-" else a * b


_BINS = {"easy": ((1, 2), (1, 2), ("+", "-")), "medium": ((2, 3), (2, 2), ("+", "-", "*")), "hard": ((3, 4), (3, 3), ("+", "-", "*"))}
_BW = (("easy", 0.4), ("medium", 0.35), ("hard", 0.25))


class ArithEnv:
    """K-step chained arithmetic (the 3-step generator of envs/arith.py extended to K steps; values capped at 7 digits)."""
    name = "arith"
    system = "Answer with a number."
    call_is_llm = False

    def __init__(self, K: int, n: int, seed: int):
        self.K = K
        rng = random.Random(seed)
        self.items = []
        for i in range(n):
            bins = [rng.choices([b for b, _ in _BW], [w for _, w in _BW])[0] for _ in range(K)]
            dl, _, _ = _BINS[bins[0]]
            d0 = rng.randint(*dl)
            first = rng.randint(10 ** (d0 - 1), 10 ** d0 - 1)
            ops, rights, vals, cur = [], [], [], first
            for b in bins:
                _, dr, allowed = _BINS[b]
                if abs(cur) > 10 ** 6:
                    allowed = ("+", "-")
                op = rng.choice(allowed)
                d = rng.randint(*dr)
                right = rng.randint(10 ** (d - 1), 10 ** d - 1)
                cur = _apply(cur, op, right)
                ops.append(op); rights.append(right); vals.append(cur)
            self.items.append({"id": f"a{K}_{seed}_{i:05d}", "first": first, "ops": ops, "rights": rights, "bins": bins, "gold": vals})
        self.tool = SymPyTool()

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.items, sort_keys=True).encode()).hexdigest()[:16]

    def _v_in(self, it, step, hist):
        return it["first"] if step == 0 else (hist[-1]["value"] if hist else None)

    def expr(self, it) -> str:
        e = str(it["first"])
        for op, r in zip(it["ops"], it["rights"]):
            e = f"({e} {op} {r})"
        return e

    def prompt(self, it, step, hist) -> str:
        lines = [f"Problem: compute {self.expr(it)} one step at a time."]
        for k, h in enumerate(hist):
            lines.append(f"Step {k + 1} result: {'?' if h['value'] is None else h['value']} (done by {h['how']}).")
        v = self._v_in(it, step, hist)
        lines.append(f"Now step {step + 1}: compute {'?' if v is None else v} {it['ops'][step]} {it['rights'][step]}.")
        lines.append("Reply with only the integer result.")
        return "\n".join(lines)

    def call_prompt(self, it, step, hist):
        return None

    def call_value(self, it, step, hist) -> Optional[int]:
        v = self._v_in(it, step, hist)
        if v is None:
            return None
        r = self.tool.execute(f"{v} {it['ops'][step]} {it['rights'][step]}")
        return int(r.output) if r.success else None

    def parse(self, text: str):
        from .tree import parse_strict
        return parse_strict(text)

    def local_ok(self, it, step, hist, ans) -> bool:
        v = self._v_in(it, step, hist)
        return ans is not None and v is not None and ans == _apply(v, it["ops"][step], it["rights"][step])

    def final_ok(self, it, ans) -> bool:
        return ans is not None and ans == it["gold"][-1]

    def v_in_for_features(self, it, step, hist):
        v = self._v_in(it, step, hist)
        return -1 if v is None else int(v)

    def question_for_features(self, it, step, hist) -> str:
        return ""


def ensure_musique(path: Path) -> Path:
    path = Path(path)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == MUSIQUE_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(MUSIQUE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    except Exception:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_dev.jsonl", repo_type="dataset", revision=MUSIQUE_URL.split("/resolve/")[1].split("/")[0])
        path.write_bytes(Path(got).read_bytes())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == MUSIQUE_SHA256, f"MuSiQue dev file checksum mismatch: {got}"
    return path


class MusiqueEnv:
    """MuSiQue-Ans dev questions with K = 2, 3 or 4 hops. Unaided = closed book; call = BM25 retrieval + reading."""
    name = "musique"
    system = ("Answer with a short phrase (a few words) and nothing else. "
              "A question of the form 'X >> relation' asks for the value of that relation for X.")
    call_is_llm = True

    def __init__(self, K: int, n: int, seed: int, path: Path):
        self.K = K
        rows = [json.loads(l) for l in open(ensure_musique(path))]
        rows = [r for r in rows if r.get("answerable") and len(r["question_decomposition"]) == K]
        rows.sort(key=lambda r: r["id"])
        random.Random(seed).shuffle(rows)
        self.items = []
        for r in rows[:n]:
            self.items.append({"id": r["id"], "question": r["question"], "steps": [d["question"] for d in r["question_decomposition"]],
                               "gold": [d["answer"] for d in r["question_decomposition"]], "aliases": r.get("answer_aliases") or [],
                               "paragraphs": [(p["title"], p["paragraph_text"]) for p in r["paragraphs"]]})
        self._bm: Dict[str, BM25] = {}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps([(i["id"]) for i in self.items]).encode()).hexdigest()[:16] + ":" + MUSIQUE_SHA256[:8]

    def _q(self, it, step, hist) -> str:
        def sub(m):
            j = int(m.group(1)) - 1
            v = hist[j]["value"] if j < len(hist) else None
            return "unknown" if v is None else str(v)
        return re.sub(r"#(\d)", sub, it["steps"][step])

    def prompt(self, it, step, hist) -> str:
        return f"Question: {self._q(it, step, hist)}\nGive only the answer."

    def call_prompt(self, it, step, hist) -> str:
        q = self._q(it, step, hist)
        bm = self._bm.get(it["id"])
        if bm is None:
            bm = self._bm[it["id"]] = BM25([t + " " + x for t, x in it["paragraphs"]])
        ctx = []
        for i in bm.top(q, 3):
            t, x = it["paragraphs"][i]
            ctx.append(f"[{t}] " + " ".join(x.split()[:110]))
        return "Passages:\n" + "\n".join(ctx) + f"\n\nQuestion: {q}\nGive only the answer, using the passages."

    def call_value(self, it, step, hist):
        raise NotImplementedError

    def parse(self, text: str):
        t = text.strip().split("\n")[0].strip().strip('"').rstrip(".")
        return t if t else None

    def _golds(self, it, step):
        g = [it["gold"][step]]
        return g + list(it["aliases"]) if step == len(it["gold"]) - 1 else g

    def local_ok(self, it, step, hist, ans) -> bool:
        return match(ans, self._golds(it, step))

    def final_ok(self, it, ans) -> bool:
        return match(ans, self._golds(it, len(it["gold"]) - 1))

    def v_in_for_features(self, it, step, hist):
        return -1 if (not hist or hist[-1]["value"] is None) else len(str(hist[-1]["value"]).split())

    def question_for_features(self, it, step, hist) -> str:
        return self._q(it, step, hist)


# ------------------------------------------------------------------------------------------------ builder
def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _run_dedup(runner, prompts: List[str], max_new: int, batch: int):
    """Generate each distinct prompt once (greedy decoding makes identical prompts identical)."""
    uniq = list(dict.fromkeys(prompts))
    outs = runner.run(uniq, [], max_new_tokens=max_new, batch_size=batch) if uniq else []
    m = dict(zip(uniq, outs))
    return [m[p] for p in prompts], len(uniq)


def build_deep(env, runner, cache_dir: Path, tag: str, max_new: int = 16, batch: int = 32, log=print) -> dict:
    """Build the outcome tree of every item in env (all items have env.K steps). Returns arrays + sidecar dict. Resumable per level."""
    K, items = env.K, env.items
    N, NN, NL = len(items), n_nodes(env.K), 2 ** env.K
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hist: List[List[Optional[list]]] = [[None] * NN for _ in range(N)]
    for i in range(N):
        hist[i][0] = []
    A = {k: np.zeros((N, NN)) for k in ("mean_logprob", "entropy0", "n_gen_tokens", "n_prompt_tokens", "call_n_gen_tokens", "call_n_prompt_tokens", "call_mean_logprob")}
    local_no = np.zeros((N, NN), bool)
    local_call = np.zeros((N, NN), bool)
    v_in_feat = np.full((N, NN), -1, np.int64)
    leaf = np.zeros((N, NL), bool)
    texts = [[""] * NN for _ in range(N)]
    call_texts = [[""] * NN for _ in range(N)]
    qs = [[""] * NN for _ in range(N)]
    gen_count = 0
    t_start = time.perf_counter()
    for lv in range(K):
        ck = cache_dir / f"{tag}_level{lv}.pkl"
        ids = [(i, 2 ** lv - 1 + p) for i in range(N) for p in range(2 ** lv)]
        pr_un = [env.prompt(items[i], lv, hist[i][k]) for i, k in ids]
        sig = _sha([pr_un, getattr(runner, "model_id", None), max_new, env.system])
        if ck.exists():
            saved = pickle.load(open(ck, "rb"))
            assert saved["sig"] == sig, f"{ck} was built from different prompts or settings; delete it to rebuild"
            o_un, o_call = saved["un"], saved["call"]
            log(f"[{tag}] level {lv}: resumed from checkpoint")
        else:
            t0 = time.perf_counter()
            o_un, n1 = _run_dedup(runner, pr_un, max_new, batch)
            if env.call_is_llm:
                pr_call = [env.call_prompt(items[i], lv, hist[i][k]) for i, k in ids]
                o_call, n2 = _run_dedup(runner, pr_call, max_new, max(4, batch // 2))
            else:
                o_call, n2 = [None] * len(ids), 0
            pickle.dump({"sig": sig, "un": o_un, "call": o_call}, open(ck, "wb"))
            gen_count += n1 + n2
            log(f"[{tag}] level {lv}: {len(ids)} nodes, {n1 + n2} distinct generations, {time.perf_counter() - t0:.0f}s")
        for (i, k), ou, oc in zip(ids, o_un, o_call):
            it, h = items[i], hist[i][k]
            ans_un = env.parse(ou.text)
            texts[i][k] = ou.text
            qs[i][k] = env.question_for_features(it, lv, h)
            v_in_feat[i, k] = env.v_in_for_features(it, lv, h)
            A["mean_logprob"][i, k], A["entropy0"][i, k] = ou.mean_logprob, ou.first_token_entropy
            A["n_gen_tokens"][i, k], A["n_prompt_tokens"][i, k] = ou.n_tokens, ou.n_prompt_tokens
            local_no[i, k] = env.local_ok(it, lv, h, ans_un)
            if env.call_is_llm:
                ans_call = env.parse(oc.text)
                call_texts[i][k] = oc.text
                A["call_mean_logprob"][i, k], A["call_n_gen_tokens"][i, k], A["call_n_prompt_tokens"][i, k] = oc.mean_logprob, oc.n_tokens, oc.n_prompt_tokens
            else:
                ans_call = env.call_value(it, lv, h)
            local_call[i, k] = env.local_ok(it, lv, h, ans_call)
            if lv < K - 1:
                hist[i][child(k, 0)] = h + [{"value": ans_un, "how": "model"}]
                hist[i][child(k, 1)] = h + [{"value": ans_call, "how": "tool"}]
            else:
                pf = k - (2 ** lv - 1)
                leaf[i, (pf << 1) | 0] = env.final_ok(it, ans_un)
                leaf[i, (pf << 1) | 1] = env.final_ok(it, ans_call)
    arrays = dict(leaf_correct=leaf, local_no_correct=local_no, local_call_correct=local_call, v_in_feat=v_in_feat,
                  **{k: v.astype(np.float32) if k.endswith("logprob") or k == "entropy0" else v.astype(np.int32) for k, v in A.items()})
    side = {"env": env.name, "K": K, "ids": [it["id"] for it in items], "fingerprint": env.fingerprint(), "texts": texts, "call_texts": call_texts,
            "questions": qs, "items": [{k: v for k, v in it.items() if k != "paragraphs"} for it in items],
            "meta": {"model_id": getattr(runner, "model_id", None), "resolved_revision": getattr(runner, "resolved_revision", None),
                     "device": getattr(runner, "device", None), "dtype": getattr(runner, "dtype_name", None),
                     "max_new_tokens": max_new, "system": env.system, "generations_this_run": gen_count,
                     "seconds_this_run": time.perf_counter() - t_start}}
    return {"arrays": arrays, "side": side}


def save_deep(res: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **res["arrays"])
    path.with_suffix(".json").write_text(json.dumps(res["side"]))


def load_deep(path: Path):
    path = Path(path)
    z = np.load(path)
    return {k: z[k] for k in z.files}, json.loads(path.with_suffix(".json").read_text())
