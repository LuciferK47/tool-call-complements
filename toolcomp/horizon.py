"""Complementarity of tool calls in conjunctive (all-steps-must-be-right) tasks, and the belief-DP rule that accounts for it.

Task: T steps. Step j answered without the tool is right with probability p_j; with the tool it is right (exact tool). The task
pays 1 iff every step is right; each call costs `price`. b = belief that everything so far is right.

  W_T(b) = b
  W_j(b) = E_p[ max( W_{j+1}(b) - price ,  W_{j+1}(b p) ) ]          p ~ empirical law of the calibrated signal at step j
  call at step j iff  W_{j+1}(b) - price > W_{j+1}(b p_j)

W_j is convex, non-decreasing, W_j(0) = 0 and slope <= 1. Rules that fix the continuation are special cases with a LINEAR W:
always-call afterwards: W(b) = b - const (slope 1); never-call afterwards: W(b) = b * prod(future p).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from .features import _digits
from .probes import Platt, SoftLogReg, Standardizer, grouped_folds
from .tree import LEVEL_OF, N_NODES, T_STEPS, TreeData

GRID = np.linspace(0.0, 1.0, 401)


# ---------------------------------------------------------------- theory helpers (homogeneous chain, calibrated p)
def set_value(p: np.ndarray, S: Sequence[int], price: float) -> float:
    """V(S) = prod_{i not in S} p_i - price |S| (exact tool)."""
    mask = np.ones(len(p), bool)
    mask[list(S)] = False
    return float(np.prod(p[mask]) - price * len(S))


def best_set(p: np.ndarray, price: float):
    """Optimal open-loop call set: for a given size, call the steps with the smallest p; scan sizes. O(T log T)."""
    order = np.argsort(p)
    best, bv = (), float(np.prod(p))
    for k in range(1, len(p) + 1):
        v = float(np.prod(p[order[k:]]) - price * k)
        if v > bv + 1e-12:
            best, bv = tuple(int(i) for i in order[:k]), v
    return best, bv


# ---------------------------------------------------------------- calibrated step-correctness signals
def surface_nobin(tree: TreeData, instances) -> np.ndarray:
    """Operator, operand digit counts, node id. (Iteration-2 S0 without the generator's bin label, which is not printed in the prompt.)"""
    out = []
    for k in range(N_NODES):
        lv = int(LEVEL_OF[k])
        op = np.array([i.ops[lv] for i in instances])
        right = np.array([i.rights[lv] for i in instances], float)
        left = np.array([i.first for i in instances], float) if lv == 0 else tree.v_in[:, k].astype(float)
        dl, dr, mul = _digits(left), _digits(right), (op == "*").astype(float)
        one = np.zeros((tree.n, N_NODES))
        one[:, k] = 1.0
        out.append(np.column_stack([op == "+", op == "-", op == "*", dl, dr, dl * dr * mul,
                                    np.log10(np.maximum(np.abs(left), 1.0)) * mul, one]).astype(float))
    return np.stack(out, 1)


class StepSignal:
    """Calibrated P(step answered correctly without the tool | features at the node). Fit on TRAIN, Platt on CAL."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, Xtr, ytr, Xca, yca):
        d = Xtr.shape[-1]
        self.st = Standardizer.fit(Xtr.reshape(-1, d))
        self.h = SoftLogReg(self.C).fit(self.st(Xtr.reshape(-1, d)), ytr.reshape(-1).astype(float))
        self.platt = Platt().fit(self.h.predict_proba(self.st(Xca.reshape(-1, d))), yca.reshape(-1).astype(float))
        return self

    def __call__(self, X):
        d = X.shape[-1]
        return np.clip(self.platt(self.h.predict_proba(self.st(X.reshape(-1, d)))).reshape(X.shape[:2]), 1e-4, 1 - 1e-4)


def pava(y: np.ndarray, w: np.ndarray):
    """Pool-adjacent-violators: non-decreasing fit. Returns fitted values and block weights per point."""
    vals, wts, lens = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
            wts[-1] += w2
            lens[-1] += l2
    return np.repeat(vals, lens), np.repeat(wts, lens)


class IsoSignal:
    """Calibrated P(step right) with isotonic calibration of out-of-fold scores (grouped by instance) on TRAIN u CAL.

    Block means are shrunk as (k+1)/(n+2) so the top block is never exactly 1 (a p of 1 would mean 'never verify')."""

    def __init__(self, C: float = 1.0, folds: int = 5, seed: int = 0):
        self.C, self.k, self.seed = C, folds, seed

    def fit(self, X, y):
        n, K, d = X.shape
        y = y.astype(float)
        oof = np.zeros((n, K))
        for f in grouped_folds(n, self.k, self.seed):
            va = np.zeros(n, bool)
            va[f] = True
            st = Standardizer.fit(X[~va].reshape(-1, d))
            h = SoftLogReg(self.C).fit(st(X[~va].reshape(-1, d)), y[~va].reshape(-1))
            oof[va] = (st(X[va].reshape(-1, d)) @ h.w + h.b).reshape(-1, K)
        self.oof = oof
        s, yy = oof.reshape(-1), y.reshape(-1)
        o = np.argsort(s, kind="stable")
        fit, wt = pava(yy[o], np.ones(len(o)))
        fit = (fit * wt + 1.0) / (wt + 2.0)
        fit = np.maximum.accumulate(fit)
        self.xs, self.ys = s[o], fit
        self.st = Standardizer.fit(X.reshape(-1, d))
        self.h = SoftLogReg(self.C).fit(self.st(X.reshape(-1, d)), yy)
        return self

    def raw(self, X):
        d = X.shape[-1]
        return (self.st(X.reshape(-1, d)) @ self.h.w + self.h.b).reshape(X.shape[:2])

    def cal(self, s):
        return np.clip(np.interp(s, self.xs, self.ys), 1e-4, 1 - 1e-4)

    def __call__(self, X):
        return self.cal(self.raw(X))


# ---------------------------------------------------------------- belief DP
def n_steps_of(P: np.ndarray) -> int:
    """Steps per part K from the number of decision nodes (2^K - 1)."""
    return int(round(np.log2(P.shape[1] + 1)))


def level_samples(P: np.ndarray, n_q: int = 60) -> List[np.ndarray]:
    """Empirical law of the signal at each within-segment level (quantile-thinned)."""
    out = []
    for lv in range(n_steps_of(P)):
        s = P[:, 2 ** lv - 1: 2 ** (lv + 1) - 1].reshape(-1)
        out.append(np.quantile(s, (np.arange(n_q) + 0.5) / n_q))
    return out


def belief_dp(samples: List[np.ndarray], T: int, price: float, q: Sequence[float] = None) -> np.ndarray:
    """W[j, g] on GRID for j = 0..T. samples[l] is the law of p at within-segment level l.
    q (optional): probability that a called step is right given a correct prefix, per within-segment level (noisy tool); a call then multiplies the belief by q
    instead of leaving it unchanged. q = None is the exact tool of the main text."""
    W = np.zeros((T + 1, len(GRID)))
    W[T] = GRID
    for j in range(T - 1, -1, -1):
        p = samples[j % len(samples)]
        nxt = W[j + 1]
        call = nxt - price if q is None else np.interp(GRID * q[j % len(q)], GRID, nxt) - price
        no = np.interp(GRID[:, None] * p[None, :], GRID, nxt)
        W[j] = np.maximum(call[:, None], no).mean(1)
    return W


# ---------------------------------------------------------------- simulation on exact trees
def simulate(P: np.ndarray, leaf: np.ndarray, tuples: np.ndarray, rule: Callable, score: np.ndarray = None, q: Sequence[float] = None):
    """Run a sequential rule on composite tasks. tuples (G, m) index instances. rule(j, b, p, s) -> bool array (call?).
    q (optional): per-level tool accuracy; the agent's belief then becomes b q_j after a call (default: unchanged, exact tool).
    Returns success (G,), calls (G,)."""
    G, m = tuples.shape
    K = n_steps_of(P)
    b = np.ones(G)
    calls = np.zeros(G)
    ok = np.ones(G, bool)
    j = 0
    for s in range(m):
        inst = tuples[:, s]
        nid = np.zeros(G, np.int64)
        bits = np.zeros(G, np.int64)
        for _ in range(K):
            p = P[inst, nid]
            sc = p if score is None else score[inst, nid]
            a = rule(j, b, p, sc).astype(np.int64)
            calls += a
            b = np.where(a == 1, b if q is None else b * q[j % len(q)], b * p)
            bits = bits * 2 + a
            nid = 2 * nid + 1 + a
            j += 1
        ok &= leaf[inst, bits]
    return ok.astype(float), calls


def rule_local(price):            # per-step expected utility (MICE / R2V style)
    return lambda j, b, p, s: (1 - p) > price


def rule_optimistic(price):       # marginal utility, always-call continuation (what MUP's fill=1 labels encode)
    return lambda j, b, p, s: b * (1 - p) > price


def rule_pessimistic(price, means, T):   # marginal utility, never-call continuation
    tail = np.ones(T + 1)
    for j in range(T - 1, -1, -1):
        tail[j] = tail[j + 1] * means[j % len(means)]
    return lambda j, b, p, s: b * (1 - p) * tail[j + 1] > price


def rule_threshold(tau):          # tuned score threshold (Anchor-lite)
    return lambda j, b, p, s: s < tau


def rule_dp(W, price, q=None):
    def f(j, b, p, s):
        nx = W[j + 1]
        bc = b if q is None else b * q[j % len(q)]
        return np.interp(bc, GRID, nx) - price > np.interp(b * p, GRID, nx) + 1e-12
    return f


def rule_all_or_nothing(price, T):      # trivial reference: call everything if it pays, otherwise never call
    return lambda j, b, p, s: np.full(len(p), 1.0 - T * price > 0)


def make_tuples(idx: np.ndarray, m: int, rng: np.random.Generator, reps: int = 1) -> np.ndarray:
    out = []
    for _ in range(reps):
        q = rng.permutation(idx)
        g = len(q) // m
        out.append(q[:g * m].reshape(g, m))
    return np.concatenate(out)


def min_calls(leaf: np.ndarray) -> np.ndarray:
    pops = np.array([bin(q).count("1") for q in range(leaf.shape[1])])
    return np.where(leaf, pops[None, :], 99).min(1)


def hindsight(leaf: np.ndarray, tuples: np.ndarray, price: float):
    k = min_calls(leaf)[tuples].sum(1)
    take = 1 - price * k > 0
    return take.astype(float), np.where(take, k, 0).astype(float)


# ---------------------------------------------------------------- evaluation of every rule on one signal / partition
from scipy.integrate import trapezoid  # noqa: E402
from .eval_harness import LAMBDAS  # noqa: E402
from .splits import make_splits  # noqa: E402

MS_DEFAULT = (1, 2, 4, 8)
RULES = ("never", "always", "all_or_nothing", "local", "optimistic", "pessimistic", "thr_transfer", "thr_retuned", "belief_dp", "hindsight")


def auc(U):
    lam = np.asarray(LAMBDAS)
    return 100 * trapezoid(U, lam, axis=-1) / (lam[-1] - lam[0])


def brier_ll(p, y):
    p, y = p.reshape(-1), y.reshape(-1).astype(float)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune(Pca, leaf_ca, m, price, rng, score):
    tp = make_tuples(np.arange(Pca.shape[0]), m, rng, reps=20)
    grid = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(score, np.linspace(0.01, 0.99, 60))]))
    best, bu = None, -np.inf
    for t in grid:
        ok, c = simulate(Pca, leaf_ca, tp, rule_threshold(t), score)
        u = float((ok - price * c).mean())
        if u > bu + 1e-12:
            best, bu = t, u
    return best


def default_sizes(n: int):
    """450/100/100/350 for n = 1000, proportional otherwise."""
    tr, ca, ga = int(round(.45 * n)), int(round(.10 * n)), int(round(.10 * n))
    return {"train": tr, "cal": ca, "gate": ga, "test": n - tr - ca - ga}


def evaluate_signals(feats, Y, leaf, ids, sizes, seed, Cs, B=0, reps=40, calib=None, MS=MS_DEFAULT, price_scale=1.0, call_ok=None, prefix_ok=None):
    """For every signal: fit the step-correctness signal (Platt or isotonic, chosen on the GATE split), then run every rule on composites of m
    parts. feats: name -> (N, 2^K-1, d); Y: (N, 2^K-1) step correctness; leaf: (N, 2^K) final correctness.
    call_ok / prefix_ok (both (N, 2^K-1), optional): adds the rule "belief_dp_noisy", Belief-DP with the tool accuracy q_l estimated on the training split
    (share of correct called steps at level l among nodes whose prefix is correct); it is not one of the nine rules of the study and is reported separately."""
    sp = make_splits(ids, sizes, seed)
    tr, ca, ga, te = sp["train"], sp["cal"], sp["gate"], sp["test"]
    tc = np.sort(np.concatenate([tr, ca]))
    qs = None
    if call_ok is not None:
        qs = []
        for lv in range(n_steps_of(Y)):
            sl = slice(2 ** lv - 1, 2 ** (lv + 1) - 1)
            ok, co = prefix_ok[tr][:, sl], call_ok[tr][:, sl]
            qs.append(float(co[ok].mean()) if ok.any() else float(co.mean()))
    out = {}
    for sig, X in feats.items():
        sp_ = StepSignal(Cs.get(sig, 1.0)).fit(X[tr], Y[tr], X[ca], Y[ca])
        si_ = IsoSignal(Cs.get(sig, 1.0)).fit(X[tc], Y[tc])
        gate = {"platt": brier_ll(sp_(X[ga]), Y[ga])[1], "isotonic": brier_ll(si_(X[ga]), Y[ga])[1]}
        pick = calib or min(gate, key=gate.get)
        P = sp_(X) if pick == "platt" else si_(X)
        fitP = sp_(X[tr]) if pick == "platt" else si_.cal(si_.oof)
        samples = level_samples(fitP)
        means = [float(x.mean()) for x in samples]
        yt, pt = Y[te].reshape(-1).astype(float), P[te].reshape(-1)
        order = np.argsort(pt)
        r = np.empty(len(pt))
        r[order] = np.arange(1, len(pt) + 1)
        npos, nneg = yt.sum(), (1 - yt).sum()
        auroc = (r[yt == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos > 0 and nneg > 0 else float("nan")
        res = {"auroc_step_test": float(auroc), "brier_step_test": float(np.mean((pt - yt) ** 2)), "calibrator": pick, "gate_logloss": gate,
               "mean_p_by_level": means, "m": {}}
        if qs is not None:
            res["q_by_level"] = qs
        rng = np.random.default_rng(1000 + seed)
        tau1 = {lam: tune(P[ca], leaf[ca], 1, price_scale * lam, np.random.default_rng(7), P[ca]) for lam in LAMBDAS}
        rules = {}
        for m in MS:
            T = len(means) * m
            for lam in LAMBDAS:
                pr = price_scale * lam / m
                W = belief_dp(samples, T, pr)
                rules[(m, lam)] = {"never": rule_threshold(-np.inf), "always": rule_threshold(np.inf), "all_or_nothing": rule_all_or_nothing(pr, T),
                                   "local": rule_local(pr), "optimistic": rule_optimistic(pr), "pessimistic": rule_pessimistic(pr, means, T),
                                   "thr_transfer": rule_threshold(tau1[lam]),
                                   "thr_retuned": rule_threshold(tune(P[ca], leaf[ca], m, pr, np.random.default_rng(7), P[ca])),
                                   "belief_dp": rule_dp(W, pr)}
                if qs is not None:
                    rules[(m, lam)]["belief_dp_noisy"] = rule_dp(belief_dp(samples, T, pr, qs), pr, qs)
        for m in MS:
            tp = make_tuples(te, m, rng, reps=reps)
            block = {}
            for name in RULES + (("belief_dp_noisy",) if qs is not None else ()):
                U, S, Kc = [], [], []
                for lam in LAMBDAS:
                    ok, c = hindsight(leaf, tp, price_scale * lam / m) if name == "hindsight" else simulate(P, leaf, tp, rules[(m, lam)][name], q=qs if name == "belief_dp_noisy" else None)
                    U.append(float((ok - price_scale * lam / m * c).mean()))
                    S.append(float(ok.mean()))
                    Kc.append(float(c.mean() / m))
                block[name] = {"u_auc": float(auc(np.array(U))), "u_by_lam": U, "success_by_lam": S, "calls_per_part_by_lam": Kc}
            names = [n for n in RULES if n not in ("never", "always", "hindsight")]
            con = {}
            if B:
                boot = {n: [] for n in names}
                brng = np.random.default_rng(31 + m)
                for _ in range(B):
                    q = brng.choice(te, size=len(te), replace=True)
                    g = len(q) // m
                    tb = q[:g * m].reshape(g, m)
                    for n in names:
                        u = [float((lambda r_: (r_[0] - price_scale * lam / m * r_[1]).mean())(simulate(P, leaf, tb, rules[(m, lam)][n]))) for lam in LAMBDAS]
                        boot[n].append(float(auc(np.array(u))))
                for n in names:
                    if n == "belief_dp":
                        continue
                    d = np.array(boot["belief_dp"]) - np.array(boot[n])
                    con[n] = {"diff": block["belief_dp"]["u_auc"] - block[n]["u_auc"], "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}
            res["m"][str(m)] = {"n_tuples": int(len(tp)), "rules": block, "belief_dp_minus": con, "signal_aware_regret": predicted_regret_signal(samples, m, price_scale)}
        out[sig] = res
    return out


# ---------------------------------------------------------------- parameter-free predictions from the step accuracies alone
def _levels_p(p_levels, T):
    return np.array([p_levels[j % len(p_levels)] for j in range(T)])


def rule_expected_utility(rule: str, p_levels, T: int, c: float) -> float:
    """Expected utility of a rule when the only information is each step's known accuracy p_j (no instance-level signal).
    The belief b is deterministic given the decisions, so the expected success is b at the end."""
    p = _levels_p(p_levels, T)
    if rule == "optimal_open_loop":
        return best_set(p, c)[1]
    if rule == "all_or_nothing":
        return max(float(np.prod(p)), 1.0 - c * T)
    tail = np.ones(T + 1)
    for j in range(T - 1, -1, -1):
        tail[j] = tail[j + 1] * p[j]
    b, calls = 1.0, 0
    for j in range(T):
        gain = {"local": 1 - p[j], "optimistic": b * (1 - p[j]), "pessimistic": b * (1 - p[j]) * tail[j + 1]}[rule]
        if gain > c:
            calls += 1
        else:
            b *= p[j]
    return b - c * calls


def predicted_regret(p_levels, m: int, rules=("local", "optimistic", "pessimistic"), price_scale=1.0):
    """U-AUC regret (points) of each per-step rule against the best open-loop call set, from p_levels alone; per-call price lambda/m."""
    T = len(p_levels) * m
    out = {}
    for r in rules:
        gaps = [rule_expected_utility("optimal_open_loop", p_levels, T, price_scale * lam / m) - rule_expected_utility(r, p_levels, T, price_scale * lam / m) for lam in LAMBDAS]
        out[r] = float(auc(np.array(gaps)))
    return out


def predicted_regret_signal(samples, m: int, price_scale: float = 1.0, G: int = 3000, seed: int = 0) -> float:
    """Signal-aware prediction of the per-step regret (U-AUC points): no test data, no fitted parameter.
    Draws G sequences of T calibrated step probabilities from the per-position training laws `samples` (as Belief-DP assumes: independent), runs the
    label-free rules on them (success probability of a sequence = product of the drawn p at unaided steps), and returns
    best label-free rule minus best per-step rule. With a constant signal this is the closed-form regret of predicted_regret."""
    K = len(samples)
    T = K * m
    rng = np.random.default_rng(seed)
    means = [float(s.mean()) for s in samples]
    P = np.stack([rng.choice(samples[j % K], G) for j in range(T)], 1)
    U = {}
    for lam in LAMBDAS:
        c = price_scale * lam / m
        W = belief_dp(samples, T, c)
        rules = {"never": None, "always": None, "all_or_nothing": rule_all_or_nothing(c, T), "local": rule_local(c), "optimistic": rule_optimistic(c),
                 "pessimistic": rule_pessimistic(c, means, T), "belief_dp": rule_dp(W, c)}
        for n, r in rules.items():
            b, k = np.ones(G), np.zeros(G)
            for j in range(T):
                a = np.zeros(G, bool) if n == "never" else np.ones(G, bool) if n == "always" else np.asarray(r(j, b, P[:, j], P[:, j]), bool)
                k += a
                b = np.where(a, b, b * P[:, j])
            U.setdefault(n, []).append(float((b - c * k).mean()))
    u = {n: float(auc(np.array(v))) for n, v in U.items()}
    return max(u.values()) - max(u[r] for r in ("local", "optimistic", "pessimistic"))
