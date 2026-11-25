#!/usr/bin/env python3
"""One-command GPU run for the scaling study. Resumable; every step is validated before the next; writes one small bundle.

    python scripts/run_gpu.py --smoke                 # 2-3 minutes: tiny version of every step with the first model, then stop
    python scripts/run_gpu.py --profile l40s          # Qwen3.5-2B, 4B, 9B
    python scripts/run_gpu.py --profile rtx4050       # Qwen2.5-0.5B, Qwen3.5-0.8B
    python scripts/run_gpu.py --profile l40s --only arith6,musique --models q35_9b   # a subset

Per model it builds, in this order (each step is skipped if already finished):
  light    compact exports (confidence, entropy, token counts, 64-d hidden PCA) of any existing tree in cache/
  arith6   600 (l40s) / 400 (rtx4050) instances of a genuine 6-step dependent arithmetic chain (63 decision nodes)
  musique  2-, 3- and 4-hop MuSiQue questions with BM25 retrieval as the (noisy) tool
  fresh    a fresh 3-step arithmetic pool (generator seed 20261201, 1000 instances; scripts/build_tree.py)  [last: least important]
If you must stop early, run:  python scripts/run_gpu.py --bundle-only   and send the bundle.
Do not edit configs or code. If a step fails the script stops and bundles what exists; send the bundle and out/status.json back.
"""

import argparse
import json
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROFILES = {
    "l40s": {"models": ["q35_2b", "q35_4b", "q35_9b"], "n_arith6": 600, "n_musique": {2: 300, 3: 200, 4: 100}, "batch": 32},
    "rtx4050": {"models": ["gpu_0p5b", "q35_0p8b"], "n_arith6": 400, "n_musique": {2: 200, 3: 150, 4: 80}, "batch": 16},
}
STEPS = ("light", "arith6", "musique", "fresh")
FRESH_SEED, ARITH6_SEED, MUSIQUE_SEED = 20261201, 20261301, 20261401
OUT = ROOT / "out"
STATUS = OUT / "status.json"


def load_status():
    return json.loads(STATUS.read_text()) if STATUS.exists() else {}


def save_status(s):
    OUT.mkdir(exist_ok=True)
    STATUS.write_text(json.dumps(s, indent=1))


def step(status, key, fn, force):
    if status.get(key, {}).get("ok") and not force:
        print(f"[skip] {key} already done", flush=True)
        return True
    t0 = time.time()
    try:
        ok, note = fn()
    except Exception as e:                                           # noqa
        traceback.print_exc()
        ok, note = False, f"exception: {e!r}"
    status[key] = {"ok": bool(ok), "note": note, "seconds": round(time.time() - t0, 1), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_status(status)
    print(f"[{'PASS' if ok else 'FAIL'}] {key}: {note}", flush=True)
    return ok


def sh(cmd, log):
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    print("$", " ".join(cmd), flush=True)
    with open(OUT / "logs" / f"{log}.log", "w") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        p.wait()
    return p.returncode


def make_runner(cfg, system):
    from toolcomp.models.real_model import HFRunner
    m = cfg["model"]
    r = HFRunner(model_id=m["id"], device=m.get("device", "auto"), dtype=m.get("dtype", "float32"), revision=m.get("revision"),
                 threads=m.get("threads"), system=system)
    return r


def do_light():
    rc = sh([sys.executable, "scripts/export_light.py", "--all", "--out", str(OUT / "light")], "light")
    return rc == 0, f"export_light exit {rc}"


def do_fresh(cfg_name, smoke):
    cmd = [sys.executable, "scripts/build_tree.py", "--config", f"configs/{cfg_name}.yaml", "--seed", str(FRESH_SEED)]
    if smoke:
        cmd += ["--n", "5"]
    rc = sh(cmd, f"{cfg_name}_fresh")
    folder = f"{cfg_name}_seed{FRESH_SEED}"
    tag = folder + ("_n5" if smoke else "")
    npz = ROOT / "cache" / folder / f"{tag}_tree.npz"
    if rc != 0 or not npz.exists():
        return False, f"build_tree exit {rc}"
    rc = sh([sys.executable, "scripts/export_light.py", "--tree", str(npz), "--out", str(OUT / "light" / f"light_{tag}.npz")], f"{tag}_light")
    return rc == 0, (f"{tag} built and exported" if rc == 0 else f"export_light exit {rc}")


def do_deep(cfg_name, kind, prof, smoke):
    from toolcomp.config import load_config
    from toolcomp.deep_tree import ArithEnv, MusiqueEnv, build_deep, save_deep
    cfg = load_config(ROOT / "configs" / f"{cfg_name}.yaml")
    runner = make_runner(cfg, "Answer with a number.")
    done = []
    if kind == "arith6":
        envs = [(f"arith6_{cfg_name}", ArithEnv(6, 4 if smoke else prof["n_arith6"], ARITH6_SEED))]
    else:
        ks = (2, 3) if smoke else (2, 3, 4)
        envs = [(f"musique_k{k}_{cfg_name}", MusiqueEnv(k, 3 if smoke else prof["n_musique"][k], MUSIQUE_SEED, ROOT / "data" / "musique_ans_v1.0_dev.jsonl")) for k in ks]
    for tag, env in envs:
        runner.system = env.system
        res = build_deep(env, runner, ROOT / "cache" / ("smoke_" if smoke else "") / tag, tag + ("_smoke" if smoke else ""),
                         max_new=cfg.get("max_new_tokens", 24) if kind == "arith6" else 16, batch=prof["batch"] if not smoke else 4)
        out = OUT / "deep" / (("smoke_" if smoke else "") + tag + ".npz")
        save_deep(res, out)
        a = res["arrays"]
        done.append(f"{tag}: n={a['leaf_correct'].shape[0]} never={a['leaf_correct'][:, 0].mean():.3f} all-call={a['leaf_correct'][:, -1].mean():.3f}")
        sane = a["leaf_correct"][:, -1].mean() >= a["leaf_correct"][:, 0].mean() - 1e-9
        if kind == "arith6" and not sane:                       # exact tool: calling every step can never be worse than calling none
            return False, f"{tag}: calling every step is worse than never calling (check outputs); " + "; ".join(done)
    return True, "; ".join(done)


def bundle(label):
    out = ROOT / f"bundle_{label}_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
    with tarfile.open(out, "w:gz") as t:
        for p in (OUT / "deep", OUT / "light", OUT / "logs", STATUS):
            if p.exists():
                t.add(p, arcname=str(p.relative_to(ROOT)))
        for p in sorted((ROOT / "results").glob("*seed20261201*")):
            t.add(p, arcname=str(p.relative_to(ROOT)))
    print(f"bundle: {out} ({out.stat().st_size / 2 ** 20:.1f} MiB)  -> send this file back", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=tuple(PROFILES), default="l40s")
    ap.add_argument("--models", default=None, help="comma-separated config names (default: the profile's)")
    ap.add_argument("--only", default=",".join(STEPS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bundle-only", action="store_true", help="just pack whatever is in out/ (use after stopping early)")
    args = ap.parse_args()
    if args.bundle_only:
        bundle("partial")
        return
    prof = PROFILES[args.profile]
    models = args.models.split(",") if args.models else prof["models"]
    only = args.only.split(",")
    try:
        import torch
        gpu = torch.cuda.is_available()
        print("torch", torch.__version__, "| cuda:", gpu, "|", torch.cuda.get_device_name(0) if gpu else "NO GPU")
    except Exception as e:
        print("torch import failed:", e)
        sys.exit(2)
    if not gpu and not args.smoke:
        print("FAIL: CUDA not available (use --smoke for a CPU check).")
        sys.exit(2)
    if args.smoke:
        models = models[:1] if not args.models else models
        globals()["STATUS"] = OUT / "status_smoke.json"
    status = {} if args.smoke else load_status()
    ok = True
    if "light" in only and not args.smoke:
        ok = step(status, "light", do_light, args.force)
    for m in models:
        for s in STEPS[1:]:
            if not ok or s not in only:
                continue
            fn = (lambda m=m: do_fresh(m, args.smoke)) if s == "fresh" else (lambda m=m, s=s: do_deep(m, s, prof, args.smoke))
            ok = step(status, f"{m}:{s}", fn, args.force or args.smoke)
    print("\n==== STATUS ====")
    for k, v in status.items():
        print(f"{'PASS' if v['ok'] else 'FAIL'}  {k:<24} {v['seconds']:>8.0f}s  {v['note'][:150]}")
    if not args.smoke:
        bundle(args.profile)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
