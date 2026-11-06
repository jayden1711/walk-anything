"""Reviewer analyses over existing result JSON / run metrics. Read-only; prints to stdout, writes nothing.
Usage: python scripts/review_stats.py RESULTS_DIR RUNS_DIR [PAIRED_SET]
"""
import json
import pathlib
import statistics as st
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
runs = pathlib.Path(sys.argv[2])
PAIR = sys.argv[3] if len(sys.argv) > 3 else "heldout12"
ARMS = ["zero", "ppo_old", "mpc", "rhythm", "rhythm_default", "residual1", "residual1_s1", "residual1_s2"]
SETS = [d.name for d in root.iterdir() if d.is_dir() and d.name not in ("gaits", "logs", "ckpt")]


def load(s, a):
    f = root / s / f"{a}.json"
    return json.load(open(f))["bodies"] if f.exists() else None


print("== training runs: KL early stop and learning signal")
for r in ["residual1", "residual1_s1", "residual1_s2"]:
    f = runs / r / "metrics.jsonl"
    if not f.exists():
        continue
    rows = [json.loads(l) for l in open(f)]
    n, q = len(rows), max(1, len(rows) // 10)
    ks = [x["kl_stopped"] for x in rows]
    os_ = [x["opt_steps"] for x in rows]
    rw = [x["reward"] for x in rows]
    print(f"{r}: iters {n} env_steps {rows[-1]['env_steps']} kl_stopped_frac {np.mean(ks):.2f} opt_steps median {st.median(os_)} "
          f"mean {np.mean(os_):.2f} reward first10% {np.mean(rw[:q]):.3f} last10% {np.mean(rw[-q:]):.3f}")

rng = np.random.default_rng(0)
print("\n== per set/arm: bodies, walk>0 bodies, speed over non-fallen episodes vs fallen episodes, body-bootstrap 95% CI of fall rate")
for s in sorted(SETS):
    for a in ARMS:
        B = load(s, a)
        if not B:
            continue
        eps = [(e["fell"], e["speed"], e["walked"]) for b in B for e in b["episodes"]]
        nf = [sp for f, sp, w in eps if not f]
        fl = [sp for f, sp, w in eps if f]
        fall = np.array([b["fall_rate"] for b in B])
        bs = [rng.choice(fall, len(fall)).mean() for _ in range(2000)]
        walkers = sum(1 for b in B if b["walk_rate"] > 0)
        print(f"{s:12s} {a:15s} bodies {len(B):2d} episodes {len(eps):3d} bodies_with_any_walk {walkers:2d} walk_episodes {sum(w for *_, w in eps):3d} "
              f"speed_nonfell {np.mean(nf) if nf else float('nan'):+.3f} (n={len(nf)}) speed_fell {np.mean(fl) if fl else float('nan'):+.3f} (n={len(fl)}) "
              f"fall {fall.mean():.2f} CI [{np.percentile(bs, 2.5):.2f},{np.percentile(bs, 97.5):.2f}]")

print("\n== paired comparison on the set named in argv[3] (default heldout12; per-body fall rate and speed difference vs zero and vs rhythm, bootstrap over bodies)")
for a in ["rhythm", "residual1", "residual1_s1", "residual1_s2", "ppo_old", "mpc"]:
    for ref in ["zero", "rhythm"]:
        A, R = load(PAIR, a), load(PAIR, ref)
        if not A or not R or a == ref:
            continue
        d = np.array([x["fall_rate"] - y["fall_rate"] for x, y in zip(A, R)])
        v = np.array([x["speed"] - y["speed"] for x, y in zip(A, R)])
        bd = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        bv = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        print(f"{a:13s} - {ref:6s}: dfall {d.mean():+.2f} CI [{np.percentile(bd, 2.5):+.2f},{np.percentile(bd, 97.5):+.2f}]  "
              f"dspeed {v.mean():+.3f} CI [{np.percentile(bv, 2.5):+.3f},{np.percentile(bv, 97.5):+.3f}]")

print("\n== residual seeds pooled over 3 seeds, heldout12, per-seed fall mean and best/worst seed")
for s in ["train12", "heldout12", "placeholder", "external"]:
    rows = []
    for a in ["residual1", "residual1_s1", "residual1_s2"]:
        B = load(s, a)
        if B:
            rows.append((np.mean([b["fall_rate"] for b in B]), np.mean([b["speed"] for b in B])))
    if rows:
        print(s, "fall per seed", [f"{r[0]:.2f}" for r in rows], "speed per seed", [f"{r[1]:+.3f}" for r in rows])
