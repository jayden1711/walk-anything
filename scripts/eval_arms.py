"""Evaluate controller arms on a body set; writes results/round2/<set>/<arm>.json (git-ignored) with settings and metadata.
Usage (always through sim-run): sim-run .venv/bin/python scripts/eval_arms.py SET ARM [--episodes N]
SET: stop1 | train12 | heldout12 | heldout_all | placeholder      ARM: zero | ppo:<ckpt dir> | mpc | rhythm | residual:<ckpt dir>
rhythm / residual read the per-body gait cache from scripts/tune_gaits.py. Same env seeds (10000+k) for every arm.
"""
import argparse
import datetime
import json
import pathlib
import subprocess
import time

from walk_anything import bodysets
from walk_anything import evalrun
from walk_anything.evalrun import evaluate, ZeroController, PolicyController

ap = argparse.ArgumentParser()
ap.add_argument("set")
ap.add_argument("arm")
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--out", default="results/round2")
ap.add_argument("--tag", default=None)
ap.add_argument("--gaits", default="results/round2/gaits")
a = ap.parse_args()

specs = bodysets.get(a.set)
out = pathlib.Path(a.out) / a.set
out.mkdir(parents=True, exist_ok=True)
tag = a.tag or a.arm.replace(":", "_").replace("/", "_")
meta = dict(set=a.set, arm=a.arm, episodes=a.episodes, sampler_seed=bodysets.SAMPLER_SEED, eval_seed0=10_000,
            eval_cmd=evalrun.EVAL_CMD, eval_steps=evalrun.EVAL_STEPS, walk_frac=evalrun.WALK_FRAC,
            date=datetime.datetime.now().isoformat(timespec="seconds"),
            git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
            command=f"scripts/eval_arms.py {a.set} {a.arm} --episodes {a.episodes}")

res = []
for s in specs:
    t0 = time.time()
    base = None
    if a.arm == "zero":
        ctrl = ZeroController()
    elif a.arm.startswith("ppo:"):
        from walk_anything import ckpt as C
        ctrl = PolicyController(C.load_latest(a.arm[4:]))
    elif a.arm == "mpc":
        from walk_anything.mpc import PredictiveSampling
        ctrl = PredictiveSampling()
    elif a.arm == "rhythm_default":  # no per-body search: first symmetry-derived pattern, fixed parameters
        from walk_anything.gait import RhythmController, body_patterns
        base, ctrl = RhythmController(s, body_patterns(s)[0], 4.5, 0.6), ZeroController()
    elif a.arm == "rhythm" or a.arm.startswith("residual:"):
        from walk_anything.gait import select_gait_cached
        base, gait = select_gait_cached(s, a.gaits)
        if a.arm == "rhythm":
            ctrl = ZeroController()  # zero residual: same code path as residual mode, rhythm alone
        else:
            from walk_anything import ckpt as C
            ctrl = PolicyController(C.load_latest(a.arm.split(":", 1)[1]))
    else:
        raise SystemExit(f"unknown arm {a.arm}")
    r = evaluate(s, ctrl, a.episodes, base=base)
    r["wall_s"] = time.time() - t0
    res.append(r)
    print(f"{s.name:24s} fall {r['fall_rate']:.2f} walk {r['walk_rate']:.2f} speed {r['speed']:+.3f} m/s ({r['speed_nd']:+.3f} nd) "
          f"steps {r['mean_steps']:.0f} [{r['wall_s']:.0f}s]", flush=True)
    json.dump(dict(meta=meta, bodies=res), open(out / f"{tag}.json", "w"), indent=1)
