"""Evaluate controller arms on the 5 stop-1 bodies; writes JSON under results/day-1106/ (git-ignored).
Usage (always through sim-run): sim-run .venv/bin/python scripts/eval_arms.py ARM [--episodes N] [--bodies i j ...]
ARM: zero | ckpt:<dir> | mpc | rhythm
"""
import argparse
import json
import pathlib
import time

import yaml

from walk_anything.evalrun import evaluate, ZeroController, PolicyController
from walk_anything.sampler import make_body

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--bodies", type=int, nargs="*", default=None)
ap.add_argument("--out", default="results/day-1106")
ap.add_argument("--tag", default=None)
a = ap.parse_args()

cfg = yaml.safe_load(open("configs/default.yaml"))
idxs = a.bodies if a.bodies is not None else cfg["bodies"]["indices"]
specs = [make_body(cfg["bodies"]["seed"], i)[0] for i in idxs]
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = a.tag or a.arm.replace(":", "_").replace("/", "_")

res = []
for s in specs:
    t0 = time.time()
    if a.arm == "zero":
        ctrl = ZeroController()
    elif a.arm.startswith("ckpt:"):
        from walk_anything import ckpt as C
        ctrl = PolicyController(C.load_latest(a.arm[5:]))
    elif a.arm == "mpc":
        from walk_anything.mpc import PredictiveSampling
        ctrl = PredictiveSampling()
    elif a.arm == "rhythm":
        from walk_anything.gait import select_gait
        ctrl, table = select_gait(s, log=print)
        json.dump(table, open(out / f"rhythm_tuning_{s.name}.json", "w"), indent=1)
        print("  selected:", ctrl.describe(), flush=True)
    else:
        raise SystemExit(f"unknown arm {a.arm}")
    r = evaluate(s, ctrl, a.episodes)
    r["wall_s"] = time.time() - t0
    res.append(r)
    print(f"{s.name:24s} fall {r['fall_rate']:.2f} speed {r['speed']:+.3f} m/s ({r['speed_nd']:+.3f} nd) "
          f"r_lin {r['r_lin']:.3f} steps {r['mean_steps']:.0f} [{r['wall_s']:.0f}s]", flush=True)
    json.dump(res, open(out / f"{tag}.json", "w"), indent=1)
