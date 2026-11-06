"""Random-action stability check for a list of body indices (prints per-body diagnostics)."""
import sys

import numpy as np

from walk_anything.sampler import make_body
from walk_anything.env import BodyEnv

idxs = [int(x) for x in sys.argv[1:]] or [0, 3, 1, 4, 2]
for idx in idxs:
    s, rep, _ = make_body(0, idx)
    e = BodyEnv(s, 1)
    rng = np.random.default_rng(0)
    early = 0
    for ep in range(3):
        e.reset()
        for t in range(200):
            _, r, fell, tr = e.step(rng.normal(0, 0.5, e.nj))
            if fell:
                break
        early += t < 199
    m = e.m
    print(idx, s.family, "nj", e.nj, "M", round(e.body.mass, 2), "L", round(s.leg_length, 2), "tau", round(s.torque_limit, 1),
          "kp", round(s.kp, 1), "kd", round(s.kd_ratio * s.kp, 2), "early_falls", early,
          "min_inertia", float(m.body_inertia[1:].min()), "min_mass", float(m.body_mass[1:].min()))
