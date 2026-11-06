"""Open-loop rhythm controller driven by a symmetry-derived phase pattern, plus per-body selection on tuning seeds.

No learning: every leg runs the same waveform, shifted by the pattern's phase. The universal residual policy of the
symmetry track (balance corrections on top of this rhythm) is not built yet.
"""
from __future__ import annotations

import itertools

import numpy as np

from .cpg import Pattern, derive_patterns
from .env import DECIMATION
from .evalrun import evaluate
from .mjcf import SIM_DT
from .spec import BodySpec
from .symmetry import detect_symmetries, leg_group

TUNE_SEED0 = 20_000   # tuning seeds; evaluation uses 10_000+ (see evalrun)


class RhythmController:
    def __init__(self, spec: BodySpec, pattern: Pattern, period_T: float = 4.5, amp: float = 0.6,
                 hip_sign: float = 1.0, knee_sign: float = 1.0):
        self.spec, self.pattern = spec, pattern
        self.period_T, self.amp, self.hip_sign, self.knee_sign = period_T, amp, hip_sign, knee_sign

    def reset(self, env):
        self.jpl = self.spec.joints_per_leg
        self.biped = self.spec.family == "biped"
        self.period_s = self.period_T * env.T
        self.frac = np.array(self.pattern.phase_fraction())

    def act(self, env, obs):
        th = 2 * np.pi * (env.t * DECIMATION * SIM_DT / self.period_s + self.frac)
        a = np.zeros(env.nj)
        for l, t in enumerate(th):
            hip = self.hip_sign * self.amp * np.sin(t)
            knee = self.knee_sign * self.amp * max(0.0, np.cos(t))
            for k in range(2, self.jpl):
                a[l * self.jpl + k] = knee
            if self.biped:
                a[l * self.jpl + self.jpl - 1] = -(hip + knee)
            a[l * self.jpl + 1] = hip
        return np.clip(a, -1, 1)

    def describe(self) -> dict:
        return dict(pattern=self.pattern.name, phases=self.pattern.phases, m=self.pattern.m, primary=self.pattern.primary,
                    period_T=self.period_T, amp=self.amp, hip_sign=self.hip_sign, knee_sign=self.knee_sign)


def body_patterns(spec: BodySpec) -> list[Pattern]:
    return derive_patterns(leg_group(detect_symmetries(spec)), spec.n_legs)


GRID = dict(period_T=(3.0, 4.5, 6.5), amp=(0.5, 1.0), hip_sign=(1.0, -1.0), knee_sign=(1.0, -1.0))


def select_gait(spec: BodySpec, n_tune: int = 3, max_patterns: int = 12, log=None):
    """Best (pattern, rhythm parameters) for this body on the tuning seeds. Returns (controller, table)."""
    pats = body_patterns(spec)[:max_patterns]
    table = []
    for p in pats:
        for combo in itertools.product(*GRID.values()):
            c = RhythmController(spec, p, **dict(zip(GRID, combo)))
            r = evaluate(spec, c, n_tune, seed0=TUNE_SEED0)
            score = float(np.mean([e["r_lin"] * e["steps"] / 500 for e in r["episodes"]]))
            table.append(dict(score=score, fall_rate=r["fall_rate"], speed_nd=r["speed_nd"], **c.describe()))
        if log:
            log(f"  {spec.name} {p.name} {p.phases}: best so far {max(t['score'] for t in table):.3f}")
    best = max(table, key=lambda t: t["score"])
    pat = next(p for p in pats if p.name == best["pattern"] and p.phases == best["phases"])
    ctrl = RhythmController(spec, pat, best["period_T"], best["amp"], best["hip_sign"], best["knee_sign"])
    return ctrl, table
