"""Seeded body sampler. A body is a pure function of (seed, index): only this generator is committed, not the data."""
from __future__ import annotations

import numpy as np

from .spec import BodySpec, FAMILIES, G
from .validity import check_body, ValidityConfig, Report

FAMILY_ORDER = ["quadruped", "biped", "hexapod"]
# torso extents relative to leg length (x, y, z) ranges, foot length ratio (biped)
_PROPS = {
    "quadruped": dict(x=(0.8, 2.2), y=(0.3, 0.9), z=(0.15, 0.5)),
    "biped": dict(x=(0.25, 0.6), y=(0.25, 0.6), z=(0.3, 0.8)),
    "hexapod": dict(x=(1.2, 3.0), y=(0.3, 0.9), z=(0.15, 0.5)),
}


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def sample_spec(rng: np.random.Generator, family: str) -> BodySpec:
    L = float(np.exp(rng.uniform(np.log(0.12), np.log(0.9))))  # total leg length, m
    n_segs = int(rng.choice([2, 3]))
    fr = rng.uniform(0.7, 1.3, size=n_segs)
    seg_l = tuple(float(x) for x in L * fr / fr.sum())
    p = _PROPS[family]
    torso = tuple(float(L * _u(rng, p[k])) for k in "xyz")
    r = float(min(seg_l) * _u(rng, (0.08, 0.2)))
    rho_leg = _u(rng, (800, 2000))
    seg_m = tuple(float(rho_leg * (np.pi * r * r * sl + 4 / 3 * np.pi * r**3)) for sl in seg_l)
    rho_t = _u(rng, (250, 700))
    torso_m = float(rho_t * torso[0] * torso[1] * torso[2])
    n_legs = FAMILIES[family]
    base = BodySpec(family=family, torso_size=torso, torso_mass=torso_m, seg_lengths=seg_l, seg_masses=seg_m,
                    seg_radius=r, stance_bend=_u(rng, (0.3, 0.7)), roll_range=_u(rng, (0.3, 0.6)),
                    pitch_range=_u(rng, (1.2, 2.0)), kd_ratio=_u(rng, (0.03, 0.08)),
                    foot_len=float(L * _u(rng, (0.2, 0.4))) if family == "biped" else 0.0)
    strength = _u(rng, (0.4, 1.0))
    torque = strength * base.total_mass * G * L / (n_legs / 2)
    return BodySpec(**{**base.__dict__, "torque_limit": float(torque)})


def make_body(seed: int, index: int, cfg: ValidityConfig | None = None, max_tries: int = 300) -> tuple[BodySpec, Report, int]:
    """Deterministic valid body for (seed, index). Returns (spec, report, number of rejected samples)."""
    family = FAMILY_ORDER[index % 3]
    rng = np.random.default_rng([seed, index])
    for t in range(max_tries):
        s = sample_spec(rng, family)
        s = BodySpec(**{**s.__dict__, "name": f"{family}_{seed}_{index:03d}"})
        rep = check_body(s, cfg)
        if rep.valid:
            return s, rep, t
    raise RuntimeError(f"no valid {family} body for seed={seed} index={index} in {max_tries} tries")


def split_of(index: int) -> str:
    """Stratified by family (index % 3): every 4th body of each family is held out for zero-shot evaluation."""
    return "heldout" if (index // 3) % 4 == 3 else "train"
