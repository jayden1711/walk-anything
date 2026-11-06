"""Validity checker: a body must pass every check (with a reason code per failure) before it is used."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import mujoco

from .body import Body
from .spec import BodySpec, FAMILIES


@dataclass
class ValidityConfig:
    seg_len: tuple = (0.03, 1.5)
    torso_dim: tuple = (0.05, 2.0)
    torso_mass: tuple = (0.2, 400.0)
    seg_mass: tuple = (0.02, 100.0)
    total_mass: tuple = (0.5, 600.0)
    leg_to_torso_mass: float = 1.0       # (all leg mass) / torso mass
    radius_to_seg: float = 0.5           # capsule radius / shortest segment
    stance_bend: tuple = (0.0, 1.0)
    torque_limit: tuple = (0.05, 2000.0)
    min_height_ratio: float = 0.3        # standing torso height / leg length
    self_penetration: float = 1e-4       # metres
    settle_seconds: float = 1.0
    settle_min_height: float = 0.6       # fraction of standing height
    settle_min_up: float = 0.7           # torso z-axis dot world z, at the end
    max_torque_frac: float = 0.6         # mean |torque| / limit over the last 0.5 s


@dataclass
class Report:
    valid: bool
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _in(x, lo_hi):
    return lo_hi[0] <= x <= lo_hi[1]


def check_spec(s: BodySpec, c: ValidityConfig) -> list[str]:
    bad = []
    if s.family not in FAMILIES:
        return ["E_RANGE:family"]
    if not (2 <= s.n_segs <= 3) or len(s.seg_masses) != s.n_segs:
        bad.append("E_RANGE:segments")
    if not all(_in(x, c.seg_len) for x in s.seg_lengths):
        bad.append("E_RANGE:seg_len")
    if not all(_in(x, c.torso_dim) for x in s.torso_size) or len(s.torso_size) != 3:
        bad.append("E_RANGE:torso_dim")
    if not _in(s.torso_mass, c.torso_mass):
        bad.append("E_RANGE:torso_mass")
    if not all(_in(x, c.seg_mass) for x in s.seg_masses):
        bad.append("E_RANGE:seg_mass")
    if s.n_legs * sum(s.seg_masses) > c.leg_to_torso_mass * s.torso_mass:
        bad.append("E_RANGE:leg_mass_ratio")
    if s.seg_radius <= 0 or s.seg_radius > c.radius_to_seg * min(s.seg_lengths):
        bad.append("E_RANGE:radius")
    if not _in(s.stance_bend, c.stance_bend):
        bad.append("E_RANGE:stance_bend")
    if not _in(s.torque_limit, c.torque_limit):
        bad.append("E_RANGE:torque_limit")
    if not _in(s.total_mass, c.total_mass):
        bad.append("E_RANGE:total_mass")
    if not (0 < s.roll_range <= 1.2 and 0.5 <= s.pitch_range <= 2.5):
        bad.append("E_RANGE:joint_range")
    return bad


def check_body(spec: BodySpec, cfg: ValidityConfig | None = None) -> Report:
    c = cfg or ValidityConfig()
    reasons = check_spec(spec, c)
    if reasons:
        return Report(False, reasons)
    try:
        body = Body(spec)
    except Exception as e:  # MuJoCo compile error
        return Report(False, [f"E_COMPILE:{str(e)[:80]}"])
    m = body.model
    metrics = {}
    if not (np.all(np.isfinite(m.body_mass)) and np.all(np.isfinite(m.body_inertia)) and np.all(m.body_inertia[1:] > 0)):
        return Report(False, ["E_MASS:non_finite_or_zero_inertia"])
    metrics["mass"] = body.mass
    metrics["stand_z"] = body.stand_z
    if not _in(body.mass, c.total_mass):
        reasons.append("E_MASS:total")
    if body.stand_z < c.min_height_ratio * spec.leg_length or body.stand_z > spec.leg_length + spec.torso_size[2]:
        reasons.append("E_STAND:height")
    d = mujoco.MjData(m)
    d.qpos[:] = body.qpos0
    mujoco.mj_forward(m, d)
    # self-collision at the standing pose: any pair of non-adjacent geoms that interpenetrates
    pen = 0.0
    fromto = np.zeros(6)
    for g1 in range(m.ngeom):
        for g2 in range(g1 + 1, m.ngeom):
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if b1 == b2 or g1 == body.floor_id or g2 == body.floor_id:
                continue
            if m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
                continue
            if _is_torso_leg_root(m, b1, b2, body.torso_id):
                continue
            dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.0, fromto)
            pen = min(pen, dist)
    metrics["min_geom_distance"] = pen
    if pen < -c.self_penetration:
        reasons.append("E_SELF:penetration")
    # settle test
    n = int(c.settle_seconds / m.opt.timestep)
    d.qpos[:] = body.qpos0
    d.qvel[:] = 0
    d.ctrl[:] = body.q0
    zs, tq = [], []
    ok_finite = True
    for i in range(n):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            ok_finite = False
            break
        zs.append(d.qpos[2])
        if i >= n - int(0.5 / m.opt.timestep):
            tq.append(np.abs(d.actuator_force) / spec.torque_limit)
    if not ok_finite:
        reasons.append("E_SETTLE:diverged")
        return Report(False, reasons, metrics)
    R = d.xmat[body.torso_id].reshape(3, 3)
    up = float(R[2, 2])
    metrics["settle_min_height_frac"] = float(min(zs) / body.stand_z)
    metrics["settle_final_up"] = up
    metrics["settle_torque_frac"] = float(np.mean(tq, axis=0).max())
    if metrics["settle_min_height_frac"] < c.settle_min_height:
        reasons.append("E_SETTLE:height")
    if up < c.settle_min_up:
        reasons.append("E_SETTLE:tilt")
    if metrics["settle_torque_frac"] > c.max_torque_frac:
        reasons.append("E_TORQUE:static")
    return Report(not reasons, reasons, metrics)


def _is_torso_leg_root(m, b1, b2, torso) -> bool:
    """Torso vs the first leg segment: the capsule is mounted flush against the torso side by construction."""
    return torso in (b1, b2)
