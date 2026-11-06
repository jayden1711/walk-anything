"""Automatic kinematic/dynamic symmetry detection for a legged body.

Candidates are the torso-rectangle symmetries (identity, left-right mirror, front-back mirror, 180 deg rotation).
Each one induces a leg permutation and per-joint sign flips from the geometry at the zero pose. A candidate is kept
only if it is also verified numerically: the mapped state evolves into the mapped state (an equivariance test of the
actual MuJoCo model), and the contact geoms (type, size, mass, friction, position) map onto each other.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .body import Body
from .spec import BodySpec

CANDIDATES = {
    "identity": np.diag([1.0, 1.0, 1.0]),
    "mirror_lr": np.diag([1.0, -1.0, 1.0]),
    "mirror_fb": np.diag([-1.0, 1.0, 1.0]),
    "rot180": np.diag([-1.0, -1.0, 1.0]),
}


@dataclass
class SymElement:
    name: str
    T: np.ndarray            # 3x3 spatial transform (diagonal, +-1)
    leg_perm: tuple          # leg l -> leg_perm[l]
    joint_perm: np.ndarray   # actuated joint i -> joint_perm[i]
    joint_sign: np.ndarray   # q'[joint_perm[i]] = joint_sign[i] * q[i]
    pose_invariant: bool     # default standing pose maps onto itself


def _zero_pose(body: Body) -> mujoco.MjData:
    d = mujoco.MjData(body.model)
    d.qpos[:] = body.qpos0
    d.qpos[body.qadr] = 0.0
    d.qpos[0:3] = (0, 0, 1.0)
    d.qpos[3:7] = (1, 0, 0, 0)
    mujoco.mj_forward(body.model, d)
    return d


def _match(points: np.ndarray, targets: np.ndarray, tol: float) -> list[int] | None:
    out = []
    for p in points:
        dist = np.linalg.norm(targets - p, axis=1)
        k = int(np.argmin(dist))
        if dist[k] > tol:
            return None
        out.append(k)
    return out if len(set(out)) == len(out) else None


def _geometric_map(body: Body, name: str, T: np.ndarray) -> SymElement | None:
    s, m = body.spec, body.model
    d = _zero_pose(body)
    tol = 1e-6 + 1e-4 * s.leg_length
    hips = np.array([d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"L{l}_roll")] for l in range(s.n_legs)])
    perm = _match(hips @ T.T, hips, tol)
    if perm is None:
        return None
    jpl, nj = s.joints_per_leg, body.nj
    jid = [m.actuator_trnid[i, 0] for i in range(nj)]
    jperm, jsign = np.zeros(nj, int), np.zeros(nj)
    det = np.linalg.det(T)
    for i in range(nj):
        i2 = perm[i // jpl] * jpl + i % jpl
        a, a2 = det * T @ d.xaxis[jid[i]], d.xaxis[jid[i2]]
        if np.linalg.norm(T @ d.xanchor[jid[i]] - d.xanchor[jid[i2]]) > tol:
            return None
        if np.linalg.norm(a - a2) < 1e-6:
            jsign[i] = 1.0
        elif np.linalg.norm(a + a2) < 1e-6:
            jsign[i] = -1.0
        else:
            return None
        jperm[i] = i2
    # joint limits, actuator gains and limits must map onto themselves under the sign flips
    for i in range(nj):
        r, r2 = body.jrange[i], body.jrange[jperm[i]]
        r_mapped = np.sort(jsign[i] * r)
        if np.abs(r_mapped - r2).max() > 1e-9:
            return None
        if (m.actuator_gainprm[i, 0] != m.actuator_gainprm[jperm[i], 0]
                or np.abs(m.actuator_forcerange[i] - m.actuator_forcerange[jperm[i]]).max() > 1e-9):
            return None
    # contact / mass geometry: every geom maps to a geom of the same type, size, mass and friction
    gpos = d.geom_xpos.copy()
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        cand = [g2 for g2 in range(m.ngeom) if m.geom_bodyid[g2] != 0 and m.geom_type[g2] == m.geom_type[g]]
        ok = False
        for g2 in cand:
            same = (np.abs(m.geom_size[g] - m.geom_size[g2]).max() < 1e-9 and abs(m.body_mass[m.geom_bodyid[g]] - m.body_mass[m.geom_bodyid[g2]]) < 1e-9
                    and np.abs(m.geom_friction[g] - m.geom_friction[g2]).max() < 1e-9
                    and np.linalg.norm(T @ gpos[g] - gpos[g2]) < tol)
            if same:
                # orientation: a box/capsule is symmetric under +-1 diagonal transforms if its mapped axes line up
                R, R2 = d.geom_xmat[g].reshape(3, 3), d.geom_xmat[g2].reshape(3, 3)
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE or np.allclose(np.abs(T @ R), np.abs(R2), atol=1e-6):
                    ok = True
                    break
        if not ok:
            return None
    q0 = body.q0
    pose_inv = bool(np.abs(_scatter(jsign * q0, jperm) - q0).max() < 1e-9)
    return SymElement(name, T, tuple(int(p) for p in perm), jperm, jsign, pose_inv)


def _scatter(q0: np.ndarray, jperm: np.ndarray) -> np.ndarray:
    out = np.zeros_like(q0)
    out[jperm] = q0
    return out


def _mapped_state(body: Body, el: SymElement, qpos, qvel):
    """Image of a full (qpos, qvel) state under a symmetry element."""
    T, det = el.T, np.linalg.det(el.T)
    q2, v2 = np.zeros_like(qpos), np.zeros_like(qvel)
    quat = qpos[3:7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    R2 = T @ R @ T
    qu2 = np.zeros(4)
    mujoco.mju_mat2Quat(qu2, R2.reshape(-1))
    q2[0:3], q2[3:7] = T @ qpos[0:3], qu2
    v2[0:3] = T @ qvel[0:3]
    w_world = R @ qvel[3:6]
    v2[3:6] = R2.T @ (det * T @ w_world)
    q2[body.qadr[el.joint_perm]] = el.joint_sign * qpos[body.qadr]
    v2[body.dadr[el.joint_perm]] = el.joint_sign * qvel[body.dadr]
    return q2, v2


def _equivariant(body: Body, el: SymElement, seed: int = 0, steps: int = 10, tol: float = 1e-6) -> bool:
    """Airborne random state + random targets: stepping then mapping equals mapping then stepping."""
    m = body.model
    rng = np.random.default_rng(seed)
    d1, d2 = mujoco.MjData(m), mujoco.MjData(m)
    qpos = body.qpos0.copy()
    qpos[2] += 1.0
    qpos[body.qadr] = rng.uniform(-0.4, 0.4, body.nj)
    ax = rng.normal(size=3)
    mujoco.mju_axisAngle2Quat(qpos[3:7], ax / np.linalg.norm(ax), 0.3)
    qvel = np.zeros(m.nv)
    qvel[:] = rng.normal(0, 0.2, m.nv)
    ctrl = rng.uniform(-0.3, 0.3, body.nj)
    d1.qpos[:], d1.qvel[:] = qpos, qvel
    q2, v2 = _mapped_state(body, el, qpos, qvel)
    d2.qpos[:], d2.qvel[:] = q2, v2
    c2 = np.zeros(body.nj)
    c2[el.joint_perm] = el.joint_sign * ctrl
    for _ in range(steps):
        d1.ctrl[:], d2.ctrl[:] = ctrl, c2
        mujoco.mj_step(m, d1)
        mujoco.mj_step(m, d2)
    e1, e2 = _mapped_state(body, el, d1.qpos, d1.qvel)
    dq = np.abs(e1[body.qadr] - d2.qpos[body.qadr]).max()
    dp = np.abs(e1[0:3] - d2.qpos[0:3]).max()
    R1, R2 = np.zeros(9), np.zeros(9)
    mujoco.mju_quat2Mat(R1, e1[3:7])
    mujoco.mju_quat2Mat(R2, d2.qpos[3:7])
    dr = np.abs(R1 - R2).max()
    return bool(max(dq, dp, dr) < tol)


def detect_symmetries(spec: BodySpec) -> list[SymElement]:
    body = Body(spec)
    out = []
    for name, T in CANDIDATES.items():
        el = _geometric_map(body, name, T)
        if el is not None and _equivariant(body, el):
            out.append(el)
    return out


def leg_group(elements: list[SymElement]) -> list[tuple]:
    """The verified leg permutations, closed under composition (asserts closure)."""
    perms = {e.leg_perm for e in elements}
    for a in perms:
        for b in perms:
            c = tuple(a[b[i]] for i in range(len(a)))
            assert c in perms, "detected symmetry set is not closed under composition"
    return sorted(perms)
