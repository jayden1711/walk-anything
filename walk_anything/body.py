"""Compiled body: MuJoCo model, default standing pose and per-joint static features."""
from __future__ import annotations

import numpy as np
import mujoco

from .mjcf import build_mjcf, default_angles
from .spec import BodySpec, G, J_MAX

STATIC_DIM = 14


class Body:
    def __init__(self, spec: BodySpec):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(spec))
        m = self.model
        self.nj = m.nu
        self.q0 = np.tile(np.array(default_angles(spec)), spec.n_legs)
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.dadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.jrange = np.array([m.jnt_range[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.foot_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"L{i}_foot") for i in range(spec.n_legs)]
        self.torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        d = mujoco.MjData(m)
        self.qpos0 = self._standing_qpos(d)
        self.stand_z = float(self.qpos0[2])
        self.mass = float(m.body_subtreemass[self.torso_id])
        self.static = self._static_features(d)

    def _standing_qpos(self, d: mujoco.MjData) -> np.ndarray:
        d.qpos[:] = 0
        d.qpos[3] = 1
        d.qpos[2] = 1.0
        d.qpos[self.qadr] = self.q0
        mujoco.mj_forward(self.model, d)
        low = min(self._geom_bottom(d, g) for g in self.foot_geoms)
        q = d.qpos.copy()
        q[2] = 1.0 - low + 0.002  # feet 2 mm above the floor
        return q

    def _geom_bottom(self, d, g) -> float:
        m = self.model
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(d.geom_xpos[g][2] - m.geom_size[g][0])
        R = d.geom_xmat[g].reshape(3, 3)
        sx, sy, sz = m.geom_size[g]
        corners = np.array([[a * sx, b * sy, c * sz] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        return float((d.geom_xpos[g] + corners @ R.T)[:, 2].min())

    def _static_features(self, d) -> np.ndarray:
        """(J_MAX, STATIC_DIM) scale-free per-joint features at the standing pose, zero padded."""
        m = self.model
        d.qpos[:] = self.qpos0
        mujoco.mj_forward(m, d)
        h, M = self.stand_z, self.mass
        torso = d.xpos[self.torso_id]
        out = np.zeros((J_MAX, STATIC_DIM), np.float32)
        s = self.spec
        for i in range(self.nj):
            jid = m.actuator_trnid[i, 0]
            bid = m.jnt_bodyid[jid]
            anchor = d.xanchor[jid]
            axis = d.xaxis[jid] - 0  # world frame at default; torso is upright so world == torso frame up to yaw
            subtree = [b for b in range(m.nbody) if self._is_descendant(b, bid)]
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in subtree]
            reach = max(np.linalg.norm(d.geom_xpos[g] - anchor) for g in geoms) if geoms else 0.0
            depth = (i % s.joints_per_leg)
            out[i, 0:3] = axis
            out[i, 3:6] = (anchor - torso) / h
            out[i, 6] = reach / h
            out[i, 7] = m.body_subtreemass[bid] / M
            out[i, 8:10] = self.jrange[i]
            out[i, 10] = self.spec.torque_limit / (M * G * h)
            out[i, 11] = depth / 3.0
            out[i, 12] = 1.0 if depth == s.joints_per_leg - 1 else 0.0
            out[i, 13] = (self.q0[i]) / 1.0
        return out

    def _is_descendant(self, b: int, anc: int) -> bool:
        m = self.model
        while b > 0:
            if b == anc:
                return True
            b = m.body_parentid[b]
        return False

    def global_static(self) -> np.ndarray:
        """Scale-free global descriptors (8)."""
        s = self.spec
        h, M = self.stand_z, self.mass
        lx, ly, lz = s.torso_size
        return np.array([np.log10(h), np.log10(M) / 2, lx / h, ly / h, lz / h, s.n_legs / 6, s.leg_length / h,
                         float(s.family == "biped")], np.float32)

    @property
    def timescale(self) -> float:
        return float(np.sqrt(self.stand_z / G))
