"""Walking environment and a multiprocess vector wrapper. No torch import here: workers stay light."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import mujoco

from .body import Body, STATIC_DIM
from .spec import BodySpec, G, J_MAX

DECIMATION = 4          # control at 1 / (4 * 0.005) = 50 Hz
ACT_SCALE = 0.4         # rad around the default pose
EP_LEN = 1000
CMD_EVERY = 250
GLOBAL_DIM = 11 + 8
JOINT_DIM = STATIC_DIM + 3
V_MAX, W_MAX = 0.4, 0.5  # nondimensional command ranges (x sqrt(g h) and x 0.5 sqrt(g / h))


class BodyEnv:
    def __init__(self, spec: BodySpec, seed: int, randomize: bool = True):
        self.body = Body(spec)
        self.spec = spec
        self.m = self.body.model
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.nj = self.body.nj
        b = self.body
        self.T = b.timescale
        self.vscale = float(np.sqrt(G * b.stand_z))
        self.gstatic = b.global_static()
        self.static = b.static
        self.mask = np.zeros(J_MAX, np.float32)
        self.mask[: self.nj] = 1
        self.fric0 = self.m.geom_friction.copy()
        self.prev_a = np.zeros(self.nj)
        self.t = 0
        self.cmd = np.zeros(2)
        self.push_left = 0
        self.ep_ret = 0.0
        self.ep_rlin = 0.0

    # ---------------------------------------------------------------- reset / step
    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        m, d, b = self.m, self.d, self.body
        mujoco.mj_resetData(m, d)
        d.qpos[:] = b.qpos0
        if self.randomize:
            d.qpos[b.qadr] += self.rng.uniform(-0.05, 0.05, self.nj)
            d.qpos[0:2] += self.rng.uniform(-0.05, 0.05, 2)
            m.geom_friction[:] = self.fric0 * self.rng.uniform(0.6, 1.4)
        d.ctrl[:] = b.q0
        mujoco.mj_forward(m, d)
        self.prev_a[:] = 0
        self.t = 0
        self.push_left = 0
        self.ep_ret = 0.0
        self.ep_rlin = 0.0
        self._sample_cmd()
        return self._obs()

    def _sample_cmd(self):
        v = 0.0 if self.rng.random() < 0.2 else self.rng.uniform(0, V_MAX)
        self.cmd[:] = (v, self.rng.uniform(-W_MAX, W_MAX))

    def step(self, a: np.ndarray):
        m, d, b = self.m, self.d, self.body
        a = np.clip(a, -1.0, 1.0)
        ctrl = np.clip(b.q0 + ACT_SCALE * a, b.jrange[:, 0], b.jrange[:, 1])
        d.ctrl[:] = ctrl
        if self.randomize:
            if self.push_left == 0 and self.rng.random() < 1 / 150:
                ang = self.rng.uniform(0, 2 * np.pi)
                f = self.rng.uniform(0, 0.3) * b.mass * G
                self._push = np.array([f * np.cos(ang), f * np.sin(ang), 0, 0, 0, 0])
                self.push_left = 5
            if self.push_left > 0:
                d.xfrc_applied[b.torso_id] = self._push
                self.push_left -= 1
                if self.push_left == 0:
                    d.xfrc_applied[b.torso_id] = 0
        tau = 0.0
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
            tau = tau + np.abs(d.actuator_force)
        tau = tau / DECIMATION
        self.t += 1
        R = d.xmat[b.torso_id].reshape(3, 3)
        up = R[2, 2]
        vel_w = d.qvel[0:3]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        vxh = c * vel_w[0] + s * vel_w[1]
        vyh = -s * vel_w[0] + c * vel_w[1]
        wz = d.qvel[5]  # body-frame angular velocity; ~ world z when upright
        vcmd = self.cmd[0] * self.vscale
        wcmd = self.cmd[1] * 0.5 / self.T
        r_lin = np.exp(-((vxh - vcmd) ** 2 + vyh**2) / (0.2 * self.vscale) ** 2)
        r_yaw = np.exp(-((wz - wcmd) * self.T / 0.3) ** 2)
        wxy = d.qvel[3:5] * self.T
        r = (1.0 * r_lin + 0.5 * r_yaw + 0.2
             - 0.05 * float(np.mean((tau / self.spec.torque_limit) ** 2))
             - 0.05 * float(np.mean((a - self.prev_a) ** 2))
             - 0.5 * (1 - up)
             - 0.02 * float(wxy @ wxy)
             - 0.1 * (vel_w[2] / self.vscale) ** 2)
        self.prev_a[:] = a
        fell = (up < 0.3) or (d.qpos[2] < 0.35 * b.stand_z) or not np.all(np.isfinite(d.qpos))
        if fell:
            r -= 2.0
        self.ep_ret += r
        self.ep_rlin += r_lin
        trunc = (not fell) and self.t >= EP_LEN
        if self.t % CMD_EVERY == 0:
            self._sample_cmd()
        return self._obs(), float(r), bool(fell), bool(trunc)

    # ---------------------------------------------------------------- observation
    def _obs(self) -> tuple[np.ndarray, np.ndarray]:
        d, b = self.d, self.body
        R = d.xmat[b.torso_id].reshape(3, 3)
        grav = R.T @ np.array([0, 0, -1.0])
        w = d.qvel[3:6] * self.T
        vb = R.T @ d.qvel[0:3] / self.vscale
        g = np.zeros(GLOBAL_DIM, np.float32)
        g[0:3], g[3:6], g[6:9] = grav, w, vb
        g[9:11] = self.cmd
        g[11:] = self.gstatic
        j = np.zeros((J_MAX, JOINT_DIM), np.float32)
        j[:, :STATIC_DIM] = self.static
        j[: self.nj, STATIC_DIM] = d.qpos[b.qadr] - b.q0
        j[: self.nj, STATIC_DIM + 1] = d.qvel[b.dadr] * self.T * 0.2
        j[: self.nj, STATIC_DIM + 2] = self.prev_a
        np.clip(g, -10, 10, out=g)
        np.clip(j, -10, 10, out=j)
        return g, j


# ------------------------------------------------------------------------ vectorised
def _worker(conn, spec_jsons, seeds, randomize):
    envs = [BodyEnv(BodySpec.from_json(s), sd, randomize) for s, sd in zip(spec_jsons, seeds)]
    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                out = [e.reset() for e in envs]
                conn.send((np.stack([o[0] for o in out]), np.stack([o[1] for o in out])))
            elif cmd == "step":
                gs, js, fg, fj, rs, dn, tr, info = [], [], [], [], [], [], [], []
                for k, (e, a) in enumerate(zip(envs, arg)):
                    (g, j), r, fell, trunc = e.step(a[: e.nj])
                    fg.append(g), fj.append(j)
                    if fell or trunc:
                        info.append((k, e.ep_ret, e.t, e.ep_rlin / max(e.t, 1), bool(fell)))
                        g, j = e.reset()
                    gs.append(g), js.append(j), rs.append(r), dn.append(fell), tr.append(trunc)
                conn.send((np.stack(gs), np.stack(js), np.array(rs, np.float32), np.array(dn), np.array(tr),
                           np.stack(fg), np.stack(fj), info))
            elif cmd == "close":
                break
    finally:
        conn.close()


class VecEnv:
    """envs[i] runs the body specs[i]. n_workers=0 runs everything in-process (tests)."""

    def __init__(self, specs: list[BodySpec], n_workers: int = 4, seed: int = 0, randomize: bool = True):
        self.specs = specs
        self.n = len(specs)
        self.n_workers = min(n_workers, self.n)
        self.nj = np.array([s.n_joints for s in specs])
        seeds = [int(np.random.default_rng([seed, i]).integers(2**31)) for i in range(self.n)]
        self.mask = np.zeros((self.n, J_MAX), np.float32)
        for i, k in enumerate(self.nj):
            self.mask[i, :k] = 1
        if self.n_workers == 0:
            self.local = [BodyEnv(s, sd, randomize) for s, sd in zip(specs, seeds)]
            self.groups = []
            return
        self.local = None
        ctx = mp.get_context("spawn")
        self.groups = [list(range(w, self.n, self.n_workers)) for w in range(self.n_workers)]
        self.conns, self.procs = [], []
        for idx in self.groups:
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, [specs[i].to_json() for i in idx], [seeds[i] for i in idx], randomize), daemon=True)
            p.start()
            b.close()
            self.conns.append(a)
            self.procs.append(p)

    def _gather(self, parts, pos):
        shape = parts[0].shape[1:]
        out = np.zeros((self.n,) + shape, parts[0].dtype)
        for idx, p in zip(self.groups, parts):
            out[idx] = p
        return out

    def reset(self):
        if self.local is not None:
            out = [e.reset() for e in self.local]
            return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])
        for c in self.conns:
            c.send(("reset", None))
        res = [c.recv() for c in self.conns]
        return self._gather([r[0] for r in res], 0), self._gather([r[1] for r in res], 1)

    def step(self, actions: np.ndarray):
        """actions (n, J_MAX). Returns g, j, reward, done, trunc, final_g, final_j, episode infos [(env, ret, len, rlin, fell)]."""
        if self.local is not None:
            res = []
            for e, a in zip(self.local, actions):
                (g, j), r, fell, trunc = e.step(a[: e.nj])
                fg, fj = g, j
                inf = []
                if fell or trunc:
                    inf = [(0, e.ep_ret, e.t, e.ep_rlin / max(e.t, 1), bool(fell))]
                    g, j = e.reset()
                res.append((g, j, r, fell, trunc, fg, fj, inf))
            infos = [(i,) + x[1:] for i, r in enumerate(res) for x in r[7]]
            st = lambda k, dt=None: np.array([r[k] for r in res], dt)
            return st(0), st(1), st(2, np.float32), st(3), st(4), st(5), st(6), infos
        for c, idx in zip(self.conns, self.groups):
            c.send(("step", actions[idx]))
        res = [c.recv() for c in self.conns]
        infos = [(idx[k],) + tuple(x[1:]) for idx, r in zip(self.groups, res) for x in r[7] for k in [x[0]]]
        parts = [self._gather([r[k] for r in res], 0) for k in range(7)]
        return (*parts, infos)

    def close(self):
        if self.local is not None:
            return
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
