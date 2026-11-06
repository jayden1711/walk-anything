"""Predictive sampling (Howell et al. 2022, arXiv 2212.00541): a no-training MPC baseline, own implementation.

Each control step: sample perturbations of the nominal action spline, roll each out in a copy of the true simulator
(same model, same friction), keep the best by summed task reward, execute its first action, shift the nominal.
The planning reward is the task reward of env.BodyEnv without its smoothness/torque terms.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .env import ACT_SCALE, DECIMATION


class PredictiveSampling:
    def __init__(self, horizon: int = 20, n_samples: int = 32, n_knots: int = 4, sigma: float = 0.5, seed: int = 0):
        self.H, self.N, self.K, self.sigma = horizon, n_samples, n_knots, sigma
        self.seed = seed
        self.kidx = np.linspace(0, horizon - 1, n_knots)
        self.W = np.stack([np.interp(np.arange(horizon), self.kidx, np.eye(n_knots)[k]) for k in range(n_knots)], 1)  # (H, K)

    def reset(self, env):
        self.rng = np.random.default_rng(self.seed)
        self.A = np.zeros((self.H, env.nj))
        self.ds = mujoco.MjData(env.m)

    def _rollout(self, env, d0, seq) -> float:
        b, m, ds = env.body, env.m, self.ds
        mujoco.mj_copyData(ds, m, d0)
        ds.xfrc_applied[:] = 0
        total = 0.0
        vcmd, wcmd = env.cmd[0] * env.vscale, env.cmd[1] * 0.5 / env.T
        for a in seq:
            ds.ctrl[:] = np.clip(b.q0 + ACT_SCALE * a, b.jrange[:, 0], b.jrange[:, 1])
            for _ in range(DECIMATION):
                mujoco.mj_step(m, ds)
            R = ds.xmat[b.torso_id].reshape(3, 3)
            up = R[2, 2]
            yaw = np.arctan2(R[1, 0], R[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            vxh = c * ds.qvel[0] + s * ds.qvel[1]
            vyh = -s * ds.qvel[0] + c * ds.qvel[1]
            finite = np.all(np.isfinite(ds.qpos)) and np.all(np.isfinite(ds.qvel))
            if not finite or up < 0.3 or ds.qpos[2] < 0.35 * b.stand_z:
                return total - 2.0
            r_lin = np.exp(-((vxh - vcmd) ** 2 + vyh**2) / (0.2 * env.vscale) ** 2)
            r_yaw = np.exp(-((ds.qvel[5] - wcmd) * env.T / 0.3) ** 2)
            wxy = np.clip(ds.qvel[3:5] * env.T, -5.0, 5.0)
            vz = float(np.clip(ds.qvel[2] / env.vscale, -5.0, 5.0))
            total += r_lin + 0.5 * r_yaw + 0.2 - 0.5 * (1 - up) - 0.02 * float(wxy @ wxy) - 0.1 * vz**2
        return total

    def act(self, env, obs):
        knots = self.A[np.round(self.kidx).astype(int)]  # (K, nj)
        cands = [knots] + [np.clip(knots + self.sigma * self.rng.standard_normal(knots.shape), -1, 1) for _ in range(self.N - 1)]
        scores = [self._rollout(env, env.d, np.clip(self.W @ k, -1, 1)) for k in cands]
        best = np.clip(self.W @ cands[int(np.argmax(scores))], -1, 1)
        a0 = best[0].copy()
        self.A = np.concatenate([best[1:], best[-1:]], 0)
        return a0
