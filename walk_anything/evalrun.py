"""Common evaluation protocol: fall rate and forward speed per body, identical for every controller arm.

A controller is any object with reset(env) and act(env, obs) -> action (nj,) in [-1, 1].
Protocol: fixed forward command, same randomisation as training (init noise, friction, pushes), same env seeds per arm.
"""
from __future__ import annotations

import numpy as np

from .env import BodyEnv, EP_LEN
from .spec import BodySpec, J_MAX

EVAL_CMD = (0.3, 0.0)   # nondimensional (x sqrt(g h), x 0.5 sqrt(g/h)); inside the training range (V_MAX = 0.4)
WALK_FRAC = 0.5         # "walked" = no fall and mean forward speed >= WALK_FRAC x commanded speed (our own threshold)
EVAL_STEPS = 500        # control steps (10 s at 50 Hz)


def make_eval_env(spec: BodySpec, seed: int, base=None) -> BodyEnv:
    e = BodyEnv(spec, seed, randomize=True, base=base)

    def fixed_cmd():
        e.cmd[:] = EVAL_CMD

    e._sample_cmd = fixed_cmd
    return e


def run_episode(env: BodyEnv, ctrl, steps: int = EVAL_STEPS) -> dict:
    assert steps <= EP_LEN
    obs = env.reset()
    ctrl.reset(env)
    vx, fell = [], False
    for _ in range(steps):
        a = np.asarray(ctrl.act(env, obs), dtype=float)
        obs, r, fell, trunc = env.step(a)
        R = env.d.xmat[env.body.torso_id].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        vx.append(np.cos(yaw) * env.d.qvel[0] + np.sin(yaw) * env.d.qvel[1])
        if fell:
            break
    speed = float(np.mean(vx))
    return dict(fell=bool(fell), steps=len(vx), speed=speed, speed_nd=speed / env.vscale,
                walked=bool((not fell) and speed >= WALK_FRAC * EVAL_CMD[0] * env.vscale),
                r_lin=env.ep_rlin / max(env.t, 1))


def evaluate(spec: BodySpec, ctrl, n_episodes: int, seed0: int = 10_000, base=None) -> dict:
    """base: gait controller for residual mode (ctrl then outputs the residual only)."""
    eps = [run_episode(make_eval_env(spec, seed0 + k, base), ctrl) for k in range(n_episodes)]
    return dict(body=spec.name, family=spec.family, n=n_episodes,
                fall_rate=float(np.mean([e["fell"] for e in eps])),
                speed=float(np.mean([e["speed"] for e in eps])),
                speed_nd=float(np.mean([e["speed_nd"] for e in eps])),
                walk_rate=float(np.mean([e["walked"] for e in eps])),
                r_lin=float(np.mean([e["r_lin"] for e in eps])),
                mean_steps=float(np.mean([e["steps"] for e in eps])), episodes=eps)


class ZeroController:
    """Hold the default standing pose."""

    def reset(self, env):
        pass

    def act(self, env, obs):
        return np.zeros(env.nj)


class PolicyController:
    """Deterministic (mean-action) universal policy from a training checkpoint, on CPU."""

    def __init__(self, ckpt_state: dict):
        import torch
        from .policy import MorphPolicy
        from .env import JOINT_DIM, BASE_FEATS
        self.torch = torch
        mc = dict(ckpt_state["cfg"]["model"])
        if ckpt_state["cfg"].get("residual"):
            mc["joint_dim"] = JOINT_DIM + BASE_FEATS
        self.pol = MorphPolicy(**mc)
        self.pol.load_state_dict(ckpt_state["model"])
        self.pol.eval()

    def reset(self, env):
        pass

    def act(self, env, obs):
        t = self.torch
        with t.no_grad():
            g, j = (t.as_tensor(x)[None] for x in obs)
            m = t.zeros(1, J_MAX)
            m[0, : env.nj] = 1
            mean, _ = self.pol(g, j, m)
        return mean[0, : env.nj].numpy()
