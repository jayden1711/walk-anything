"""Preemptible PPO trainer. Run it directly under `gpu-run --preemptible` (see scripts/run_trainer.sh).

Yield contract: SIGUSR1 (or SIGTERM) or the existence of $GPU_RUN_YIELD_FILE -> atomic checkpoint, clean exit.
Start-up: resume from the newest valid checkpoint.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import collections
import hashlib
import json
import pathlib
import signal
import sys
import time

import numpy as np
import yaml


class Yield:
    def __init__(self):
        self.flag = False
        self.path = os.environ.get("GPU_RUN_YIELD_FILE", "")
        for s in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(s, self._h)

    def _h(self, signum, frame):
        self.flag = True

    def __bool__(self):
        return self.flag or bool(self.path and os.path.exists(self.path))


def cfg_hash(cfg: dict) -> str:
    keep = {k: v for k, v in cfg.items() if k not in ("ckpt", "workers", "need_mib")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def log(msg: str):
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="stop1")
    ap.add_argument("--max-iters", type=int, default=10**9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force-config", action="store_true", help="resume even if the config hash differs")
    ap.add_argument("--seed", type=int, default=0, help="env + torch seed for a fresh start (0 = the original stop1 seeds)")
    a = ap.parse_args(argv)

    Y = Yield()  # handlers first: a signal during start-up must not kill us before we can save
    cfg = yaml.safe_load(open(a.config))
    if a.workers is not None:
        cfg["workers"] = a.workers
    chash = cfg_hash(cfg)
    ckdir = pathlib.Path("checkpoints") / a.run
    rundir = pathlib.Path("runs") / a.run
    rundir.mkdir(parents=True, exist_ok=True)

    from .env import VecEnv, GLOBAL_DIM, JOINT_DIM, BASE_FEATS
    from .sampler import make_body
    from .spec import J_MAX

    specs_body = [make_body(cfg["bodies"]["seed"], i)[0] for i in cfg["bodies"]["indices"]]
    epb = cfg["envs_per_body"]
    specs = [s for s in specs_body for _ in range(epb)]
    gaits, jdim, mkw = None, JOINT_DIM, {}
    if cfg.get("residual"):  # residual mode: per-body rhythm from the gait cache (scripts/tune_gaits.py), policy adds a correction
        gpath = pathlib.Path(cfg["gait_dir"])
        gaits_body = [json.load(open(gpath / f"{s.name}.json"))["selected"] for s in specs_body]
        gaits = [g for g in gaits_body for _ in range(epb)]
        jdim, mkw = JOINT_DIM + BASE_FEATS, dict(joint_dim=JOINT_DIM + BASE_FEATS)
    venv = VecEnv(specs, n_workers=cfg["workers"], seed=1234 + a.seed, gaits=gaits)  # workers spawn before CUDA is touched

    import torch
    from . import ckpt as C
    from .policy import MorphPolicy
    from .ppo import Rollout, gae, update

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    policy = MorphPolicy(**cfg["model"], **mkw).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    N, T = venv.n, cfg["horizon"]
    mask = torch.as_tensor(venv.mask, device=dev)
    state = dict(iteration=0, env_steps=0, ret_scale=10.0)

    saved = C.load_latest(ckdir)
    if saved is not None:
        if saved["cfg_hash"] != chash and not a.force_config:
            venv.close()
            sys.exit(f"checkpoint config hash {saved['cfg_hash']} != current {chash}; use --force-config to override")
        policy.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        state.update(saved["state"])
        torch.set_rng_state(saved["rng_cpu"])
        if dev.type == "cuda" and saved.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(saved["rng_cuda"])
        log(f"resumed from iteration {state['iteration']} ({state['env_steps']} env steps)")
    else:
        log("no checkpoint: starting fresh")

    last_saved_it = state["iteration"] if saved is not None else -1

    def save(reason: str):
        nonlocal last_saved_it
        t0 = time.time()
        st = dict(model=policy.state_dict(), opt=opt.state_dict(), state=dict(state), cfg_hash=chash, cfg=cfg,
                  rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                  bodies=[s.name for s in specs_body])
        path = C.save_atomic(st, ckdir, state["iteration"])
        C.prune(ckdir, cfg["ckpt"]["keep_last"], cfg["ckpt"]["milestone_every"], cfg["ckpt"]["max_milestones"])
        last_saved_it = state["iteration"]
        log(f"checkpoint {path.name} ({reason}) saved in {time.time() - t0:.2f}s; dir {C.dir_size_mb(ckdir):.1f} MB")

    ro = Rollout(T, N, GLOBAL_DIM, jdim, J_MAX, dev)
    g_np, j_np = venv.reset()
    last_ckpt_t = time.time()
    metrics = open(rundir / "metrics.jsonl", "a")
    ep_hist = collections.deque(maxlen=200)
    fam = [s.family for s in specs]

    try:
        while state["iteration"] < a.max_iters:
            t_roll = time.time()
            interrupted = False
            ep_new = []
            for t in range(T):
                if Y:
                    interrupted = True
                    break
                g = torch.as_tensor(g_np, device=dev)
                j = torch.as_tensor(j_np, device=dev)
                with torch.no_grad():
                    mean, v = policy(g, j, mask)
                    act = mean + policy.log_std.exp() * torch.randn_like(mean) * mask
                    lp, _ = policy.dist_stats(mean, mask, act)
                g_np, j_np, r, fell, trunc, fg, fj, infos = venv.step(act.cpu().numpy())
                r = torch.as_tensor(r, device=dev)
                done = torch.as_tensor(fell | trunc, device=dev).float()
                if trunc.any():
                    ti = torch.as_tensor(np.nonzero(trunc)[0], device=dev)
                    with torch.no_grad():
                        _, vf = policy(torch.as_tensor(fg, device=dev)[ti], torch.as_tensor(fj, device=dev)[ti], mask[ti])
                    r[ti] += cfg["gamma"] * vf * state["ret_scale"]
                ro.g[t], ro.j[t], ro.a[t], ro.logp[t], ro.v[t], ro.r[t], ro.done[t] = g, j, act, lp, v * state["ret_scale"], r, done
                ep_new += infos
            if interrupted:
                log("yield requested: saving checkpoint")
                if state["iteration"] != last_saved_it:
                    save("yield")
                else:
                    log(f"state unchanged since iteration {last_saved_it} checkpoint; nothing new to save")
                break
            t_sim = time.time() - t_roll
            with torch.no_grad():
                _, lv = policy(torch.as_tensor(g_np, device=dev), torch.as_tensor(j_np, device=dev), mask)
            adv, ret = gae(ro.r, ro.v, ro.done, lv * state["ret_scale"], cfg["gamma"], cfg["lam"])
            state["ret_scale"] = float(max(1.0, 0.98 * state["ret_scale"] + 0.02 * ret.std().item()))
            t_up = time.time()
            st = update(policy, opt, ro, adv, ret, mask, state["ret_scale"], cfg)
            t_up = time.time() - t_up
            state["iteration"] += 1
            state["env_steps"] += N * T
            for k, ret_e, ln, rlin, fell_e in ep_new:
                ep_hist.append((fam[k], k // epb, ret_e, ln, rlin, fell_e))
            per_body = {}
            for b in range(len(specs_body)):
                rows = [e for e in ep_hist if e[1] == b]
                if rows:
                    per_body[specs_body[b].name] = dict(n=len(rows), ret=float(np.mean([e[2] for e in rows])),
                                                        len=float(np.mean([e[3] for e in rows])),
                                                        r_lin=float(np.mean([e[4] for e in rows])),
                                                        fell=float(np.mean([e[5] for e in rows])))
            rec = dict(it=state["iteration"], env_steps=state["env_steps"], reward=float(ro.r.mean()), sim_s=t_sim,
                       update_s=t_up, sps=N * T / (t_sim + t_up), ret_scale=state["ret_scale"], **st, per_body=per_body,
                       t=time.time())
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            if state["iteration"] % 10 == 0:
                log(f"it {state['iteration']} steps {state['env_steps']} rew/step {rec['reward']:.3f} sps {rec['sps']:.0f} (sim {t_sim:.2f}s upd {t_up:.2f}s)")
            if time.time() - last_ckpt_t > cfg["ckpt"]["every_seconds"]:
                save("periodic")
                last_ckpt_t = time.time()
        else:
            save("final")
    finally:
        venv.close()
    log("exit")


if __name__ == "__main__":
    main()
