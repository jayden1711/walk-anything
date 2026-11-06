"""Batching study (PLAN section 5). Subcommands write JSON + a markdown table to results/batching/ (git-ignored).

  sim     CPU: env steps/s vs worker processes and envs per worker   (run via sim-run)
  policy  GPU: policy fwd / fwd+bwd vs batch size, padded vs per-family grouped, update time vs horizon,
               checkpoint save time                                   (run via gpu-run)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

OUT = pathlib.Path("results/batching")
IDX = [0, 3, 1, 4, 2]


def bodies():
    from walk_anything.sampler import make_body
    return [make_body(0, i)[0] for i in IDX]


def table(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return s


def sim(a):
    from walk_anything.env import VecEnv
    from walk_anything.spec import J_MAX
    sp = bodies()
    rows = []
    for workers in (0, 1, 2, 3, 4):
        for epw in (1, 4, 16):
            n = max(workers, 1) * epw
            specs = [sp[i % len(sp)] for i in range(n)]
            v = VecEnv(specs, n_workers=workers, seed=1)
            v.reset()
            rng = np.random.default_rng(0)
            act = lambda: rng.normal(0, 0.3, (n, J_MAX)).astype(np.float32)
            for _ in range(10):
                v.step(act())
            t0, steps = time.time(), 0
            while time.time() - t0 < a.seconds:
                v.step(act())
                steps += 1
            dt = time.time() - t0
            v.close()
            rows.append(dict(workers=workers, envs=n, envs_per_worker=epw, batch_steps_per_s=steps / dt, env_steps_per_s=steps * n / dt))
            print(rows[-1], flush=True)
    return rows


def policy(a):
    import torch
    from walk_anything import ckpt as C
    from walk_anything.env import GLOBAL_DIM, JOINT_DIM
    from walk_anything.policy import MorphPolicy
    from walk_anything.ppo import Rollout, update
    from walk_anything.spec import J_MAX
    import yaml
    cfg = yaml.safe_load(open("configs/default.yaml"))
    dev = torch.device("cuda")
    torch.set_num_threads(1)
    pol = MorphPolicy(**cfg["model"]).to(dev)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-4)
    sp = bodies()
    nj = {s.family: s.n_joints for s in sp}

    def sync():
        torch.cuda.synchronize()

    def bench(fn, reps=30):
        for _ in range(5):
            fn()
        sync()
        t0 = time.time()
        for _ in range(reps):
            fn()
        sync()
        return (time.time() - t0) / reps

    def inputs(B, J):
        g = torch.randn(B, GLOBAL_DIM, device=dev)
        j = torch.randn(B, J, JOINT_DIM, device=dev)
        m = torch.ones(B, J, device=dev)
        return g, j, m

    rows = []
    for B in (80, 320, 1280, 5120, 20480):
        g, j, m = inputs(B, J_MAX)

        def inf():
            with torch.no_grad():
                pol(g, j, m)

        def trn():
            mean, v = pol(g, j, m)
            (mean.sum() + v.sum()).backward()
            opt.zero_grad(set_to_none=True)

        # grouped: three forwards, each trimmed to the family's joint count
        parts = [inputs(B // 3, nj[f]) for f in nj]

        def inf_grouped():
            with torch.no_grad():
                for p in parts:
                    pol(*p)

        rows.append(dict(batch=B, padded_J=J_MAX, infer_ms=1e3 * bench(inf), train_ms=1e3 * bench(trn),
                         grouped_infer_ms=1e3 * bench(inf_grouped), infer_samples_per_s=B / bench(inf)))
        print(rows[-1], flush=True)

    # per-step rollout latency at the real env count and update time vs horizon
    N = len(sp) * cfg["envs_per_body"]
    upd = []
    mask = torch.ones(N, J_MAX, device=dev)
    for T in (16, 32, 64, 128):
        ro = Rollout(T, N, GLOBAL_DIM, JOINT_DIM, J_MAX, dev)
        ro.g.normal_(); ro.j.normal_(); ro.a.normal_(); ro.r.normal_()
        adv, ret = torch.randn(T, N, device=dev), torch.randn(T, N, device=dev)
        with torch.no_grad():
            pol(ro.g[0], ro.j[0], mask)
        sync()
        t0 = time.time()
        for _ in range(5):
            update(pol, opt, ro, adv, ret, mask, 10.0, cfg)
        sync()
        upd.append(dict(horizon=T, envs=N, update_s=(time.time() - t0) / 5, samples=T * N))
        print(upd[-1], flush=True)
    # checkpoint save time
    d = pathlib.Path(a.ckdir)
    ts = []
    for k in range(5):
        st = dict(model=pol.state_dict(), opt=opt.state_dict(), state=dict(iteration=k), cfg=cfg)
        t0 = time.time()
        C.save_atomic(st, d, 10**6 + k)
        ts.append(time.time() - t0)
    ck = dict(save_s_mean=float(np.mean(ts)), save_s_max=float(np.max(ts)), size_mb=C.dir_size_mb(d) / 5)
    print(ck)
    return dict(forward=rows, update=upd, ckpt=ck)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.add_argument("--seconds", type=float, default=8.0)
    p = sub.add_parser("policy"); p.add_argument("--ckdir", default="results/batching/ckpt_probe")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = sim(a) if a.cmd == "sim" else policy(a)
    (OUT / f"{a.cmd}.json").write_text(json.dumps(res, indent=1))
    md = ""
    if a.cmd == "sim":
        md = table(res, ["workers", "envs", "envs_per_worker", "batch_steps_per_s", "env_steps_per_s"])
    else:
        md = table(res["forward"], ["batch", "padded_J", "infer_ms", "grouped_infer_ms", "train_ms", "infer_samples_per_s"])
        md += "\n" + table(res["update"], ["horizon", "envs", "samples", "update_s"]) + f"\ncheckpoint: {res['ckpt']}\n"
    (OUT / f"{a.cmd}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
