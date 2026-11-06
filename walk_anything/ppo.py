"""Compact PPO (GAE, clipped objective, return-scaled value loss) for the morphology policy."""
from __future__ import annotations

import torch


class Rollout:
    def __init__(self, T, N, gd, jd, jmax, dev):
        z = lambda *s: torch.zeros(T, N, *s, device=dev)
        self.g, self.j, self.a = z(gd), z(jmax, jd), z(jmax)
        self.logp, self.v, self.r, self.done = z(), z(), z(), z()
        self.T = T


def gae(r, v, done, last_v, gamma, lam):
    """r, v, done: (T, N); last_v: (N,) value after the final step. Truncation bootstrap is already inside r."""
    adv = torch.zeros_like(r)
    last = torch.zeros_like(last_v)
    for t in reversed(range(r.shape[0])):
        nv = last_v if t == r.shape[0] - 1 else v[t + 1]
        nonterm = 1.0 - done[t]
        delta = r[t] + gamma * nv * nonterm - v[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + v


def update(policy, opt, ro: Rollout, adv, ret, mask, ret_scale, cfg):
    T, N = ro.r.shape
    B = T * N
    flat = lambda x: x.reshape(B, *x.shape[2:])
    g, j, a, lp_old = flat(ro.g), flat(ro.j), flat(ro.a), flat(ro.logp)
    m = mask[None].expand(T, N, -1).reshape(B, -1)
    adv = flat(adv)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret_n = flat(ret) / ret_scale
    stats = dict(pg=0.0, vf=0.0, ent=0.0, kl=0.0, clipfrac=0.0)
    n_mb = cfg["minibatches"]
    n = 0
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(B, device=g.device)
        for idx in perm.chunk(n_mb):
            mean, v = policy(g[idx], j[idx], m[idx])
            lp, ent = policy.dist_stats(mean, m[idx], a[idx])
            ratio = (lp - lp_old[idx]).exp()
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv[idx]).mean()
            vf = 0.5 * (v - ret_n[idx]).pow(2).mean()
            loss = pg + cfg["vf_coef"] * vf - cfg["ent_coef"] * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            opt.step()
            with torch.no_grad():
                stats["pg"] += pg.item()
                stats["vf"] += vf.item()
                stats["ent"] += ent.mean().item()
                stats["kl"] += ((lp_old[idx] - lp).mean()).item()
                stats["clipfrac"] += ((ratio - 1).abs() > cfg["clip"]).float().mean().item()
            n += 1
    return {k: v / n for k, v in stats.items()}
