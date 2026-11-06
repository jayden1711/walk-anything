import torch

from walk_anything.env import GLOBAL_DIM, JOINT_DIM
from walk_anything.policy import MorphPolicy
from walk_anything.ppo import Rollout, update
from walk_anything.spec import J_MAX


def _setup(seed=0, T=8, N=16):
    torch.manual_seed(seed)
    pol = MorphPolicy(d=16, layers=1, heads=2)
    ro = Rollout(T, N, GLOBAL_DIM, JOINT_DIM, J_MAX, "cpu")
    ro.g.normal_(), ro.j.normal_()
    mask = torch.zeros(N, J_MAX)
    mask[:, :8] = 1
    with torch.no_grad():
        for t in range(T):
            mean, v = pol(ro.g[t], ro.j[t], mask)
            ro.a[t] = (mean + pol.log_std.exp() * torch.randn_like(mean)) * mask
            ro.logp[t], _ = pol.dist_stats(mean, mask, ro.a[t])
            ro.v[t] = v
    adv = torch.randn(T, N)
    return pol, ro, adv, adv + ro.v, mask


def _cfg(**kw):
    c = dict(minibatches=4, epochs=10, clip=0.2, vf_coef=1.0, ent_coef=0.0, max_grad_norm=1.0)
    c.update(kw)
    return c


def test_kl_early_stop_cuts_epochs_and_logs():
    pol, ro, adv, ret, mask = _setup()
    opt = torch.optim.Adam(pol.parameters(), lr=3e-2)
    st = update(pol, opt, ro, adv, ret, mask, 1.0, _cfg(target_kl=1e-4))
    assert st["kl_stopped"] == 1.0
    assert 1 <= st["opt_steps"] < 10 * 4


def test_no_target_runs_all_epochs():
    pol, ro, adv, ret, mask = _setup()
    opt = torch.optim.Adam(pol.parameters(), lr=1e-4)
    st = update(pol, opt, ro, adv, ret, mask, 1.0, _cfg())
    assert st["opt_steps"] == 40 and st["kl_stopped"] == 0.0


def test_loose_target_not_triggered_and_kl_nonneg():
    pol, ro, adv, ret, mask = _setup()
    opt = torch.optim.Adam(pol.parameters(), lr=1e-4)
    st = update(pol, opt, ro, adv, ret, mask, 1.0, _cfg(target_kl=10.0))
    assert st["opt_steps"] == 40 and st["kl_stopped"] == 0.0 and st["kl"] >= 0.0
