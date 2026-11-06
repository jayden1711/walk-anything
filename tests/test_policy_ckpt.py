import numpy as np
import pytest
import torch

from walk_anything import ckpt as C
from walk_anything.env import VecEnv, GLOBAL_DIM, JOINT_DIM
from walk_anything.policy import MorphPolicy
from walk_anything.sampler import make_body
from walk_anything.spec import J_MAX


def test_vecenv_local_shapes_and_autoreset():
    specs = [make_body(0, i)[0] for i in (0, 1, 2)]
    v = VecEnv(specs, n_workers=0, seed=1)
    g, j = v.reset()
    assert g.shape == (3, GLOBAL_DIM) and j.shape == (3, J_MAX, JOINT_DIM)
    saw_done = False
    for _ in range(400):
        out = v.step(np.random.default_rng(0).normal(0, 1.0, (3, J_MAX)))
        g, j, r, fell, trunc, fg, fj, infos = out
        assert np.isfinite(g).all() and np.isfinite(r).all()
        saw_done |= bool(fell.any())
    assert saw_done  # random actions with std 1 knock these bodies over


def test_vecenv_workers_match_local():
    specs = [make_body(0, i)[0] for i in (0, 1)]
    a = VecEnv(specs, n_workers=0, seed=3)
    b = VecEnv(specs, n_workers=2, seed=3)
    try:
        ga, _ = a.reset()
        gb, _ = b.reset()
        np.testing.assert_allclose(ga, gb, atol=1e-6)
        act = np.zeros((2, J_MAX))
        for _ in range(5):
            ra = a.step(act)
            rb = b.step(act)
        np.testing.assert_allclose(ra[0], rb[0], atol=1e-5)
    finally:
        b.close()


def test_policy_ignores_padding_and_handles_mixed_joint_counts():
    torch.manual_seed(0)
    p = MorphPolicy()
    specs = [make_body(0, i)[0] for i in (0, 1, 2)]
    v = VecEnv(specs, n_workers=0, seed=1)
    g, j = v.reset()
    g, j, m = map(torch.as_tensor, (g, j, v.mask))
    mean, val = p(g, j, m)
    j2 = j.clone()
    j2[m == 0] = torch.randn_like(j2[m == 0])  # garbage in the padded slots
    mean2, val2 = p(g, j2, m)
    torch.testing.assert_close(mean, mean2)
    torch.testing.assert_close(val, val2)
    assert (mean[m == 0] == 0).all()
    lp, ent = p.dist_stats(mean, m, torch.zeros_like(mean))
    assert lp.shape == (3,) and torch.isfinite(lp).all()


def test_ckpt_atomic_prune_and_corrupt_fallback(tmp_path):
    for it in (1, 2, 3, 4, 5, 10, 20):
        C.save_atomic({"it": it}, tmp_path, it)
    assert not list(tmp_path.glob("*.tmp*"))
    C.prune(tmp_path, keep_last=2, milestone_every=10, max_milestones=1)
    assert [i for i, _ in C.list_ckpts(tmp_path)] == [10, 20]
    (tmp_path / "ckpt_00000030.pt").write_bytes(b"truncated garbage")
    assert C.load_latest(tmp_path)["it"] == 20


def test_ckpt_refuses_when_disk_nearly_full(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MIN_FREE_GB", 10**9)
    with pytest.raises(RuntimeError):
        C.save_atomic({"x": 1}, tmp_path, 1)
