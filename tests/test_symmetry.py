import dataclasses

import numpy as np

from walk_anything.body import Body
from walk_anything.cpg import derive_patterns, symmetry_set
from walk_anything.gait import RhythmController, body_patterns
from walk_anything.sampler import make_body
from walk_anything.symmetry import _equivariant, detect_symmetries, leg_group


def test_quadruped_group_is_klein_four():
    s = make_body(0, 0)[0]
    els = detect_symmetries(s)
    assert [e.name for e in els] == ["identity", "mirror_lr", "mirror_fb", "rot180"]
    assert len(leg_group(els)) == 4
    # default pose is left-right symmetric but the front-back mirror flips the pitch angles
    assert {e.name: e.pose_invariant for e in els} == {"identity": True, "mirror_lr": True, "mirror_fb": False, "rot180": False}


def test_biped_rejects_front_back_mirror():
    s = make_body(0, 1)[0]
    assert [e.name for e in detect_symmetries(s)] == ["identity", "mirror_lr"]  # offset foot breaks front-back symmetry


def test_equivariance_check_rejects_a_wrong_element():
    s = make_body(0, 0)[0]
    el = next(e for e in detect_symmetries(s) if e.name == "mirror_lr")
    body = Body(s)
    assert _equivariant(body, el)
    bad = dataclasses.replace(el, joint_sign=-el.joint_sign)
    assert not _equivariant(body, bad)


def test_quadruped_primary_gaits():
    pats = body_patterns(make_body(0, 0)[0])
    prim = {p.name for p in pats if p.primary}
    assert prim == {"pronk", "bound", "pace", "trot"}
    assert any(p.name == "walk-like" for p in pats)


def test_biped_and_hexapod_patterns():
    assert {p.name for p in body_patterns(make_body(0, 1)[0]) if p.primary} == {"hop", "alternate"}
    assert "tripod" in {p.name for p in body_patterns(make_body(0, 2)[0]) if p.primary}


def test_symmetry_set_definition():
    group = [(0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]
    trot = (0, 2, 2, 0)
    assert len(symmetry_set(trot, 4, group)) == 4
    assert len(symmetry_set((0, 1, 2, 3), 4, group)) == 2  # walk-like keeps only a subgroup
    assert derive_patterns(group, 4)[0].primary


def test_rhythm_action_shape_and_bounds():
    from walk_anything.evalrun import make_eval_env
    s = make_body(0, 1)[0]
    c = RhythmController(s, derive_patterns([(0, 1), (1, 0)], 2)[1])
    e = make_eval_env(s, 0)
    e.reset()
    c.reset(e)
    a = c.act(e, None)
    assert a.shape == (e.nj,) and np.abs(a).max() <= 1
