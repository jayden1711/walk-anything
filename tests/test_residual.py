import numpy as np
import pytest

from walk_anything import bodysets
from walk_anything.env import BodyEnv, JOINT_DIM, BASE_FEATS, RES_SCALE
from walk_anything.evalrun import evaluate, ZeroController
from walk_anything.gait import RhythmController, body_patterns
from walk_anything.parse import parse_body


def _gait(spec):
    p = body_patterns(spec)[0]
    return RhythmController(spec, p, 4.5, 0.6)


def test_body_sets_disjoint_and_split():
    assert not set(bodysets.TRAIN12) & set(bodysets.HELDOUT12)


def test_parse_ignores_bare_numbers_after_segments():
    s = parse_body("quadruped, torso 0.37 x 0.19 x 0.09 m, legs of 2 segments 0.2 m and 0.2 m, total mass 12 kg")
    assert s.seg_lengths == pytest.approx((0.2, 0.2))


def test_residual_env_shapes_and_zero_residual_equals_rhythm():
    spec = bodysets.get("stop1")[0]
    g = _gait(spec)
    env = BodyEnv(spec, 0, randomize=False, base=g)
    (_, j) = env.reset()
    assert j.shape[1] == JOINT_DIM + BASE_FEATS
    r_zero = evaluate(spec, ZeroController(), 1, base=_gait(spec))
    r_plain = evaluate(spec, _gait(spec), 1)
    assert r_zero["speed"] == r_plain["speed"] and r_zero["steps"] == r_plain["steps"] if "steps" in r_zero else True
    assert r_zero["episodes"][0]["steps"] == r_plain["episodes"][0]["steps"]


def test_residual_is_bounded_by_res_scale():
    spec = bodysets.get("stop1")[0]
    env = BodyEnv(spec, 0, randomize=False, base=_gait(spec))
    env.reset()
    base_a = env.base.act(env, None)
    env.step(np.ones(env.nj))
    assert np.all(np.abs(env.prev_a - np.clip(base_a, -1, 1)) <= RES_SCALE + 1e-9)
