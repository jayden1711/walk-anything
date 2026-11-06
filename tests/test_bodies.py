import numpy as np
import pytest

from walk_anything.parse import parse_body
from walk_anything.sampler import make_body, split_of, sample_spec, FAMILY_ORDER
from walk_anything.spec import BodySpec, J_MAX
from walk_anything.validity import check_body, check_spec, ValidityConfig
from walk_anything.body import Body


@pytest.mark.parametrize("text,family,joints", [
    ("quadruped, torso 0.5 x 0.25 x 0.12 m, 15 kg, legs of 2 segments 0.25 m and 0.25 m", "quadruped", 12),
    ("biped with 0.9 m legs, total mass 40 kg", "biped", 8),
    ("six-legged robot, three segments per leg, leg length 0.3 m, strong", "hexapod", 24),
])
def test_parse_and_validate(text, family, joints):
    s = parse_body(text)
    assert s.family == family
    assert s.n_joints == joints
    if family == "quadruped":
        assert s.seg_lengths == (0.25, 0.25)
    rep = check_body(s)
    assert rep.valid, rep.reasons


def test_parse_values():
    s = parse_body("quadruped, torso 50 x 25 x 12 cm, legs of 2 segments 0.25 m and 0.3 m")
    assert s.seg_lengths == (0.25, 0.3)
    assert s.torso_size == pytest.approx((0.5, 0.25, 0.12))


def test_parse_rejects_unknown():
    with pytest.raises(ValueError):
        parse_body("a blob with some wheels")


def test_spec_json_roundtrip():
    s, _, _ = make_body(0, 1)
    assert BodySpec.from_json(s.to_json()) == s


def test_validity_rejects_bad_specs():
    ok = parse_body("quadruped with 0.4 m legs")
    assert check_body(ok).valid
    bad = BodySpec(**{**ok.__dict__, "torque_limit": ok.torque_limit * 0.02})   # cannot hold itself up
    rep = check_body(bad)
    assert not rep.valid and any(r.startswith("E_TORQUE") or r.startswith("E_SETTLE") for r in rep.reasons)
    assert "E_RANGE:seg_len" in check_spec(BodySpec(**{**ok.__dict__, "seg_lengths": (5.0, 5.0)}), ValidityConfig())
    heavy = BodySpec(**{**ok.__dict__, "seg_masses": (50.0, 50.0)})
    assert "E_RANGE:leg_mass_ratio" in check_spec(heavy, ValidityConfig())


def test_sampler_deterministic_and_valid():
    a, ra, _ = make_body(7, 4)
    b, rb, _ = make_body(7, 4)
    assert a == b and ra.valid
    assert a.family == FAMILY_ORDER[4 % 3]


def test_split_is_stratified():
    held = [i for i in range(200) if split_of(i) == "heldout"]
    for k in range(3):
        n = sum(1 for i in held if i % 3 == k)
        assert 15 <= n <= 18
    assert all(split_of(i) == "train" for i in (0, 1, 2, 3, 4))


@pytest.mark.parametrize("fam,segs", [("quadruped", (0.2, 0.2)), ("biped", (0.3, 0.3)), ("hexapod", (0.1, 0.1, 0.1))])
def test_body_features(fam, segs):
    s = BodySpec(family=fam, seg_lengths=segs, seg_masses=(0.6, 0.4, 0.2)[: len(segs)], foot_len=0.1)
    b = Body(s)
    assert b.nj == s.n_joints <= J_MAX
    assert b.static.shape[0] == J_MAX and not b.static[b.nj:].any()
    assert np.isfinite(b.static).all() and b.stand_z > 0
