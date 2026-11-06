import json
import pathlib
import re
import subprocess
import sys

import pytest

from walk_anything import bodysets
from walk_anything import heldout as H
from walk_anything.gait import select_gait, select_gait_cached
from walk_anything.sampler import make_body, split_of
from walk_anything.spec import BodySpec

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_manifest_hash_matches_code_and_readme():
    assert H.manifest_sha256() == H.MANIFEST_SHA256
    assert H.MANIFEST_SHA256 in (ROOT / "README.md").read_text()


def test_manifest_is_the_sampler_output_and_balanced():
    specs = H.heldout_specs()
    assert len(specs) == H.N_BODIES == 48
    assert [s.family for s in specs].count("biped") == 16
    for i in (0, 1, 2, 47):
        assert make_body(H.SAMPLER_SEED, i)[0] == specs[i]
    assert H.SAMPLER_SEED != bodysets.SAMPLER_SEED


def test_heldout_disjoint_from_tuning_and_train_sets():
    keys = {H._content_key(s) for s in H.heldout_specs()}
    for name in ("train12", "tuning48"):
        assert not keys & {H._content_key(s) for s in bodysets.get(name)}
    assert split_of(3) == "train" and split_of(9) == "tuning"


def test_hash_mismatch_is_refused(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(H.MANIFEST.read_text().replace("quadruped", "biped", 1))
    with pytest.raises(H.HeldoutViolation):
        H._verified(bad)


def test_guard_refuses_gait_search_and_cache_for_heldout_bodies(tmp_path):
    s = H.heldout_specs()[0]
    with pytest.raises(H.HeldoutViolation):
        select_gait(s)
    with pytest.raises(H.HeldoutViolation):
        select_gait_cached(s, tmp_path)
    renamed = BodySpec(**{**s.__dict__, "name": "innocent"})   # a renamed copy is still caught by content
    with pytest.raises(H.HeldoutViolation):
        H.refuse_heldout([renamed], "test")
    H.refuse_heldout(bodysets.get("stop1"), "test")           # ordinary bodies pass


def test_training_and_tuning_scripts_refuse_the_set(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(re.sub(r"seed: \d+", f"seed: {H.SAMPLER_SEED}", (ROOT / "configs/default.yaml").read_text(), count=1))
    r = subprocess.run([sys.executable, "-m", "walk_anything.train", "--config", str(cfg), "--run", "guardtest", "--device", "cpu", "--max-iters", "1"],
                       cwd=tmp_path, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "HeldoutViolation" in r.stderr
    r = subprocess.run([sys.executable, str(ROOT / "scripts/tune_gaits.py"), "heldout_v2", "--gaits", str(tmp_path / "g")],
                       cwd=tmp_path, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "HeldoutViolation" in r.stderr
    assert not (tmp_path / "g").exists()


def test_eval_refuses_arms_that_need_per_body_search():
    for arm in ("rhythm", "residual:checkpoints/x"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_arms.py"), "heldout_v2", arm], cwd=ROOT, capture_output=True, text=True,
                           env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
        assert r.returncode != 0 and "refused" in (r.stderr + r.stdout)
