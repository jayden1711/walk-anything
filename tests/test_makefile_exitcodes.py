import os
import subprocess
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_eval_loop_failure_propagates(tmp_path):
    # fake sim-run: fails only for the first body set, succeeds for the rest
    fake = tmp_path / "fakesim"
    fake.write_text('#!/bin/sh\ncase "$*" in *" stop1 "*) exit 7;; esac\nexit 0\n')
    fake.chmod(0o755)
    r = subprocess.run(["make", "-C", str(ROOT), "eval-zero", f"SIM={fake}", "PY=py"],
                       capture_output=True, text=True, env={**os.environ, "MAKEFLAGS": ""})
    assert r.returncode != 0, r.stdout + r.stderr
