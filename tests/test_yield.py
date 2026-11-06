"""Yield contract on a CPU trainer: SIGUSR1 and the yield file both produce an atomic checkpoint and a clean exit
well inside 60 s, and a restart resumes from it."""
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _config(tmp_path):
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    cfg.update(envs_per_body=1, workers=2, horizon=8, epochs=1, minibatches=2)
    cfg["bodies"]["indices"] = [0, 1]
    cfg["ckpt"]["every_seconds"] = 10**6          # no periodic saves: only the yield path may write one
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _launch(cfg, run, env_extra=None):
    env = {**os.environ, "OMP_NUM_THREADS": "1", **(env_extra or {})}
    return subprocess.Popen([sys.executable, "-m", "walk_anything.train", "--config", str(cfg), "--run", run, "--device", "cpu"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait_iters(run, n, timeout=120):
    f = ROOT / "runs" / run / "metrics.jsonl"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if f.exists() and sum(1 for _ in open(f)) >= n:
            return
        time.sleep(0.5)
    raise TimeoutError("trainer made no progress")


def _ckpt_iters(run):
    d = ROOT / "checkpoints" / run
    return sorted(int(re.findall(r"\d+", p.name)[0]) for p in d.glob("ckpt_*.pt")) if d.exists() else []


@pytest.mark.parametrize("mode", ["sigusr1", "yieldfile"])
def test_yield_then_resume(tmp_path, mode):
    run = f"pytest_yield_{mode}_{os.getpid()}"
    cfg = _config(tmp_path)
    yf = tmp_path / "gpu.yield"
    p = _launch(cfg, run, {"GPU_RUN_YIELD_FILE": str(yf)})
    try:
        _wait_iters(run, 3)
        assert _ckpt_iters(run) == []
        t0 = time.time()
        if mode == "sigusr1":
            p.send_signal(signal.SIGUSR1)
        else:
            yf.touch()
        out, _ = p.communicate(timeout=60)
        took = time.time() - t0
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, out
    assert took < 30, took
    its = _ckpt_iters(run)
    assert its, out
    assert not list((ROOT / "checkpoints" / run).glob("*.tmp*"))
    yf.unlink(missing_ok=True)

    p2 = _launch(cfg, run, {"GPU_RUN_YIELD_FILE": str(yf)})
    try:
        _wait_iters(run, len(open(ROOT / "runs" / run / "metrics.jsonl").readlines()) + 2)
        p2.send_signal(signal.SIGUSR1)
        out2, _ = p2.communicate(timeout=60)
    finally:
        if p2.poll() is None:
            p2.kill()
    assert f"resumed from iteration {its[-1]}" in out2, out2
    assert max(_ckpt_iters(run)) > its[-1]
