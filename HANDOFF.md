# HANDOFF: walk-anything
Status: stop point 1 code complete in worktree branch worktree-overnight-stop1 (local commits, not pushed); GPU-side items see "Open" below
Next: review + push the branch, then stop point 2 (train on the full train split, zero-shot eval on held-out bodies)
Updated: 2026-11-06

## What exists (stop point 1)
- `walk_anything/`: spec, text parser, procedural MJCF for 3 families (quadruped / biped / hexapod), validity checker,
  seeded sampler (200 bodies, stratified train/held-out split), MuJoCo env + multiprocess VecEnv, morphology-token
  transformer policy, compact PPO, atomic checkpoints, preemptible trainer (`python -m walk_anything.train`).
- `scripts/`: `make_bodies.py` (writes `results/bodies/`), `batching_study.py` (`sim` and `policy` subcommands),
  `run_trainer.sh` (the only way to launch the trainer: preemptible gpu-run), `yield_test.sh` (end to end).
- `tests/`: 19 pytest tests (bodies, policy/checkpoint round trip, yield by SIGUSR1 and by yield file).
- Venv: `.venv` in the worktree (torch, mujoco, numpy, pyyaml, pytest); listed in THIRD_PARTY.md.

## Verified tonight
- pytest: all pass (run under sim-run).
- `scripts/yield_test.sh`: PASS. A normal gpu-run job forced the preemptible trainer to yield, the checkpoint
  appeared and the trainer exited within a few seconds, and the restart resumed from that checkpoint. It uses a
  private gpu-run lock dir and a CPU trainer, so it never touched the real GPU lock.
- 200 bodies generated into `results/bodies/0/` (git-ignored).
- CPU batching study written to `results/batching/sim.md` (git-ignored). Numbers stay there.

## Decisions taken overnight
- Fixed a reward blow-up seen in the previous session's smoke run (one quadruped had an absurd episode return):
  velocity penalties are now clipped and a non-finite state counts as a fall.
- No `rm` used (per instruction). The yield test leaves its temp dir under `$CLAUDE_JOB_DIR` and its run
  dirs under `runs/` and `checkpoints/`.
- Batching GPU study and trainer were queued behind another project's non-preemptible GPU job instead of being
  forced; see Open.
- The yield test runs the trainer on CPU with a private lock dir (conservative: no GPU use outside the real lock).

## Open / blocked
- Left-over git-ignored dirs from the previous session's tests, to delete by hand
  (`rm -rf runs/pytest_yield_* checkpoints/pytest_yield_* runs/smoke checkpoints/smoke`), plus `runs/yieldtest_*`
  and `checkpoints/yieldtest_*` from tonight's yield test.
- GPU status: see the final section below.

## Disk
Checkpoints are small (policy + optimiser). Usage is logged by the trainer at each save. NUC disk had ~195 GB free
at the start of the night (limit: keep > 100 GB free).

## GPU status at end of the night
- Another project's (edge-vla-bench) non-preemptible gpu-run job held the GPU lock for the whole session, so
  no GPU work ran: the GPU half of the batching study (`scripts/batching_study.py policy`, queued under gpu-run
  with a 3000 s wait; may have timed out) and PPO training on the 5 bodies are NOT done.
- The preemptible trainer wrapper is running detached and waiting for its turn
  (`scripts/run_trainer.sh --run stop1`; log `runs/stop1_wrapper.log`, metrics `runs/stop1/metrics.jsonl`).
  It starts by itself when the GPU frees up and yields to any normal gpu-run job or Quatern.
  To stop it: `pkill -TERM -f 'gpu-run --preemptible'` (it checkpoints on SIGTERM).
- To run the GPU batching study by hand: `~/projects/bin/gpu-run --need-mib 3072 --label walk-anything --wait 600 --
  ~/projects/bin/sim-run .venv/bin/python scripts/batching_study.py policy` (the trainer yields to it).
- After the study, set final defaults in `configs/default.yaml` from `results/batching/` (current ones are
  untuned guesses; the CPU study says sim throughput scales with workers up to 4 and larger envs-per-worker).
