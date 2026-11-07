# HANDOFF: walk-anything
Status: branch seeds (local, not pushed): all 10 seeds exist (3-9 trained on CPU as residual1_s<N>_cpu, GPU unavailable); tuning and heldout_v2 evaluated for all 10; reports regenerated, CRITIQUES R18-R19
Next: owner review of results/heldout_v2/seeds/report.md and results/seeds/report.md; stop the stalled GPU queue (results/_scratch/train_seeds.sh, seed 3 waiting) once the owner agrees; goal still unmet, decide next method step
Updated: 2026-11-07

## Owner decisions of 2026-11-06 and where they landed
1. Tuning set: `tuning12` / `tuning48` (old held-out split; split label `tuning`; `ph_` prefix for placeholder bodies); renamed in code, Makefile, report, CRITIQUES, PLAN, README.
2. Fresh held-out set `heldout_v2`: 48 bodies (16 per family), sampler seed 20261106, same sampler and validity rules, `bodies_heldout_v2/manifest.json` (chmod a-w), SHA-256 in
   `walk_anything/heldout.py` and README. Size fixed up front: matches the tuning set, 16 per family, not changed after seeing results. Guard: `refuse_heldout()` in gait
   search (`select_gait`, `select_gait_cached`) and `train.py`; `eval_arms.py` allows only search-free arms, fixed episodes, never overwrites a complete result.
   Tests: `tests/test_heldout.py`.
3. Evaluation of every arm on the held-out set per family (fall rate and speed, CIs over seeds and bodies): `make eval-heldout`, `make eval-heldout-mpc`, `make eval-heldout-report`
   -> `results/heldout_v2/report.md`. Old tuning set kept and labelled in `results/round2/report.md` and `results/seeds/report.md`.
4. `make report` rerun; old `external` folder, report and `ext_` gait files moved to `results/_trash/pre_rename/`, placeholder results regenerated under the new names.
5. `scripts/review_compare.py` kept (docstring, `make compare`); `scripts/review_stats.py` moved to `results/_scratch/` (hard-coded old arms, superseded by `scripts/seeds_report.py`).
   The yield test's inline heredoc is now `scripts/yield_test_cfg.py`. No repo command relies on an inline heredoc any more.
6. CRITIQUES.md: R13 to R17 (tuning vs held-out, the guard, hash, arms changed, seed intervals), no numbers. Results: `results/seeds/`, `results/heldout_v2/`.

## Decisions taken
- The held-out set may not be searched or tuned on, but `rhythm` and `rhythm+residual` need a per-body gait search. Conservative choice: on `heldout_v2` the arms are zero, old PPO,
  predictive sampling, `rhythm_default` and `residual_default` (residual policy on the fixed default rhythm). Tuned `rhythm` / `residual` are reported on the tuning set only (CRITIQUES R16).
  A fair held-out test of tuned rhythms would need a rule fixed on the tuning set and applied unchanged (e.g. one gait per family); not done.
- Held-out sampler seed 20261106 (the date), episodes 10 per body (5 for predictive sampling, as elsewhere). The held-out manifest is committed (specs only, no results).
- Seeds 3-9 (2026-11-07 overnight): trained on CPU, one process at a time through sim-run (`results/_scratch/cpu_seeds.sh`, chunks under 10 min, resumes; seeds in descending order 9..3), run names `residual1_s<N>_cpu`,
  1500 iterations, same config. `results/seeds/done_s<N>` contains "cpu". The GPU queue (`train_seeds.sh`) stayed stalled at seed 3 (GPU not usable: nvidia-smi "Unknown Error"); seed 3 was
  therefore also trained on CPU (FORCE_SEED=3, the GPU run was waiting, not training). The first to finish is used (CPU for all of 3-9). The Makefile picks the checkpoint dir from `done_s<N>` (`CKDIR`).
  Older checkpoints of the CPU runs moved to `results/_trash/ckpt_prune/` (kept: final only, which is also latest). Checkpoints for seeds 0-2 were copied from the round2 worktree.
  Gotcha: `timeout` must be `--foreground`, else the env worker also gets SIGUSR1 and dies. `eval-residual-seeds` restarts at seed 3 each call and a kill mid-run leaves a partial JSON (one tuning48 file had to be redone).
- Held-out seeds report now goes to `results/heldout_v2/seeds/report.md` (make_report overwrote it at `results/heldout_v2/report.md`).
- Reports use two-level bootstrap (seeds and bodies, 4000 draws); `scripts/seeds_report.py` rewritten in numpy.
- One sim process of my own at a time, started only when load < 20 (`results/_scratch/rerun.sh` gates on it, idempotent, chunks under 10 min).

## Seed study (branch seeds)
- Seeds 0-2 exist; seeds 3-9 queue `results/_scratch/train_seeds.sh` (gpu-run --preemptible, run names `residual1_s<N>`, resumes; touches `results/seeds/done_s<N>`), blocked as above.
  Check `pgrep -af train_seeds`; do not start a second copy.
- Eval new seeds: `make eval-residual-seeds`; held-out: `make eval-heldout`.

## Earlier work (branch `round2`, merged to main)
Residual on the rhythm (`env.py`, `policy.py`, `train.py`, `configs/residual.yaml`), per-family evaluation of every arm (`scripts/eval_arms.py`, `make_report.py`, `bodysets.py`),
12 train + tuning bodies, placeholder descriptions (7, AI-written, `[verify]`), `CRITIQUES.md`, Makefile with one entry point per result. Residual runs: 1 env worker, 1500 iterations, seeds 0-2.
Walk threshold (half the commanded speed) is my choice; raw per-episode speeds are in the JSON. Old PPO arm = frozen stop1 checkpoint at `results/round2/ckpt/ppo_old` (git-ignored).
`.venv` is a symlink to the overnight worktree's venv (scripts run with `PYTHONPATH=.`); recreate a venv in the main checkout before that worktree is deleted.

## Open / blocked
- Seeds are done. Descriptions from other people for the public post (owner to collect). Not done: foot-contact analysis, shared circuit-compiler import, LLM text front-end, terrain, long residual runs.

## Disk
Checkpoints about 1 MB per kept run directory (final only for the CPU runs); results small. NUC disk free was about 87 GB on 2026-11-07 (floor now 50 GB per NUC_RULES; this repo uses well under 100 MB).

## Run results (2026-11-06, evening)
- Done: tuning12/tuning48/placeholder re-run under the new names for zero, old PPO, rhythm, rhythm_default, residual seeds 0-2 (MPC on tuning12 and placeholder); `make report`, `make seeds-report`
  regenerated (`results/round2/report.md`, `results/seeds/report.md`). Old `external`/`heldout12` outputs, `ext_` gait files and logs that name `external` are in `results/_trash/pre_rename/`.
- Done: `heldout_v2` evaluated once with zero, old PPO, MPC, rhythm_default, residual_default for seeds 0-2: `results/heldout_v2/report.md` (+ `make eval-heldout-report`). Update 2026-11-07: seeds 3-9 added, all arms re-reported with 10 seeds (`results/heldout_v2/seeds/report.md`, `results/seeds/report.md`, `results/round2/report.md`); no held-out arm was re-run.
- Test suite: 3 tests in `tests/test_yield.py` failed on 2026-11-06 under the old 100 GB floor; rerun 2026-11-07: all 40 pass.
- Held-out results were read after the run; no arm was re-run, and nothing was tuned on them.
