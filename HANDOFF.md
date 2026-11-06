# HANDOFF: walk-anything
Status: branch seeds (local, not pushed): seed-variance study of the residual started; seeds 3-9 NOT yet trained (GPU never free for the preemptible trainer), report script ready
Next: let the queue finish (results/_scratch/train_seeds.sh, background; rerun it if dead), then eval seeds 3-9 on 4 sets, run scripts/seeds_report.py -> results/seeds/report.md
Updated: 2026-11-06


## Seed study (branch seeds)
- `scripts/seeds_report.py`: per set/family per-seed values, mean/median/range, bootstrap CI over seeds, paired comparison with rhythm/zero/MPC/PPO. Written, tested on seeds 0-2.
- Training seeds 3-9 queue: `results/_scratch/train_seeds.sh` (gpu-run --preemptible, run names `residual1_s<N>`, resumes; touches results/seeds/done_s<N>). Eval each with
  `make eval-residual RUN=residual1_s<N>` (tag must be residual1_s<N>; checkpoints are in this worktree's checkpoints/), then `.venv/bin/python scripts/seeds_report.py`.
- Old `external` results were copied to results/round2/placeholder (rename only; names and settings identical).

## Decisions taken (seed study)
- GPU was held by robot-framing-test / robot-arena-render and others queued ahead; a preemptible job waits behind them, so no seed was trained in this session. A CPU
  trial was about 7x slower (about 110 min per seed, 1 thread), so it was dropped rather than run for 13 h; its partial output is in results/_trash/.
- Seeds 0-2 are the existing runs; seeds 3-9 will use the same config and command, differing only in `--seed`.

## What was done (branch `round2`, from main which has day-1106 merged)
1. **Residual on the rhythm** (`env.py` residual mode, `policy.py` `joint_dim`, `train.py` `residual`/`--seed`, `configs/residual.yaml`): the applied
   action is `clip(rhythm + RES_SCALE * policy)`; joint tokens get 3 extra features (rhythm action, sin/cos leg phase). Zero residual equals the
   plain rhythm controller (`tests/test_residual.py`). Trained through `gpu-run --preemptible`, 1 env worker, 1500 iterations per run, seeds 0/1/2
   (`make train-residual RUN=... SEED=...`). The KL early stop is active in these runs (`kl_stopped`, `opt_steps` in `runs/*/metrics.jsonl`),
   which verifies the day-1106 fix beyond unit tests.
2. **Per-family evaluation of every arm** (zero, old PPO, predictive sampling, rhythm, rhythm_default control, rhythm+residual) on stop1, train12,
   heldout12 and placeholder sets: `scripts/eval_arms.py`, `scripts/make_report.py`, `walk_anything/bodysets.py`. Fall rate, walk rate (no fall and
   speed >= half the command) and forward speed always together. Output: `results/round2/report.md`.
3. **Bodies**: 12 train-split and 12 held-out generated bodies (fixed index rules in `bodysets.py`); placeholder set `bodies_placeholder/` = 7 placeholder
   descriptions **written by the AI assistant, not by other people**, flagged `[verify]`. Fixed a parser bug found while writing them (bare numbers
   such as "12 kg" after "segments ..." were read as segment lengths; `parse.py`, tested).
4. **Docs**: `CRITIQUES.md` (22 items, each A/B/C), README disclosure, simulation statement and related work (`[verify]` marks), PLAN section 9,
   `Makefile` with one entry point per result.

## Decisions taken
- Residual runs use 1 env worker (owner rule: 1 sim process of my own while humanoid-on-a-chip sims run; none were running when I checked, but 1 worker
  is the conservative choice and is recorded in `configs/residual.yaml`). Runs are short (1500 iterations); no long brute-force training.
- Residual trained on 12 train-split bodies (4 per family) instead of the 5 of stop1, so that held-out generalisation can be reported; rhythm per body is
  taken from the gait cache (grid search on tuning seeds 20000+), exactly as in day-1106.
- Walk threshold (half the commanded speed) is my choice; raw per-episode speeds are in the JSON.
- Predictive sampling kept at 5 episodes per body; other arms 10.
- Old PPO arm = frozen copy of the stop1 checkpoint (iteration 74571) at `results/round2/ckpt/ppo_old` (git-ignored).
- External descriptions: no web access to cite real specs and the owner asked for descriptions by other people, which I cannot produce. I wrote clearly
  labelled placeholders so the pipeline is tested, and did not use them for any generalisation claim.
- Related work: cited only items relevant to this repo. PicoGPT, vla.cpp, Hirundo / CrowdStrike / Booz Allen are not related to this repo and are not cited.
  RoboStriker (2601.22517) and the MIT weight-space post are listed with `[verify relevance]` because I could not check them offline.
- Added an extra control arm (`rhythm_default`) because the per-body rhythm search contradicts "no training per body" (CRITIQUES item 10).
- No `rm` used. `.venv` is a symlink to the overnight worktree's venv (scripts run with `PYTHONPATH=.`); `results` and `.venv` added to `.gitignore`.
  `results/day-1106` was copied (not moved) into this worktree's `results/`.

## Open / blocked
- External test set from other people: not available to me; needs the owner to collect descriptions (format in the existing files).
- Not done: bootstrap intervals, foot-contact analysis, shared circuit-compiler import, LLM text front-end, terrain.
- Recreate a venv in the main checkout before the overnight worktree is deleted.

## Disk
Checkpoints are about 3 MB per run directory; results/round2 is small. NUC disk was far above the 100 GB free limit.
