# HANDOFF: walk-anything
Status: day-1106 branch (local commits, not pushed): KL fix coded + unit tested, symmetry-gait track started, predictive-sampling baseline built; 5-body report in git-ignored results/day-1106/report.md
Next: owner review. Then: re-run the short KL verification when the GPU is free; residual policy on top of the derived rhythm
Updated: 2026-11-06

## What was done today (branch `day-1106`, branched from main)
1. **KL fix** (`walk_anything/ppo.py`, `configs/default.yaml`): lr 3e-4 -> 1e-4, new `target_kl: 0.02`. The epoch loop stops once the
   minibatch approx-KL (k3 estimator, now also the logged `kl`) exceeds the target; `opt_steps` and `kl_stopped` are logged.
   Unit tests: `tests/test_ppo_kl.py`. The old stop1 trainer wrapper was stopped (pid 1388615) and not restarted.
2. **Symmetry-gait track, first stage** (no learning yet):
   - `symmetry.py`: detects body symmetries (identity, left-right, front-back, 180 deg) from geometry, then verifies each one
     numerically (state equivariance of the MuJoCo model + contact-geom match). Outputs leg permutation and joint sign flips.
   - `cpg.py`: enumerates phase patterns from the leg-permutation group; "primary" = every symmetry kept as a spatio-temporal symmetry.
   - `gait.py`: open-loop rhythm controller; picks pattern + (period, amplitude, signs) per body on tuning seeds (20000+),
     separate from the evaluation seeds (10000+).
   - Not built: the universal residual policy, and the shared circuit-compiler import (zero-step-robots).
   - Tests: `tests/test_symmetry.py`.
3. **Predictive-sampling baseline** (`mpc.py`): own implementation of arXiv 2212.00541, CPU only, sim-run. MJPC itself was
   not used (see THIRD_PARTY.md). It plans with the true model and friction.
4. **Common eval** (`evalrun.py`, `scripts/eval_arms.py`, `scripts/make_report.py`): same protocol for every arm
   (fixed forward command, 500 steps, training randomisation). Arms run: zero action, old stop1 PPO checkpoint (read from the
   overnight worktree), predictive sampling, symmetry rhythm. Output: `results/day-1106/` (git-ignored).

## Decisions taken
- Own predictive-sampling implementation instead of building MJPC (conservative: no new build, no checkout, no pinned third party).
- Treated front-back mirror and 180 deg rotation as part of the symmetry group even though they reverse the travel direction
  (this is what the symmetric-CPG theory uses for the network); the default pose is not invariant under them (recorded per element).
- Candidate symmetries limited to the torso-rectangle group; the bodies here are not radial.
- Rhythm waveform and its parameter grid (3 periods x 2 amplitudes x 2 hip signs x 2 knee signs) were my choice; selection uses
  tuning seeds only. A sparse, hand-made waveform: treat the arm as a first rhythm-only reference, not a tuned one.
- Evaluation protocol (command 0.3, 500 steps, 10 episodes; 5 for predictive sampling because it is slow) is my choice for the
  "same metrics as stop1" request, since stop1 had no eval script. The old checkpoint was evaluated with the same code.
- CPU budget: another project already ran 4 workers, so I ran only one sim process at a time (MPC, then rhythm), sequentially.
- No `rm` used. The pytest yield tests left `checkpoints/pytest_yield_*` and `runs/pytest_yield_*` dirs in this worktree (git-ignored).
- `.venv` in this worktree is a symlink to the overnight worktree's venv (its editable install points there, so scripts run
  with `PYTHONPATH=.`). Recreate a venv in the main checkout before the overnight worktree is deleted.

## Open / blocked
- **KL verification run not done**: the single <=15 min run under `gpu-run --preemptible` never got the GPU (held by the
  robot-framing-test server for the whole window; 900 s limit expired while waiting). Only the unit tests verify the fix so far.
  Command when the GPU is free: `timeout -s TERM 900 scripts/run_trainer.sh --run klfix --workers 2`
- No further tuning of the brute-force universal policy (per owner).
- PLAN.md was not changed; docs/symmetry-gait-track.md still needs folding in at the next review.

## Disk
Checkpoints are small. Nothing large was added today. NUC disk had well over 100 GB free (limit: keep > 100 GB free).
