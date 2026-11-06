# walk-anything: PLAN

_Status: **approved by the owner (2026-11-06, pre-approved for the overnight run).** Stop point 1 below is the
scope of the first build. Later stop points are proposals and need a fresh look._

Goal: a text description of a robot body (legs, joints, sizes, masses) → a body that one shared policy
makes walk right away, with no training per body.

Fixed constraints (from the owner):
- Always-on, lowest-priority background GPU trainer on the NUC, always through
  `~/projects/bin/gpu-run --preemptible` (see `~/projects/NUC_RULES.md`). On SIGUSR1 / `$GPU_RUN_YIELD_FILE`
  it checkpoints and exits within 60 s; on restart it resumes from the latest checkpoint.
- Public repo, Apache-2.0. Checkpoints, runs, logs, videos and eval numbers stay in git-ignored dirs until
  the owner approves. No result numbers in README/PLAN/HANDOFF.

## 1. Pipeline

```
text ──parse──▶ BodySpec (validated dataclass / JSON) ──generate──▶ MJCF ──check──▶ valid body
                                                                         │
                       one shared body-conditioned policy ◀──────────────┘  (no per-body training)
```

### 1.1 Body description
- **BodySpec**: family, torso box size and mass, per-leg-group parameters (leg count, segment count, segment
  lengths, radii, masses, hip offsets, joint ranges, torque limit, PD gains), foot geometry.
- **Text → BodySpec**: a deterministic grammar/keyword parser for the first stop point
  ("quadruped, torso 0.4 m x 0.2 m, 12 kg, legs 2 segments of 0.2 m, ..."). Unspecified fields take family
  defaults. An LLM front-end that emits the same BodySpec JSON is a later stop point; the schema is the contract.
- Every spec passes through the **validity checker** before it can be used for training or evaluation.

### 1.2 Body families (stop point 1: three)
| Family | Legs | Joints per leg | Notes |
|---|---|---|---|
| quadruped | 4 | hip roll, hip pitch, then (segments-1) pitch joints | trot-style gaits |
| biped | 2 | hip roll, hip pitch, then pitch joints (+ ankle) | the hard balance case |
| hexapod | 6 | same pattern | statically stable |

All are generated procedurally (no URDF/third-party model needed), so there is nothing to put in THIRD_PARTY
beyond the Python dependencies. Max joints (hexapod, 3 per leg) = 18; the policy pads to `J_MAX`.

### 1.3 Validity checker (`walk_anything/validity.py`)
A body is **valid** only if all of these hold; each failure has a reason code:
1. spec fields in documented ranges (lengths, masses, ratios such as leg mass / torso mass);
2. MJCF compiles in MuJoCo; no NaNs in mass/inertia; total mass within bounds;
3. default standing pose is collision-free (no self-contact) and feet reach the ground with the torso at a
   sane height;
4. static torque feasibility: gravity-compensating torques at the standing pose are below a fraction of the
   torque limit (so a body that cannot hold itself up is rejected);
5. settle test: hold the default pose with PD for ~1 s; torso must stay above a height threshold and within a
   tilt limit (bipeds use a short stabilised settle, see below), no divergence.
Bipeds that cannot stand passively get no free pass: the settle test uses a fixed-window pass criterion that
is the same for all families, and thresholds are config values, recorded alongside the dataset.

### 1.4 Body dataset
- **200 bodies**: seeded sampler over the three families (roughly balanced), rejection-sampled through the
  validity checker. Deterministic from `(seed, index)`, so the dataset is reproducible and only the generator
  is committed. Materialised specs go in git-ignored `results/bodies/`.
- Fixed split: train / held-out, stratified by family. The held-out bodies are never used for training
  decisions; they are for zero-shot evaluation.

## 2. Simulator
**MuJoCo (Apache-2.0), CPU**, driven by ≤4 worker processes at one thread each (NUC rules), nice 10 via
`sim-run`. Policy inference and PPO updates run on the GPU inside the preemptible gpu-run job.
Reasons: license, procedural MJCF is easy, one dependency, robust on Blackwell without CUDA sim builds.
Trade-off: CPU sim throughput is the bottleneck. MuJoCo Warp / MJX (GPU physics) is the stated upgrade path if
the batching study shows the sim dominating; it is *not* in stop point 1.

## 3. Policy and training
- **Policy**: morphology-token transformer. One token per joint (features: joint axis and position in the parent
  frame, limb length/mass, range, torque limit, current q, qd, previous action) + one global token (gravity
  direction, base angular velocity, command). Small (2 layers, width 64). Outputs per-joint PD target offsets.
  Padding mask handles different joint counts, so the same weights drive all bodies. The body description
  enters only through the token features: that is the "no training per body" conditioning.
- **Command**: forward speed and yaw rate. **Reward**: velocity tracking, upright, torque and action-rate
  penalties, fall termination (a joint-limit penalty was planned and is not implemented).
- **PPO** (own compact implementation, GAE, clipped objective, per-body observation/return normalisation is
  *not* used; global normalisation only). Randomisation: friction, push disturbances (small mass noise: planned, not implemented).
- **Stop point 1 training scope**: PPO on **5 bodies** (2 quadrupeds, 2 bipeds, 1 hexapod) from the train
  split, to prove the loop, the yield contract and the batching choices. Training on all train bodies and
  zero-shot evaluation on held-out bodies is stop point 2.

## 4. Always-on trainer: yield, checkpoints, disk
- Launch: `~/projects/bin/gpu-run --preemptible --need-mib <N> --label walk-anything -- python -m walk_anything.train ...`
  (the trainer runs directly under gpu-run, no intermediate shell, so SIGUSR1 arrives). Env workers are started
  inside the trainer with `sim-run`-equivalent settings (nice, 1 thread, ≤4).
- **Yield**: SIGUSR1 handler sets a flag; the trainer also polls `$GPU_RUN_YIELD_FILE` every iteration and
  between rollout chunks. On yield: finish the current step, write an atomic checkpoint (temp file + `os.replace`
  + fsync), shut down workers, exit 0 within the 60 s budget. Rollouts are chunked so the flag is seen within
  seconds. Checkpoints are also written periodically so a hard kill loses little.
- **Resume**: at startup load the newest valid checkpoint (model, optimiser, normaliser, iteration, RNG,
  config hash); refuse to resume under a different config unless forced.
- **Yield test** (`scripts/yield_test.sh`): (a) start the trainer under gpu-run --preemptible, (b) start a normal
  gpu-run job that forces a yield, (c) assert a checkpoint appears and the trainer exits well within 60 s,
  (d) assert it resumes from that checkpoint afterwards. Also a unit test that signals the trainer directly
  with SIGUSR1 and with the yield file.
- **Disk**: keep the last 3 checkpoints plus a milestone every N iterations (milestones thinned to a cap).
  Checkpoints are small (policy + optimiser). Usage reported in HANDOFF.md (the NUC disk is shared, keep
  >100 GB free; the trainer refuses to write if free space is below that).

## 5. Batching study (stop point 1)
Question: how to batch heterogeneous bodies to keep the GPU fed without starving the 4 CPU workers.
Measure, for the real sim and policy, with the machine otherwise as it is:
1. sim steps/s vs env workers (1..4) and envs per worker;
2. policy forward/backward throughput vs batch size and padding (`J_MAX`) on the GPU;
3. mixed-body batches (padded, one forward) vs per-family grouped batches;
4. rollout length vs update frequency, and checkpoint save time (for the 20 s rule).
Output: a short write-up and raw tables in git-ignored `results/batching/`, and the chosen defaults in
`configs/`. No numbers in git docs.

## 6. Repository layout
```
walk_anything/   spec.py, parse.py, mjcf.py, validity.py, sampler.py, env.py, policy.py, ppo.py, train.py, ckpt.py
configs/         default.yaml (+ thresholds)
scripts/         make_bodies.py, batching_study.py, yield_test.sh, run_trainer.sh
tests/           pytest
results/ runs/ checkpoints/   git-ignored
```
Python deps (torch, numpy, mujoco, pyyaml, pytest) go into a local `.venv`, listed in THIRD_PARTY.md.

## 7. Stop points
1. **(this build)** 3 families + validity checker; 200 bodies; PPO on 5 bodies; batching study; yield test.
2. Train on the full train split; zero-shot evaluation on held-out bodies; failure analysis per family.
3. Text front-end (LLM → BodySpec), more families / terrain, sim-to-real or Jetson deployment questions.

## 8. Risks
- A procedural biped may not learn to walk in the overnight budget; stop point 1 only needs the loop to run
  correctly end to end, learning quality is judged later.
- CPU sim throughput; mitigation above (Warp/MJX path).
- The always-on trainer is preempted often by Quatern; resume correctness matters more than speed.

## 9. Symmetry-derived rhythm + learned residual (owner decision 2026-11-06; supersedes scaling the brute-force policy for now)
1. `symmetry.py` detects a body's symmetries and verifies them numerically; `cpg.py` turns the leg-permutation group into phase patterns
   (symmetric-CPG theory); `gait.py` is an open-loop rhythm built from a pattern. The rhythm parameters (pattern, period, amplitude, signs) are
   currently **selected per body by a small grid search in simulation on tuning seeds**. That is per-body tuning and is stated as a limitation;
   the `rhythm_default` arm (no search) measures how much it matters.
2. The residual policy is the same morphology-token transformer with 3 extra per-joint features (the rhythm action and the sine / cosine of its
   leg phase). The applied action is `clip(rhythm + RES_SCALE * residual)`. Short, preemptible runs (`configs/residual.yaml`,
   `make train-residual`), 1 env worker, no long brute-force training.
3. Evaluation (same protocol for every arm: zero action, old PPO, predictive sampling, rhythm, rhythm + residual): fall rate, walk rate and forward
   speed **per body family**, on train bodies, held-out bodies and the placeholder descriptions. Fall rate is never reported without forward speed.
4. Body sets: generated bodies are a pure function of (sampler seed, index); the held-out list is a fixed rule (`bodysets.py`). The placeholder set
   (`bodies_placeholder/`) is meant to be replaced by descriptions written by other people; the current files are placeholders written by the AI assistant
   and support no generalisation claim. The held-out list is held out from PPO / residual training only: the per-body rhythm search is also run on held-out bodies.
5. Still not done: import of the shared circuit compiler (zero-step-robots), an LLM text front-end, terrain, any real hardware.

Related work is listed in README.md; see CRITIQUES.md for the open criticisms and how each one is handled.
