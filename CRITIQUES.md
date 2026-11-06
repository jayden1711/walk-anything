# CRITIQUES: what a skeptical expert would say, and how each point is handled

No result numbers in this file. Numbers live in the git-ignored `results/` and are regenerated with the `make` targets named below.
Each item is exactly one of: **[A] addressed** (where), **[B] extra experiment** (what, and whether it was run), **[C] limitation** we state up front.
Everything in this repo is simulation (MuJoCo, procedurally generated bodies). No real robot was used.

Result files (all git-ignored, each JSON carries seeds, settings, date, git revision and the command that made it):
`results/round2/<set>/<arm>.json`, `results/round2/report.md` (`make report`), `results/round2/gaits/<body>.json` (rhythm selection tables),
`runs/<run>/metrics.jsonl` (training), `results/round2/logs/`.

## Evaluation validity

1. **"Standing still counts as not falling."** [A] Every table reports forward speed and a `walk` rate (no fall AND mean forward speed at least half
   the commanded speed) next to the fall rate; the `zero` arm (hold the standing pose) is in every table as the control. `walk_anything/evalrun.py`
   (`run_episode`, `WALK_FRAC`), `scripts/make_report.py`. The 0.5 threshold is our choice [C]; the raw per-episode speeds are in the JSON so
   others can apply their own.
2. **"Bodies were cherry-picked."** [A] Generated bodies are a pure function of `(sampler seed 0, index)` (`walk_anything/sampler.py`); every one of the
   200 is materialised by `make bodies` into `results/bodies/0/` (spec, validity report, number of rejected draws, split) and nothing is dropped after
   generation. The train and held-out lists are fixed index rules written before evaluation (`walk_anything/bodysets.py`: the first 4 bodies per
   family of each split). To be published together with the sampler seed. [C] The validity checker rejects bodies that cannot stand; the number of
   rejected draws per body is recorded, and the surviving distribution is not "all possible robots".
3. **"The test set was written by the authors."** [C] The `external` set (`bodies_external/*.txt`) currently holds seven descriptions written by the AI
   assistant from memory of public robot sizes or invented size classes; each file says so and is marked `[verify]`. They are placeholders, not an independent
   test set. [B] Not run: collect descriptions from other people (or copy published robot descriptions with the source cited), drop them into
   `bodies_external/` with `source`/`author` headers, run `make eval-*` again. The set is evaluated and reported separately and not mixed into the
   held-out numbers. Descriptions must fit the rule-based grammar (`walk_anything/parse.py`) until an LLM front-end exists, which limits how free-form
   they can be [C].
4. **"Train/test leakage."** [A] Reports are split into `stop1` and `train12` (in-sample for the residual; PPO saw only the five stop1 bodies),
   `heldout12` (never used for training or rhythm selection), and `external`. Only `heldout12` and `external` support generalisation claims. Rhythm
   selection uses tuning seeds 20000+, evaluation uses 10000+ (`walk_anything/gait.py`, `evalrun.py`).
5. **"Averages hide the failures."** [A] Per-family tables (quadruped, biped, hexapod) for every arm, with mean, median, and min..max over bodies plus a
   per-body table (`results/round2/report.md`). Bad arms and bad bodies are shown, not removed.
6. **"Few episodes, no uncertainty."** [A, partly] Episodes use fixed env seeds, identical for every arm; spreads across bodies are reported. [C] 10
   episodes per body (5 for predictive sampling, which is slow), so per-body rates are coarse; pooling per family gives more episodes but the bodies
   are the real unit of variation. [B] Not run: bootstrap intervals over bodies; cheap, can be added to `make_report.py`.
7. **"Fall criteria and episode length are arbitrary."** [C] A fall is torso up-vector below 0.3 or height below 35% of standing height (`env.py`,
   `BodyEnv.step`); episodes are 10 s at 50 Hz with a forward command of 0.3 (nondimensional). A body that falls at 12 s is not caught. Only one
   command, no turning, flat ground, no sensor noise or latency.
8. **"Forward speed does not mean walking."** [C] Speed is measured, gait quality is not (no foot-contact analysis, a hopping or sliding gait counts).
   [B] Not run: log foot-contact patterns and report duty factors; videos are possible (`mp4` stays git-ignored).
9. **"Randomisation is mild or hides the problem."** [A] The same init noise, friction scaling and random pushes as training are applied to every arm
   (`BodyEnv.reset`/`step`). [C] No mass or motor-strength randomisation at evaluation, no sim-to-real gap study.

## Baselines and fairness

10. **"The 'no training per body' claim: the rhythm is tuned per body."** [A, partly] The rhythm pattern comes from the symmetry-derived list, but the
    period, amplitude and signs are chosen by a grid search in simulation on tuning seeds (`gait.select_gait`). That *is* per-body tuning and the
    claim "no training per body" does not hold for the `rhythm` and `residual` arms as built [C]. [B] The `rhythm_default` arm (first derived pattern,
    fixed parameters, no search) is the control: `make eval-rhythm` plus `scripts/eval_arms.py <set> rhythm_default`.
11. **"Predictive sampling is a straw man."** [C] Own small implementation (32 samples, 4 knots, horizon 20, one seed, untuned), not MuJoCo MPC
    (`walk_anything/mpc.py`, `THIRD_PARTY.md`). It plans with the true model and true friction, so it is privileged information, not a like-for-like
    zero-shot method. Treat it as a reference point only. It also takes far more compute per step.
12. **"The PPO baseline is outdated and tiny."** [C] The `ppo_old` arm is the stop-point-1 checkpoint (5 bodies, old learning rate, no KL stop),
    frozen at `results/round2/ckpt/ppo_old`. It is not a tuned universal policy. The KL-stopped trainer has unit tests (`tests/test_ppo_kl.py`) and the
    residual runs log the KL and stop flag (`runs/<run>/metrics.jsonl`); a long brute-force run was deliberately not done (owner decision).
13. **"One training run, no seed spread."** [B] `train.py --seed` exists; short residual runs with several seeds were run if time allowed (see HANDOFF
    for which; seeds, steps and checkpoints are in `runs/` and `checkpoints/`). Results from a single seed must be described as such.
14. **"Compute comparison missing."** [C] Wall time per body is stored in each JSON (`wall_s`), env steps and checkpoint iteration in the training
    metrics; a like-for-like compute table was not made.

## Method claims

15. **"Novelty claim."** [C] A quick search found no work combining automatic symmetry detection, symmetric-CPG phase patterns and a universal residual
    policy. That was not a systematic literature search; no novelty claim is made. Related work is listed in `README.md`, with unverified items marked
    `[verify]`.
16. **"Symmetry detection is only for these box-torso bodies."** [A, partly] Candidates are the torso-rectangle group; each is verified numerically
    (state equivariance of the MuJoCo model and contact-geometry match), `tests/test_symmetry.py`. [C] Radial or irregular bodies are not covered.
    Front-back mirror and 180 degree rotation are included in the group although they reverse travel direction (documented in HANDOFF decisions).
17. **"'Any body described in words' is overstated."** [C] The parser is a rule-based grammar for three families (biped, quadruped, hexapod) with
    2-3 segments per leg; the text-to-body front-end using a language model is not built. Parsing failures are not counted as results.
18. **"The residual might do nothing."** [A] The `rhythm` arm is the same code path with a zero residual (`tests/test_residual.py` checks the
    two are identical), so the difference between `rhythm` and `residual` in every table is the residual's effect.

## Reproducibility and disclosure

19. **"Cannot reproduce the numbers."** [A] One command per result in the `Makefile` (`bodies`, `gaits`, `eval-*`, `train-residual`, `report`).
    Versions in `THIRD_PARTY.md`. [C] Hardware: CPU simulation on one machine, GPU RTX 5080 for training; bit-exact equality across machines was not
    tested. Training is preemptible and resumes from checkpoints, so the exact step count of a run depends on interruptions (recorded in
    `runs/<run>/metrics.jsonl`).
20. **"Prompts and attempt logs."** [A] This repo has no language-model prompts in its results: bodies come from a seeded sampler or a rule-based parser.
    If an LLM front-end is added, its prompts, outputs and failures go in `results/` verbatim and are listed here. All attempts of the rhythm selection are
    kept (`results/round2/gaits/*.json` hold the full grid table); training metrics are kept in full.
21. **"Conflict of interest."** [A] Disclosure in `README.md`: the author was an NVIDIA intern until 2026-11-13 and founded Quatern. Applies to anything
    about NVIDIA hardware or software, including the training GPU.
22. **"Only best runs shown."** [A] Reports include all arms, all bodies and every evaluated set; failed arms and bad bodies stay. The report script
    reads whatever JSON exists, and the command and date in each JSON show which runs were made.
