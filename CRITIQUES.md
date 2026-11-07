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
   200 is written by `make bodies` into `results/bodies/0/` (spec, validity report, number of rejected draws, split) and nothing is dropped after
   generation. (Reviewer pass: that directory did not exist in the round2 results tree, so the claim was untested; `make bodies` was run into a local
   git-ignored directory and takes about a second. Run it and publish the output with the post.) The train and tuning-set lists are fixed index rules
   written before evaluation (`walk_anything/bodysets.py`: the first 4 bodies per family of each split), so `tuning12` is 12 of the 48 tuning-set bodies;
   the other 36 are now evaluated too (set `tuning48`, see the reviewer pass, item R3). To be published together with the sampler seed. [C] The validity checker rejects bodies that cannot stand; the number of
   rejected draws per body is recorded, and the surviving distribution is not "all possible robots".
3. **"The test set was written by the authors."** [C] The `placeholder` set (`bodies_placeholder/*.txt`) currently holds seven descriptions written by the AI
   assistant from memory of public robot sizes or invented size classes; each file says so and is marked `[verify]`. They are placeholders, not an independent
   test set. [B] Not run: collect descriptions from other people (or copy published robot descriptions with the source cited), drop them into
   `bodies_placeholder/` with `source`/`author` headers, run `make eval-*` again. The set is evaluated and reported separately and not mixed into the
   tuning-set numbers. Descriptions must fit the rule-based grammar (`walk_anything/parse.py`) until an LLM front-end exists, which limits how free-form
   they can be [C].
4. **"Train/test leakage."** [A] Reports are split into `stop1` and `train12` (in-sample for the residual; PPO saw only the five stop1 bodies),
   `tuning12` and `tuning48` (the TUNING set: 12 resp. all 48 bodies of the tuning split; not used for PPO or residual training, but **the per-body rhythm
   search was run on each of them**, so they are not held out and not zero-shot for the `rhythm` and `residual` arms, see R4; they were called "held-out"
   until 2026-11-06, which was wrong), `placeholder` (AI-written descriptions, supports no claim) and `heldout_v2` (the fresh held-out set, see R13). Only
   `heldout_v2` can support a generalisation claim, and only for arms that involve no per-body search. Rhythm selection uses tuning seeds 20000+,
   evaluation uses 10000+ (`walk_anything/gait.py`, `evalrun.py`). Descriptions written by other people will be collected as part of the public post.
5. **"Averages hide the failures."** [A] Per-family tables (quadruped, biped, hexapod) for every arm, with mean, median, and min..max over bodies plus a
   per-body table (`results/round2/report.md`). Bad arms and bad bodies are shown, not removed.
6. **"Few episodes, no uncertainty."** [A, partly] Episodes use fixed env seeds, identical for every arm; spreads across bodies are reported. [C] 10
   episodes per body (5 for predictive sampling, which is slow), so per-body rates are coarse; pooling per family gives more episodes but the bodies
   are the real unit of variation. [A] Bootstrap intervals over bodies and over training seeds are in `scripts/seeds_report.py` (`make seeds-report`): with 12 bodies most arm differences are inside the interval, with all 48 tuning-set bodies several are not (R3).
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
13. **"One training run, no seed spread."** [A, partly] `train.py --seed` exists and three residual runs were made (seeds 0, 1, 2, same config, 1500
    iterations each; the seed is in the command line at the top of `results/round2/logs/train_residual1*.log`, steps and checkpoints in `runs/` and
    `checkpoints/`). All three seeds are reported side by side in every table (`residual1`, `residual1_s1`, `residual1_s2`); do not pick one. Three seeds is
    still few, and another session is adding more. The seeds differ a lot per family, so quote per-seed numbers or the spread, never one seed.
14. **"Compute comparison missing."** [C] Wall time per body is stored in each JSON (`wall_s`), env steps and checkpoint iteration in the training
    metrics; a like-for-like compute table was not made.

## Method claims

15. **"Novelty claim."** [C] No systematic literature search was done and no search record exists (this repo had no web access when the list was written),
    so no novelty claim is made. (Corrected: an earlier version said "a quick search found no work combining ...", which cannot be traced to anything.) Related work is listed in `README.md`, with unverified items marked
    `[verify]`.
16. **"Symmetry detection is only for these box-torso bodies."** [A, partly] Candidates are the torso-rectangle group; each is verified numerically
    (state equivariance of the MuJoCo model and contact-geometry match), `tests/test_symmetry.py`. [C] Radial or irregular bodies are not covered.
    Front-back mirror and 180 degree rotation are included in the group although they reverse travel direction (documented in HANDOFF decisions).
17. **"'Any body described in words' is overstated."** [C] The parser is a rule-based grammar for three families (biped, quadruped, hexapod) with
    2-3 segments per leg; the text-to-body front-end using a language model is not built. For descriptions from other people, parse failures and
    invalid bodies must be reported as failures of the pipeline, not dropped: `bodysets.get("placeholder")` used to drop invalid descriptions silently and
    now prints a warning for each (all seven placeholders were valid). The placeholders were written by the same assistant that wrote the parser (and fixed
    a parser bug while writing them), so they fit the grammar by construction (R6).
18. **"The residual might do nothing."** [A] The `rhythm` arm is the same code path with a zero residual (`tests/test_residual.py` checks the
    two are identical), so the difference between `rhythm` and `residual` in every table is the residual's effect.

## Reproducibility

19. **"Cannot reproduce the numbers."** [A] One command per result in the `Makefile` (`bodies`, `gaits`, `eval-*`, `train-residual`, `report`).
    Versions in `THIRD_PARTY.md` (checked against the venv in the reviewer pass: they match). The `git` field in the result JSONs is the HEAD at run time, not a
    clean-tree marker: most round2 evaluations record a revision older than the commit that added the evaluation code (they ran from an uncommitted tree), so
    that field alone does not identify the code (R5). The `ppo_old` checkpoint is a manual copy of the overnight stop-point-1 run; the Makefile has no
    command that recreates it [C]. [C] Hardware: CPU simulation on one machine, GPU RTX 5080 for training; bit-exact equality across machines was not
    tested. Training is preemptible and resumes from checkpoints, so the exact step count of a run depends on interruptions (recorded in
    `runs/<run>/metrics.jsonl`).
20. **"Prompts and attempt logs."** [A] This repo has no language-model prompts in its results: bodies come from a seeded sampler or a rule-based parser.
    If an LLM front-end is added, its prompts, outputs and failures go in `results/` verbatim and are listed here. All attempts of the rhythm selection are
    kept (`results/round2/gaits/*.json` hold the full grid table); training metrics are kept in full.

22. **"Only best runs shown."** [A] Reports include all arms, all bodies and every evaluated set; failed arms and bad bodies stay. The report script
    reads whatever JSON exists, and the command and date in each JSON show which runs were made. [C] One thing was moved aside rather than reported:
    `results/_trash/round2_residual_stale/` holds evaluations of the `residual1` checkpoint taken while it was still training (an intermediate checkpoint),
    later superseded by evaluations of the final checkpoint. They were not reported; it is not recorded whether any decision was based on them. The
    three residual runs are all the residual training that was done with this config.

## Reviewer pass (2026-11-06)

Fresh outside review (an AI assistant session with no memory of the earlier work, acting as a skeptical expert, before anything is posted).
Read: CLAUDE.md, NUC_RULES.md, README, PLAN, this file, the code, `results/round2/` (read-only), `runs/`, `checkpoints/`. No web access, so
nothing external was checked: all references stay `[verify]`. Still no result numbers here; where a number matters it is printed by
`scripts/review_compare.py` (and the one-off `review_stats.py`, now in `results/_scratch/`, superseded by `scripts/seeds_report.py`) / `make report` and lives in git-ignored files or the terminal.
Extra runs (sim-run, one process at a time, outputs in the git-ignored `runs/review_bodies` and `runs/review_results`, none under `results/round2`):
all 200 bodies materialised; rhythm search and 7 arms (`zero`, `ppo_old`, `rhythm`, `rhythm_default`, `residual1`, `residual1_s1`, `residual1_s2`) on all 48
tuning-set bodies (`tuning48`; predictive sampling not run, about 10 minutes per 12 bodies). Commands: `make bodies`, `make gaits`, `make eval-*` with
`--out runs/review_results --gaits runs/review_results/gaits` (new options), `python scripts/review_stats.py <results> <runs> tuning48`.

### Headline, to be stated first in any post
In these simulations **no controller walks the generated bodies reliably**. Under the repo's own `walk` definition (no fall and at least half the commanded
speed), almost no episode of any arm counts as walking, on train bodies, on tuning-set bodies and on the placeholders. Reported speed is driven by falling:
the mean speed over episodes that ended in a fall is far above the mean over non-fallen episodes, for nearly every arm. Bipeds are the clearest case: the
rhythm and residual arms fall almost every episode and the standing-pose control falls less. The learned residual does not beat the tuned rhythm it
sits on, and over all 48 tuning-set bodies it falls more than the rhythm alone and more than the standing pose. The old PPO baseline produces the most
walking episodes. A post must say this; the pipeline (parser, sampler, validity check, symmetry detection, evaluation harness) is the contribution so far,
not a walking result.

### 1. Claims that fail the traceability / reproducibility / evidence test
| claim (where) | problem | status |
|---|---|---|
| README tagline: one policy makes any described body walk instantly, no per-body training | Not shown by any result; the best arms do not walk reliably, and the arms that come closest (`rhythm`, `residual`) use a per-body simulated search | README reworded as a goal "not met yet" |
| "held-out bodies never used for tuning" [old name of the tuning set] (CRITIQUES 4, `make_report.py` header, PLAN 1.4 wording) | False for the rhythm search: `results/round2/gaits/` has tables for every body of the old 'held-out' split, now the tuning set | CRITIQUES 4 and `make_report.py` text fixed; PLAN gets a sentence (R4) |
| "every one of the 200 bodies is materialised by `make bodies`" (item 2) | `results/bodies/` did not exist | made locally, one second; item 2 amended; must be run and published with the post |
| "A quick search found no work combining..." (item 15) | no record of any search | removed (item 15) |
| "the KL early stop ... verifies the day-1106 fix beyond unit tests" (HANDOFF) | The stop fires in most iterations of all three runs, so only a few optimiser steps happen per iteration and the training reward is nearly flat from start to end. That shows the stop is active, not that training is healthy (R7) | stated here; HANDOFF not rewritten |
| "`report.md` is regenerated by `make report`" (item 19) | the round2 `report.md` and result folder used the old set name `external` and the label "bodies_external", which reads as written by other people | fixed 2026-11-06: set renamed `placeholder`, body prefix `ph_`, all placeholder results re-run and `make report` regenerated; no `external` name remains in code, reports or results (R6) |
| PLAN section 3: "small mass noise", "joint-limit penalty" | not in `env.py` | PLAN edited to say planned, not implemented; item 9 already said no mass randomisation |
| result JSON `git` field as provenance | HEAD at run time, not a clean-tree check; most evaluations record a commit that predates the evaluation code | R5 |

Checked and true: seeds, episode count, eval seeds (10000+k, same for every arm), 500 steps, command, fall rule, 5 episodes for predictive sampling, the
`ppo_old` checkpoint iteration, the three training commands with `--seed 0/1/2`, `--max-iters 1500`, 1 worker (log headers), library versions in
THIRD_PARTY.md, and that the unit tests (31 here) pass. No model IDs or prompts are involved (no language model is used).

### 2. Existing entries: is the answer true, is the label honest
- 1 true, A honest. 2 had the unmade `results/bodies` and the 12-of-48 subset (fixed, R3). 3 true (seven files, each marked AI-written and `[verify]`); C honest.
- 4 wrong on rhythm selection (fixed). 5 true. 6 now partly A (bootstrap run), see R3. 7 true; also the rhythm ignores the command (R9). 8 true.
- 9 true (init noise 0.05, friction scale 0.6..1.4, pushes up to 0.3 m g for 5 steps; no mass or motor randomisation, no latency or noise).
- 10 true and honest (a 24-point parameter grid times up to 12 patterns times 3 tuning episodes per body is per-body training in all but name; the
  `rhythm_default` control is clearly worse on falls, which confirms the search carries the result). 11 true: the planner copies the true model including the
  sampled friction (privileged). 12 true. 13 relabelled A partly. 14 true. 15 fixed. 16 plausible, tests pass; the front-back mirror caveat is real.
- 17 amended (parse failures must count). 18 true (`tests/test_residual.py` passes), but the equality shows only that the code paths agree. 19 amended. 20 true
  (grid tables are stored per body). 21 fixed. 22 amended with the trashed intermediate evaluations.

### 3. Added criticisms (R1 to R12)
- **R1 "Standing still counts as not falling" (owner-required).** [A] The `zero` arm is in every table; `walk` requires a forward speed. New in the reviewer
  pass: speed over non-fallen episodes only is printed next to the all-episode speed (`make_report.py`), because a body that falls forward has a large
  positive mean speed over its few steps. Finding: most non-fallen episodes move very little, i.e. the bodies stand or shuffle. The 0.5 threshold is arbitrary [C]; raw speeds are in the JSONs.
- **R2 "Cherry-picked bodies" (owner-required).** [A] bodies are `(seed 0, index)`, the split is by index rule; train12/tuning12 are the first 4 per family of
  each split, stop1 is indices 0 to 4. Every body is now listed by `make bodies` (index, family, split, validity report, number of rejected draws); publish
  that directory. [C] About half of all draws are rejected by the validity checker, so results are for the surviving distribution, not for all possible robots,
  and the sampler ranges (leg length 0.12 to 0.9 m, torso proportions, torque rule) are the author's choices.
- **R3 "Only 12 of 48 tuning-set bodies, 10 episodes each."** [B, run] All 48 evaluated for every arm except predictive sampling. Tuning and evaluation on the 12
  already-evaluated bodies reproduce bit for bit (`review_compare.py`), so the larger set is consistent with the old one. Over 48 bodies the picture is
  worse for the proposed arms than over 12: the 12-body subset flattered the residual and the rhythm. Body-bootstrap intervals (reviewer script, now `scripts/seeds_report.py`): the residual's extra falls
  over the standing pose and over the rhythm are outside the interval for all three seeds; the rhythm's speed gain over the standing pose is outside it, but
  mostly reflects falls. Do not quote the 12-body numbers alone.
- **R4 "The rhythm arms are not zero-shot."** [C] The grid search (pattern, period, amplitude, signs) is run on each evaluated body, including tuning-set ones, on
  simulated tuning episodes. The residual is trained on tuned rhythms of train bodies and evaluated on tuned rhythms of tuning-set bodies. So `rhythm` and
  `residual` results are "with per-body simulated tuning", and the title claim (no training per body) does not apply to them. `rhythm_default` is the only
  search-free rhythm arm. Predictive sampling and the PPO baseline involve no per-body fitting (the planner does use privileged model information, item 11).
- **R5 Provenance of results.** [C] See table above. Mitigation done: all seven arms were re-run on the 12 tuning-set bodies with the current code and agree exactly
  with the stored JSONs, so the stored results are reproducible by the current commit, for that set. Not re-run: stop1, train12, placeholder, predictive sampling.
  Future runs should record `git status --porcelain` in the JSON.
- **R6 "The placeholder set is AI-written."** [C] Stated in: each `bodies_placeholder/*.txt` header, README (added in this pass; it was missing), CRITIQUES 3, 4,
  17, PLAN section 9, `bodysets.py`, `make_report.py` section title. The stale `external` names (report, result folder, JSON `set` field, body prefix `ext_`, cached gait files) were removed on 2026-11-06 by re-running the placeholder set under the new names (not by editing numbers).
  The set cannot support any claim about descriptions written by others: the same assistant wrote the descriptions and the parser. For the public post: collect
  descriptions from other people without showing them the grammar, count unparsable and invalid ones as failures, and list all of them.
- **R7 "The residual is under-trained, so its negative result says little."** [C] Each run is 1500 iterations (about 4.6 million environment steps) against the
  roughly 75 thousand iterations of the old PPO run. The KL stop ends the optimisation epoch after a few minibatch steps in most iterations, entropy and reward
  barely move, and seeds differ widely. The finding is "this short, KL-limited run learns no useful correction", not "residual learning does not help".
  [B] Not run (owner decision against long runs; GPU): a run with the KL target relaxed or a longer run. Another session is adding seeds; not duplicated.
- **R8 Baseline ordering.** [A] The old PPO baseline (trained on 5 bodies, a different observation, no rhythm) is at least as good as the proposed arms on
  tuning-set bodies on the walk measure, and worse than the standing pose on bipeds' falls. Reported as is in the tables. The comparison is not like for like
  (training budget, tuning, observation), see 12 and 14.
- **R9 Open-loop rhythm and a single command.** [C] The rhythm does not read the command, so it cannot track speed; only forward 0.3 is evaluated, no turning,
  no speed sweep, flat ground. A command-following claim cannot be made.
- **R10 Simulation realism.** [C] MuJoCo defaults, PD position control with the author's gains, no actuator dynamics, latency, sensor noise or terrain; bodies
  are procedural boxes and capsules with a torque rule from mass and leg length, not real robots; no hardware. The word "simulation" must appear in any post
  title or first line (README does).
- **R11 Multiple comparisons.** [C] Many arms, sets and metrics are shown with no correction; the walk threshold, fall rule, episode length and grid are the author's
  choices fixed before the runs as far as the record shows, but nothing pre-registers that. Only differences whose body-bootstrap interval excludes zero
  on all 48 tuning-set bodies should be called differences.
- **R12 Prior work and wording.** [C] Related work in README is from memory and marked `[verify]` (added: CPG review by Ijspeert, RMA by Kumar et al., "My Body is
  a Cage" by Kurin et al., all `[verify]`; the RoboStriker and MIT-post items stay "[verify relevance: not read]"). Predictive sampling and symmetry-detection
  citations were also written offline; verify authors, years, venues before the post. Wording: "skeptical", "cherry-picked" etc. appear only in this file; README and PLAN
  state goals as goals. A post should avoid "walks anything" and "zero-shot" for anything involving the rhythm search.

### 4. Checks requested
- **All attempts with medians and spreads:** yes for arms, bodies and seeds (reports give mean, median, min..max over bodies, per-body tables, all three residual
  seeds). Not reported before: the trashed intermediate-checkpoint evaluations (item 22) and the number of body draws rejected (now in `make bodies` output).
  Per-family tables have only 4 bodies per cell in `tuning12`; `tuning48` has 16.
- **Simulation stated plainly:** README line 1 after the title, CRITIQUES header, PLAN section 2, `make_report.py` title. Add to any post's first sentence.
- **Neutral wording:** README tagline rewritten as an unmet goal; "walk anything" style claims removed from the first lines.

### 5. What I changed
CRITIQUES items 2, 4, 6, 13, 15, 17, 19, 22 corrected or amended (marked "Corrected" where a statement was false); this section added. README: goal
wording, three `[verify]` references. PLAN: placeholder wording, tuning/rhythm sentence, two unimplemented items marked. Code: `bodysets.py`
(`tuning48` set, warning on dropped placeholders), `eval_arms.py` and `tune_gaits.py` (`--gaits`, `--out`), `make_report.py` (non-fallen speed column,
tuning-set text fixed, `tuning48`, legacy `external/` fallback), Makefile (`ALLSETS`), new reviewer scripts `review_stats.py` (later moved to `results/_scratch/`) and `scripts/review_compare.py` (read-only).
Not changed: `results/round2/*` (read-only), HANDOFF body (status line updated only).

## Update 2026-11-06 (owner decisions): tuning set versus a fresh held-out set
Still no result numbers in this file; results are in git-ignored `results/seeds/` (tuning sets, seed variance) and `results/heldout_v2/` (fresh held-out set).
Older sections above were edited only to rename the set; where they say "tuning set" they mean the old 48-body "held-out" split.
- **R13 "The 'held-out' bodies were used for tuning."** [A, fixed] The rhythm search ran on every body of the old held-out split, so that split is renamed the
  TUNING set everywhere (`tuning12`, `tuning48`, split label `tuning`, report sections, README, PLAN). It supports no generalisation claim. It stays in the
  report, labelled, because it is where per-body-tuned arms (`rhythm`, `residual`) can be compared with each other and with the other arms.
- **R14 The fresh held-out set (`heldout_v2`).** [A] 48 bodies (16 per family), sampler seed 20261106 (a date, arbitrary), indices 0..47, same sampler and
  validity rules as the tuning set, written once by `scripts/make_heldout_v2.py` into `bodies_heldout_v2/manifest.json` (read-only file mode, SHA-256 in
  `walk_anything/heldout.py` and README.md). Size fixed before any evaluation: 48 matches the tuning set so intervals are comparable, and 16 per family is
  enough for per-family intervals; it is not changed after seeing results. [C] Same sampler and same author as the tuning set, so it is a fresh sample of the
  same distribution, not an independent distribution, and still not real robots.
- **R15 The guard.** [A] `walk_anything/heldout.py`: `refuse_heldout()` raises when a body of the set (matched by name or by content, so renamed copies are
  caught) reaches per-body gait search (`select_gait`, `select_gait_cached`) or training (`train.py`); `scripts/tune_gaits.py` goes through that code;
  `scripts/eval_arms.py` on `heldout_v2` refuses arms that need a per-body search (`rhythm`, `residual:`), refuses other episode counts, writes only to
  `results/heldout_v2/`, and never overwrites a complete result. The manifest hash is verified on every read. Tests: `tests/test_heldout.py`.
  [C] The guard cannot stop someone editing the guard, reading the held-out results and then changing a hyperparameter, or deleting a result file by
  hand. That part is discipline: results on this set are read once, after all choices are frozen, and no arm is re-run because of how it came out.
- **R16 No per-body rhythm search on the held-out set, so two arms changed.** [C] The owner rule (nothing tunes on the set) conflicts with evaluating
  `rhythm` and `rhythm+residual`, which need a per-body search. The conservative choice was taken: on `heldout_v2` the arms are `zero`, old PPO, predictive
  sampling, `rhythm_default` (fixed pattern and parameters) and `residual_default` (the residual policy, trained on tuned rhythms, applied on the fixed default
  rhythm; a distribution shift it was not trained for). The per-body-tuned `rhythm` and `residual` arms are reported on the tuning set only. A held-out
  comparison of tuned rhythms would need a rule fixed on the tuning set and then applied unchanged (for example one gait choice per family); not done.
- **R17 Seeds and intervals.** [A] The held-out and tuning reports give, per family, the residual over all 10 training seeds with a bootstrap over seeds and a
  two-level bootstrap over seeds and bodies; fall rate, forward speed and walk rate are always shown together. [C] Seeds 3 to 9 were trained in a
  background queue behind other jobs on the shared GPU; the report states how many seeds it actually contains.
- **R18 Seeds 3 to 9 trained on CPU, seeds 0 to 2 on GPU.** [A] All ten seeds now exist and both reports contain all ten. Seeds 3 to 9 ran on CPU (same config, 1500
  iterations) because the shared GPU was unavailable overnight; the seeds report has a per-seed device table. [C] CPU and GPU float arithmetic differ, so the seeds are
  not bit-for-bit comparable across devices and a device effect is mixed into the seed spread (3 GPU seeds against 7 CPU seeds); not separated. Seed 3 also had a
  GPU run queued; the CPU run finished first and is the one used.
- **R19 The held-out reading, stated plainly.** [A] On the fresh held-out set, per family, the residual policy on the default rhythm does not separate from the default
  rhythm alone on speed in any family or pooled, nor on fall rate except for bipeds (where the seeds+bodies interval excludes zero in the better direction), and the
  seeds+bodies intervals are wide because there are 16 bodies per family. One family out of three on one metric is not a general result; no multiplicity correction. [C] The stated goal
  (one policy makes an unseen body walk without training) is not met by any arm on this set; see `results/heldout_v2/seeds/report.md`. Do not describe the residual as
  a generalisation result. No arm was re-run and nothing was tuned after reading these results.
