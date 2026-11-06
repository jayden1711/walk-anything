# walk-anything: PLAN (skeleton, not yet reviewed by the owner)

_Status: skeleton only. No code yet. The first session writes the full plan for approval and builds nothing before that._

Goal: a text description of a robot body (legs, joints, sizes, masses) → a body that one shared policy
makes walk right away, with no training per body.

Fixed constraints (from the owner):
- Runs as an **always-on, lowest-priority background GPU trainer** on the NUC, always through
  `~/projects/bin/gpu-run --preemptible` (see `~/projects/NUC_RULES.md`). On SIGUSR1 / `$GPU_RUN_YIELD_FILE`
  it checkpoints and exits within 60 s, and on restart it resumes from the latest checkpoint.
- Public repo, Apache-2.0. Checkpoints, runs, logs, videos and eval numbers stay in git-ignored dirs until
  the owner approves.

Open questions for the plan: body description format and generator, simulator choice and license, training
recipe (body-conditioned policy), evaluation on unseen bodies, checkpoint cadence and disk budget.
