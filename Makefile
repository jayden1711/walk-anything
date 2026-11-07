# One command per result. Everything writes to the git-ignored results/ (see CLAUDE.md). Simulation only.
# CPU sims go through sim-run (nice 10, 1 thread); GPU training through gpu-run --preemptible. At most 4 sim processes machine-wide.
.SHELLFLAGS := -ec   # a failing command inside a recipe's `for` loop must fail the target, not just the last iteration
export PYTHONPATH := .
PY      = .venv/bin/python
SIM     = $(HOME)/projects/bin/sim-run
SEED    = 0
ITERS   = 1500
SETS    = stop1 train12 heldout12 placeholder
ALLSETS = $(SETS) heldout_all   # heldout_all: all 48 held-out bodies (tuning ~20 s per body, evals seconds; not for mpc)
PPO_OLD = results/round2/ckpt/ppo_old   # stop1 checkpoint (iteration 74571), copied from the overnight run

.PHONY: bodies gaits eval-zero eval-ppo eval-mpc eval-rhythm eval-residual train-residual report test

bodies:           ## materialise every generated body (sampler seed 0, indices 0..199) into results/bodies/0
	$(SIM) $(PY) scripts/make_bodies.py --seed 0 --n 200

gaits:            ## per-body rhythm selection on tuning seeds (20000+), cached as results/round2/gaits/<body>.json
	$(SIM) $(PY) scripts/tune_gaits.py $(ALLSETS)

eval-zero:
	for s in $(ALLSETS); do $(SIM) $(PY) scripts/eval_arms.py $$s zero; done

eval-ppo:
	for s in $(ALLSETS); do $(SIM) $(PY) scripts/eval_arms.py $$s ppo:$(PPO_OLD) --tag ppo_old; done

eval-mpc:         ## slow (CPU); 5 episodes per body
	for s in $(SETS); do $(SIM) $(PY) scripts/eval_arms.py $$s mpc --episodes 5; done

eval-rhythm:
	for s in $(ALLSETS); do $(SIM) $(PY) scripts/eval_arms.py $$s rhythm; done

eval-rhythm-default:  ## control: no per-body search
	for s in $(ALLSETS); do $(SIM) $(PY) scripts/eval_arms.py $$s rhythm_default; done

eval-residual:    ## needs a residual checkpoint dir: make eval-residual RUN=residual1
	for s in $(ALLSETS); do $(SIM) $(PY) scripts/eval_arms.py $$s residual:checkpoints/$(RUN) --tag $(RUN); done

train-residual:   ## short preemptible run; stop with SIGUSR1/yield file, rerun resumes. make train-residual RUN=residual1
	PYTHONPATH=. $(HOME)/projects/bin/gpu-run --preemptible --need-mib 2048 --label walk-anything -- \
	  .venv/bin/python -m walk_anything.train --config configs/residual.yaml --run $(RUN) --workers 1 --seed $(SEED) --max-iters $(ITERS)

report:
	$(PY) scripts/make_report.py results/round2

test:
	$(SIM) $(PY) -m pytest -q tests
