# One command per result. Everything writes to the git-ignored results/ (see CLAUDE.md). Simulation only.
# CPU sims go through sim-run (nice 10, 1 thread); GPU training through gpu-run --preemptible. At most 4 sim processes machine-wide.
export PYTHONPATH := .
PY      = .venv/bin/python
SIM     = $(HOME)/projects/bin/sim-run
SEED    = 0
ITERS   = 1500
SETS    = stop1 train12 tuning12 placeholder
ALLSETS = $(SETS) tuning48   # tuning48: all 48 tuning-split bodies (the old 'held-out' bodies, used by the rhythm search; not for mpc)
SEED_RUNS = residual1 $(foreach n,1 2 3 4 5 6 7 8 9,residual1_s$(n))   # training seeds 0..9
HELD    = heldout_v2   # locked fresh held-out set (final evaluation only; see README and walk_anything/heldout.py)
PPO_OLD = results/round2/ckpt/ppo_old   # stop1 checkpoint (iteration 74571), copied from the overnight run

.PHONY: bodies gaits eval-zero eval-ppo eval-mpc eval-rhythm eval-rhythm-default eval-residual eval-residual-seeds train-residual report seeds-report compare heldout-bodies eval-heldout eval-heldout-mpc eval-heldout-report test

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

eval-residual:    ## needs a residual checkpoint dir: make eval-residual RUN=residual1_s3
	for s in $(ALLSETS); do $(SIM) $(PY) scripts/eval_arms.py $$s residual:checkpoints/$(RUN) --tag $(RUN); done

train-residual:   ## short preemptible run; stop with SIGUSR1/yield file, rerun resumes. make train-residual RUN=residual1
	PYTHONPATH=. $(HOME)/projects/bin/gpu-run --preemptible --need-mib 2048 --label walk-anything -- \
	  .venv/bin/python -m walk_anything.train --config configs/residual.yaml --run $(RUN) --workers 1 --seed $(SEED) --max-iters $(ITERS)

eval-residual-seeds:  ## residual seeds 3..9 on every tuning/train set (checkpoints/residual1_s<N> must exist)
	for r in $(filter-out residual1 residual1_s1 residual1_s2,$(SEED_RUNS)); do $(MAKE) eval-residual RUN=$$r; done

report:           ## regenerates results/round2/report.md from the per-set JSON
	$(PY) scripts/make_report.py results/round2

seeds-report:     ## seed-variance report: results/seeds/report.md (tuning sets)
	$(PY) scripts/seeds_report.py results/round2 results/seeds

compare:          ## reproducibility check of two result roots: make compare A=results/round2 SA=tuning48 B=other/root SB=tuning48
	$(PY) scripts/review_compare.py $(A) $(SA) $(B) $(SB)

heldout-bodies:   ## writes the locked fresh held-out manifest ONCE (refuses if it exists); put the printed hash in heldout.py and README
	$(PY) scripts/make_heldout_v2.py

eval-heldout:     ## every arm that needs no per-body search, residual over all 10 seeds -> results/heldout_v2/ (never re-run complete files)
	$(SIM) $(PY) scripts/eval_arms.py $(HELD) zero
	$(SIM) $(PY) scripts/eval_arms.py $(HELD) ppo:$(PPO_OLD) --tag ppo_old
	$(SIM) $(PY) scripts/eval_arms.py $(HELD) rhythm_default
	for r in $(SEED_RUNS); do $(SIM) $(PY) scripts/eval_arms.py $(HELD) residual_default:checkpoints/$$r --tag $$r; done

eval-heldout-mpc: ## slow (CPU); 5 episodes per body
	$(SIM) $(PY) scripts/eval_arms.py $(HELD) mpc

eval-heldout-report:  ## per family fall rate and speed with CIs over seeds and bodies -> results/heldout_v2/report.md
	$(PY) scripts/seeds_report.py results/heldout_v2 results/heldout_v2 heldout_v2 --residual-label "residual policy on the default rhythm (no per-body search)"
	$(PY) scripts/make_report.py results/heldout_v2

test:
	$(SIM) $(PY) -m pytest -q tests
