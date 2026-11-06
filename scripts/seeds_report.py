"""Seed-variance report for the rhythm+residual arm: results/round2/<set>/<arm>.json -> results/seeds/report.md (git-ignored).
Per set and body family (and all bodies pooled): one value per training seed (mean over the family's bodies of the per-body mean),
then mean / median / range over seeds and a bootstrap 95% CI over seeds (resampling seeds, 20000 draws). Baselines are deterministic
controllers (no training seed), so they get one value. The paired comparison is (residual seed value - baseline value); its CI is a
bootstrap over seeds. Bodies and evaluation episodes are fixed, so the CI covers training-seed variance only, not body sampling.
Usage: python scripts/seeds_report.py [results/round2] [results/seeds]
"""
import json
import pathlib
import random
import statistics as st
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/round2")
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "results/seeds")
SEEDS = [(0, "residual1")] + [(s, f"residual1_s{s}") for s in range(1, 10)]
BASE = [("rhythm", "rhythm (per-body tuned)"), ("rhythm_default", "rhythm default (no per-body search)"),
        ("zero", "zero action"), ("mpc", "predictive sampling (MPC)"), ("ppo_old", "old PPO (stop1)")]
SETS = ["stop1", "train12", "heldout12", "placeholder"]
FAMS = ["quadruped", "biped", "hexapod", "all"]
METRICS = [("fall_rate", "fall rate", False), ("speed", "speed m/s", True), ("walk_rate", "walk rate", True)]  # (key, label, higher_is_better)
rng = random.Random(0)


def load(set_, tag):
    f = root / set_ / f"{tag}.json"
    return json.load(open(f))["bodies"] if f.exists() else None


def val(bodies, fam, key):
    bs = [b for b in bodies if fam == "all" or b["family"] == fam]
    return st.mean(b[key] for b in bs) if bs else None


def boot(xs, f=st.mean, n=20000):
    ds = sorted(f([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


L = ["# Seed variance of rhythm + residual (simulation only)", "",
     "Residual = rhythm + learned residual, identical config, training seeds 0..9 (1500 iterations each). Bodies, commanded speed and evaluation episodes",
     "(env seeds 10000+k, 10 episodes per body) are fixed, so intervals cover training-seed variance only; they say nothing about how bodies were sampled.",
     "Per-seed value = mean over the bodies of the family. 'CI' = bootstrap 95% over seeds (percentile). A set with 4 bodies per family is small: read the pooled 'all' rows first.",
     "'placeholder' bodies were written by the AI assistant, not by other people: no generalisation claim.", ""]
verdict = {}
for set_ in SETS:
    seeds = [(s, t, load(set_, t)) for s, t in SEEDS if load(set_, t)]
    if not seeds:
        continue
    bases = [(t, d, load(set_, t)) for t, d in BASE if load(set_, t)]
    L += [f"## {set_} ({len(seeds)} seeds: {[s for s, _, _ in seeds]})", ""]
    for fam in FAMS:
        n_b = len({b['body'] for b in seeds[0][2] if fam == 'all' or b['family'] == fam})
        if not n_b:
            continue
        L += [f"### {set_} / {fam} ({n_b} bodies)", "",
              "| metric | residual mean | median | range over seeds | CI95 over seeds | " + " | ".join(t for t, _, _ in bases) + " |",
              "|---|---|---|---|---|" + "---|" * len(bases)]
        for key, lab, hib in METRICS:
            xs = [val(b, fam, key) for _, _, b in seeds]
            lo, hi = boot(xs)
            bv = [val(b, fam, key) for _, _, b in bases]
            L.append(f"| {lab} | {st.mean(xs):+.3f} | {st.median(xs):+.3f} | {min(xs):+.3f}..{max(xs):+.3f} | {lo:+.3f}..{hi:+.3f} | " +
                     " | ".join(f"{v:+.3f}" for v in bv) + " |")
        L += [""]
        L += ["Paired: residual seed value minus baseline (CI over seeds; 'beats' = CI excludes 0 in the better direction; seeds better = how many of the seeds are better than the baseline)", "",
              "| baseline | metric | mean diff | CI95 | seeds better | beats |", "|---|---|---|---|---|---|"]
        for t, _, bb in bases:
            for key, lab, hib in METRICS[:2]:
                xs = [val(b, fam, key) for _, _, b in seeds]
                d = [x - val(bb, fam, key) for x in xs]
                lo, hi = boot(d)
                better = sum((x > 0) if hib else (x < 0) for x in d)
                beats = (lo > 0) if hib else (hi < 0)
                worse = (hi < 0) if hib else (lo > 0)
                L.append(f"| {t} | {lab} | {st.mean(d):+.3f} | {lo:+.3f}..{hi:+.3f} | {better}/{len(d)} | {'yes' if beats else ('WORSE' if worse else 'no (CI includes 0)')} |")
                verdict[(set_, fam, t, key)] = 'yes' if beats else ('WORSE' if worse else 'no')
        L += [""]
    L += [f"### {set_} per seed, all bodies pooled", "", "| seed | fall | walk | speed m/s |", "|---|---|---|---|"]
    for s, t, b in seeds:
        L.append(f"| {s} | {val(b, 'all', 'fall_rate'):.3f} | {val(b, 'all', 'walk_rate'):.3f} | {val(b, 'all', 'speed'):+.3f} |")
    L.append("")

L += ["## Answer on held-out bodies (pooled over the 12 generated held-out bodies, seed variance counted)", ""]
for t, d in BASE:
    f, s = verdict.get(("heldout12", "all", t, "fall_rate")), verdict.get(("heldout12", "all", t, "speed"))
    if f:
        L.append(f"- vs {d}: fall rate {f}; forward speed {s}.")
L += ["", "(Verdicts are mechanical from the CI rule above. 'no' = the interval over seeds includes 0, so seed variance does not rule out no difference.)"]
out.mkdir(parents=True, exist_ok=True)
(out / "report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
