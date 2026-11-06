"""Seed-variance report for the rhythm+residual arm: per-set result JSON -> a report in a git-ignored results folder.
Per set and body family (and all bodies pooled): one value per training seed (mean over the family's bodies of the per-body mean), then mean / median /
range over seeds, a bootstrap 95% CI over seeds, and a bootstrap 95% CI over seeds AND bodies (two-level: resample seeds, resample bodies of the family,
average the per-seed per-body values). Baselines are deterministic controllers (no training seed): bodies-only bootstrap. Paired comparison = residual
minus baseline, resampling the same bodies for both. Per family: fall rate, forward speed, walk rate, always together.
Files are read from ROOT/<set>/<tag>.json, or ROOT/<tag>.json (the heldout_v2 layout).
Usage: python scripts/seeds_report.py ROOT OUT [SET,SET,...] [--residual-label TEXT]
  tuning sets (default):  python scripts/seeds_report.py results/round2 results/seeds
  fresh held-out set:     python scripts/seeds_report.py results/heldout_v2 results/heldout_v2 heldout_v2 --residual-label "residual policy on the default rhythm (no per-body search)"
"""
import json
import pathlib
import statistics as st
import sys

import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
label_res = "residual"
if "--residual-label" in sys.argv:
    label_res = sys.argv[sys.argv.index("--residual-label") + 1]
    args.remove(label_res)
root = pathlib.Path(args[0] if args else "results/round2")
out = pathlib.Path(args[1] if len(args) > 1 else "results/seeds")
SETS = args[2].split(",") if len(args) > 2 else ["stop1", "train12", "tuning12", "tuning48", "placeholder"]
SEEDS = [(0, "residual1")] + [(s, f"residual1_s{s}") for s in range(1, 10)]
BASE = [("rhythm", "rhythm (per-body tuned)"), ("rhythm_default", "rhythm default (no per-body search)"),
        ("zero", "zero action"), ("mpc", "predictive sampling (MPC)"), ("ppo_old", "old PPO (stop1)")]
FAMS = ["quadruped", "biped", "hexapod", "all"]
METRICS = [("fall_rate", "fall rate", False), ("speed", "speed m/s", True), ("walk_rate", "walk rate", True)]  # (key, label, higher_is_better)
NB = 4000
rng = np.random.default_rng(0)


def load(set_, tag):
    for f in (root / set_ / f"{tag}.json", root / f"{tag}.json"):
        if f.exists():
            d = json.load(open(f))
            if d["meta"]["set"] == set_:
                return d["bodies"]
    return None


def vec(bodies, fam, key):
    return np.array([b[key] for b in bodies if fam == "all" or b["family"] == fam])


def ci(xs):
    return np.percentile(xs, 2.5), np.percentile(xs, 97.5)


def boot_seeds(per_seed):          # per_seed: (S,) values
    return ci([rng.choice(per_seed, len(per_seed)).mean() for _ in range(NB)])


def boot_two_level(M, base=None):  # M: (S, B) per-seed per-body values; base: (B,) or None. Returns draws of mean(M[S*,B*]) [- mean(base[B*])]
    S, B = M.shape
    d = np.empty(NB)
    for i in range(NB):
        bi, si = rng.integers(0, B, B), rng.integers(0, S, S)
        d[i] = M[np.ix_(si, bi)].mean() - (base[bi].mean() if base is not None else 0.0)
    return d


def boot_bodies(v):
    return ci([rng.choice(v, len(v)).mean() for _ in range(NB)])


L = ["# Seed variance of rhythm + residual (simulation only)", "",
     f"Residual arm here: {label_res}. Identical config, training seeds 0..9 (1500 iterations each). Commanded speed and evaluation episodes",
     "(env seeds 10000+k, fixed episodes per body) are fixed. Per-seed value = mean over the bodies of the family.",
     "'CI seeds' = percentile bootstrap over training seeds only (bodies fixed). 'CI seeds+bodies' = two-level bootstrap: resample seeds and resample bodies",
     "of the family (4000 draws). With few bodies per family the second interval is the honest one. 'beats' = the seeds+bodies CI excludes 0 in the better direction.",
     "'placeholder' bodies were written by the AI assistant, not by other people: no generalisation claim.", ""]
for set_ in SETS:
    seeds = [(s, t, load(set_, t)) for s, t in SEEDS if load(set_, t)]
    if not seeds:
        continue
    bases = [(t, d, load(set_, t)) for t, d in BASE if load(set_, t)]
    L += [f"## {set_} ({len(seeds)} seeds: {[s for s, _, _ in seeds]})", ""]
    for fam in FAMS:
        n_b = len(vec(seeds[0][2], fam, "fall_rate"))
        if not n_b:
            continue
        L += [f"### {set_} / {fam} ({n_b} bodies)", "",
              "| metric | residual mean | median | range over seeds | CI seeds | CI seeds+bodies | " + " | ".join(t for t, _, _ in bases) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(bases)]
        for key, lab, hib in METRICS:
            M = np.stack([vec(b, fam, key) for _, _, b in seeds])
            xs = M.mean(1)
            lo, hi = boot_seeds(xs)
            lo2, hi2 = ci(boot_two_level(M))
            L.append(f"| {lab} | {xs.mean():+.3f} | {np.median(xs):+.3f} | {xs.min():+.3f}..{xs.max():+.3f} | {lo:+.3f}..{hi:+.3f} | {lo2:+.3f}..{hi2:+.3f} | " +
                     " | ".join(f"{vec(bb, fam, key).mean():+.3f} [{boot_bodies(vec(bb, fam, key))[0]:+.2f}..{boot_bodies(vec(bb, fam, key))[1]:+.2f}]" for _, _, bb in bases) + " |")
        L += ["", "Baseline cells: mean [bootstrap 95% over bodies]. Paired: residual minus baseline on the same bodies (seeds better = seeds whose family mean is better than the baseline's)", "",
              "| baseline | metric | mean diff | CI seeds+bodies | seeds better | beats |", "|---|---|---|---|---|---|"]
        for t, _, bb in bases:
            for key, lab, hib in METRICS[:2]:
                M = np.stack([vec(b, fam, key) for _, _, b in seeds])
                base = vec(bb, fam, key)
                d = ci(boot_two_level(M, base))
                diff = M.mean() - base.mean()
                better = int(sum((x > base.mean()) if hib else (x < base.mean()) for x in M.mean(1)))
                beats = d[0] > 0 if hib else d[1] < 0
                worse = d[1] < 0 if hib else d[0] > 0
                L.append(f"| {t} | {lab} | {diff:+.3f} | {d[0]:+.3f}..{d[1]:+.3f} | {better}/{len(seeds)} | {'yes' if beats else ('WORSE' if worse else 'no (CI includes 0)')} |")
        L += [""]
    L += [f"### {set_} per seed, all bodies pooled", "", "| seed | fall | walk | speed m/s |", "|---|---|---|---|"]
    for s, t, b in seeds:
        L.append(f"| {s} | {vec(b, 'all', 'fall_rate').mean():.3f} | {vec(b, 'all', 'walk_rate').mean():.3f} | {vec(b, 'all', 'speed').mean():+.3f} |")
    L.append("")

L += ["(Verdicts are mechanical from the CI rule. 'no' = the interval includes 0, i.e. the data do not separate the arms.)"]
out.mkdir(parents=True, exist_ok=True)
(out / "report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
