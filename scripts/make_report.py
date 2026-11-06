"""Collect <root>/<set>/<arm>.json (or <root>/<arm>.json for heldout_v2) into <root>/report.md (git-ignored): per body family, every arm.
Per family: number of bodies and episodes, fall rate, walk rate (no fall and speed >= WALK_FRAC x commanded), forward speed.
Fall rate is never shown without forward speed. Spread = median and min..max across bodies.
Usage: python scripts/make_report.py [results/round2]      (fresh held-out: python scripts/make_report.py results/heldout_v2)
"""
import json
import pathlib
import statistics as st
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/round2")
ARMS = [("zero", "zero action (hold standing pose)"),
        ("ppo_old", "old universal PPO (stop1, lr 3e-4, no KL stop), trained on 5 bodies"),
        ("mpc", "predictive sampling, no training, plans with the true model"),
        ("rhythm", "symmetry-derived rhythm, open loop, tuned per body on tuning seeds"),
        ("rhythm_default", "rhythm, first derived pattern, fixed parameters, no per-body search (control)"),
        ("residual1", "rhythm + learned residual, training seed 0 (1500 iterations)"),
        ("residual1_s1", "rhythm + learned residual, training seed 1"),
        ("residual1_s2", "rhythm + learned residual, training seed 2")] + \
       [(f"residual1_s{n}", f"rhythm + learned residual, training seed {n}") for n in range(3, 10)]
SETS = [("stop1", "5 bodies of stop point 1 (train split; PPO and residual were trained on them)"),
        ("train12", "12 train-split bodies (residual trained on them; PPO saw only 5)"),
        ("tuning12", "12 TUNING bodies (first 4 per family of the tuning split; not used to train PPO or the residual, but the per-body rhythm search WAS run on each of them: not held out)"),
        ("tuning48", "all 48 TUNING bodies of the 200 (same rule; the rhythm search was run on each: not held out)"),
        ("heldout_v2", "48 fresh HELD-OUT bodies (locked manifest, sha256 in README): no tuning, search or training touched them; arms: zero, old PPO, MPC, rhythm_default, residual on the default rhythm"),
        ("placeholder", "placeholder descriptions written by the AI assistant, NOT by other people (see bodies_placeholder/); no generalisation claim")]
FAMS = ["quadruped", "biped", "hexapod"]


def load(set_, tag):
    for f in (root / set_ / f"{tag}.json", root / f"{tag}.json"):
        if f.exists():
            d = json.load(open(f))
            if d["meta"]["set"] == set_:
                return d
    return None


def fmt(v):
    return f"{v:+.3f}"


lines = [f"# {root.name} report (simulation only)", "",
         "Fixed forward command, 500 control steps (10 s), same randomisation as training, env seeds 10000+k identical for every arm.",
         "fall = fraction of episodes that fell. walk = no fall AND mean forward speed >= 0.5 x commanded speed (our threshold).",
         "speed = mean forward speed in m/s over the episode (including the steps before a fall, which can be a forward lunge; the last column uses non-fallen episodes only) (nd = divided by sqrt(g h), h = standing height). Commanded: 0.3 nd.",
         "Spread columns: median and min..max over bodies of the per-body mean. Bodies listed in the per-body tables.", ""]
for set_, desc in SETS:
    present = [(t, d, load(set_, t)) for t, d in ARMS if load(set_, t)]
    if not present:
        continue
    lines += [f"## {set_}: {desc}", ""]
    if set_ == "heldout_v2":
        lines += ["In this set every residual1* row is the residual policy applied on the fixed default rhythm (arm residual_default), not on a per-body tuned rhythm.", ""]
    for fam in FAMS:
        rows = []
        for tag, d, data in present:
            bs = [b for b in data["bodies"] if b["family"] == fam]
            if not bs:
                continue
            eps = sum(b["n"] for b in bs)
            fall = [b["fall_rate"] for b in bs]
            walk = [b["walk_rate"] for b in bs]
            sp = [b["speed"] for b in bs]
            nd = [b["speed_nd"] for b in bs]
            nf = [e["speed"] for b in bs for e in b["episodes"] if not e["fell"]]
            rows.append(f"| {tag} | {len(bs)} | {eps} | {st.mean(fall):.2f} | {st.median(fall):.2f} ({min(fall):.2f}..{max(fall):.2f}) | "
                        f"{st.mean(walk):.2f} | {fmt(st.mean(sp))} | {fmt(st.median(sp))} ({fmt(min(sp))}..{fmt(max(sp))}) | {fmt(st.mean(nd))} | {fmt(st.mean(nf)) if nf else "n/a"} |")
        if rows:
            lines += [f"### {set_} / {fam}", "",
                      "| arm | bodies | episodes | fall mean | fall median (range) | walk mean | speed m/s mean | speed median (range) | speed nd mean | speed m/s, non-fallen episodes only |",
                      "|---|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    lines += [f"### {set_} per body", "", "| body | " + " | ".join(f"{t} fall/walk/speed" for t, _, _ in present) + " |",
              "|---|" + "---|" * len(present)]
    names = [b["body"] for b in present[0][2]["bodies"]]
    for n in names:
        cells = []
        for _, _, data in present:
            b = next((x for x in data["bodies"] if x["body"] == n), None)
            cells.append(f"{b['fall_rate']:.2f} / {b['walk_rate']:.2f} / {b['speed']:+.3f}" if b else "-")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("")
lines += ["## Arms", ""] + [f"- {t}: {d}" for t, d in ARMS]
(root / "report.md").write_text("\n".join(lines))
print("\n".join(lines))
