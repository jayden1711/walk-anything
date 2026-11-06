"""Collect results/day-1106/*.json into one markdown table (git-ignored output)."""
import json
import pathlib

d = pathlib.Path("results/day-1106")
arms = [("zero", "hold standing pose"), ("ppo_stop1_old", "universal PPO, old settings (lr 3e-4, no KL stop), checkpoint it 74571"),
        ("mpc", "predictive sampling, no training"), ("rhythm", "symmetry-derived rhythm, open loop, tuned per body")]
lines = ["# Day 1106: 5 bodies, fixed forward command 0.3, 500 steps, same randomisation as training", "",
         "fall = fraction of episodes that fell; speed = mean forward speed in m/s (nd = in units of sqrt(g h)); r_lin = tracking reward term.", ""]
for tag, desc in arms:
    p = d / f"{tag}.json"
    if not p.exists():
        continue
    rows = json.load(open(p))
    lines += [f"## {tag}: {desc}", "", "| body | episodes | fall | speed m/s | speed nd | r_lin |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['body']} | {r['n']} | {r['fall_rate']:.2f} | {r['speed']:+.3f} | {r['speed_nd']:+.3f} | {r['r_lin']:.3f} |")
    lines.append(f"| mean | | {sum(r['fall_rate'] for r in rows) / len(rows):.2f} | {sum(r['speed'] for r in rows) / len(rows):+.3f} | "
                 f"{sum(r['speed_nd'] for r in rows) / len(rows):+.3f} | {sum(r['r_lin'] for r in rows) / len(rows):.3f} |")
    lines.append("")
(d / "report.md").write_text("\n".join(lines))
print("\n".join(lines))
