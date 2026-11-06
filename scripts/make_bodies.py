"""Materialise the body dataset: N valid bodies from (seed, index) into results/bodies/<seed>/ (git-ignored)."""
import argparse
import collections
import json
import pathlib

from walk_anything.sampler import make_body, split_of

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", default="results/bodies")
a = ap.parse_args()

out = pathlib.Path(a.out) / str(a.seed)
out.mkdir(parents=True, exist_ok=True)
rejected = 0
fam = collections.Counter()
index = []
for i in range(a.n):
    spec, rep, tries = make_body(a.seed, i)
    rejected += tries
    fam[(spec.family, split_of(i))] += 1
    (out / f"{spec.name}.json").write_text(json.dumps({"spec": json.loads(spec.to_json()), "report": rep.to_dict(), "rejected_before": tries, "split": split_of(i)}, indent=1))
    index.append({"index": i, "name": spec.name, "family": spec.family, "split": split_of(i)})
(out / "index.json").write_text(json.dumps(index, indent=1))
print(f"{a.n} bodies, {rejected} rejected samples ({rejected / (a.n + rejected):.0%} of draws)")
for k, v in sorted(fam.items()):
    print(k, v)
