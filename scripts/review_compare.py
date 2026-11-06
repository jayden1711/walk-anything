"""Reviewer check: do two result roots agree on the bodies they share (reproducibility)? Prints max abs difference per arm.
Usage: python scripts/review_compare.py ROOT_A SET_A ROOT_B SET_B
"""
import json
import pathlib
import sys

ra, sa, rb, sb = sys.argv[1:5]
for f in sorted((pathlib.Path(rb) / sb).glob("*.json")):
    g = pathlib.Path(ra) / sa / f.name
    if not g.exists():
        continue
    A = {b["body"]: b for b in json.load(open(g))["bodies"]}
    B = {b["body"]: b for b in json.load(open(f))["bodies"]}
    sh = sorted(set(A) & set(B))
    d = max((abs(A[k]["speed"] - B[k]["speed"]) + abs(A[k]["fall_rate"] - B[k]["fall_rate"]) for k in sh), default=float("nan"))
    print(f"{f.stem:15s} shared bodies {len(sh):2d} max abs diff (fall+speed) {d:.6f}")
