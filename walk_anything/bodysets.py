"""Named body sets. Generated bodies are pure functions of (SAMPLER_SEED, index); placeholder bodies are parsed from text files in bodies_placeholder/.
The fresh held-out set (heldout_v2) is read from a locked, hash-checked manifest by heldout.py and is deliberately NOT listed in get() for tuning code."""
from __future__ import annotations

import pathlib
import sys

from .parse import parse_body
from .sampler import make_body, split_of
from .spec import BodySpec
from .validity import check_body

SAMPLER_SEED = 0
STOP1_INDICES = [0, 3, 1, 4, 2]                                   # the 5 bodies of stop point 1 (all train split)
TRAIN12 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]                 # 4 per family, train split
TUNING12 = [9, 10, 11, 21, 22, 23, 33, 34, 35, 45, 46, 47]        # 4 per family, tuning split: not used to train PPO/residual, but used by the rhythm search (NOT held out)
PLACEHOLDER_DIR = pathlib.Path(__file__).resolve().parent.parent / "bodies_placeholder"

assert all(split_of(i) == "train" for i in TRAIN12) and all(split_of(i) == "tuning" for i in TUNING12)


def generated(indices: list[int]) -> list[BodySpec]:
    return [make_body(SAMPLER_SEED, i)[0] for i in indices]


def placeholder() -> list[tuple[BodySpec, dict]]:
    """Each bodies_placeholder/<name>.txt: header lines 'key: value' (source, author, verified), a blank line, then the description."""
    out = []
    for f in sorted(PLACEHOLDER_DIR.glob("*.txt")):
        head, _, text = f.read_text().partition("\n\n")
        meta = dict(line.split(":", 1) for line in head.splitlines() if ":" in line)
        meta = {k.strip(): v.strip() for k, v in meta.items()}
        spec = parse_body(text.strip())
        spec = BodySpec(**{**spec.__dict__, "name": f"ph_{f.stem}"})
        meta["valid"] = check_body(spec).valid
        out.append((spec, meta))
    return out


def get(name: str) -> list[BodySpec]:
    if name == "stop1":
        return generated(STOP1_INDICES)
    if name == "train12":
        return generated(TRAIN12)
    if name == "tuning12":
        return generated(TUNING12)
    if name == "tuning48":  # every tuning-split body of the 200 (16 per family), not only the first 4 per family
        return generated([i for i in range(200) if split_of(i) == "tuning"])
    if name == "heldout_v2":
        from .heldout import heldout_specs
        return heldout_specs()
    if name == "placeholder":
        ps = placeholder()
        for s, m in ps:
            if not m["valid"]:
                print(f"WARNING: placeholder description {s.name} is not a valid body and is excluded; count it as a failure", file=sys.stderr)
        return [s for s, m in ps if m["valid"]]
    raise KeyError(name)
