"""Named body sets. Generated bodies are pure functions of (SAMPLER_SEED, index); placeholder bodies are parsed from text files in bodies_placeholder/."""
from __future__ import annotations

import pathlib

from .parse import parse_body
from .sampler import make_body, split_of
from .spec import BodySpec
from .validity import check_body

SAMPLER_SEED = 0
STOP1_INDICES = [0, 3, 1, 4, 2]                                   # the 5 bodies of stop point 1 (all train split)
TRAIN12 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]                 # 4 per family, train split
HELDOUT12 = [9, 10, 11, 21, 22, 23, 33, 34, 35, 45, 46, 47]       # 4 per family, held-out split, never used for tuning/training
EXTERNAL_DIR = pathlib.Path(__file__).resolve().parent.parent / "bodies_placeholder"

assert all(split_of(i) == "train" for i in TRAIN12) and all(split_of(i) == "heldout" for i in HELDOUT12)


def generated(indices: list[int]) -> list[BodySpec]:
    return [make_body(SAMPLER_SEED, i)[0] for i in indices]


def placeholder() -> list[tuple[BodySpec, dict]]:
    """Each bodies_placeholder/<name>.txt: header lines 'key: value' (source, author, verified), a blank line, then the description."""
    out = []
    for f in sorted(EXTERNAL_DIR.glob("*.txt")):
        head, _, text = f.read_text().partition("\n\n")
        meta = dict(line.split(":", 1) for line in head.splitlines() if ":" in line)
        meta = {k.strip(): v.strip() for k, v in meta.items()}
        spec = parse_body(text.strip())
        spec = BodySpec(**{**spec.__dict__, "name": f"ext_{f.stem}"})
        meta["valid"] = check_body(spec).valid
        out.append((spec, meta))
    return out


def get(name: str) -> list[BodySpec]:
    if name == "stop1":
        return generated(STOP1_INDICES)
    if name == "train12":
        return generated(TRAIN12)
    if name == "heldout12":
        return generated(HELDOUT12)
    if name == "placeholder":
        return [s for s, m in placeholder() if m["valid"]]
    raise KeyError(name)
