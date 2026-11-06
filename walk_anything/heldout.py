"""Fresh held-out body set (v2): written once to a locked manifest, hash-checked, and refused by every tuning path.

The 48 bodies of the old 'held-out' split were used by the rhythm search, so they are the TUNING set (bodysets.py).
This set is generated once by scripts/make_heldout_v2.py (sampler seed below, indices 0..N-1, same sampler and validity rules as the tuning
set), stored in bodies_heldout_v2/manifest.json (chmod a-w) and identified by its SHA-256 (also in README.md).

Rules: no tuning, rhythm/gait search, gait selection, hyperparameter choice, training or debugging ever touches it. Tuning and search code
calls refuse_heldout(); evaluation code may only read the set through heldout_specs(), which verifies the hash first.
Size: N = 48 (16 per family) fixed up front: it matches the 48-body tuning set so intervals are comparable, and 16 bodies per family is the smallest
size that gave usable per-family intervals on the tuning set; it is not changed after seeing results.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from .spec import BodySpec

SAMPLER_SEED = 20261106          # arbitrary (the date it was fixed); different from the tuning set's seed 0
N_BODIES = 48
EPISODES = 10                    # per body, fixed up front (predictive sampling: 5, as elsewhere)
MANIFEST = pathlib.Path(__file__).resolve().parent.parent / "bodies_heldout_v2" / "manifest.json"
MANIFEST_SHA256 = "41173bec84dbe54ef99f286592a20693c8a981b1d3ddec1985021c5907eb0c9f"
ALLOWED_ARMS = ("zero", "ppo", "mpc", "rhythm_default", "residual_default")  # arms that need no per-body search


class HeldoutViolation(RuntimeError):
    pass


def manifest_sha256(path: pathlib.Path = MANIFEST) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified(path: pathlib.Path = MANIFEST) -> dict:
    got = manifest_sha256(path)
    if got != MANIFEST_SHA256:
        raise HeldoutViolation(f"held-out manifest {path} has sha256 {got}, expected {MANIFEST_SHA256}: the locked set was changed or is not the registered one")
    return json.loads(path.read_text())


def heldout_specs() -> list[BodySpec]:
    """The fresh held-out bodies, after verifying the manifest hash. Evaluation only."""
    return [BodySpec.from_json(b["spec"]) for b in _verified()["bodies"]]


def is_heldout(spec: BodySpec) -> bool:
    """True if the spec carries a held-out name or has the same body content as a held-out body (renamed copies are caught too)."""
    if f"_{SAMPLER_SEED}_" in spec.name:
        return True
    try:
        key = _content_key(spec)
        return any(_content_key(BodySpec.from_json(b["spec"])) == key for b in _verified()["bodies"])
    except FileNotFoundError:
        return False


def _content_key(spec: BodySpec) -> str:
    d = json.loads(spec.to_json())
    d.pop("name", None)
    return json.dumps(d, sort_keys=True)


def refuse_heldout(specs, purpose: str) -> None:
    """Call at the top of any tuning, search, training or selection path. Raises HeldoutViolation if a held-out body is among the specs."""
    bad = [s.name for s in specs if is_heldout(s)]
    if bad:
        raise HeldoutViolation(f"refusing {purpose}: {len(bad)} body/bodies belong to the locked held-out set v2 (e.g. {bad[0]}). "
                               "That set is for final evaluation only; use the tuning set (bodysets.get('tuning48')).")
