"""Generate the fresh held-out set ONCE: bodies_heldout_v2/manifest.json (read-only) and print its SHA-256.
Refuses to run if the manifest exists. After running: put the printed hash in walk_anything/heldout.py (MANIFEST_SHA256) and README.md.
Usage: python scripts/make_heldout_v2.py
"""
import json
import os
import sys

from walk_anything import heldout as H
from walk_anything.sampler import make_body

if H.MANIFEST.exists():
    sys.exit(f"{H.MANIFEST} already exists: the held-out set is written once and never regenerated")
bodies, rejected = [], 0
for i in range(H.N_BODIES):
    spec, rep, tries = make_body(H.SAMPLER_SEED, i)
    rejected += tries
    bodies.append({"index": i, "name": spec.name, "family": spec.family, "spec": json.loads(spec.to_json())})
doc = {"description": "Fresh held-out body set v2: final evaluation only. See walk_anything/heldout.py.", "sampler_seed": H.SAMPLER_SEED,
       "n_bodies": H.N_BODIES, "rejected_samples": rejected, "bodies": bodies}
H.MANIFEST.parent.mkdir(exist_ok=True)
H.MANIFEST.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
os.chmod(H.MANIFEST, 0o444)
print(f"{len(bodies)} bodies, {rejected} rejected draws\nsha256 {H.manifest_sha256()}")
