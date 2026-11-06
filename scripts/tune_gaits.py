"""Select the rhythm (pattern + parameters) per body on tuning seeds and cache it in results/round2/gaits/<body>.json.
Usage (via sim-run): sim-run .venv/bin/python scripts/tune_gaits.py SET [SET ...]   SET: stop1 | train12 | heldout12 | external
"""
import sys
import time

from walk_anything import bodysets
from walk_anything.gait import select_gait_cached

for name in sys.argv[1:]:
    for s in bodysets.get(name):
        t0 = time.time()
        _, d = select_gait_cached(s, "results/round2/gaits")
        print(f"{name} {s.name}: {d['pattern']} {d['phases']} T={d['period_T']} amp={d['amp']} [{time.time() - t0:.0f}s]", flush=True)
