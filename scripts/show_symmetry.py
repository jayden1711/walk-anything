"""Print detected symmetries and derived phase patterns for the 5 stop-1 bodies."""
import yaml

from walk_anything.sampler import make_body
from walk_anything.symmetry import detect_symmetries, leg_group

cfg = yaml.safe_load(open("configs/default.yaml"))
for i in cfg["bodies"]["indices"]:
    s = make_body(cfg["bodies"]["seed"], i)[0]
    els = detect_symmetries(s)
    print(s.name, [(e.name, e.leg_perm, "pose_inv" if e.pose_invariant else "pose_flips") for e in els])
    try:
        from walk_anything.cpg import derive_patterns
        for p in derive_patterns(leg_group(els), s.n_legs)[:12]:
            print("   ", p.name, p.phases, "H", p.h_size, "primary" if p.primary else "")
    except ImportError:
        pass
