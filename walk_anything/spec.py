"""BodySpec: the structured description of a legged body. The text parser and the sampler both produce it."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

FAMILIES = {"quadruped": 4, "biped": 2, "hexapod": 6}
G = 9.81


@dataclass(frozen=True)
class BodySpec:
    family: str = "quadruped"
    torso_size: tuple = (0.4, 0.2, 0.1)      # full extents x (forward), y (lateral), z, metres
    torso_mass: float = 10.0                 # kg
    seg_lengths: tuple = (0.2, 0.2)          # leg segments from hip down, metres
    seg_masses: tuple = (0.6, 0.4)           # kg per segment (per leg)
    seg_radius: float = 0.02                 # capsule radius, metres
    stance_bend: float = 0.5                 # default hip pitch (rad); knee = -2x, so the shank is vertical
    roll_range: float = 0.5                  # hip roll limit +- (rad)
    pitch_range: float = 1.5                 # pitch joint limit +- (rad)
    torque_limit: float = 20.0               # Nm, every joint
    kd_ratio: float = 0.05                   # kd = kd_ratio * kp (s)
    foot_len: float = 0.0                    # biped flat-foot length (m); 0 = sphere foot (ignored for non-bipeds)
    name: str = ""

    @property
    def n_legs(self) -> int:
        return FAMILIES[self.family]

    @property
    def n_segs(self) -> int:
        return len(self.seg_lengths)

    @property
    def joints_per_leg(self) -> int:
        return 2 + (self.n_segs - 1) + (1 if self.family == "biped" else 0)

    @property
    def n_joints(self) -> int:
        return self.n_legs * self.joints_per_leg

    @property
    def leg_length(self) -> float:
        return float(sum(self.seg_lengths))

    @property
    def total_mass(self) -> float:
        extra = 0.5 * self.seg_masses[-1] if self.family == "biped" else 0.02 * self.seg_masses[-1]
        return self.torso_mass + self.n_legs * (sum(self.seg_masses) + extra + 0.05 * self.seg_masses[0] * 2)

    @property
    def kp(self) -> float:
        return self.torque_limit / 0.5

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str | dict) -> "BodySpec":
        d = json.loads(s) if isinstance(s, str) else dict(s)
        for k in ("torso_size", "seg_lengths", "seg_masses"):
            d[k] = tuple(d[k])
        return BodySpec(**d)


J_MAX = 24  # hexapod with 3 segments: 6 legs x 4 joints
