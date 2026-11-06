"""Deterministic text -> BodySpec parser for a small body-description grammar.

Examples:
  "quadruped, torso 0.5 x 0.25 x 0.12 m, 15 kg, legs of 2 segments 0.25 m and 0.25 m"
  "biped with 0.9 m legs, total mass 40 kg"
  "six-legged robot, three segments per leg, leg length 0.3 m, strong"
Unstated fields take family defaults scaled from the leg length and mass. An LLM front-end can emit the same
BodySpec JSON; this parser is the offline reference.
"""
from __future__ import annotations

import re

import numpy as np

from .spec import BodySpec, FAMILIES, G

_FAMILY_WORDS = {
    "quadruped": "quadruped", "four-legged": "quadruped", "four legged": "quadruped", "4-legged": "quadruped",
    "biped": "biped", "two-legged": "biped", "two legged": "biped", "humanoid": "biped", "2-legged": "biped",
    "hexapod": "hexapod", "six-legged": "hexapod", "six legged": "hexapod", "6-legged": "hexapod",
}
_NUM = r"(\d+(?:\.\d+)?)"
_NUMWORDS = {"two": 2, "three": 3, "2": 2, "3": 3}
_UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def _len(val: str, unit: str | None) -> float:
    return float(val) * _UNIT[(unit or "m").lower()]


def parse_body(text: str) -> BodySpec:
    t = text.lower().strip()
    fam = next((v for k, v in _FAMILY_WORDS.items() if k in t), None)
    if fam is None:
        raise ValueError("no body family found (quadruped / biped / hexapod)")

    n_segs = 2
    m = re.search(r"(two|three|2|3)[ -]segment", t)
    if m:
        n_segs = _NUMWORDS[m.group(1)]

    seg_lengths = None
    m = re.search(r"segments?\s*(?:of)?\s*" + _NUM + r"\s*(cm|mm|m)?(?:\s*(?:,|and)\s*" + _NUM + r"\s*(cm|mm|m)?)+", t)
    if m:
        seg_part = t[m.start():]
        seg_part = re.split(r";|\.\s|\bkg\b", seg_part)[0]
        vals = re.findall(_NUM + r"\s*(cm|mm|m)?\b", seg_part)
        if len(vals) >= 2:
            seg_lengths = tuple(_len(v, u) for v, u in vals[:3])
            n_segs = len(seg_lengths)
    leg_len = None
    m = re.search(r"(?:leg length|legs? of|(?:legs?)\s*)\s*" + _NUM + r"\s*(cm|mm|m)\b", t) or re.search(_NUM + r"\s*(cm|mm|m)\s*legs?", t)
    if m and seg_lengths is None:
        leg_len = _len(m.group(1), m.group(2))
    if seg_lengths is None:
        L = leg_len if leg_len is not None else {"quadruped": 0.4, "biped": 0.8, "hexapod": 0.3}[fam]
        seg_lengths = tuple(L / n_segs for _ in range(n_segs))
    L = sum(seg_lengths)

    torso = None
    m = re.search(r"torso\s*" + _NUM + r"\s*(?:x|by|\*)\s*" + _NUM + r"\s*(?:x|by|\*)\s*" + _NUM + r"\s*(cm|mm|m)?", t)
    if m:
        u = m.group(4)
        torso = tuple(_len(m.group(i), u) for i in (1, 2, 3))
    if torso is None:
        f = {"quadruped": (1.4, 0.5, 0.25), "biped": (0.35, 0.4, 0.5), "hexapod": (2.0, 0.5, 0.25)}[fam]
        torso = tuple(L * x for x in f)

    total = None
    m = re.search(r"total mass\s*" + _NUM + r"\s*kg", t)
    if m:
        total = float(m.group(1))
    torso_mass = None
    m = re.search(r"torso\s*(?:mass)?\s*" + _NUM + r"\s*kg", t) or re.search(_NUM + r"\s*kg\s*torso", t)
    if m:
        torso_mass = float(m.group(1))
    if torso_mass is None:
        m = re.search(_NUM + r"\s*kg", t)
        if m and total is None:
            torso_mass = float(m.group(1)) if "legs" not in t[max(0, m.start() - 8): m.start()] else None
    n_legs = FAMILIES[fam]
    r = min(seg_lengths) * 0.12
    leg_share = 0.3
    if total is not None:
        torso_mass = total * (1 - leg_share)
    if torso_mass is None:
        torso_mass = 350.0 * torso[0] * torso[1] * torso[2]
    if total is not None:
        seg_total = total * leg_share / n_legs
    else:
        seg_total = min(0.3 * torso_mass / n_legs, 1500.0 * np.pi * r * r * L)
    w = np.array(seg_lengths)
    seg_masses = tuple(float(x) for x in seg_total * w / w.sum())

    strength = 0.9 if "strong" in t else 0.45 if "weak" in t else 0.7
    base = BodySpec(family=fam, torso_size=torso, torso_mass=float(torso_mass), seg_lengths=tuple(seg_lengths),
                    seg_masses=seg_masses, seg_radius=float(r), foot_len=float(0.3 * L) if fam == "biped" else 0.0)
    torque = strength * base.total_mass * G * L / (n_legs / 2)
    return BodySpec(**{**base.__dict__, "torque_limit": float(torque), "name": "parsed"})
