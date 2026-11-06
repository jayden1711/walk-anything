"""Phase patterns of a rhythm generator from the leg-permutation symmetry group (Golubitsky-Stewart style).

A pattern assigns each leg a phase in Z_m (fractions k/m of the gait cycle). Its spatio-temporal symmetry set is
{(g, s): phase[g(i)] = phase[i] + s for all legs i}. H = the g that occur; K = those with s = 0; H/K is cyclic.
A pattern is *primary* when H is the whole group (every symmetry of the body is kept as a spatio-temporal symmetry),
which is the symmetry-derived list of gaits: pronk / bound / pace / trot for a quadruped, hop / alternate for a biped,
tripod-like patterns for a hexapod. Secondary patterns with a smaller H (walk-like, all phases distinct) are listed after.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    name: str
    phases: tuple        # ints in Z_m, one per leg (leg order as in mjcf.hip_positions)
    m: int
    h_size: int
    k_size: int
    primary: bool

    def phase_fraction(self) -> list[float]:
        return [p / self.m for p in self.phases]


def symmetry_set(phases: tuple, m: int, group: list[tuple]) -> list[tuple]:
    out = []
    for g in group:
        shifts = {(phases[g[i]] - phases[i]) % m for i in range(len(phases))}
        if len(shifts) == 1:
            out.append((g, shifts.pop()))
    return out


def _name(phases: tuple, n_legs: int) -> str:
    p = phases
    if len(set(p)) == 1:
        return "pronk" if n_legs > 2 else "hop"
    if n_legs == 2:
        return "alternate"
    if n_legs == 4:  # legs: 0 LF, 1 RF, 2 LH, 3 RH
        if p[0] == p[1] != p[2] == p[3]:
            return "bound"
        if p[0] == p[2] != p[1] == p[3]:
            return "pace"
        if p[0] == p[3] != p[1] == p[2]:
            return "trot"
        if len(set(p)) == 4:
            return "walk-like"
    if n_legs == 6 and len(set(p)) == 2 and p[0] == p[3] == p[4] != p[1]:  # 0 LF, 1 RF, 2 LM, 3 RM, 4 LH, 5 RH
        return "tripod"
    return "pattern"


def derive_patterns(group: list[tuple], n_legs: int, m: int | None = None, max_patterns: int = 12) -> list[Pattern]:
    """Patterns with a non-trivial spatio-temporal symmetry, primary ones first, capped at max_patterns."""
    m = m or (2 if n_legs == 2 else n_legs)
    full = len(group)
    found = []
    for rest in itertools.product(range(m), repeat=n_legs - 1):
        ph = (0,) + rest
        sym = symmetry_set(ph, m, group)
        h = len(sym)
        if h < 2 and full > 1:
            continue
        k = sum(1 for _, s in sym if s == 0)
        found.append(Pattern(_name(ph, n_legs), ph, m, h, k, h == full))
    found.sort(key=lambda p: (not p.primary, -p.h_size, len(set(p.phases)), p.phases))
    return found[:max_patterns]
