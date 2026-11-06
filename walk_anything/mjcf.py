"""BodySpec -> MJCF string. Procedural, no third-party model files."""
from __future__ import annotations

from .spec import BodySpec

SIM_DT = 0.005


def hip_positions(s: BodySpec) -> list[tuple[float, float, float]]:
    lx, ly, _ = s.torso_size
    n = s.n_legs // 2
    xs = [0.0] if n == 1 else [lx / 2 * (1 - 2 * i / (n - 1)) for i in range(n)]
    out = []
    for x in xs:
        for side in (1, -1):
            out.append((x, side * (ly / 2 + s.seg_radius), 0.0))
    return out


def default_angles(s: BodySpec) -> list[float]:
    """Per-leg default joint angles: roll, hip pitch, then the pitch chain.

    Absolute segment angles are [b, -b, 0, 0...], so the shank is vertical; a biped ankle keeps the sole level."""
    b = s.stance_bend
    absang = ([b, -b] + [0.0] * s.n_segs)[: s.n_segs]
    q = [0.0, absang[0]] + [absang[i] - absang[i - 1] for i in range(1, s.n_segs)]
    if s.family == "biped":
        q.append(-absang[-1])
    return q


def build_mjcf(s: BodySpec) -> str:
    lx, ly, lz = s.torso_size
    r = s.seg_radius
    rng_roll, rng_pitch = s.roll_range, s.pitch_range
    q0 = default_angles(s)
    tl = s.torque_limit
    arm = 0.001 + 8 * s.kp * SIM_DT**2  # reflected rotor inertia: keeps the explicit stiffness term stable
    body, acts = [], []
    small = lambda m: f'<inertial pos="0 0 0" mass="{0.05 * m:.6g}" diaginertia="{1e-5 + 0.05 * m * r * r:.6g} {1e-5 + 0.05 * m * r * r:.6g} {1e-5 + 0.05 * m * r * r:.6g}"/>'

    for li, (hx, hy, hz) in enumerate(hip_positions(s)):
        lines = [f'<body name="L{li}_roll" pos="{hx:.6g} {hy:.6g} {hz:.6g}">', small(s.seg_masses[0])]
        lines.append(f'<joint name="L{li}_j0" type="hinge" axis="1 0 0" range="{-rng_roll:.6g} {rng_roll:.6g}" damping="0.01" armature="{arm:.6g}"/>')
        lines.append(f'<body name="L{li}_hip">')
        ji = 1
        closers = 2
        for k, (L, m) in enumerate(zip(s.seg_lengths, s.seg_masses)):
            if k == 0:
                lines.append(f'<joint name="L{li}_j{ji}" type="hinge" axis="0 1 0" range="{-rng_pitch:.6g} {rng_pitch:.6g}" damping="0.01" armature="{arm:.6g}"/>')
            else:
                lines.append(f'<body name="L{li}_s{k}" pos="0 0 {-s.seg_lengths[k-1]:.6g}">')
                lines.append(f'<joint name="L{li}_j{ji}" type="hinge" axis="0 1 0" range="{-rng_pitch:.6g} {rng_pitch:.6g}" damping="0.01" armature="{arm:.6g}"/>')
                closers += 1
            lines.append(f'<geom type="capsule" fromto="0 0 0 0 0 {-L:.6g}" size="{r:.6g}" mass="{m:.6g}" contype="2" conaffinity="1" friction="1 0.005 0.0001" group="1"/>')
            ji += 1
        # foot at the tip of the last segment
        tip = s.seg_lengths[-1]
        if s.family == "biped":
            lines.append(f'<body name="L{li}_ankle" pos="0 0 {-tip:.6g}">')
            lines.append(f'<joint name="L{li}_j{ji}" type="hinge" axis="0 1 0" range="{-rng_pitch:.6g} {rng_pitch:.6g}" damping="0.01" armature="{arm:.6g}"/>')
            fl = s.foot_len if s.foot_len > 0 else 4 * r
            th = max(0.5 * r, 0.004)
            lines.append(f'<geom name="L{li}_foot" type="box" pos="{fl * 0.25:.6g} 0 {-th:.6g}" size="{fl / 2:.6g} {1.5 * r:.6g} {th:.6g}" mass="{0.5 * s.seg_masses[-1]:.6g}" contype="2" conaffinity="1" friction="1.2 0.005 0.0001" group="1"/>')
            lines.append("</body>")
            ji += 1
        else:
            lines.append(f'<geom name="L{li}_foot" type="sphere" pos="0 0 {-tip:.6g}" size="{1.2 * r:.6g}" mass="{0.02 * s.seg_masses[-1]:.6g}" contype="2" conaffinity="1" friction="1.2 0.005 0.0001" group="1"/>')
        lines.append("</body>" * closers)  # segments chain + hip + roll bodies
        body.append("\n".join(lines))
        for j in range(ji):
            acts.append(f'<position name="L{li}_a{j}" joint="L{li}_j{j}" kp="{s.kp:.6g}" kv="{s.kd_ratio * s.kp:.6g}" forcerange="{-tl:.6g} {tl:.6g}"/>')
    assert len(q0) == s.joints_per_leg, (len(q0), s.joints_per_leg)
    sx, sy, sz = lx / 2, ly / 2, lz / 2
    return f"""<mujoco model="{s.name or s.family}">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option timestep="{SIM_DT}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
    <body name="torso" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="{sx:.6g} {sy:.6g} {sz:.6g}" mass="{s.torso_mass:.6g}" contype="2" conaffinity="1" group="1"/>
{chr(10).join(body)}
    </body>
  </worldbody>
  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
"""
