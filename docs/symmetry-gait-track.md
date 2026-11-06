# Method direction: symmetry-derived gait track (owner, 2026-11-06)

Fold into PLAN.md at the next review. Do it **before scaling the universal policy**. It doesn't change tonight's
first stop point.

## Idea
1. **Detect symmetries.** For each generated body, find its kinematic symmetries automatically: permutations of
   legs/joints, with sign flips, that map the body (and its joint limits, masses and contact geometry) onto itself
   (left-right mirror, front-back, rotational for radial bodies).
2. **Derive the rhythm.** Use Golubitsky-Stewart symmetric CPG theory to turn the symmetry group into the allowed
   phase patterns of a rhythm generator (walk / trot / pace / bound / pronk-style patterns: which legs move in
   phase, out of phase, or with fractional phase shifts).
3. **Learn only a residual.** A small universal residual policy, trained across bodies, adds balance and
   corrections on top of the derived rhythm. The heavy universal policy is then only needed where the rhythm
   isn't enough.
4. **Share the circuit compiler.** The rhythm generator and simple reflexes are a natural fit for the circuit compiler
   being built in zero-step-robots. Plan to import it as a shared package (no copy-paste), so the derived rhythm
   compiles into exact weights.

## Baselines to add
- **MuJoCo MPC (MJPC) predictive sampling** as a **no-training** baseline per body: Howell et al. 2022,
  "Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo", arXiv 2212.00541,
  github.com/google-deepmind/mujoco_mpc (Apache-2.0). It's slow, but it shows what's achievable with zero training.
- Existing arms: universal policy without the symmetry prior, and per-body PPO.

## Prior art (starting points)
- Golubitsky, Stewart, Buono, Collins, "Symmetry in locomotor central pattern generators and animal gaits",
  Nature 401:693-695 (1999), doi:10.1038/44416. The CPG network structure follows from the symmetry of the gaits;
  biped and quadruped gait predictions. Follow-up: Buono & Golubitsky, J. Math. Biol. (2001), quadruped primary gaits.
- Ordoñez-Apraez et al., "On discrete symmetries of robotics systems: a group-theoretic and data-driven analysis",
  RSS 2023 (arXiv 2303.04857): identifies a robot's morphological symmetry group and uses it via data augmentation
  and equivariant networks. This is the closest prior art for automatic symmetry detection.
- Symmetry-guided legged RL: IROS 2024 (Berkeley Hybrid Robotics, G-invariant rewards/dynamics). arXiv 2510.10455 (2025):
  rewards encoding temporal, morphological and time-reversal symmetry, and it notes that CPG controllers rely on
  prescribed foot trajectories. Ding & Gan (2023): group-theoretic definitions of legged gait symmetries.
- Universal / morphology-agnostic policies to compare against (from memory, verify): "One Policy to Control Them All"
  (Huang et al. 2020), MetaMorph (Gupta et al. 2022).
- A quick search found nothing that combines automatic symmetry detection → Golubitsky-Stewart CPG phase
  patterns → a universal residual policy. If that holds after a proper search, that combination is the contribution.
