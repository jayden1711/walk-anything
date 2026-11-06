# walk-anything

Goal: describe any robot body in words and have one policy make it walk instantly, with no training per body. **This goal is not met yet**: in the simulation tests so far, no controller here walks the generated bodies reliably (see [CRITIQUES.md](CRITIQUES.md)).

Early stage; see [PLAN.md](PLAN.md). Code is Apache-2.0 ([LICENSE](LICENSE)). Third-party models and assets: [THIRD_PARTY.md](THIRD_PARTY.md).

**Everything here is simulation** (MuJoCo, procedurally generated bodies). Nothing has been run on a real robot.

The seven descriptions in `bodies_placeholder/` were written by an AI assistant, not by other people, and support no generalisation claim; descriptions from other people will be collected for the public post.

**Body sets.** `train12` / `stop1`: training bodies. `tuning12` / `tuning48`: the old "held-out" split; the per-body rhythm search was run on it, so it is a TUNING set, not held out.
`heldout_v2`: 48 fresh bodies (sampler seed 20261106, same sampler and validity rules), written once to `bodies_heldout_v2/manifest.json` (read-only), used only for final evaluation
(no tuning, gait search, training or debugging; guarded in code, `walk_anything/heldout.py`). Manifest SHA-256:
`41173bec84dbe54ef99f286592a20693c8a981b1d3ddec1985021c5907eb0c9f`

Known holes and how each is handled: [CRITIQUES.md](CRITIQUES.md). Results are not stored in git; every result has a one-command
entry point in the [Makefile](Makefile) and is written to a local, git-ignored `results/` with its seeds, settings and date.

## Disclosure
The author is an NVIDIA intern until 2026-11-13 (this work was done in that period) and founded Quatern. Development runs on a shared RTX 5080 machine; this project is not an
NVIDIA product and nothing here has been reviewed or endorsed by NVIDIA or Quatern.

## Related work
Cited where used; items marked [verify] were written from memory without web access and need a check before publication.
- Symmetric CPG theory: Golubitsky, Stewart, Buono, Collins, "Symmetry in locomotor central pattern generators and animal gaits", Nature 401:693-695 (1999).
- Automatic morphological symmetry detection / use in robot learning: Ordonez-Apraez et al., RSS 2023, arXiv 2303.04857.
- Predictive sampling (our no-training baseline, own implementation): Howell et al., arXiv 2212.00541. MuJoCo MPC itself is not used.
- Central pattern generators for legged robots, review: Ijspeert, Neural Networks 21(4), 2008 [verify].
- Adaptation to varying dynamics: RMA, Kumar et al. (RSS 2021) [verify]; morphology-agnostic policies: Kurin et al., "My Body is a Cage" (ICLR 2021) [verify].
- Morphology-conditioned / universal policies: Huang et al., "One Policy to Control Them All" (ICML 2020) [verify]; MetaMorph, Gupta et al. (ICLR 2022) [verify];
  URMA, Bohlinger et al. (2024) [verify].
- Text to robot: Text2Robot, Ringel et al. [verify year and venue].
- RoboStriker, arXiv 2601.22517 [verify relevance: not read].
- MIT weight-space post (Takehi, Teoh, Zhang, Khattab, 2026-11-02) [verify relevance: not read].
