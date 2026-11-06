# Third-party models, assets and tools

Nothing listed here is committed or redistributed. Add a row **before** first use: upstream, pinned
revision, license, how it is fetched. Training dependencies and robot assets must be Apache/BSD/MIT unless the
owner approves otherwise.

| Item | Upstream | Revision | License | Use / notes |
|---|---|---|---|---|
| MuJoCo (Python `mujoco`) | github.com/google-deepmind/mujoco | pip 3.17.0 | Apache-2.0 | CPU physics; installed into local `.venv`, not redistributed |
| PyTorch (`torch`) | pytorch.org | pip 2.15.0+cu132 | BSD-3-Clause | policy + PPO on GPU; `.venv` only |
| NumPy | numpy.org | pip 2.5.4 | BSD-3-Clause | `.venv` only |
| PyYAML | pyyaml.org | pip 6.0.3 | MIT | configs; `.venv` only |
| pytest | pytest.org | pip 9.2.0 | MIT | dev/test only; `.venv` only |
| Predictive sampling (method) | arXiv 2212.00541, Howell et al. 2022 | n/a (algorithm only) | n/a | `walk_anything/mpc.py` is our own implementation on the `mujoco` Python bindings; no code from the paper's repo |
| MuJoCo MPC (MJPC), reference only | github.com/google-deepmind/mujoco_mpc | not pinned, not used | Apache-2.0 | Considered as the baseline; skipped because it needs a C++/Bazel build and a per-body task definition. If adopted later, pin a commit here first |

All robot bodies are generated procedurally by `walk_anything/mjcf.py`; no robot models or datasets are used.
