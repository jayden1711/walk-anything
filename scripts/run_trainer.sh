#!/usr/bin/env bash
# Always-on background trainer. Lowest priority: preemptible gpu-run job, trainer runs directly under it.
# Usage: scripts/run_trainer.sh [train args, e.g. --run stop1 --config configs/default.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
NEED_MIB="${NEED_MIB:-2048}"
exec "$HOME/projects/bin/gpu-run" --preemptible --need-mib "$NEED_MIB" --label walk-anything -- \
  .venv/bin/python -m walk_anything.train "$@"
