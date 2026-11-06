#!/usr/bin/env bash
# End-to-end yield test: preemptible trainer under gpu-run, a normal gpu-run job forces a yield, assert
# (c) checkpoint + exit well inside 60 s, (d) the restarted trainer resumes from that checkpoint.
# Uses a private gpu-run lock dir and a CPU trainer, so it never touches the real GPU lock or the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
T="$(mktemp -d "${CLAUDE_JOB_DIR:-/tmp}/yieldtest.XXXXXX")"
export GPU_RUN_LOCKDIR="$T/locks"
RUN="yieldtest_$$"
GR="$HOME/projects/bin/gpu-run"
.venv/bin/python - "$T/cfg.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open("configs/default.yaml"))
c.update(envs_per_body=2, workers=2, horizon=16, epochs=1, minibatches=2)
c["bodies"]["indices"] = [0, 1]
c["ckpt"]["every_seconds"] = 10**6
yaml.safe_dump(c, open(sys.argv[1], "w"))
PY
"$GR" --preemptible --need-mib 100 --label yieldtest -- \
  .venv/bin/python -m walk_anything.train --config "$T/cfg.yaml" --run "$RUN" --device cpu > "$T/trainer.log" 2>&1 &
WRAP=$!
fail() { echo "FAIL: $*"; tail -20 "$T/trainer.log"; kill -TERM "$WRAP" 2>/dev/null || true; exit 1; }

for _ in $(seq 120); do [ "$(cat runs/$RUN/metrics.jsonl 2>/dev/null | wc -l)" -ge 3 ] && break; sleep 1; done
[ "$(cat runs/$RUN/metrics.jsonl 2>/dev/null | wc -l)" -ge 3 ] || fail "trainer made no progress"
ls checkpoints/$RUN/ckpt_*.pt >/dev/null 2>&1 && fail "checkpoint existed before the yield"

t0=$(date +%s)
"$GR" --need-mib 100 --label yieldtest-waiter --wait 120 -- bash -c 'sleep 8' > "$T/waiter.log" 2>&1 || fail "waiter job failed"
t1=$(date +%s)
echo "waiter job finished after $((t1 - t0)) s (includes its 8 s of work)"

ls checkpoints/$RUN/ckpt_*.pt >/dev/null 2>&1 || fail "no checkpoint after yield"
grep -q "yield requested" "$T/trainer.log" || fail "wrapper never requested a yield"
y=$(grep -o "yielded after [0-9]*s" "$T/trainer.log" | grep -o "[0-9]*" | head -1)
[ -n "$y" ] && [ "$y" -lt 30 ] || fail "yield took '${y:-?}' s (want < 30)"
echo "yielded in ${y} s"
ck=$(ls checkpoints/$RUN | grep -o 'ckpt_[0-9]*' | sed 's/ckpt_0*//' | sort -n | tail -1)
ls checkpoints/$RUN | grep -q tmp && fail "temp file left in checkpoint dir"

for _ in $(seq 120); do grep -q "resumed from iteration" "$T/trainer.log" && break; sleep 1; done
grep -q "resumed from iteration ${ck:-?} " "$T/trainer.log" || fail "restart did not resume from iteration $ck"
echo "resumed from iteration $ck"
kill -TERM "$WRAP"; wait "$WRAP" 2>/dev/null || true
echo "PASS (logs: $T, run: $RUN)"
