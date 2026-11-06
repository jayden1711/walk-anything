# walk-anything: Claude rules

**Read `~/projects/NUC_RULES.md` first.** It covers Quatern priority, sim-run/gpu-run, the results policy and publishing.

Describe any robot body in words; one policy makes it walk instantly, with no training per body.
Plan: PLAN.md (draft until approved). State: HANDOFF.md.

- **Lowest priority on the NUC.** The trainer is an always-on background job and must run through
  `~/projects/bin/gpu-run --preemptible --need-mib <N> --label walk-anything -- <python train cmd>`.
  Never run it under plain gpu-run or outside gpu-run.
- **Yield contract** (tested in gpu-run): on SIGUSR1 or when `$GPU_RUN_YIELD_FILE` exists, save an atomic checkpoint
  (temp file + rename) and exit within 60 s; always resume from the latest checkpoint at startup. Quatern
  always preempts. Keep checkpoint saves fast; make them more frequent if a save takes more than ~20 s.
- **CPU side:** env/sim workers at most 4 processes, one thread each (`OMP_NUM_THREADS=1` etc., as `sim-run` sets);
  they inherit nice 19 from the wrapper.
- **Disk:** keep only the last few checkpoints plus milestones. Report usage in HANDOFF.md. The NUC disk is
  shared (keep > 100 GB free).
- **Public repo, Apache-2.0.** Code only. `checkpoints/`, `runs/`, `results/`, logs and videos are git-ignored.
  No numbers in README/PLAN/HANDOFF until approved. Nothing goes public without the owner's go-ahead.
- Don't start long training until the owner has approved PLAN.md.
