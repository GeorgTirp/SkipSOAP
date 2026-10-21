# Deviations

## D1 — replay progress line referenced a moved variable (2026-10-12, before any replay output)

- **What happened.** All 24 training units of run `20261012T084322-39ae45d3` completed. The first replay
  (`replay --unit soap__cpu64__s0 --device cpu`) then stopped with `NameError: name 'refreshed' is not defined`. The
  progress `print` in `replay` still referred to a variable that had moved into `replay_step` when the single-step
  core was factored out for the tests. The tests call `replay_step` directly, so they did not reach the line.
  Nothing had been written: `replay` writes its JSON only after the loop. Because `run_all.sh` stops on errors,
  finalization and the cost run C1 had not started.
- **Change.** `scripts/train.py`: the print now reads `rows[-1]['soap_refresh']`. `replay` now calls
  `verify_amended`, which accepts a file listed in the run's `amendments.json` only at its amended checksum and
  against its archived amended copy (`source_amended/`). No computation changed: the training loop, `replay_step`
  and every gate and reading are as before.
- **Checksums.** `scripts/train.py` as archived and run: `b9315a240e4f3bc4e91e42908648582d6f0435668f2c6add1e0c905f8cb31e74`;
  amended: see the run's `amendments.json`.
- **Continuation.** Replays, finalization and C1 are run from the existing run directory. The completed units are
  not rerun.

## D2 — two cost runs for C1 (2026-10-12)

- **What happened.** The first orchestration chained the cost run after `run_all.sh` with `;`, not `&&`. So after the
  D1 failure it created and ran a cost run anyway (`20261012T085237-8f184eba`, cells 10:52:38–10:55:11 local time).
  The continuation after D1 started at 10:55:14. It ran the replays and finalization, then a second cost run in the
  protocol's order (`20261012T085621-870cc249`, 10:56:22–10:59:00).
- **Overlap.** None on the GPU. The first run's last cell completed at 10:55:11, before the continuation started, and
  the GPU replays ran 10:55:30–10:56:08. During the first cost run only CPU-only study tests of the next study ran
  (one thread, CUDA hidden).
- **Use.** The second run is reported as C1. The first is reported as a repetition: per-cell mean seconds per step
  agree within 2.5% and peak memory is identical.
