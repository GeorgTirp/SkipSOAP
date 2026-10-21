# Deviations: ot_1000_20261013

## D1. Stopped by the owner at 17:36Z before any unit completed; to be resumed (2026-10-13)

- **Why:** the owner's machine time was ending ("I have only 5 mins left on this machine so lets stop and resume
  tomorrow"). The orchestration's process group was terminated at 17:36Z.
- **State:**
  - Before-manifest and before-suites taken (`logs/baseline_suite.log`).
  - Run `20261013T173455-a0a150ee` created.
  - `soap__lr0.03__s1` interrupted early (started 17:34:58Z).
  - `topfree__lr0.03__s1` and `topfree_ot__lr0.03__s1` not started.
- **Resume** (from `residual-curvature-lab/`):
  1. Run `sh studies/ot_1000_20261013/scripts/run_queue.sh <run>` on the same run. The archive is verified, the
     partial unit is kept as `*.interrupted-*` and rerun from step 0, and units are deterministic.
  2. Then run `train.py report --run <run>`, the after-manifest and the after-suites (see `scripts/orchestrate.sh`,
     from `RUN=` on).
  3. Launch it detached (`setsid nohup … &`).

## D2. Superseded and closed without results (2026-10-14)

The owner chose a tuned 1,000-step study instead (`ot_tuned_1000_20261014`: OT-probe top-free and SOAP, both tuned on
seed 0, evaluated on seeds 1–3). This exploratory check is not resumed. Run `20261013T173455-a0a150ee` holds only the
interrupted `soap__lr0.03__s1`. No results.
