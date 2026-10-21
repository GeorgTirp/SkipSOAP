# Deviations: long_run_20261013

## D1. Evaluation interrupted by the end of the Claude Code session (2026-10-13, about 14:31Z)

- **What happened:** the chain script that ran this study was a background process of the Claude Code session that
  launched it. It ended with that session, about 14:31Z.
  - Not a code, gate or numerical failure: no failure.json was written, and no process remained.
  - The queued `ggn_c_control_20261013` chain ended with it before starting.
- **State at the stop:**
  - Tuning `20261013T120559-5d587678`: 9 units, complete.
  - Extension `20261013T134808-5bd891bd`: 3 units at lr 9e-2, complete.
  - Evaluation `20261013T142233-20ef534e`:
    - `soap__lr0.09__s1` complete;
    - `soap__lr0.09__s2` interrupted at about step 100;
    - 13 units not started.
- **Selection:** all three arms selected the extension's lr 9e-2: `soap` with EMA 0.98, `topfree` with EMA 0.995,
  `topfree_n9` with EMA 0.98. That is still the grid's top edge. The protocol allows one extension per arm, so the
  evaluation runs at 9e-2 as declared, and the unbracketed optimum is a limitation (see `RESULTS.md`).
- **Resumption:** pending the owner's decision.
- **Owner decision (14:40Z): resume as declared.**
  - `scripts/resume.sh` runs the 14 pending evaluation units with the run's own queue (archive verified unchanged
    before resuming), then the summary.
  - Because every unit is deterministic, the results equal those of an uninterrupted run. The partial unit is kept as
    `soap__lr0.09__s2.interrupted-*`.
  - The processes are now detached from the session (`setsid nohup`), so a session ending no longer stops them.

## D2. Stopped by the owner at 17:25Z, after every unit had completed (2026-10-13)

- **What happened:** the owner asked to stop the current runs ("this is not enough to argue for building on this
  exact PARSEC idea"). The chain's process group was terminated at 17:25:00Z.
- **What had already finished:** the last evaluation unit (`topfree_n9__lr0.09__s5`) had completed at 17:24:59Z, and
  the queue had finalized the run. All 15 units verify by checksum, and the run's `complete.json` exists.
- **What the stop prevented:** only the summary step and the queued `ggn_c_control_20261013`.
- **Summary:** run by hand with `scripts/summarize.py` (read-only, CPU). The readings are therefore complete and
  unchanged by the stop.
