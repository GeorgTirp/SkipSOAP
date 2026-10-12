Isolated side study. Nothing here changes the main lab, its scripts or its runs, or the gap-audit study. Large artifacts live outside the synced folder at `~/skipsoap-study-data/geometry_followup_20261008/` (`$GEO_FOLLOWUP_DATA`). Start with RESULTS.md.

# Geometry follow-up (`geometry_followup_20261008`)

Questions, selected by the project owner on 8 October 2026: tuned PARSEC (`pns_eigenadam`, both
branches), tuned OT (graph-Wasserstein) training, and full (non-block) GGN and OT steps compared with
their tensor-block versions. Prospective protocol: [PROTOCOL.md](PROTOCOL.md) (SHA-256
`abc5998f95241e838bff6c1556aa69d74ca734d815753f84c452c91c54b56ad3`, written 2026-10-08T09:15Z before
any code or run). Departures: [DEVIATIONS.md](DEVIATIONS.md). Results: [RESULTS.md](RESULTS.md).
Draft ledger entry: [HANDOFF_PROPOSED_ENTRY.md](HANDOFF_PROPOSED_ENTRY.md).

Baselines are the gap-audit evaluation units (`~/skipsoap-study-data/gap_audit_20261007/runs/20261007T143302-baba2664`,
read-only): same seeds 1–3, data order, harness and validation schedule.

## Status

- [x] Isolation manifest (1,157 files), baseline suite (106 passed), protocol, study tests (6 passed)
- [x] Cost dry run `20261008T091935-6d9ca4b0`: projected 11.9 CPU-h (budget 18); no fallback
- [x] Reproduction gates `20261008T092039-cdeb20f3`: all three bitwise
- [x] Tuning `20261008T092438-2c2b66dd` (seed 0, staged)
- [x] Evaluation, `ggn_full` `20261008T092440-b7620df1` (seeds 1–3, 500 steps)
- [x] Evaluation, PARSEC/OT/`ot_full` `20261008T103332-f9ee97b7`
- [x] Summary, final suite (106 passed), study tests (6 passed), isolation manifests identical, reports
