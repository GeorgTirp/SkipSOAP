Isolated side study. Nothing here changes the main lab, its scripts or its runs. Large artifacts live outside the synced folder at `~/skipsoap-study-data/gap_audit_20261007/` (`$GAP_AUDIT_DATA`). Start with RESULTS.md.

# Gap audit study (`gap_audit_20261007`)

Prospective protocol: [AGENT_INSTRUCTIONS.md](AGENT_INSTRUCTIONS.md) (frozen; SHA-256
`469b6eed9857f15c6d6933e132086b9a73dd6e688ebc3a3fd389eb529e3c5838`). Departures are
listed in [DEVIATIONS.md](DEVIATIONS.md). Results for all checks are in
[RESULTS.md](RESULTS.md). A draft entry for the main ledger is in
[HANDOFF_PROPOSED_ENTRY.md](HANDOFF_PROPOSED_ENTRY.md); a human merges it.

## Status and task list

Final state (2026-10-07, 16:50Z). Summary and verdicts: [RESULTS.md](RESULTS.md) section 1.

- [x] Section 4: documents read, isolation manifest, baseline suite (106 passed), protocol hash,
      cost dry run (all projections within budget; no fallback)
- [x] Check 1 step 1: reproduction gate (passed, bitwise)
- [x] Check 1 step 2: seed-0 tuning, with the one-time edge extension
- [x] Check 1 step 3: evaluation, seeds 1–3, 500 steps. Verdict **C (mixed)**
- [x] Check 2a/2b (inside the evaluation): `dense_ef` **no**; `soap_gradient_map` far worse than searched SOAP
- [x] Check 2c: best cheap family `kron_basis_diag` 46.0%, **insufficient**
- [x] Check 3: **OT is easier to compress** by the pre-declared reading (thin margins)
- [x] Check 4a: 22 trajectories, 66 states; reproduction gate passed
- [ ] Check 4b: **stopped** at 21 of 22 units (study-internal assertion; see DEVIATIONS.md #11).
      Partial results reported. A rerun needs the project owner's approval.
- [x] Check 4c: skipped, gate failed (46.0% < 60%)
- [x] Final suite 106 passed; study tests 12 passed; isolation manifests identical; mirrors 27.9 MB

## Runs

| Purpose | Run ID (external; mirror under `runs/`) |
| --- | --- |
| Section 4.5 cost dry run | `20261007T121137-28c507f9` |
| Check 1 reproduction gate | `20261007T121557-dfef403c` |
| Check 1 tuning (seed 0) | `20261007T122018-998da6a3` |
| Check 1 evaluation (seeds 1–3, 500 steps; includes 2a/2b) | `20261007T143302-baba2664` |
| Check 2c capacity | `20261007T123608-7f39721a` |
| Check 3 OT/GGN capacity | `20261007T124647-1d8487a2` |
| Check 4a architecture-family dataset | `20261007T125950-63edd820` |
| Check 4b headroom across architectures | `20261007T143410-b579ef62` |

Each mirror holds the run's json/md/png/yaml files and `EXTERNAL_PATH.txt`. Checkpoints,
saved states (`arrays.pt`), compressed full histories and source archives with checksums are only
in the external run directories.

## Layout

`configs/` study configs, `scripts/` study code, `tests/` study tests (run by explicit
path), `runs/` light mirrors of external runs (json/md/png/yaml), `figures/` PNGs rebuilt
from stored JSON, `logs/` stdout/stderr of long runs.
