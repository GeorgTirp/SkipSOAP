# stage_d_20261009

PARSEC-L training test: the fixed-step brake, Early-Newton and exact GGN with selected step, against tuned
SOAP and all earlier optimizers on the same seeds.

- [PROTOCOL.md](PROTOCOL.md): approved design and pre-declared readings.
- [RESULTS.md](RESULTS.md): readings, comparison table, seed-0 grid, diagnostics, limitations.
- [DEVIATIONS.md](DEVIATIONS.md): logging-bug relaunch; SOAP reproduction gate failure (seed 1).
- [HANDOFF_PROPOSED_ENTRY.md](HANDOFF_PROPOSED_ENTRY.md): draft ledger row for `HANDOFF.md`.
- `scripts/`:
  - `train.py` (runner) and `ggn_selected.py` (the GGN arm);
  - `summarize.py` and `orchestrate.sh`;
  - `arms.py` (gap-audit copy, unchanged) and `common.py`;
  - queue and manifest scripts.
- `tests/`: harness tests, run by explicit path; the PARSEC-L tests live in `studies/parsec_l_20261009/tests`.
- `runs/`: light mirrors; `figures/stage_d.png`; `logs/`.
