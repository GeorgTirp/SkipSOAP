# damping_sweep_20261011

Two pre-declared checks that gate a per-direction Kalman version of the momentum-free top-space brake:
- a sweep of its damping strength c ∈ {256, 1024, 4096, 10⁹} against c = 64 (seeds 4–6);
- an offline check of the gradient-noise (Ornstein–Uhlenbeck) model on the seed-4 checkpoints of
  `topfree_confirm_20261011`.

- [PROTOCOL.md](PROTOCOL.md): design, readings S1, S2, O1 and the Kalman decision rule.
- [DEVIATIONS.md](DEVIATIONS.md): pre-run clarification of the lag-10 prediction, implementation details and one
  stopped launch.
- [RESULTS.md](RESULTS.md): readings, tables, decision (no Kalman version), limitations.
- `scripts/`: `train.py` (runner, copied from `topfree_confirm`), `summarize.py` (S1, S2, gate), `ou_check.py`
  (part 2), queue runner and isolation manifest.
- `results/`: `sweep_summary.json`, `ou_check.json`.
- `tests/`: harness test. `runs/`: light mirror of the sweep run.
