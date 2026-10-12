# topfree_confirm_20261011

Pre-declared confirmation, on fresh seeds 4–6, that damping the top whitened directions inside training (no
momentum there, strong brake) beats weight averaging and a final learning-rate decay.

- [PROTOCOL.md](PROTOCOL.md): design and readings.
- [RESULTS.md](RESULTS.md): verdict, tables, sharpness, limitations.
- `scripts/`: `train.py` (runner, copied from the noise controls) and `summarize.py`.
- `tests/`: harness test.
- `runs/`: light mirrors. Seed-4 checkpoints, every 10 steps, are stored externally for the offline OU check.
