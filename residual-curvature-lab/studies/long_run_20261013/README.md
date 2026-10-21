# long_run_20261013

SOAP vs top-free PARSEC-L (and top-free + N9, added by protocol amendment) on CIFAR-10 over 2,000 steps (warm-up 100, WSD decay over the last 400 steps to 1%). Both
arms are tuned on seed 0 and evaluated on seeds 1–5.

- [PROTOCOL.md](PROTOCOL.md): schedule, tuning, readings R1–R5.
- [RESULTS.md](RESULTS.md): at 2,000 steps, with every arm at lr 9e-2, top-free no longer beats SOAP (EMA 0.462 vs
  0.457, lower in 1 of 5); N9 is mixed.
- [DEVIATIONS.md](DEVIATIONS.md): D1 (session end, resumed), D2 (owner stop after completion).
- `scripts/`: `train.py` (the arms are `cifar_ggn`'s, with the schedule constants changed; `topfree_n9` uses `n9.py`,
  copied unchanged from `n9_training`), `summarize.py`,
  `run_queue.sh`, `orchestrate.sh`, `common.py`.
- `tests/`: schedule and unit plan; the arms bitwise equal to `cifar_ggn`'s under the same schedule factor.
- External data: `~/skipsoap-study-data/long_run_20261013/` (`$LR_DATA`).
