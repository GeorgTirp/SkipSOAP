# ggn_c_control_20261013

Fairness control for `probe_variants_20261013`'s OT result: top-free PARSEC-L with GGN probes and c tuned on CIFAR
(seed 0, c ∈ {64, 256, 1024, 4096}), against top-free with OT probes at c = 4096. Seeds 1–3, 500 steps.

**Status: cancelled before any run (owner, 13 October; see [DEVIATIONS.md](DEVIATIONS.md)).**

- [PROTOCOL.md](PROTOCOL.md): arms, G1 (bitwise against `probe_variants` and `cifar_ggn`), readings C1–C3.
- `scripts/`: `train.py` (derived from `probe_variants`'), `probes_ot.py` (copied unchanged), `g1.py`,
  `summarize.py`, `run_queue.sh`, `orchestrate.sh`, `common.py`.
- External data: `~/skipsoap-study-data/ggn_c_control_20261013/` (`$CC_DATA`).
