# ot_1000_20261013

Exploratory, one seed: SOAP, top-free PARSEC-L (GGN probes, c = 64) and top-free with OT-metric probes (c = 4096) on
CIFAR-10 over 1,000 steps (lr 3e-2, warm-up 50, decay over steps 801–1000). No verdict.

**Status: superseded by `ot_tuned_1000_20261014` and closed without results (see [DEVIATIONS.md](DEVIATIONS.md)).**

- [PROTOCOL.md](PROTOCOL.md)
- `scripts/`:
  - `train.py`: derived from `long_run`'s;
  - `probes_ot.py`, `n9.py`: copied unchanged;
  - `orchestrate.sh`: manifests, suites, runs, report.
- External data: `~/skipsoap-study-data/ot_1000_20261013/` (`$OT1K_DATA`).
