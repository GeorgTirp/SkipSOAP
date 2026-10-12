# sn_ggn_20261008

Isolated side study: per-tensor exact GGN and native SOAP training with and without SNGP spectral
normalization (c = 0.95) on the main ViT, seeds 1–3, 250 steps.

- [PROTOCOL.md](PROTOCOL.md): design, gates and pre-declared readings.
- [RESULTS.md](RESULTS.md): findings, tables, audits and reproduction commands.
- [DEVIATIONS.md](DEVIATIONS.md): departures from the protocol (none).
- [HANDOFF_PROPOSED_ENTRY.md](HANDOFF_PROPOSED_ENTRY.md): draft ledger row for `HANDOFF.md`.
- `scripts/`: `train.py` (runner), `summarize.py` (readings and figure), `arms.py` (gap-audit copy,
  unchanged), `sn.py` (SN-study copy, unchanged), `common.py` (paths), queue and manifest scripts.
- `tests/`: study tests, run by explicit path only.
- `runs/`: light mirror (json/md/png/yaml) of the external run; `figures/sn_ggn.png`; `logs/`.
