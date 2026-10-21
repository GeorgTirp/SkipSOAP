# ggn_ot_fairness_20261014

GGN-probe top-free PARSEC-L tuned over lr {3e-2, 9e-2} × c {64, 256, 1024} at `ot_tuned_1000_20261014`'s 1,000-step
schedule, evaluated on seeds 1–3. It is compared with that study's tuned OT top-free and tuned SOAP; one OT unit is
rerun bitwise (G1).

- [PROTOCOL.md](PROTOCOL.md): readings F1 (does the OT metric matter?) and F2 (tuned GGN top-free vs SOAP).
- [RESULTS.md](RESULTS.md): F1 *better* (OT by 0.006 in every seed), F2 *mixed*; G1 passed.
- [HANDOFF_PROPOSED_ENTRY.md](HANDOFF_PROPOSED_ENTRY.md): proposed ledger entry.
- `scripts/`: `train.py`, `g1.py`, `summarize.py`, `orchestrate.sh`, `run_queue.sh`, `common.py`; `probes_ot.py` and
  `n9.py` copied unchanged.
- External data: `~/skipsoap-study-data/ggn_ot_fairness_20261014/` (`$GOF_DATA`).
