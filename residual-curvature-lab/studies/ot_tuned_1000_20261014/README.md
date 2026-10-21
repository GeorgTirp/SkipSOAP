# ot_tuned_1000_20261014

Tuned comparison on CIFAR-10 over 1,000 steps (warm-up 50, decay over steps 801–1000):
- top-free PARSEC-L with OT-metric probes, tuned on seed 0 over lr {3e-2, 9e-2} × c {1024, 4096};
- SOAP, tuned over lr {3e-2, 9e-2, 0.27}.

Each arm gets one extension past a selected lr edge. Both are evaluated on seeds 1–3.

- [PROTOCOL.md](PROTOCOL.md): arms, tuning and extension rule, readings T1–T3.
- [RESULTS.md](RESULTS.md): tuned OT top-free beats tuned SOAP in every seed (EMA 0.460 vs 0.483); it runs at
  3× SOAP's selected lr.
- `scripts/`:
  - `train.py`: derived from `ot_1000_20261013`'s;
  - `probes_ot.py`, `n9.py`: copied unchanged;
  - `orchestrate.sh`: manifests, suites, all phases, summary; `summarize.py`.
- External data: `~/skipsoap-study-data/ot_tuned_1000_20261014/` (`$OTT_DATA`).
