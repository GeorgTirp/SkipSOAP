# probe_variants_20261013

Top-free PARSEC-L on CIFAR-10 with two alternative curvature sources:
- OT-metric probes, which sketch the lab's dynamic graph-Wasserstein pullback, to close the OT chapter;
- GGN probes drawn on an independent batch rather than the step's batch.

- [PROTOCOL.md](PROTOCOL.md): arms, the c grid for OT, gates G0–G1, readings P1–P3.
- [RESULTS.md](RESULTS.md): OT probes beat GGN probes in every seed (EMA 0.521 vs 0.541), but c was tuned for OT only (control
  queued); probes from another batch are mixed.
- `scripts/`:
  - `probes_ot.py`: OT probes, which with exact draws reproduce `curvature/wasserstein.py`'s blocks;
  - `train.py`: arms; `topfree` is `cifar_ggn`'s unchanged;
  - `g1.py`, `summarize.py`, `run_queue.sh`, `orchestrate.sh`, `common.py`.
- `tests/`: OT probes against dense blocks, unbiasedness, the top-free arm bitwise, the probe stream, the unit plan.
- External data: `~/skipsoap-study-data/probe_variants_20261013/` (`$PV_DATA`).
