# ladder_mechanism_20261013

Offline, at replayed CIFAR states:
- **Part L (the ladder):** does PARSEC-L's held information contain the early per-tensor GGN gain? The gap
  between top-free's step and the exact step is split into coefficients, staleness, estimation and missing
  directions.
- **Part N (the N21 control):** is the ten-step curvature staleness along SOAP's step a selection effect?

- [PROTOCOL.md](PROTOCOL.md): rungs, gap decomposition, readings L1–L5 and N1–N3, gates S1 and L0.
- [RESULTS.md](RESULTS.md): 56–80% of the gap to the exact per-tensor GGN step lies outside a rank-16 top space
  (L4, 9 of 9), so the goal of closing the GGN gap with PARSEC-L is formally dropped. N21 is mostly selection.
- `scripts/`:
  - `states.py`: replays of `cifar_ggn`'s units and snapshots;
  - `algebra.py`: Galerkin steps, subspace iteration, the held-brake reconstruction, directional GGN;
  - `ladder.py`, `control.py`: Parts L and N;
  - `run.py`: the driver; `summarize.py`.
- `tests/`: Galerkin optimality, subspace iteration, the brake reconstruction against PARSEC-L's logs, whitening
  round trips, snapshot round trips.
- External data: `~/skipsoap-study-data/ladder_mechanism_20261013/` (`$LM_DATA`).
