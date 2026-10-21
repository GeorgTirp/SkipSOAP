# basis_refresh_snr_20261013

Offline tests of three PARSEC ideas at states of the CIFAR-10 depth-6 ViT runs:
- **N9:** SOAP bases from the probe GGN (Part A, the basis ladder);
- **I10, I15a and N8:** triggered SOAP refresh (Part B, refresh benefit against cheap signals);
- **I11:** signal-to-noise of top-free PARSEC-L's braked directions (Part C).

- [PROTOCOL.md](PROTOCOL.md): questions, states, candidates, signals, gates (S1, A0) and readings A1–A4, B1–B3,
  C1–C3.
- [RESULTS.md](RESULTS.md):
  - N9 holds on the per-tensor quadratic (9 of 9), and post hoc on held-out loss at steps 10–50.
  - No refresh signal beats the clock.
  - Top-free brakes its cleanest directions.
- [DEVIATIONS.md](DEVIATIONS.md): D1 (Part C loader bug, rerun), D2 (transfer grid never bracketed; post-hoc check).
- `scripts/`:
  - `common.py`: paths, archives and determinism;
  - `states.py`: bitwise replays, snapshots, SOAP contexts, `refresh_now`, top-free loader;
  - `quad.py`: Kronecker bases, exact, probe and damping diagonals, the per-tensor quadratic, and Jacobian
    quadratics;
  - `ladder.py`, `refresh.py`, `snr.py`: Parts A, B and C;
  - `run.py`: the driver; `queue.sh`: the queue;
  - `summarize.py`, `posthoc.py`: the readings and the D2 check.
- `tests/`: 9 tests against float64 references, SOAP's own refresh and PARSEC-L's logged brakes.
- External data: `~/skipsoap-study-data/basis_refresh_snr_20261013/` (`$BRS_DATA`): snapshots and full outputs.
  Copies of the outputs are in `results/`.

Commands, from `residual-curvature-lab/`:

```sh
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -m pytest -p no:cacheprovider -q studies/basis_refresh_snr_20261013/tests
.venv/bin/python studies/basis_refresh_snr_20261013/scripts/run.py replay
sh studies/basis_refresh_snr_20261013/scripts/queue.sh
.venv/bin/python studies/basis_refresh_snr_20261013/scripts/run.py ladder --posthoc
.venv/bin/python studies/basis_refresh_snr_20261013/scripts/summarize.py
.venv/bin/python studies/basis_refresh_snr_20261013/scripts/posthoc.py
```
