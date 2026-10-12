# transfer_law_20261011

Quick offline test of the transfer law for batch-fitted steps, share = (1 − r)/(1 + r) with the finite-corpus
correction r = (1 − f)·B_noise/B. It also tests whether step sizes computed from statistics alone (γ̂/λ̂, Wiener
shrinkage) can replace second-batch selection. Six banked states; no training.

- [PROTOCOL.md](PROTOCOL.md): theory, design and readings A1–A3, B1–B2.
- [DEVIATIONS.md](DEVIATIONS.md): Wiener gate failure (B2 not evaluated), boundary rule for reported-only
  searches, and the `soap_map` Ô scale bug (fixed and rerun).
- [RESULTS.md](RESULTS.md): verdicts, tables, post-hoc curvature decomposition, consequences for PARSEC.
- `scripts/`:
  - `corpus.py STATE` (Part A, one state per process);
  - `steps.py` (Part B);
  - `summarize.py`;
  - `run.sh` (four-worker driver);
  - `isolation_manifest.sh`.
- `results/`: per-state Part A JSON, Part B JSON (final and pre-fix), `summary.json`, `summary_stdout.txt`.
