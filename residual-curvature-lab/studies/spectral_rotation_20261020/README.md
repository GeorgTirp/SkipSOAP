# spectral_rotation_20261020

Offline study: along `cifar_ggn_20261012`'s SOAP and top-free runs (seeds 1–3, 500 steps, replayed bitwise), is
the per-tensor GGN spectrum stable in the second half of training while its eigenvectors rotate? Read `PROTOCOL.md`
first (SHA-256 in `logs/protocol_sha256.txt`).

- [RESULTS.md](RESULTS.md): both optimizers give *stable spectrum, rotating eigenvectors* late in training; early
  training changes both; the power law straightens to α ≈ 1.
- [HANDOFF_PROPOSED_ENTRY.md](HANDOFF_PROPOSED_ENTRY.md): proposed ledger entry.

## Layout

- `scripts/run.py`: replays each unit from step 0, checks gate G1 against the saved checkpoints at steps 50, 100,
  250 and 500, and measures the raw per-tensor GGN at the declared states on a fixed batch A. At the anchors it also
  measures a second batch B. Outputs go to `$SR_DATA/spectra/` (default `~/skipsoap-study-data/spectral_rotation_20261020/`),
  mirrored to `results/spectra/`.
- `scripts/rotation.py`: per-state spectra (top 200 eigenpairs, trace, fits) and anchor-pair capture and overlap.
- `scripts/summarize.py`: readings M1–M3, V, E, T, B and S into `results/summary.json` and `results/summary.md`.
- `scripts/orchestrate.sh`: manifests and suites before and after, the run and the summary.
- `scripts/common.py`, `states.py`, `spectral.py`, `isolation_manifest.sh`: copied from `bulk_krylov_20261014` (paths
  changed in `common.py`; the others carry a header note).
- `tests/`: run by explicit path, `pytest -p no:cacheprovider studies/spectral_rotation_20261020/tests`.

## Implementation choices the protocol leaves open (fixed before any run)

- **Eigen generator:** seeded per state (4242 + step); batch B's pass at an anchor continues the same generator.
- **Captures:** computed for every anchor against every later state on batch A (the declared pairs are a subset).
- **Shape distance and medians:** computed in `summarize.py` from the stored top-200 eigenvalues (7 significant
  digits) and traces.
