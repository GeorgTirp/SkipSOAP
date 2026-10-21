# Deviations: cifar_sn_soap_20261012

## D1. Bytecode written by a pre-run test call without PYTHONDONTWRITEBYTECODE (2026-10-12, before any run)

- **What happened:** at 16:54Z, while preparing this study, a pytest call (study tests, for `logs/study_tests_before.log`)
  ran without `PYTHONDONTWRITEBYTECODE=1`. It wrote `__pycache__` folders into:
  - `scripts/` and `src/residual_curvature/` (`curvature`, `experiments`);
  - `studies/gpu_port_20261012/gpu_parsec/`;
  - this study's `scripts/` and `tests/`.

  An earlier call during `cifar_ggn_20261012`'s D1 diagnosis did the same (that study's DEVIATIONS.md, D2).
- **Removed:** every `__pycache__` folder outside `.venv` was created today by this session's calls, and all were
  removed. The study tests were rerun with the variable set (5 passed), and that run is the one recorded in
  `logs/study_tests_before.log`.
- **Effect:** none. Caches are derived from the sources, no source changed, the isolation manifests exclude
  `__pycache__`, and no training had run.

## D2. Isolation manifest: one file outside this study changed (2026-10-12)

- **What changed:** the after-manifest (1,415 files) differs from the before-manifest in exactly one file,
  `studies/cifar_ggn_20261012/DEVIATIONS.md`.
- **Why:** after this study's before-manifest was taken, its D2 entry was appended. That entry records the
  bytecode slip of D1 above, which also occurred during that study's diagnosis.
- **Judgment:** the write recorded a deviation of that study in that study's own file, which is where the rules put
  it. No code, data or results changed. `ideas/parsec.md` changes after the after-manifest, as rule 5 requires.
