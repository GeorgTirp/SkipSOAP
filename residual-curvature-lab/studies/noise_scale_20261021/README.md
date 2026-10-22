# noise_scale_20261021

Offline test of predictions 1 and 2 of [ideas/batch_noise.md](../../ideas/batch_noise.md) along `cifar_ggn_20261012`'s
SOAP and top-free runs (seeds 1–3, steps 10–400, replayed bitwise).
- **Break time:** do the noise scales of steps −A^{−ν}z grow with the power ν and with training, so that stronger
  preconditioning stops paying at B = 2,048?
- **Fractional Newton:** is the best power interior, and does it fall during training?

Read `PROTOCOL.md` first (SHA-256 in `logs/protocol_sha256.txt`).

- [RESULTS.md](RESULTS.md): noise scales grow about tenfold during training and with the power. Full Newton stops
  paying at B = 2,048 from step 25–50. The best power is interior (¾ early, ½ late). The theory places it about ¼
  too high and overstates the gains.
- [DEVIATIONS.md](DEVIATIONS.md): D1, the pre-launch smoke ran twice and the Lanczos storage layout changed for speed.
- [HANDOFF_PROPOSED_ENTRY.md](HANDOFF_PROPOSED_ENTRY.md): proposed ledger entry.

## Layout

- `scripts/run.py`: replays each unit and checks gate G1 at steps 50, 100 and 250. At the declared states it runs
  `noise.analyse` (gate G2 at steps 10 and 400) and `noise.score_steps`. Outputs go to `$NS_DATA/noise/` (default
  `~/skipsoap-study-data/noise_scale_20261021/`), mirrored to `results/noise/`.
- `scripts/noise.py`: per-state measurements; non-consuming peeks of the step, selection and evaluation batches.
- `scripts/lanczos.py`: batched Lanczos with full reorthogonalization; Gauss-quadrature forms and matrix functions.
- `scripts/summarize.py`: readings N1–N3, F1, F2, P1 and tables into `results/summary.json` and `summary.md`.
- `scripts/describe.py` (written after the run): the declared descriptive tables D into `results/descriptive.json`
  and `descriptive.md`. It checks its pooled theory values against `summary.json` exactly.
- `scripts/orchestrate.sh`: manifests and suites before and after, the run and the summary.
- `scripts/common.py`, `states.py`, `isolation_manifest.sh`: copies (paths changed in `common.py`).
- `tests/`: run by explicit path.

## Implementation choices the protocol leaves open (fixed before the run)

- **Lanczos columns:** per tensor, [corpus gradient, step-batch gradient, 43 deviations]. Breakdown tolerance
  100·eps (relative); Ritz values are clamped at the smallest positive float only for the power evaluation.
- **Partition and reference batches:** partition positions come from `DataOrder(seed 20261021)`, the first 43
  batches of 1,024. The reference curvature uses `spectral_rotation`'s batches A and B (`DataOrder(seed 20261020)`,
  the first two batches of 2,048).
- **Scale terms:** α₀ uses the step-batch gradient and the damped block A: −Σ z_Bᵀy/Σ yᵀAy.
