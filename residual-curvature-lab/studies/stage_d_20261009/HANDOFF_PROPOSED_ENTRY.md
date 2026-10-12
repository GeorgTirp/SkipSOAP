# Proposed HANDOFF entry (stage D, PARSEC-L training test, 10–11 October 2026)

Draft for a human to merge into `HANDOFF.md`; nothing was written to `HANDOFF.md` or `STATUS.md`.
Study `studies/stage_d_20261009/`; external runs `~/skipsoap-study-data/stage_d_20261009/runs/`
(`20261010T171813-415c766d`, `20261010T171814-46f283ad`, `20261010T180135-101694d8`).

## Ledger row

| Question / experiment | Finding | Artifacts |
| --- | --- | --- |
| PARSEC-L in training (main ViT, batch 512, seeds 1–3, 500 steps, settings chosen on seed 0) against tuned SOAP and earlier optimizers. Arms: fixed-step brake (lr or damping calibration × lr), Early-Newton (step chosen on a 128-example selection batch, optional model-free hand-over), exact GGN with step chosen on the selection batch | **Brake** (lr calibration, c = 4, lr 1e-2) beats tuned SOAP at step 500 in all seeds (0.149 ± 0.006 vs 0.161 ± 0.010), with fewer steps to targets (T50 117 vs 142). It has the lowest step-500 loss of all optimizers on these seeds, but is slower than SOAP in wall-clock time on this CPU workload. **Early-Newton**: mixed at steps 50/100, unstable in one seed, worse at 500. **GGN with a selected step**: better than same-batch GGN at step 200, mixed at 500. One SOAP reproduction unit failed its 1e-6 gate from rounding growth (2.2e-6). | the three runs above; `studies/stage_d_20261009/RESULTS.md` |

## Candidate insights to preserve

0. **Qualified on 11 October:** the brake's win holds only against constant-lr baselines. SOAP with a final
   learning-rate decay or weight averaging matches or beats it (`studies/noise_controls_20261011`).

1. **A learning-rate-calibrated shrink of SOAP's sharpest whitened directions (κ = 4ηλ) improves SOAP
   throughout training, but not via edge of stability.** SOAP's full preconditioned sharpness is 10–27,
   against the EMA-momentum limit of 78. According to `studies/brake_mechanism_20261011`:
   - those directions carry 57–70% of the whitened gradient noise in 11% of the dimensions;
   - under β₁ = 0.95 they are underdamped, so their speed does not depend on the learning rate;
   - the brake removes noise there without slowing them, and lowers the training loss too.
   An earlier draft said "edge of stability" here; that was an error.
2. **The brake stabilizes but does not move the optimal learning rate.** On seed 0, every brake beats SOAP at
   lr 3e-2, but the best configuration is at SOAP's own optimum.
3. **Large out-of-batch step lengths (Early-Newton) give early speed in some seeds but are unstable.**
   Same-batch fitting is not the only obstacle to GGN-like early gains.
