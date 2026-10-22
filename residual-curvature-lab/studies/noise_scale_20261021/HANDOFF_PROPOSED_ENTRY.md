# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-21 — Noise scales, break time and fractional Newton (`studies/noise_scale_20261021`).**
Offline. `cifar_ggn`'s SOAP and top-free runs (seeds 1–3) were replayed bitwise and measured at steps 10, 25, 50,
100, 200 and 400 (36 states). The steps −A^{−ν}z are in SOAP-whitened coordinates, ν ∈ {0, ¼, ½, ¾, 1}, with
A = K + δI the damped per-tensor GGN. Noise forms come from 43 batch gradients; matrix functions use batched
Lanczos.
- **Noise scales grow about tenfold during training and grow with the power.** Newton's K₁ goes from
  1,340–2,170 to 17,300–20,400 examples (N2 holds). The declared N1 (K₁ > K½ > K₀ in 36 of 36) fails in two
  step-10 states, where K½ is below K₀ by 7–9%.
- **The break comes early (N3 holds):** at B = 2,048 the full Newton power stops paying from step 25 (five
  trajectories) or step 50 (one). The required batch size levels off near 10⁴.
- **The best power is interior and falls (F1 35 of 36, F2 holds):** about ¾ early and ½ late. ¾ beats 1 in 35 of
  36 states; ½ beats 0 in 35 of 36. Late, ν = ½ gives 1.3–2.0× the one-step evaluation decrease of SOAP's
  direction.
- **The theory (batch_noise.md T1–T2) is optimistic about strong preconditioning.** P1 holds (32 of 36 within one
  grid step), but the predicted optimum is about ¼ too high, and the predicted gains over ν = 0 are too large:
  1.5–3.8× too large at ν = ½ and 7–9× at ν = 1 early (ratios of the medians).
- **Identity:** the family is the geodesic between SOAP's metric and the damped GGN, so ν = ½ is their matrix
  geometric mean.
- **Deviation:** D1, before launch (the smoke ran twice; Lanczos storage layout changed). G1 18 of 18, G2 worst
  1.2e-4 (tolerance 1e-2), suites pass, manifest unchanged. Report: `studies/noise_scale_20261021/RESULTS.md`.
