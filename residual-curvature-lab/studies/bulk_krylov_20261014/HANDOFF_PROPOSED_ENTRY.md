# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-14 — Power-law spectra and cheap bulk preconditioning, offline (`studies/bulk_krylov_20261014`).**
`ladder_mechanism`'s 12 top-free states on CIFAR (seeds 1–3, steps 10/25/50/100), per-tensor whitened GGN blocks.
- **Spectrum:** roughly a power law, but broken early. The top 16 fall as about k^−1, ranks 17–200 about twice as
  steeply. It straightens by step 100. A single law fitted from the probes' top-16 and trace underestimates λ₁₇ by
  1.6–2.4× (P1, P2 not supported), although the probes measure λ₁ and the trace accurately.
- **Spectrum-only preconditioning (owner's idea):** a power-law diagonal closes a median 6% of the gap to the exact
  per-tensor GGN step, the same as the exact (Jacobi) diagonal (K1 supported).
- **Krylov:** the top-16 probe directions plus 8 deflated products close 67–76% of the gap (K2: m = 8; m = 4 gives a
  median of 51%). On held-out loss these steps match or beat the exact per-tensor step.
- **Chebyshev** with the probe-predicted interval fails (K3). **5-step full-GGN CG on 256 examples** matches the
  exact step on held-out loss at median, but beats top-free in only 7 of 9 states (K4). It costs about 0.07 s per
  product (SOAP step 0.22 s).
- **Deviation:** D1, a fit-helper crash before any output; fixed and rerun. Suites pass; manifest unchanged.
  Report: `studies/bulk_krylov_20261014/RESULTS.md`.
