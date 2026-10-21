# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-13 — Reduced-subspace ladder and N21 control (`studies/ladder_mechanism_20261013`, offline).** Run at
replayed CIFAR top-free and SOAP states; replays bitwise.
- **The gap is mostly missing directions:** 56–80% of the gap between top-free's step and the exact per-tensor GGN
  step lies outside a rank-16 top space plus momentum and gradient (L4, 9 of 9). It is measured on the damped
  per-tensor quadratic at steps 10–50. The other parts are small:
  - better coefficients in the held space: 10–28%;
  - fresher probes: 3–17%;
  - exact instead of probe top directions: ≤ 3%.
- **Pre-declared consequence:** "close the GGN gap with PARSEC-L" is formally dropped.
- **Held-out, step 10:** the reduced solve in the fresh subspace recovers 60–87% of the exact GGN's single-step
  held-out advantage. This fades by step 25–50.
- **N21 control:** the ten-step curvature staleness along the step is mostly selection (11 of 18 points), with
  global sharpening around step 60.
- **Deviations:** none. Report: `studies/ladder_mechanism_20261013/RESULTS.md`.
