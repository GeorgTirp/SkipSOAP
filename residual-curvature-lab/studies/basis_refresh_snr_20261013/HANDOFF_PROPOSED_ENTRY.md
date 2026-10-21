# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-13 — Offline tests of N9, triggered refresh and I11 (`studies/basis_refresh_snr_20261013`).** Run at states
of the CIFAR-10 depth-6 ViT runs. SOAP states are replayed bitwise (gate S1, 9 of 9); top-free states come from the
stored checkpoints.
- **N9, SOAP bases from the probe GGN.**
  - Pre-declared, on the damped per-tensor quadratic: the probe-factor eigenbasis recovers 12–18% of the exact
    per-tensor GGN's advantage over SOAP's map at steps 10–50, against 2–5% for SOAP's own basis (A1, 9 of 9). The
    deployable probe diagonal is as good as the exact one (A2, 9 of 9).
  - Held out: the pre-declared step grid never bracketed (A3 not shown, D2). Post hoc, with an extended grid, the
    deployable step beats SOAP's map in 9 of 9 states at steps 10–50 and recovers 31–73% of the exact GGN's held-out
    advantage at steps 10–25. Not at steps 100–250.
- **Refresh.** SOAP's basis goes stale from the first step on: the refresh benefit grows with age in 9 of 9 windows.
  No signal (factor rotation, probe ρ rise, probe spread) beats the trivial age signal (B2 not shown).
- **I11.** Top-free brakes its *highest*-SNR directions (C1 and C2 reversed, 0 of 9). In the weight matrices they
  carry 29–57% of the gradient's signal energy and 8–32% of its noise in 0.4% of the dimensions, and top-free moves
  along them at about 1/60 of the static SNR optimum. The static noise-filtering explanation of top-free's gain is
  refuted.
- **Deviations:** D1 (a Part C loader bug, rerun) and D2 (post-hoc transfer grid).
- **Suites** pass before and after.
- **Data:** `~/skipsoap-study-data/basis_refresh_snr_20261013/`. Report: `studies/basis_refresh_snr_20261013/RESULTS.md`.
