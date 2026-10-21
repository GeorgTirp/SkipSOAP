# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-14 — GGN vs OT probes at matched tuning, top-free PARSEC-L, 1,000 steps (`studies/ggn_ot_fairness_20261014`).**
CIFAR-10 depth-6 ViT, batch 2048. GGN-probe top-free tuned on seed 0 over lr {3e-2, 9e-2} × c {64, 256, 1024} (plus
lr 0.27), evaluated on seeds 1–3 against `ot_tuned_1000`'s tuned OT top-free and tuned SOAP (G1 rerun bitwise).
- **Selection:** lr 9e-2, c = 256, EMA 0.98 (interior). This matches OT's c = 4096 in brake strength.
- **F1, OT vs tuned GGN:** better in every seed, so by the declared reading *the OT metric matters*. The margin is
  small: EMA 1000 0.460 vs 0.466, per seed −0.007, −0.008, −0.002.
- **F2, tuned GGN vs SOAP:** mixed (0.466 vs 0.483; −0.026, +0.001, −0.025).
- **Reading:** tuning c and lr gives 74% of OT top-free's gain over SOAP. Both top-free arms cost about 1.65× SOAP's
  wall time per step (`ot_tuned_1000`'s 1.3× was the median step, without the probe steps).
- **Deviations:** none. Suites pass. The manifest changed only in the concurrently prepared
  `studies/bulk_krylov_20261014/`. Report: `studies/ggn_ot_fairness_20261014/RESULTS.md`.
