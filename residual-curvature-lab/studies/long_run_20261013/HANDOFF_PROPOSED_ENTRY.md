# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-13 — Top-free PARSEC-L vs SOAP over 2,000 steps (`studies/long_run_20261013`).** CIFAR-10 depth-6 ViT, batch
2048, warm-up 100 and WSD decay over the last 400 steps. Each arm was retuned on seed 0 and evaluated on seeds 1–5.
- **The lr optimum moved:** every arm selected lr 9e-2, still the grid's top edge after the one allowed extension.
- **Top-free's 500-step advantage does not survive** (R1 mixed): EMA 0.462 vs SOAP 0.457, lower in 1 of 5 seeds.
  Raw losses are tied (0.488 vs 0.489). Accuracy is higher in all 5 (0.847 vs 0.844).
- **Top-free + N9:** mixed against both (EMA 0.460).
- **Deviations:** D1 (session end; resumed bitwise-equivalently) and D2 (owner stop after all units had
  completed). The queued `ggn_c_control_20261013` was cancelled before running.
- **Owner's decision:** not enough to build further on this PARSEC idea.

Report: `studies/long_run_20261013/RESULTS.md`.
