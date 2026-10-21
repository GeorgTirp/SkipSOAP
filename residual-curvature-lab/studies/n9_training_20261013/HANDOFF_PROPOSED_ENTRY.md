# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-13 — N9 in training (`studies/n9_training_20261013`).** CIFAR-10 depth-6 ViT, batch 2048, 500 steps,
seeds 1–3. SOAP and top-free PARSEC-L with SOAP's bases rebuilt every 10 steps from the probe factors (N9), with
momentum and second moments carried into the new basis, against the baselines and a schedule control (exact
eigenvectors of SOAP's own factors).
- **G1:** the baselines reproduce `cifar_ggn` bitwise.
- **G2:** passes at all 450 rebuilds (momentum to 3e-7, second-moment sum to 1e-6).
- **N9 does not help SOAP** (N1 not shown): it is worse raw at steps 25, 50 and 200 in every seed, at its selected
  lr 3e-2, and mixed at EMA 500 (0.563 vs 0.554). The own-factor control behaves the same.
- **Under top-free, N9 helps modestly** (N3): better raw at steps 25 and 200 in every seed; EMA 500 lower in 2 of 3
  (0.531 vs 0.541); accuracy higher in all 3. It is also lower than SOAP + EMA in every seed (descriptive).
- **No deviations.** Report: `studies/n9_training_20261013/RESULTS.md`.
