# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-20 — Stable spectrum or rotating eigenvectors? Per-tensor GGN spectra along CIFAR training (`studies/spectral_rotation_20261020`).**
Offline. `cifar_ggn`'s SOAP and top-free 500-step runs (seeds 1–3) were replayed bitwise (G1: 24 of 24 checkpoints).
The raw per-tensor GGN (top 200 eigenpairs) was measured on a fixed batch at 14 states.
- **Verdict, both optimizers: stable spectrum, rotating eigenvectors** (M1, M2 hold).
  - Late (steps 250–500), the spectrum's shape changes by only 0.03–0.08 in log units over 10–50 steps.
  - A 10-step-old top-16 subspace still captures 0.84–0.88 of the current top-16 curvature; a 50-step-old one
    captures 0.70–0.81.
  - Early training (steps 10–100) changes both.
- **The power law straightens to α ≈ 1** (R² 0.99). This is close to the reported language-model value; that paper
  was not verified here.
- **Rotation dominates batch noise for the top subspace** (noise/drift ratio 0.02–0.6). Pooling batches should buy
  little; refreshing directions matters more. Top-free rotates faster than SOAP (its lr is 3× larger).
- **Deviations:** none. Suites pass. The manifest changed only in the concurrently prepared next study. Report:
  `studies/spectral_rotation_20261020/RESULTS.md`.
