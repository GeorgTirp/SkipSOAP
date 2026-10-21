# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-12 — CIFAR-10 scale-up (`studies/cifar_ggn_20261012`), closed early.** CIFAR-10 (45k/5k split of the
training set; the test split is never used), depth-6 ViT (width 64, 0.31M parameters), batch 2048, 500 steps, GPU
float32. Every arm uses crop + flip, warm-up + WSD, weight decay 0.05 and weight EMA.
- Top-free PARSEC-L (c = 64, lr 3e-2) + EMA beats tuned SOAP (lr 1e-2) + EMA in each of seeds 1–3: 0.541 vs 0.554
  EMA validation loss at step 500 (R2 *better*). Tuning seed 0 tied, and top-free's raw weights are worse
  mid-training.
- The exact per-tensor GGN ± SN arms (γ = 1e-2) stopped at the step-100 action audit of `ggn` seed 1 (`head.weight`
  3.6e-4 vs 1e-4).
  - Float64 diagnosis: the solved whitened block is accurate (7e-7). The audit's float32 reference through original
    coordinates is not (2.7e-4).
  - The owner kept the gate and stopped the GGN arms (DEVIATIONS.md D1). R1 and R3–R5 are not available.
- Descriptive: along SOAP and top-free steps the next-batch full-GGN and Hessian quadratic models are accurate (median
  ratio 0.97–1.06 after step 100). The per-tensor model is not (cross-tensor coupling 15–21×). GGN seed 1 was well
  ahead of SOAP per step to step 75, at 110× SOAP's cost per step.
- Pre- and post-study suites pass (106, 22, 11). The isolation manifest is unchanged apart from `ideas/parsec.md`.
- Runs `~/skipsoap-study-data/cifar_ggn_20261012/runs/` (`20261012T103803-7d7d6274`, `20261012T110142-5e9e0779`,
  `20261012T110529-d747f9fc`); report `studies/cifar_ggn_20261012/RESULTS.md`.
