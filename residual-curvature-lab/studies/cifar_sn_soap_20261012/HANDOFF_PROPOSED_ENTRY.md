# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-12 — SOAP with and without SN on CIFAR-10 (`studies/cifar_sn_soap_20261012`).** Setup: the CIFAR depth-6
ViT (0.31M parameters), batch 2048, 500 steps, GPU float32. Arms: SOAP at lr 1e-2 and SOAP + SN (c = 1, branch
weights; tuned lr 1e-2). Seeds 1–3, with distance preservation, quadratic-model and added nonlinearity measures.
- **G1:** the SOAP units reproduce `cifar_ggn`'s bitwise.
- **Loss:** SN is worse at raw steps 50 and 200 and at EMA step 500 in every seed (0.580 vs 0.554). It also costs 2×
  per step.
- **SN hypotheses:** none of the five passes its pre-declared rule.
  - Distance preservation: improved after block 6 and at step 500, but mixed at the head input at step 100.
  - Full-GGN model accuracy: not improved.
  - Cross-tensor coupling: *higher*, 17–19 vs 13–15.
  - Network nonlinearity at matched logit step: about equal.
  - Ten-step curvature persistence: about equal.
- **Descriptive:**
  - SN slows representation drift and deformation at every depth, and keeps the blocks near identity.
  - Along the current step, the curvature ten steps earlier was only 32–62% of the current value in both arms.
  - Prediction 3 (linearization error vs distortion as predictors of quadratic-model error): weakly supported in 3
    of 6 units, not shown.
- **Checks:** the suites pass before and after (106, 22, 11, 12).
- **Deviations:** D1 (bytecode from a pre-run test call, removed) and D2 (a manifest difference from recording
  `cifar_ggn`'s D2).
- **Runs:** `~/skipsoap-study-data/cifar_sn_soap_20261012/runs/` (`20261012T171855-1adb9b1e`,
  `20261012T173632-18f98b9d`). Report: `studies/cifar_sn_soap_20261012/RESULTS.md`.
