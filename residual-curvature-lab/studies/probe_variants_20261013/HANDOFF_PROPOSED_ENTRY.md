# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-13 — Top-free PARSEC-L probe variants (`studies/probe_variants_20261013`).** CIFAR-10 depth-6 ViT, 500 steps,
seeds 1–3, lr 3e-2.
- **OT-metric probes** (the lab's graph-Wasserstein pullback, sketched by probes; c tuned on seed 0, 4096 selected)
  beat GGN-probe top-free (c = 64) in every seed at raw steps 50 and 200 and at EMA 500 (0.521 vs 0.541; P1).
  - **Confound:** GGN top-free's c was never tuned on CIFAR. The control `ggn_c_control_20261013` is queued.
  - **Scale:** OT's curvature is about 9× smaller than the GGN's, so c = 4096 brakes about 7× harder than GGN at
    c = 64.
- **Probes from an independent batch:** mixed at every point (EMA 0.537 vs 0.541; P2 not shown).
- **G1:** bitwise. **Deviations:** none. Report: `studies/probe_variants_20261013/RESULTS.md`.
