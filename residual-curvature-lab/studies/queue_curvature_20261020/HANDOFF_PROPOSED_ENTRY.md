# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-20 — Curvature along the data queue (`studies/queue_curvature_20261020`).**
Offline. `cifar_ggn`'s top-free runs (seeds 1–3) were replayed bitwise with queue bookkeeping (random reshuffling).
States 50, 100, 250 and 400 were measured.
- **Visit phase:** per-example loss rises with the time since the last visit, by about 0.04–0.08 nats from step 100
  on (placebo flat). The declared reading V1 fails (absent at step 50; one large placebo value). The curvature
  version (V2) is too noisy to read.
- **For curvature estimates the queue does not matter:** next 8 ≈ last 8 ≈ random 8 batches (differences ≤ 0.007
  capture; S2 and S5 fail, S1 negligible).
- **Staleness and pooling do matter.** Probing at old weights costs 0.03–0.09 capture (S3); pooling 8 batches beats
  1 by 0.014–0.073 (S4).
- **Trailing pool vs fresh batch:** the trailing 8-batch pool loses to a fresh batch early and wins late, crossing
  between steps 100 and 250. The declared floor-vs-drift rule (F1) placed the break-even too high and is not
  supported.
- **Probe-row Krylov works one step ahead:** 8-step CG on probe-row Grams roughly doubles the held-out decrease of
  the plain whitened step (median 2.1×). Rademacher probes are marginally better than sampled labels.
- **Deviation:** D1, an out-of-memory crash before any output; fixed and rerun. Suites pass; manifest unchanged.
  Report: `studies/queue_curvature_20261020/RESULTS.md`.
