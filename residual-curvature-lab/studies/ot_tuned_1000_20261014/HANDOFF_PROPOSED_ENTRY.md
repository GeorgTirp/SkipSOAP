# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-14 — Tuned OT-probe top-free PARSEC-L vs tuned SOAP, 1,000 steps (`studies/ot_tuned_1000_20261014`).**
CIFAR-10 depth-6 ViT, batch 2048. Both arms tuned on seed 0, evaluated on seeds 1–3.
- **Tuned vs tuned, every seed:** OT top-free (lr 9e-2, c = 4096) beats SOAP (lr 3e-2) at EMA 1000 (0.460 vs 0.483,
  T1) and raw 1000 (0.484 vs 0.608, T2). Accuracy 0.841 vs 0.828.
- **The gain rests on the lr:** on seed 0 at matched lr, SOAP is slightly better at 3e-2 and OT is far better at 9e-2.
  OT top-free uses a 3× larger lr productively.
- **Open:** the 2,000-step behaviour (SOAP's optimum moves up there) and the metric-vs-c question (N23).
- **Deviations:** none. Suites pass; manifest unchanged. Report: `studies/ot_tuned_1000_20261014/RESULTS.md`.
