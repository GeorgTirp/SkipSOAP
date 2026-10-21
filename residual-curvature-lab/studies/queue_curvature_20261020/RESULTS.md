# Curvature along the data queue: visit phase, pooled sources and steps — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, fixed before any run. One deviation:
[DEVIATIONS.md](DEVIATIONS.md) D1. The first attempt ran out of CUDA memory in the capture normalization (a
full-size float64 copy) before any output. The normalization is now accumulated in blocks, and the expandable-segments
allocator is enabled; the study was rerun from the start.

Outputs: `results/queue/topfree_s{1,2,3}.json` (copies of `~/skipsoap-study-data/queue_curvature_20261020/queue/`;
the per-example arrays stay external) and `results/summary.json` / `summary.md`, from `scripts/summarize.py`.

**Status (2026-10-20):** all 12 cases completed (3 seeds × 4 states), at about 225 s per state.
- **G1:** passed bitwise at steps 50, 100 and 250 for every seed.
- **Suites:** passed before and after (16 suites, including this study's 8 tests).
- **Isolation manifest:** unchanged.

Capture statistics are medians over the 39 matrix tensors of a case.

## 1. Summary

- **The visit phase is visible in losses, not decisively by the declared reading, and not at all in curvature
  estimates.**
  - **V1 is not supported.** At step 50 the effect is absent (one negative case). One placebo difference (0.040)
    exceeds the smallest real one.
  - **From step 100 on, every case shows it:** examples about to be revisited have higher loss than just-trained
    ones, D_loss 0.016–0.067 nats (9 of 9). The seed-averaged age profile (post hoc) rises steadily from about
    −0.03 (ages 2–5) to +0.01–0.05 (ages 26–30) relative to the training mean, while the placebo profile stays
    within ±0.017.
  - **V2 (curvature) is not supported:** the one-draw trace estimate is too noisy (D_trace from −15 to +17, placebo
    from −31 to +16).
- **Which batches feed the curvature estimate does not matter; when they were probed does.**
  - At current weights, the next 8, the last 8 and 8 random batches give the same subspaces: capture differences of
    at most 0.007 (S1 holds, 11 of 12, but the effect is negligible).
  - There is no visit-phase penalty for the just-consumed batches (S2: 9 of 12, differences of about 0.001).
  - Steps from the coming queue do not fit the coming queue's loss better (S5: 3 of 12).
  - As the theory predicted, they do not do better on held-out data either (S6 not supported: 6 of 12).
  - The queue corrections of the theory (finite population, anti-correlation, visit phase) are too small to matter.
    For curvature the batches behave as exchangeable.
- **Staleness and pooling both matter:**
  - **Staleness (S3, 12 of 12):** the same 8 batches probed at their own, older weights capture 0.03–0.09 less of
    the next steps' curvature than when probed at current weights.
  - **Pooling (S4, 12 of 12):** 8 batches beat 1. The gain grows with training: held-out capture +0.014–0.022 at
    step 50, +0.059–0.073 at steps 250–400.
- **Trailing pool vs single fresh batch: a crossover during training.** The realistic trailing pool (the last 8
  batches at their own weights) loses to a single fresh batch early and wins late:

  | Steps | Held-out capture, trailing minus single (k = 64) |
  | --- | --- |
  | 50–100 | −0.011 to −0.057 |
  | 250–400 | +0.006 to +0.038 |

  This matches `spectral_rotation`'s finding that drift dominates early while batch noise grows late.
  - **F1 is not supported.** The crossover comes earlier than predicted (noise/drift ratio R ≈ 0.15–0.3), so the
    declared break-even of 0.3–0.8 undervalued pooling. With R ≤ 0.3 everywhere, 8 of 11 predictions agree; the
    three late disagreements are all at steps 250–400.
- **Probe-row curvature steps work one step ahead.**
  - Eight CG steps on the probe-row Gram (2,048–16,384 rows per tensor) roughly double the held-out loss decrease of
    the plain whitened-gradient step, each at its own best size: median ratio 2.1, in 11–12 of 12 cases for every
    fresh source.
  - The stale trailing pool is weaker (median ratio 1.8).
  - Every selected step size was interior. Ratios range from 1.1 to 10 across cases.
- **Rademacher probes:** slightly better subspaces than sampled-label probes (held-out capture higher in 10 of 12,
  by at most 0.008). Steps are indistinguishable.

## 2. Readings

| Code | Result | Count / values |
| --- | --- | --- |
| V1 visit phase, loss | **not supported** | D_loss −0.002–0.067 (positive in 11 of 12; all 9 at steps ≥ 100); placebo −0.040–0.019 |
| V2 visit phase, trace | **not supported** | D_trace −15.1–16.9; placebo −30.5–16.0 |
| S1 next8 > rand8 on T_queue | **holds** (negligible size) | 11 of 12; Δ −0.0002–0.0072 |
| S2 rand8 > last8 on T_held | **not supported** | 9 of 12; Δ about 0.001 |
| S3 last8 > last8own on T_queue | **holds** | 12 of 12; Δ 0.03–0.09 |
| S4 rand8 > cur on T_held | **holds** | 12 of 12; Δ 0.014–0.073 |
| S5 next8 > rand8, queue-loss decrease | **not supported** | 3 of 12 |
| S6 next8 > rand8, held-out decrease | **not supported** (as predicted) | 6 of 12 |
| F1 floor-vs-drift prediction | **not supported** | 11 predicted ("single"), 8 agree; the 3 disagreements are at steps 250–400 |

## 3. Descriptive (S7, V3)

**Capture fraction, k = 64** (T_queue / T_held; ranges over seeds):

| State | cur | next8 | last8 | last8own | rand8 | rand8_rad |
| ---: | --- | --- | --- | --- | --- | --- |
| 50 | 0.67–0.70 / 0.74–0.78 | 0.68–0.72 / 0.77–0.79 | 0.68–0.72 / 0.77–0.79 | 0.61–0.64 / 0.71–0.74 | 0.68–0.72 / 0.77–0.79 | 0.68–0.72 / 0.77–0.79 |
| 100 | 0.43–0.47 / 0.49–0.52 | 0.47–0.50 / 0.52–0.56 | 0.46–0.50 / 0.52–0.56 | 0.40–0.44 / 0.47–0.51 | 0.47–0.50 / 0.52–0.56 | 0.47–0.50 / 0.53–0.56 |
| 250 | 0.23–0.27 / 0.26–0.29 | 0.29–0.32 / 0.32–0.35 | 0.28–0.32 / 0.32–0.35 | 0.24–0.26 / 0.28–0.30 | 0.28–0.32 / 0.32–0.35 | 0.29–0.32 / 0.33–0.35 |
| 400 | 0.20–0.21 / 0.21–0.23 | 0.26–0.27 / 0.28–0.30 | 0.25–0.27 / 0.28–0.30 | 0.21–0.23 / 0.25–0.26 | 0.26–0.27 / 0.29–0.30 | 0.26–0.27 / 0.29–0.31 |

- **Spread of the curvature:** captures fall over training at fixed k (0.7 → 0.25 for k = 64). Later curvature is
  spread over many more directions, consistent with the falling top-16 trace share in `spectral_rotation`.
- **Steps:** the ratio of `rand8`'s held-out decrease to `identity`'s ranges from 1.1 to 10 across cases (median
  2.1). Examples: about 1.1 at step 50 for seed 1 (0.147 vs 0.139), and 2.5–4 at step 100. The full table is in
  `results/summary.md`.
- **Visit-phase profile:** seed means of the loss deviation from the training mean, by age band:

  | State | 0–1 | 2–5 | 6–10 | 11–15 | 16–20 | 21–25 | 26–30 |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 100 | −0.021 | −0.023 | −0.001 | −0.007 | +0.015 | +0.028 | +0.013 |
  | 250 | −0.020 | −0.037 | −0.021 | +0.013 | +0.015 | +0.028 | +0.036 |
  | 400 | −0.020 | −0.031 | −0.006 | +0.009 | +0.018 | +0.013 | +0.050 |
  | placebo (all states) | | | | | | | within ±0.017 |

  The just-trained bands (0–1) sit slightly above ages 2–5. This is not explained here; training saw augmented
  copies, the measurement uses unaugmented images.

## 4. Interpretation and limitations

- **For the queue theory:**
  - Random reshuffling leaves a measurable footprint in per-example losses: about 0.04–0.08 nats between ages 2–5
    and ages 26–30, from step 100 on.
  - It leaves no footprint in curvature estimates at this batch size. The "two moving things" reduce to one: for
    curvature, the parameters' drift is what matters, and the batches can be treated as exchangeable.
  - The theory deliverable therefore simplifies to a drift-versus-noise filter whose optimal pooling window grows
    during training. Early, use fresh probes only; late, pooling several batches pays even with stale weights.
    Re-probing the pooled batches at current weights (`last8` at θ_t) beats both, at the cost of 8 probe passes.
- **For the Hennig route:** curvature from probe rows is good enough for 8-step CG to double the one-step held-out
  gain over the plain step, without dense blocks. This is the online-cost question `bulk_krylov` left open, answered
  positively one step ahead.
- **Not shown:** any training benefit. The comparison step is the plain whitened gradient (no momentum, no
  top-free brake). The CG steps' natural scale was 30–250× too large (selected multipliers 2^−8 to 2^−5).
- **Limitations:**
  - **Model:** one model and optimizer (top-free), states 50–400 of 500-step runs.
  - **Augmentation:** measurements use unaugmented images, while visits used augmented ones.
  - **Trace noise:** one probe draw per example, which made V2 uninformative.
  - **Step:** the step affects matrix tensors only.
  - **Calibration:** F1's break-even calibration was a rough model, and it was off.
