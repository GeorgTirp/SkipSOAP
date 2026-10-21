# Tuned OT-probe top-free PARSEC-L vs tuned SOAP over 1,000 steps on CIFAR-10 — protocol

Study `ot_tuned_1000_20261014`. Written on 14 October 2026, before any run.

**Request.** The owner asked on 14 October: "okay now lets start a tuned OPT probe run here". They chose "1,000 steps, all
tuned": OT tuned over lr {3e-2, 9e-2} × c {1024, 4096} and SOAP over lr {3e-2, 9e-2, 0.27} on seed 0, then both on
seeds 1–3. This replaces the untuned exploratory check `ot_1000_20261013`, closed without results.

**Context.**
- **At 500 steps** (`probe_variants`): OT-probe top-free (c = 4096 tuned, lr 3e-2) beat GGN top-free and SOAP in every
  seed (EMA 0.521 vs 0.541 vs 0.554). But only OT's c was tuned on CIFAR.
- **At 2,000 steps** (`long_run`): plain top-free's edge over SOAP was gone, and every arm's lr optimum had moved up
  to ≥ 9e-2.
- **This study:** both arms tuned at an intermediate length.

## 1. Question

At 1,000 steps, with lr (and c) tuned for each arm on the same seed, does top-free PARSEC-L with OT-metric probes beat
SOAP?

## 2. Workload

CIFAR-10 (45k/5k split of the training set; the test split is never accessed; crop + flip), the M1d6 ViT, batch
2048, GPU float32 with deterministic algorithms:
- **Length:** 1,000 steps.
- **Schedule:** linear warm-up over steps 1–50, constant, linear decay over steps 801–1000 to 1%. These are the
  proportions of the 500- and 2,000-step runs.
- **Regularization:** decoupled weight decay 0.05; EMAs d ∈ {0.98, 0.995, 0.998}.
- **Validation:** at steps 0, 10, 25, 50, 100, 200, …, 1000.
- **Runner:** `ot_1000_20261013`'s. Its `soap` path is `cifar_ggn`'s, and `topfree_ot` is `probe_variants`' arm with
  the arm's c; both are checked over CPU steps. Tuning phases are added.

## 3. Arms and tuning

| Arm | Settings | Seed-0 grid |
| --- | --- | --- |
| `soap` | SOAP (betas 0.95/0.95, refresh every 10 steps) | lr ∈ {3e-2, 9e-2, 0.27} |
| `topfree_ot` | top-free PARSEC-L with OT-metric probes (rank 16, `top_rhs='gradient'`, lr calibration, probes at steps 2, 12, …) | lr ∈ {3e-2, 9e-2} × c ∈ {1024, 4096} |

- **Selection:** per arm, the setting and EMA decay with the lowest step-1000 EMA validation loss. A diverged unit
  counts as worse than any finite one.
- **Extension:** if the selected lr lies at an edge of the arm's lr grid, one extension unit at ×3 or ÷3 of that edge
  is run, at the arm's selected c, and the selection is repeated.
  - For `topfree_ot` every lr is an edge, so its selection always takes one extension (0.27 or 1e-2).
  - c is not extended.
- **Evaluation:** seeds 1–3, both arms at their selections (6 units).

## 4. Gates

- **G0:** suites before and after:
  - all earlier studies' tests;
  - this study's tests: the schedule; the unit plan and selection rule; `soap` equal to `cifar_ggn`'s and
    `topfree_ot` equal to `probe_variants`' over CPU steps under the same schedule factor.
- **Divergence** is reported, not a gate failure.

## 5. Pre-declared readings (seeds 1–3)

*Better* or *worse* means lower or higher in all three seeds; otherwise *mixed*.

| Code | Reading |
| --- | --- |
| T1 | **Main:** `topfree_ot` + EMA vs `soap` + EMA at step 1000, each with its selected decay. *OT top-free beats SOAP* only if better. |
| T2 | Raw weights at step 1000: same comparison. |
| T3 | Descriptive: the raw curves (steps 100, 200, 500, 800, 1000); accuracy; seconds per step; the tuning table; the comparison with the 500- and 2,000-step results. |

## 6. Cost, isolation and limitations

- **Cost:** SOAP about 4 min per unit, `topfree_ot` about 6.5 min. Tuning 7 units plus 1–2 extensions (about
  45 min); evaluation 6 units (about 32 min). **About 1.3 GPU-hours.**
- **Running:** detached (`setsid nohup`), so it survives the session.
- **Writes:** only to `studies/ot_tuned_1000_20261014/` and `~/skipsoap-study-data/ot_tuned_1000_20261014/`.
  Hygiene as before.
- **Limitations:**
  - Three seeds; one tuning seed.
  - A two-point c grid.
  - No GGN-probe top-free arm, so the metric-vs-c question of `probe_variants` (N23) remains open.
