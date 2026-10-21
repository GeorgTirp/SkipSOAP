# GGN vs OT probes at matched tuning: top-free PARSEC-L over 1,000 steps — protocol

Study `ggn_ot_fairness_20261014`. Written on 14 October 2026, before any run.

**Request.** The owner asked on 14 October: "Do the GGN OT fairness test and the power-law experiment now."

**Context.** `ot_tuned_1000_20261014` found that tuned OT-probe top-free beats tuned SOAP in every seed at 1,000 steps:
EMA 0.460 vs 0.483, with OT at lr 9e-2 and c = 4096. GGN-probe top-free was never tuned on CIFAR beyond lr at 500
and 2,000 steps with c fixed at 64. N23's open question is whether the OT *metric* matters, or whether any
top-free brake tuned in c and lr does as well.

## 1. Questions

1. **F1:** At 1,000 steps with lr and c tuned the same way, does OT-probe top-free beat GGN-probe top-free?
2. **F2:** Does tuned GGN-probe top-free also beat tuned SOAP?

## 2. Workload

`ot_tuned_1000_20261014`'s exactly:
- **Data:** CIFAR-10, 45k/5k split of the training set (the test split is never accessed); crop + flip.
- **Model:** M1d6; batch 2048; GPU float32, deterministic.
- **Schedule:** 1,000 steps; warm-up over 50; decay over steps 801–1000 to 1%.
- **Regularization:** weight decay 0.05; EMAs {0.98, 0.995, 0.998}.
- **Runner:** `ot_tuned_1000_20261014`'s. Its `topfree` path (GGN probes, `micro.lean_probes`) is `cifar_ggn`'s with
  the arm's c.

## 3. Arms, tuning and references

| Arm | Seed-0 grid | Evaluation |
| --- | --- | --- |
| `topfree` (GGN probes) | lr ∈ {3e-2, 9e-2} × c ∈ {64, 256, 1024} | seeds 1–3 at its selection |
| `topfree_ot` (OT probes) | none; `ot_tuned_1000`'s selection lr 9e-2, c = 4096, EMA 0.98 | **reference:** `ot_tuned_1000`'s units, seeds 1–3; seed 1 is rerun here for G1 |
| `soap` | none; `ot_tuned_1000`'s selection lr 3e-2, EMA 0.995 | **reference:** `ot_tuned_1000`'s units, seeds 1–3 |

- **Why this c grid:** GGN curvature is about 9× OT's (`probe_variants`: median η·λ_max 9.7 vs 1.1). GGN c ∈
  {64, 256, 1024} therefore brackets OT's selected c = 4096 in brake strength (OT 4096 ≈ GGN 450). It also includes
  the c = 64 used so far.
- **Selection:** the lowest step-1000 EMA validation loss over decays and grid. A diverged unit counts as worse.
- **Extension:** if the selected lr is at a grid edge, one unit at ×3 or ÷3 at the selected c, then the selection is
  repeated. If the selected c is at a grid edge (64 or 1024), one unit at c/4 or 4c at the selected lr.

## 4. Gates

- **G0:** suites before and after, including this study's tests:
  - the unit plan and extension rule;
  - `topfree` at c = 64 equal to `cifar_ggn`'s `topfree` over CPU steps (same schedule factor);
  - `topfree_ot` equal to `ot_tuned_1000`'s arm.
- **G1:** the rerun `topfree_ot__lr0.09__c4096__s1` reproduces `ot_tuned_1000`'s unit
  (`20261014T162934-085e376e`) bitwise in `validation.json` and `training.json`, timings excluded. This licenses the
  reference units.

## 5. Pre-declared readings (seeds 1–3)

*Better* or *worse* means lower or higher in all three seeds; otherwise *mixed*.

| Code | Reading |
| --- | --- |
| F1 | `topfree_ot` + EMA vs tuned `topfree` + EMA at step 1000. *The OT metric matters* only if better. Otherwise N23's advantage is attributed to tuning the brake and lr, and OT is not pursued as a metric. |
| F2 | Tuned `topfree` + EMA vs tuned `soap` + EMA at step 1000. |
| F3 | Descriptive: raw at step 1000, raw at steps 200 and 500, accuracy, the tuning table, and the selected lr and c. |

## 6. Cost, isolation and limitations

- **Cost:** about 6.5 min per unit. Tuning 6 units plus 1–2 extensions; evaluation 3 units plus the G1 rerun.
  **About 75 GPU-minutes.**
- **Running:** detached, followed by `bulk_krylov_20261014`.
- **Writes:** only to `studies/ggn_ot_fairness_20261014/` and `~/skipsoap-study-data/ggn_ot_fairness_20261014/`.
  `ot_tuned_1000`'s runs are read, not modified.
- **Limitations:**
  - Three seeds; one tuning seed.
  - Grids of 2 × 3 points with one extension per axis.
  - The references' tuning used a 2 × 2 grid for OT.
