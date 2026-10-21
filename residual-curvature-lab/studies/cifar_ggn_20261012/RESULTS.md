# CIFAR-10 on a depth-6 ViT: exact per-tensor GGN with and without SN, tuned SOAP and top-free PARSEC-L — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, approved with γ = 1e-2 and no GGN
tuning, and fixed before any run. Deviations: [DEVIATIONS.md](DEVIATIONS.md) (D1: an action-audit gate failure in
`ggn` seed 1, with a diagnosis and the owner's decision).

External runs `~/skipsoap-study-data/cifar_ggn_20261012/runs/`:
- tuning (seed 0): `20261012T103803-7d7d6274`;
- edge extension (`topfree` lr 9e-2): `20261012T110142-5e9e0779`;
- evaluation (seeds 1–3): `20261012T110529-d747f9fc`, closed after D1 (`closed.json`).

Light mirrors are in `runs/`. Summary: `results/summary.json` (`summarize.py --partial`). D1 diagnosis:
`results/diagnosis_D1_step51.json`.

**Status (2026-10-12): closed early.**
- `ggn` seed 1 stopped at step 100 on the block-against-action audit: `head.weight` measured 3.6e-4 against a gate
  of 1e-4.
- The owner decided to keep the gate and stop the GGN arms. `ggn` seeds 2 and 3 and all `ggn_sn` units never ran.
- Readings R1, R3, R4 and R5 are therefore **not available**. R2 is complete, and R6 is reported for the seven
  units that exist.
- Questions 1, 3 and 4 of the protocol stay open.

## 1. Summary

- **Top-free PARSEC-L + EMA beat tuned SOAP + EMA in all three evaluation seeds (R2: *better*).**
  - EMA validation loss at step 500 (d = 0.98): 0.5408 vs 0.5536 (seed means). Per-seed margins: 0.007, 0.019 and
    0.012.
  - The raw weights at step 500 were also lower in every seed (0.567 vs 0.588).
  - Mid-training, top-free's raw weights were higher than SOAP's in every seed, at step 50 (1.547 vs 1.458) and at
    step 200 (0.891 vs 0.826). Its tuned lr is 3× SOAP's (3e-2 vs 1e-2).
  - The tuning seed 0 showed no gap: 0.5427 vs 0.5430.
  - The run-to-run floor of this workload was not measured. On MNIST it was 0.002–0.008 (rule 11), so the seed-1
    margin is at that scale.
  - This is the first replication of the MNIST result on a different dataset and a model 54× larger.
- **Exact per-tensor GGN, seed 1 only, steps 0–75 (descriptive; the unit failed a gate).**
  - The step was clearly more efficient than SOAP's per step: validation loss 1.48, 1.28, 1.18 and 1.11 at steps
    10, 25, 50 and 75, against SOAP's 1.93, 1.71, 1.43 and 1.19 on the same seed and data order. SOAP first reached
    GGN's step-75 loss at its step-125 validation.
  - It was not more efficient per wall-clock second: 24 s per step against 0.22 s, so 1,844 s for 75 steps against
    28 s for SOAP's 125.
  - It was not more efficient per example seen either, counting the selection batch: 307k examples against 256k.
  - The gap to SOAP narrowed from 0.42 at step 25 to 0.08 at step 75.
- **D1: the audit's float32 reference fails, not the solved block.**
  - At the step-50 checkpoint, the solved whitened block of `head.weight` agrees with a float64 recomputation to
    7e-7. The float32 reference F0ᵀ·C·F0·u, which passes through original coordinates, is off by 2.7e-4. In float64
    the two sides agree to 4e-13.
  - `head.weight`'s SOAP denominators span 6.1 decades by then, against at most 4.8 for the other tensors checked.
  - As written, the audit could not have passed at later steps.
- **Quadratic models along the steps taken (descriptive).**
  - **Along SOAP and top-free steps:**
    - The full-GGN model predicts the next batch's loss change well after step 100: median ratio 0.99–1.06 over
      steps 110–500. The Hessian model does as well.
    - The per-tensor GGN model does not: median ratio −0.03 to 0.15. Along these steps the cross-tensor curvature
      is 15–21× the per-tensor terms (coupling).
  - **Along the GGN steps (seed 1, steps 10–70):**
    - Next batch: full-GGN median ratio 0.75, Hessian 0.98, per-tensor 0.47.
    - Coupling 9.9, and the Hessian term is 1.42× the GGN term (Gauss–Newton gap). On its own batch the full-GGN
      model is accurate (0.99).
- **Distance preservation and blocks (descriptive).**
  - Local distortion rises with depth in every arm.
  - Top-free distorts more early (steps 10–70) and settles to SOAP's level by step 250.
  - At step 50, `ggn`'s blocks 2–6 are far from identity: the RMS of R_k − I is 0.74–0.85, against 0.12–0.62
    for SOAP and 0.07–0.71 for top-free.
- **R6 (descriptive): no monotone relation** between the head input's local distortion and the next-batch
  full-GGN model error in the units that exist. The Spearman correlation is −0.09 pooled (307 rows, 7 units), and
  −0.31 to +0.08 per unit.

## 2. Tuning (seed 0, 500 steps)

Validation loss at step 500. The selection uses EMA d = 0.98, and the selected lr is marked.

| Arm | lr | Raw | EMA 0.9 | EMA 0.98 | EMA 0.995 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `soap` | 1e-3 | 0.8249 | 0.8240 | 0.8282 | 0.9381 | 0.717 |
| `soap` | 3e-3 | 0.6683 | 0.6637 | 0.6489 | 0.7211 | 0.777 |
| `soap` | **1e-2** | 0.5784 | 0.5736 | **0.5430** | 0.5983 | 0.812 |
| `soap` | 3e-2 | 0.5694 | 0.5642 | 0.5432 | 0.6620 | 0.807 |
| `topfree` | 1e-3 | 0.8132 | 0.8150 | 0.8176 | 0.9154 | 0.717 |
| `topfree` | 3e-3 | 0.6773 | 0.6753 | 0.6563 | 0.7251 | 0.774 |
| `topfree` | 1e-2 | 0.5984 | 0.5942 | 0.5592 | 0.5862 | 0.809 |
| `topfree` | **3e-2** | 0.5735 | 0.5709 | **0.5427** | 0.6391 | 0.810 |
| `topfree` | 9e-2 (extension) | 0.6674 | 0.6648 | 0.6973 | 1.1183 | 0.773 |

- **SOAP:** 1e-2 and 3e-2 tie to 2e-4.
- **Top-free:** 3e-2 was at the grid edge. The pre-declared extension to 9e-2 was clearly worse, so 3e-2 stands.
- **EMA decay:** 0.98 was best for both selected settings.

## 3. Evaluation, `soap` and `topfree` (seeds 1–3)

| Validation loss | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| `soap` raw, step 50 | 1.4340 | 1.4504 | 1.4896 | 1.4580 |
| `topfree` raw, step 50 | 1.6308 | 1.5069 | 1.5045 | 1.5474 |
| `soap` raw, step 200 | 0.8175 | 0.7924 | 0.8684 | 0.8261 |
| `topfree` raw, step 200 | 0.9172 | 0.8629 | 0.8929 | 0.8910 |
| `soap` raw, step 500 | 0.5780 | 0.5772 | 0.6092 | 0.5881 |
| `topfree` raw, step 500 | **0.5711** | **0.5442** | **0.5860** | **0.5671** |
| `soap` EMA 0.98, step 500 | 0.5474 | 0.5380 | 0.5755 | 0.5536 |
| `topfree` EMA 0.98, step 500 | **0.5402** | **0.5189** | **0.5633** | **0.5408** |
| accuracy at step 500, raw weights, `soap` / `topfree` | 0.807 / 0.807 | 0.809 / 0.816 | 0.798 / 0.804 | 0.805 / 0.809 |

**R2 (`topfree` + EMA vs `soap` + EMA, step 500): *better* in all three seeds.**
- Accuracy is the raw model's. It is higher for top-free in seeds 2 and 3, and lower by 0.0006 in seed 1.
- With seed 0's tie, top-free is lower in three of the four seeds run.

## 4. `ggn` seed 1 (failed unit; descriptive, not a reading)

Same seed, data order and augmentation as the other seed-1 units. The GGN arm also draws a 2,048-example selection
batch per step for its step-length search.

| Step | 10 | 25 | 50 | 75 |
| --- | ---: | ---: | ---: | ---: |
| `ggn` raw | **1.4816** | **1.2843** | **1.1774** | **1.1137** |
| `soap` raw | 1.9337 | 1.7089 | 1.4340 | 1.1899 |
| `topfree` raw | 2.0572 | 1.8491 | 1.6308 | 1.2823 |
| `ggn` EMA 0.98 | 1.7266 | 1.4262 | 1.2705 | 1.1917 |
| `soap` EMA 0.98 | 1.9732 | 1.8150 | 1.6060 | 1.4154 |
| `ggn` accuracy | 0.491 | 0.545 | 0.584 | 0.614 |
| `soap` accuracy | 0.273 | 0.370 | 0.480 | 0.576 |

- **Step trace (steps 2–75):**
  - μ rose from 1.9e-5 to 6.3e-4.
  - The selected step multiplier α had a median of 4.9 (range 2.8–20.4), and every search bracketed.
  - The whitened solve residual was at most 2.8e-5 (gate 1e-4). The original-coordinate residual was at most
    1.7e-5.
  - At the step-2 audit: κ_max was 1.33e6 and the maximum audit error 2.4e-6.
- **Cost:** 24.2 s per step (median), against 0.22 s for `soap` and 0.29–0.31 s for `topfree`.
- **Reach:** `ggn` reached SOAP's seed-mean step-50 loss (1.458) at its first validation after it, step 25.
- **Wall-clock and data:**
  - SOAP passed `ggn`'s step-75 loss at its step-125 validation, after 28 s of training. `ggn` took 1,844 s for its
    75 steps.
  - Counting the selection batches, `ggn` had seen 307k examples by step 75, and SOAP 256k by step 125.
- **Records:** they end at step 75 because units write records only at validation steps. Steps 76–99 ran but were
  not saved (DEVIATIONS.md D1).

## 5. D1: the audit failure

Details are in DEVIATIONS.md (D1, *D1 diagnosis*).

- **What failed:** at step 100 the block-against-action audit measured 3.6e-4 for `head.weight` against 1e-4. Every
  other tensor was at most 2.4e-6, measured at the step-50 checkpoint.
- **Float64 recomputation at that checkpoint, same u and F0:**
  - The float32 block, which is what the step solves, is accurate to 7.4e-7.
  - The float32 reference is off by 2.7e-4.
  - The two float64 sides agree to 4.1e-13.
- **Cause:** the reference F0ᵀ·C·(F0u) is formed in original coordinates. With `head.weight`'s SOAP denominators
  spanning 6.1 decades, this amplifies float32 rounding by the spread of d^-½ on both sides. The block is
  accumulated from whitened rows and avoids that path.
- **The steps taken are not in doubt.** The audit as specified cannot certify this tensor at 1e-4 in float32 once its
  preconditioner spreads this far.
- **Fix, if wanted:** a float64 recheck of the failing tensors would cost seconds per audit step (`logs/diagnosis_D1_float64_timing.log`: one
  float64 GGN action takes 4 s for `ggn` and 12–21 s for `ggn_sn`). The owner chose to keep the gate and stop the
  GGN arms.

## 6. Quadratic models along the steps taken (descriptive)

Ratio = actual loss change / predicted change gᵀd + q/2, on the next training batch (not augmented). Values are
medians over measurement steps (every 10 steps).
- *Coupling* = dᵀCd / Σ_b d_bᵀC_bb d_b.
- *GN gap* = dᵀHd / dᵀCd.

Means are not used, because single rows with tiny predicted changes dominate them; they are in `summary.json`.

| Unit | Steps | Full GGN | Per-tensor GGN | Hessian | Coupling | GN gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `soap` s1 / s2 / s3 | 10–100 | 1.00 / 0.98 / 0.94 | 0.17 / −0.11 / 0.38 | 0.95 / 0.85 / 0.99 | 13.2 / 14.7 / 12.7 | 0.99 / 0.95 / 0.95 |
| `soap` s1 / s2 / s3 | 110–500 | 0.99 / 1.00 / 1.04 | 0.08 / 0.15 / 0.12 | 1.00 / 1.02 / 1.01 | 15.0 / 15.1 / 15.2 | 1.02 / 1.02 / 1.01 |
| `topfree` s1 / s2 / s3 | 10–100 | 0.97 / 0.65 / 1.22 | 0.30 / 0.01 / 0.18 | 0.91 / 0.86 / 1.00 | 18.6 / 20.8 / 20.0 | 0.93 / 0.95 / 0.95 |
| `topfree` s1 / s2 / s3 | 110–500 | 1.06 / 1.01 / 0.99 | −0.03 / −0.03 / 0.00 | 1.06 / 0.97 / 1.00 | 18.8 / 21.5 / 17.2 | 1.01 / 1.00 / 1.01 |
| `ggn` s1 | 10–70 | 0.75 | 0.47 | 0.98 | 9.9 | 1.42 |

- **On the step's own batch,** the full-GGN median ratio is 0.99–1.03 for `soap`, and 0.99 for `ggn` over steps
  10–70.
- **For `ggn`, the next-batch full-GGN error** comes with a Hessian term 1.4× the GGN term along its step. SOAP and
  top-free steps show a GN gap of about 1.0.
- **No conclusion about SN.** The comparison with and without SN (R5) did not run.

## 7. Distance preservation (descriptive)

Local distortion q95/q5 of ‖φ(x+ξ)−φ(x)‖/‖ξ‖, over 512 validation inputs with relative perturbation 1e-3.

| Unit | Step | Stream after block 1 | Block 3 | Block 6 | Head input |
| --- | ---: | ---: | ---: | ---: | ---: |
| `soap` s1 | 10 / 100 / 500 | 1.20 / 1.52 / 1.54 | 1.43 / 1.73 / 1.68 | 1.58 / 1.93 / 1.80 | 4.04 / 3.70 / 3.97 |
| `topfree` s1 | 10 / 100 / 500 | 1.28 / 2.12 / 1.62 | 2.00 / 2.13 / 1.68 | 3.65 / 2.27 / 1.73 | 20.5 / 3.75 / 3.69 |
| `ggn` s1 | 10 / 50 / 70 | 1.47 / 1.48 / 1.47 | 2.01 / 1.91 / 1.88 | 2.34 / 2.23 / 2.14 | 4.66 / 4.21 / 4.44 |

Seeds 2 and 3 follow the same pattern; every step is in `summary.json`.

**Block-Jacobian deviation,** RMS singular value of R_k − I per block (4 probes):

| Unit | Step | Blocks 1–6 |
| --- | ---: | --- |
| `soap` s1 | 50 / 500 | 1.13 0.62 0.37 0.24 0.17 0.13 / 2.16 0.82 0.61 0.50 0.51 0.69 |
| `topfree` s1 | 50 / 500 | 2.92 0.55 0.20 0.14 0.09 0.07 / 3.69 0.62 0.43 0.41 0.42 0.60 |
| `ggn` s1 | 50 | 1.84 0.84 0.80 0.80 0.85 0.74 |

- **SOAP and top-free:** the deep blocks move away from identity gradually over 500 steps. Top-free's first block
  moves furthest (3.3–3.7 at step 500, against 2.2 for SOAP).
- **GGN:** it reaches deep-block deviations of 0.7–0.85 by step 50, more than SOAP has at step 500.

## 8. R6 (descriptive)

Spearman correlation between the head input's local distortion and |1 − ratio| of the next-batch full-GGN model, over
each unit's measurement steps:

| Unit | ρ | Rows |
| --- | ---: | ---: |
| `soap` s1 / s2 / s3 | −0.31 / +0.07 / −0.28 | 50 each |
| `topfree` s1 / s2 / s3 | −0.00 / +0.08 / −0.08 | 50 each |
| `ggn` s1 | −0.07 | 7 |
| pooled, 7 units | −0.09 | 307 |

The protocol's pooled correlations over the GGN units (with and without SN) are not available.

## 9. Cost

| Arm | Seconds per step (median, training only) | Unit wall time (evaluation, with diagnostics every 10 steps) |
| --- | ---: | ---: |
| `soap` | 0.22 | 8.3 min |
| `topfree` | 0.29–0.31 | 9.5–10.4 min |
| `ggn` | 24.2 | 43 min to the stop at step 100 (1,844 s of training to step 75); about 3.5 h per unit projected |

- **Diagnostics:** a row takes 7.1–7.5 s.
- **GPU time used:**
  - tuning and extension: 27 min;
  - `soap` and `topfree` evaluation: 55 min;
  - `ggn` seed 1 up to its stop: 43 min.
- **D1 diagnosis:** a few more minutes.

## 10. Limitations

- **Missing arms:** with the GGN arms stopped, the study cannot answer whether exact GGN keeps its per-step
  advantage to step 500 (Q1), whether SN changes GGN training (Q3), or how SN relates to distance preservation and
  the quadratic model (Q4). The `ggn` numbers above come from one seed, cover 75 steps, and come from a unit that
  failed a gate.
- **Noise floor:** R2 rests on three seeds with margins of 0.007–0.019. This workload's run-to-run floor is unknown.
  Learning rates were tuned on one seed (seed 0), where the two arms tied.
- **Run length:** 500 steps are about 23 epochs at batch 2048. Longer runs could change the comparison.
- **Quadratic-model ratios:** next-batch ratios are noisy when the predicted change is small, so the tables use
  medians. Coupling and GN gap are measured along each arm's own steps, so the arms are not compared at matched
  directions.
- **Hardware limits:** float32 on an 8 GB GPU. Exact per-tensor GGN at about 0.3M parameters takes 24 s per step
  (39 s with SN, from the dry run).
