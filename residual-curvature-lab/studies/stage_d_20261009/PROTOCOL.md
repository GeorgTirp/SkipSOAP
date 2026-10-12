# Stage D: PARSEC-L training test — protocol

Study `stage_d_20261009`. **Status: approved by the owner on 10 October 2026** ("what we actually need to
do is now a full PARSEC-L run to compare it with the other optimizers, so lets do that here please"). It
was drafted on 9 October; the harness was written after approval and the text below is unchanged. The
finished arms are also compared with every optimizer already evaluated on these seeds and data order (gap
audit and geometry follow-up), without rerunning them.

## 1. Questions

1. **Brake.** Does PARSEC-L with SOAP's fixed learning rate stabilize SOAP enough that a larger learning
   rate beats tuned SOAP? Variant: learning-rate calibration β = c·lr·√(1 − β₂ᵗ), or damping calibration.
2. **Early-Newton.** Does the PARSEC-L direction, with its step length chosen each step on a separate
   selection batch, keep exact GGN's early per-step advantage?
3. **GGN remedy.** Does choosing the exact-GGN step length on a selection batch remove GGN's late
   collapse? Under same-batch search, its next-batch transfer is under 1% after step 50.

## 2. Isolation rules

As in the earlier side studies:
- write only to `studies/stage_d_20261009/` and `~/skipsoap-study-data/stage_d_20261009/`;
- read and import elsewhere only, with no git writes, no new packages, no downloads and no official MNIST
  test split;
- `PYTHONDONTWRITEBYTECODE=1`; pytest with `-p no:cacheprovider`;
- at most four concurrent single-threaded processes;
- numerical tolerance 1e-8; never loosen a gate;
- isolation manifests and repository suite before and after.

## 3. Workload and data

- **Model and training:** main ViT (`configs/mini_vit_allocation.yaml`), depth 2, **branch scaling off**
  (`branch_scale` 1, the plain model). MNIST 28x28, batch 512, CPU float64, all 40 tensors, 500 consumed
  steps.
- **Seeds:** settings are chosen on seed 0 only; evaluation runs on seeds 1, 2, 3 (model init s, data
  order 102 + 1000 s, split 101). Gradient batches follow the gap audit's data order, so the baselines
  stay comparable.
- **Selection batches:** 128 examples, drawn fresh each step from a separate stream over the training
  split (order seed 5102 + 1000 s). There is no fixed pool, so no pool can be overfitted over 500
  selections. Examples used for gradients and for selection are counted and reported separately; probes
  reuse the gradient batch.
- **Validation:** the reserved 5,000 examples at steps 0, 10, 25, 50, 75, …, 500.

## 4. Arms (`PARSECL` from `studies/parsec_l_20261009`, 21 tests passing)

Norm blend and calibration are set explicitly in every arm and never taken from defaults.

**A. Native SOAP.** `PARSECL` with rank 0, which is upstream SOAP bit for bit. Betas (0.95, 0.95),
precondition frequency 10, weight decay 0; lr ∈ {1e-2, 2e-2, 3e-2}, run on seed 0 (references for the
seed-0 grid) and evaluated on seeds 1–3.
- Reproduction gate: lr 1e-2 seeds 1–3 within 1e-6 of the gap audit's `native_soap`. The lab chain
  differs from in-place SOAP by rounding (2e-12 after 12 steps).

**B. PARSEC-L, fixed step (brake).** Rank 16; s = 4 sampled-label probes on the gradient batch every 10
steps; `refresh='extract'`; `norm_blend=0`; RHS momentum (SOAP's own update).
- Calibration ∈ {lr with c = 1, lr with c = 4, lr with c = 30, damping with γ = 0.01}, crossed with
  lr ∈ {1e-2, 2e-2, 3e-2}: 12 runs on seed 0.
- The configuration with the lowest step-500 validation loss is evaluated on seeds 1–3.

**C. PARSEC-L, Early-Newton.** Damping calibration (γ 0.01), rank 16, s = 4, probes every 10 steps,
`norm_blend=0`, base lr 1e-2.
- Step length chosen on the selection batch each step: α ∈ {0} ∪ {2^k : k = −4…8}, then 12 golden-section
  refinements in log₂ α (about 26 evaluations of 128 examples).
- Variants: RHS ∈ {momentum, gradient} × hand-over ∈ {off, on}.
  - *On:* SOAP's own fixed-lr step from the same start is also evaluated on the selection batch, and the
    better step is taken. After k = 10 consecutive SOAP wins the arm switches to plain SOAP permanently
    (`select_compare_soap`, `select_handover=10`). This rule is model free.
- On seed 0, four runs choose the RHS by step-100 validation loss averaged over the two hand-over
  variants. Both hand-over variants are evaluated with that RHS on seeds 1–3.

**D. Exact tensor GGN, step chosen on the selection batch.** The gap audit's `ggn_exact_trace`:
- exact per-tensor blocks, γ 0.001, exact trace, shadow-SOAP metric, gradient RHS;
- the same 52-evaluation ray search (`search_candidate`), but on the selection batch instead of the
  training batch;
- no tuning; seeds 1–3.
- Baseline: the gap audit's same-batch GGN on the same seeds (existing runs, no rerun).

## 5. Measurements

- **Every step, all arms:**
  - training-batch loss before and after;
  - the next gradient batch's loss before and after the update (paired transfer, outside the timed
    region);
  - step norm; training seconds (diagnostics excluded); examples used.
- **PARSEC-L probe steps:**
  - μ; κ and ηλ_max per tensor (ηλ_max = κ_max · lr·√(1−β₂ᵗ)/β, the edge-of-stability ratio);
  - correction energy (every step); held-subspace residual.
- **Selection arms:** chosen α, selection losses, winner (PARSEC-L or SOAP), hand-over step.
- **Every 10 steps, all arms:** diagnostic r̂ from a true-label per-example pass on the training batch,
  for SOAP's metric and for the arm's own operator (for GGN, A = C + μΩ). Signal is cross-fitted from
  the two 256-example halves. The pass is not used by any arm.

## 6. Gates

- PARSEC-L test suite (21 tests) and repository suite pass before the runs.
- SOAP reproduction (section 4A).
- GGN action checks and dense-solve residuals ≤ 1e-8 at steps 2, 50, 200 and 500.
- Any failed numerical check stops the arm, and the arm is reported as failed. Divergence (non-finite
  loss) is a result, not a failure.

## 7. Pre-declared readings

Seeds 1–3, settings chosen on seed 0. Better or worse means lower or higher validation loss in all three
seeds.

| Arm | Counts as working if |
| --- | --- |
| B. Brake | The selected configuration beats tuned SOAP (lr 1e-2) at step 500. |
| C. Early-Newton | Each hand-over variant beats tuned SOAP at steps 50 and 100. Steps-to-target and seconds-to-target (gap-audit T50 = 0.355037, T100 = 0.265401) are reported side by side, with examples used (gradient plus selection). |
| D. GGN, selected step | Beats the gap audit's same-batch GGN at steps 200 and 500. |

Exploratory, labelled as such:
- the c-by-lr grid on seed 0;
- whether SOAP's failing learning rate coincides with ηλ_max > 2;
- r̂ trajectories, and whether r̂ crossing about 0.5–1 predicts the model-free hand-over step;
- next-batch transfer by phase (1–50, 51–200, 201–500).

## 8. Cost projection (from existing measurements, sequential)

| Arm | Seconds per step | Runs | CPU-time |
| --- | ---: | ---: | ---: |
| A. SOAP | 0.04 | 3 (seed 0) + 9 | ≈ 4 min |
| B. Brake | 0.085 | 12 + 3 | ≈ 11 min |
| C. Early-Newton | ≈ 0.22 (probes, correction, ~26 forward passes on 128) | 4 + 6 | ≈ 20 min |
| D. GGN, selected step | ≈ 4.2 (blocks 3.9, search on 128) | 3 | ≈ 1.8 h |
| r̂ diagnostics | ≈ 0.1 per 10 steps | all | negligible |

Total ≈ 2.4 CPU-h sequential, about 3.5 with four workers (contention factor 1.6). Budget 5 CPU-h.

A dry run of 5 steps per arm precedes the runs, once approved.

## 9. Limitations known in advance

- One workload (ViT width 16, depth 2, MNIST), 500 steps, three seeds; settings chosen on seed 0 only.
- The selection arms use 25% more examples per step; they are counted, not hidden.
- The GGN arm is not retuned for the new step policy.
- 128-example selection batches are noisier than the 512-example panels used offline.
- r̂ uses 256/256 halves and is a diagnostic only.
