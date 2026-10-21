# CIFAR-10 on a depth-6 ViT: SOAP with and without SN — protocol

Study `cifar_sn_soap_20261012`. Written on 12 October 2026, before any run.
- **First version:** SHA `11b2fbc2…`, held by the owner before launch.
- **This version:** adds measures A–D and readings S8–S12, at the owner's request ("add the measures and then
  launch", 12 October). The motivation is a theoretical draft the owner shared, which separates network nonlinearity
  from distance preservation. Its prediction 3 reads: compare logit-linearization error, tangent-curvature drift and
  distance preservation at matched states.

**The owner's request** (12 October, after `cifar_ggn_20261012` closed early): "Can we run the SN question for a
SOAP run please." In `cifar_ggn_20261012` the SN questions were posed for the exact per-tensor GGN, whose arms
stopped at a float32 audit gate (that study's DEVIATIONS.md, D1). This study asks the same questions for tuned SOAP
on the same workload, with the same diagnostics. SOAP is cheap, so the study takes about 2 GPU-hours instead of 17.

## 1. Questions

1. Does SN (c = 1) change how SOAP trains? Asked at SN's own tuned learning rate, and at SOAP's learning rate.
2. Does SN raise distance preservation through depth?
3. Does SN make the quadratic model along SOAP's step more accurate? And does it lower the cross-tensor coupling
   along the step? In `cifar_ggn` that coupling was 15–21×, so the per-tensor GGN model missed most of the
   curvature along SOAP steps.
4. Does distance preservation track quadratic-model fidelity over training? (descriptive)
5. Where does the quadratic model fail: in the network's nonlinearity in its parameters, or in the cross-entropy
   loss beyond second order?
   - Does SN lower the network nonlinearity at a matched function-space step?
   - Does SN keep the directional curvature closer to its value ten steps earlier?
6. **Prediction 3:** does the logit-linearization error along the step track the quadratic-model error better than
   distance preservation does?

## 2. Workload

Identical to `cifar_ggn_20261012` (its PROTOCOL.md, section 2):
- **Data:** CIFAR-10, 45k/5k split of the official training set (seed 101). The test split is never accessed.
- **Augmentation:** crop (pad 4) + flip.
- **Model:** M1d6 ViT, width 64, depth 6, 4 heads, MLP 256, 308,042 parameters.
- **Training:** batch 2048 in microbatches of 512, 500 steps, GPU float32 with TF32 off and deterministic
  algorithms.

The model, data, measurement and microbatch code is imported unchanged from `studies/cifar_ggn_20261012/scripts`
(`cifar.py`, `measure.py`, `micro.py`, `ggn.py`) and archived with each run. `scripts/train.py` is that study's
runner with the GGN and top-free arms removed and an SN arm added. Its `soap` path is unchanged.

## 3. Arms

| Arm | Settings |
| --- | --- |
| `soap` | SOAP as PARSEC-L with rank 0 (the lab's `soap` arm). Betas (0.95, 0.95), preconditioner refresh every 10 steps, decoupled weight decay 0.05. **lr 1e-2 and EMA decay 0.98 as selected on seed 0 in `cifar_ggn`; not retuned.** There, 3e-2 tied with 1e-2 to 2e-4. |
| `soap_sn` | The same SOAP on `cifar.CifarViTSN`, with W_eff = W·min(1, c/σ(W)) and **c = 1**. σ is exact (an SVD at every forward pass). It acts on the 36 residual-branch weights: attention query, key, value and output, and MLP w1 and w2. The optimizer, its weight decay and its preconditioner act on the raw weights. Parameter names equal the plain model's. **Peak lr tuned on seed 0.** |

- **Schedules (both arms):** linear warm-up over steps 1–25, constant, then linear decay over steps 401–500 to 1%
  (WSD).
- **Weight EMAs:** d ∈ {0.9, 0.98, 0.995}, bias-corrected, used only for evaluation.
- **SN is active from the start.**
  - At initialization the raw spectral norms of the 36 weights have a median of 1.11 (range 0.83–1.70).
  - The `soap_sn` dry run clipped 30 and 31 of the 36 at steps 10 and 20.
  - Under plain SOAP in `cifar_ggn`, every one of the 36 exceeded 1 from step 50 on, and the median reached 2.4 by
    step 500.
  - So c = 1 constrains essentially every branch weight throughout.
- **Known interaction, kept by declared choice:** for a clipped weight the loss is invariant to the raw weight's
  scale. Decoupled weight decay then only shrinks the raw norm, and that enlarges SOAP's step relative to the
  effective weight.

## 4. Tuning and evaluation

- **Tuning (seed 0, 500 steps, no diagnostics):** `soap_sn` with lr ∈ {1e-3, 3e-3, 1e-2, 3e-2}.
  - Selection: the lowest step-500 EMA validation loss over the three decays, which also selects the decay.
  - If the selected lr lies at a grid edge, one extension unit at ×3 or ÷3 is run, and the selection is repeated
    over both runs.
- **Evaluation (seeds 1–3, 500 steps, diagnostics):**
  - `soap` at lr 1e-2;
  - `soap_sn` at its tuned lr;
  - if that differs from 1e-2, also `soap_sn` at lr 1e-2 (*matched*).
- **Seeds:** seed s sets model initialization s, data order 102 + 1000 s and augmentation 5000 + s. Diagnostics use
  fixed validation pairs (seed 9001) and Jacobian probes (seed 9002).
- **Queue:** one GPU process at a time, `soap` units first. `scripts/orchestrate.sh` chains tuning, the extension
  if needed, evaluation, G1 and the summary.

## 5. Diagnostics (evaluation units)

As in `cifar_ggn_20261012` (its PROTOCOL.md, section 5), along the step actually taken (d = θ_{t+1} − θ_t, raw
parameters, including schedule and weight decay):

- **Every 10 steps:**
  - quadratic-model fidelity on the step's own augmented batch and on the next 2,048 unaugmented training examples:
    - actual change, gᵀd, the full GGN dᵀCd, the per-tensor GGN Σ_b d_bᵀC_bb d_b and the Hessian dᵀHd (double
      backward);
    - the ratios actual / (gᵀd + q/2);
    - coupling dᵀCd / Σ_b d_bᵀC_bb d_b;
    - Gauss–Newton gap dᵀHd / dᵀCd;
  - distance preservation:
    - local (512 validation inputs, x′ = x + 10⁻³‖x‖ξ) and global (1,000 validation pairs) distance ratios;
    - representations: the stream after the embedding and after each block, the CLS row from block 1 on, and the
      head input;
    - reported as quantiles and distortion q95/q5;
  - spectral norms of the 36 branch weights, raw and effective, and SN activity.
- **Added measures, every 10 steps** (`scripts/nonlinear.py`; recorded under `nonlinear` in each diagnostic row):
  - **A. Error split along d,** on the next batch and on the own batch, from logits only. With f = f(θ), Jd
    (forward mode), f⁺ = f(θ+d) and the remainder r = f⁺ − f − Jd:
    - *actual* = L(f⁺) − L(f), and *linearized* = L(f + Jd) − L(f), the linearized network with the exact loss;
    - actual − (gᵀd + ½dᵀCd) = **network error** (actual − linearized) + **loss-Taylor error** (linearized −
      gᵀd − ½dᵀCd), plus the network share |network| / (|network| + |loss-Taylor|);
    - the **logit-linearization error** ‖r‖/‖Jd‖ over the batch;
    - the residual term 2·mean(∂ℓ/∂f · r), equal to dᵀHd − dᵀCd to second order in d;
    - the function-space step size, the RMS over examples of ‖J_i d‖.
  - **B. Matched function-space step,** next batch: the quantities of A along d rescaled so that the RMS of ‖J_i d‖
    is **0.1** and **1.0** logits.
    - These sizes are fixed from the dry runs, where SOAP's own steps had an RMS of 0.6–2 at full lr.
    - They compare the arms at equal logit movement, a size the raw-versus-effective parametrization does not
      change.
  - **C. Curvature persistence along d,** next batch:
    - one-step ratio dᵀC(θ+d)d / dᵀC(θ)d;
    - ten-step ratio dᵀC(θ_old)d / dᵀC(θ)d, with θ_old the state ten steps earlier (from step 20 on).

    Ten steps is SOAP's preconditioner refresh and PARSEC-L's probe interval.
  - **D. Representations by depth** (stream after the embedding and after each block, head input), on the 512
    local validation inputs:
    - the linearization error along d;
    - the temporal drift since the previous measurement, ten steps earlier: relative change; relative change of the
      pairwise-distance matrix of the first 256 inputs; and linear CKA, where 1 − CKA is the deformation not
      explained by rotation and isotropic scale.
- **Every 50 steps:** block-Jacobian deviation, the RMS singular value of R_k − I per block.
- **Validation** at steps 0, 10, 25, …, 500, raw and EMA.
- **Checkpoints** at steps 50, 100, 250 and 500.

## 6. Gates

- **G0:** before and after the study these suites pass:
  - the repository suite (106);
  - the PARSEC-L tests (22);
  - the `cifar_ggn` study tests (11);
  - this study's tests (12): unit plan; arm models; this study's `soap` arm equal to `cifar_ggn`'s bitwise for 3
    CPU steps; `soap_sn` with an inactive bound equal to plain SOAP to 1e-6; the G1 and reading helpers; and measures
    A–D against independent float64 references, for both models. The references are: loss differences, gradient,
    GGN action, central-difference Jd, double-backward Hessian, first-order scaling of the linearization errors,
    and rotation invariance of CKA and distance change.
- **G1:** the three evaluation `soap` units reproduce the `soap` units of `cifar_ggn`'s evaluation run
  `20261012T110529-d747f9fc` **bitwise**, in every field of `validation.json`, `training.json` and
  `measurements.json` except wall-clock timings and this study's added measures under `nonlinear`
  (`scripts/g1.py`). The added measures run after `cifar_ggn`'s diagnostics, use no random numbers and do not
  touch the training state.
  - Pre-run check: a 25-step `soap` dry run on seed 0 equals `cifar_ggn`'s tuning unit `soap__lr0.01__s0` bitwise
    in training and validation records (`logs/dryrun_soap.log`).
  - A G1 failure is reported, and the SN readings are then flagged as resting on unverified code.
- **Divergence** (a non-finite loss or parameters) ends a unit as diverged. It is reported, not a gate failure, and
  it counts as worse in every comparison.

## 7. Pre-declared readings (evaluation seeds 1–3)

*Better* or *worse* means lower or higher in all three seeds; otherwise *mixed*. EMA readings use each arm's
selected decay. Median windows are over the measurement steps 10–100 and 110–500.

| Code | Reading |
| --- | --- |
| S1 | `soap_sn` (tuned lr) vs `soap`: raw validation loss at steps 50 and 200, EMA at step 500 |
| S2 | `soap_sn` (lr 1e-2) vs `soap`, same three comparisons. If the tuned lr is 1e-2, S2 is S1. |
| S3 | Distance preservation: local distortion q95/q5 of the head input and of the stream after block 6, at steps 100 and 500. *SN raises preservation* if `soap_sn` is lower in all seeds, for both representations at both steps. |
| S4 | Quadratic model: per-unit median of \|1 − ratio\| of the full-GGN model on the *next* batch, per window. *SN improves the full-GGN model* if `soap_sn` is lower in all seeds in both windows. |
| S5 | Coupling: per-unit median of the next-batch coupling, per window. *SN lowers cross-tensor coupling along the step* if `soap_sn` is lower in all seeds in both windows. |
| S6 | Descriptive, no verdict: |
| | – the same medians for the per-tensor and Hessian models, on both batches; |
| | – the Gauss–Newton gap and the step norm; |
| | – the full depth profile of local and global distortion; |
| | – block-Jacobian deviations; |
| | – SN activity and raw spectral norms. |
| S7 | Interaction (descriptive, no verdict): per unit, the Spearman correlation over the 50 measurement steps between the head input's local distortion and next-batch \|1 − ratio_full\|. Also pooled per arm and over all units. |
| S8 | Network nonlinearity at matched function-space step (B): per-unit median of the next-batch logit-linearization error, at each step size (0.1 and 1.0), per window. *SN lowers network nonlinearity at matched step* if `soap_sn` is lower in all seeds, at both sizes, in both windows. |
| S9 | Error split (descriptive): the median network share along the own step (next and own batch) and at both matched sizes, with the matched-size full-GGN ratio. |
| S10 | Curvature persistence (C): per-unit median of \|1 − ten-step ratio\| (next batch), per window. In the window 10–100 this covers steps 20–100. *SN keeps the directional curvature closer over ten steps* if `soap_sn` is lower in all seeds in both windows. The one-step ratio is reported alongside. |
| S11 | Prediction 3: per unit, the Spearman correlations over the measurement steps of next-batch \|1 − ratio_full\| with (a) the next-batch logit-linearization error along the step and (b) the head input's local distortion. *Linearization error tracks quadratic-model error better than distortion* if ρ_a > 0 and ρ_a > \|ρ_b\| in every evaluation unit. |
| S12 | Representations (D, descriptive): per depth, median linearization error, drift, distance change and CKA, per window. Also the transfer decomposition from existing fields: γ = next gᵀd / own gᵀd and ρ = next dᵀCd / own dᵀCd. |

S3–S5, S8 and S10 are read for `soap_sn` at its tuned lr. With a matched arm, they are also read at lr 1e-2.

Also reported:
- accuracy;
- seconds per step;
- the tuning table.

Medians are used for S4, S5, S8 and S10 because the next-batch ratios of `cifar_ggn`'s SOAP units had single rows with tiny
predicted changes that dominated the means.

## 8. Cost (dry runs, `logs/dryrun_*.log`; 40 steps with all measures, `logs/dryrun_measures_*.log`)

| Arm | Seconds per step | Diagnostic row | Peak GPU | Per evaluation unit |
| --- | ---: | ---: | ---: | ---: |
| `soap` | 0.22 | 8.8 s (7.1 s before A–D) | 2.6 GiB | about 10 min |
| `soap_sn` | 0.43 | 22 s (18 s before A–D) | 2.6 GiB | about 23 min |

**Budget:**
- tuning: about 18 min, plus about 5 min for an extension;
- evaluation: about 30 min for `soap` and 70 min for `soap_sn`, plus 70 min if a matched arm is needed;
- **total about 2–3.2 GPU-hours.**

## 9. Isolation and resources

- **Writes:** only to `studies/cifar_sn_soap_20261012/` and `~/skipsoap-study-data/cifar_sn_soap_20261012/`
  (`$CIFAR_SN_DATA`).
- **Reads:** the lab; `studies/gpu_port_20261012/gpu_parsec`; `studies/cifar_ggn_20261012/scripts`; and, for G1,
  `cifar_ggn`'s run directories. These are imported or read, never modified.
- **Hygiene:** `PYTHONDONTWRITEBYTECODE=1`; pytest with `-p no:cacheprovider`; isolation manifests and suites
  before and after. The before-manifest has 1,415 files, and the baseline suites pass.
- **Processes:** one GPU process at a time.

## 10. Limitations known in advance

- One model, three evaluation seeds, 500 steps (about 23 epochs), c = 1 only.
- SOAP is not retuned here. Its lr comes from `cifar_ggn`'s seed-0 grid, where 1e-2 and 3e-2 tied.
- **Quadratic models along different steps:** the fidelity and coupling comparisons are along each arm's own
  steps, which differ in size and direction, so they do not compare the two models at matched directions. They
  are measured in raw-parameter coordinates, the optimizer's.
- **Raw versus effective coordinates:** with SN, the raw-parameter GGN is singular along the scale of every clipped
  weight. Distances and Jacobians are those of the effective network.
- **This run-to-run floor is unknown:** the floor was measured on MNIST (0.002–0.008 at step 500), not on this
  workload. Loss differences of that size are not interpreted.
- The attention key biases have zero gradient. In float32 their coordinates may drift without effect on the loss.
- **Added measures (A–D):**
  - The error split and the linearization errors are exact identities, but they are measured in float32. At
    matched size 0.1 the expected remainders (a few percent of Jd) lie well above float32 rounding of the logits.
  - The residual term matches dᵀHd − dᵀCd only to second order. At the dry run's step sizes (RMS logit change
    0.6–2) it differs, as expected.
  - Prediction 3 is tested along SOAP's steps only, not along curvature steps.
