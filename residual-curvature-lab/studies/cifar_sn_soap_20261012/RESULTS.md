# CIFAR-10 on a depth-6 ViT: SOAP with and without SN — results

Protocol: [PROTOCOL.md](PROTOCOL.md). The approved version's SHA-256 is in `logs/protocol_sha256.txt`. It is the
first version plus measures A–D and readings S8–S12, added at the owner's request before launch. Deviations:
[DEVIATIONS.md](DEVIATIONS.md) (D1: bytecode written by a pre-run test call; D2: the isolation-manifest
difference).

External runs `~/skipsoap-study-data/cifar_sn_soap_20261012/runs/`:
- tuning (seed 0): `20261012T171855-1adb9b1e`;
- evaluation (seeds 1–3): `20261012T173632-18f98b9d`.

Light mirrors are in `runs/`. Summary: `results/summary.json`. G1: `results/g1.json`.

**Status (2026-10-12):** all 10 units completed without divergence. G1 passed: the three `soap` units equal
`cifar_ggn`'s bitwise in every recorded field. The suites pass before and after (106, 22, 11, 12).
- SN's tuned lr is 1e-2, SOAP's lr, so there is no extension and no separate matched arm. S2 is S1.
- SN clipped 30 of the 36 branch weights at step 10, and all 36 at steps 50, 100, 250 and 500, in every seed.

## 1. Summary

- **SN (c = 1) makes SOAP worse (S1: *worse* on all three comparisons).**
  - Raw loss at step 50: 1.507 vs 1.458. At step 200: 0.874 vs 0.826.
  - EMA loss at step 500: 0.580 vs 0.554, worse in each seed, by 0.030, 0.040 and 0.009.
  - On tuning seed 0 the gap is 0.049 (0.592 vs 0.543).
  - The raw weights end nearly level at step 500 (0.586 vs 0.588; mixed), so most of the gap at the end is a
    smaller EMA gain: 0.007 for SN against 0.035 for SOAP.
  - SN costs 2× per step (0.43 s vs 0.22 s).
- **None of the five SN hypotheses passes its pre-declared rule (S3, S4, S5, S8, S10).** Two go the other way:
  - **Distance preservation (S3): improved, but not in all four parts.** Local distortion is lower with SN after
    block 6 at steps 100 and 500 and at the head input at step 500, in all seeds. At the head input at step 100 it
    is mixed. The gain is largest in shallow blocks (after block 1: 1.21–1.25 vs 1.51–1.54) and nearly gone after
    block 6 (1.69–1.74 vs 1.71–1.81).
  - **Full-GGN quadratic model (S4): not improved.** Mixed over steps 10–100. *Worse* over steps 110–500 (median
    \|1 − ratio\| 0.062–0.092 vs 0.060–0.069).
  - **Cross-tensor coupling (S5): higher with SN, the opposite of the hypothesis.** *Worse* in all seeds in both
    windows: 16.7–17.2 vs 12.7–14.7 early, and 17.8–18.7 vs 15.0–15.2 late.
  - **Network nonlinearity at matched function-space step (S8): no real difference.** Seven of eight parts are
    mixed. The eighth (step size 1.0, late) is lower with SN by about 3% (0.111–0.114 vs 0.115–0.118). At step
    size 0.1 the logit-linearization error is 1.1–1.7% in both arms; at 1.0 it is 11–17%.
  - **Curvature persistence (S10): no difference.** Mixed in both windows.
- **SN slows and smooths representation change (S12, descriptive).**
  - Over ten steps (110–500), every depth moves less with SN:
    - relative drift after block 6: 0.29–0.30 vs 0.37–0.39;
    - after block 1: 0.15–0.17 vs 0.21–0.22.
  - Representations also deform less: pairwise-distance change after block 6 is 0.029–0.033 vs 0.043–0.045, and CKA
    is higher.
  - The blocks stay much closer to identity: at step 500 the RMS ‖R_k − I‖ of block 1 is 0.71–0.79 vs 2.16.
  - None of this made the quadratic model more accurate or training better.
- **Prediction 3 (S11): not shown. Linearization error tracks quadratic-model error more often than distortion
  does, but not in every unit.**
  - The Spearman correlation of next-batch \|1 − ratio_full\| with the logit-linearization error is positive in 5
    of 6 units (−0.02 to 0.39).
  - It exceeds the absolute correlation with head-input distortion in 4 of 6 units. Distortion correlations are
    −0.31 to 0.07.
  - Both relations are weak.
- **Curvature along the current step was much lower ten steps earlier, with or without SN (C, descriptive).**
  - Along the step just taken, the GGN curvature at the state ten steps earlier was 32–42% of its current value over
    steps 20–100, and 56–62% over steps 110–500.
  - One step later it is 93–101%.
  - Part of this may be a selection effect: the step is built from the current gradient, which tends to point into
    currently sharp directions. A random-direction control would separate the two.
  - Ten steps is SOAP's preconditioner refresh and PARSEC-L's probe interval.
- **Where the quadratic model fails (S9, descriptive).**
  - Along SOAP's own steps (RMS logit change about 0.8–1.0), network nonlinearity and the cross-entropy's
    beyond-second-order terms contribute about equally. The network share is 0.48–0.76 on the next batch.
  - At a step size of 0.1, network nonlinearity carries 92–98% of a much smaller error (full-GGN ratio
    0.998–1.003). That is expected from the orders: the network term is second order in the step, the loss-Taylor
    term third.

## 2. Tuning (seed 0, `soap_sn`, 500 steps)

| lr | Raw | EMA 0.9 | EMA 0.98 | EMA 0.995 | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 0.8278 | 0.8274 | 0.8320 | 0.9454 | 0.714 |
| 3e-3 | 0.6948 | 0.6928 | 0.6922 | 0.7726 | 0.757 |
| **1e-2** | 0.6000 | 0.5931 | **0.5917** | 0.6806 | 0.798 |
| 3e-2 | 0.6115 | 0.6064 | 0.6107 | 0.7623 | 0.788 |

- **Selection:** lr 1e-2 with EMA 0.98, an interior point of the grid, so no extension.
- **SOAP's reference on seed 0** (`cifar_ggn`): 0.5430 at lr 1e-2.

## 3. Loss readings (S1; S2 is the same units)

| Validation loss | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| `soap` raw, step 50 | **1.4340** | **1.4504** | **1.4896** | **1.4580** |
| `soap_sn` raw, step 50 | 1.5137 | 1.4993 | 1.5076 | 1.5069 |
| `soap` raw, step 200 | **0.8175** | **0.7924** | **0.8684** | **0.8261** |
| `soap_sn` raw, step 200 | 0.8824 | 0.8447 | 0.8961 | 0.8744 |
| `soap` raw, step 500 | **0.5780** | **0.5772** | 0.6092 | 0.5881 |
| `soap_sn` raw, step 500 | 0.5811 | 0.5869 | **0.5913** | **0.5864** |
| `soap` EMA 0.98, step 500 | **0.5474** | **0.5380** | **0.5755** | **0.5536** |
| `soap_sn` EMA 0.98, step 500 | 0.5770 | 0.5779 | 0.5848 | 0.5799 |
| accuracy at step 500 (raw), `soap` / `soap_sn` | 0.807 / 0.804 | 0.809 / 0.806 | 0.798 / 0.800 | 0.805 / 0.803 |

**S1: *worse* at raw step 50, raw step 200 and EMA step 500.** Raw at step 500 (not a reading) is mixed: SN is
higher in seeds 1 and 2 (by 0.003 and 0.010) and lower in seed 3 (by 0.018).

## 4. SN readings (S3–S5, S8, S10)

Values are per seed (1 / 2 / 3). The verdict needs `soap_sn` lower in all seeds in every part.

| Reading | Part | `soap_sn` | `soap` | Part reading |
| --- | --- | --- | --- | --- |
| S3 distance preservation (local q95/q5) | head input, step 100 | 4.18 / 3.40 / 3.67 | 3.70 / 3.68 / 4.19 | mixed |
| | after block 6, step 100 | 1.54 / 1.52 / 1.51 | 1.93 / 1.76 / 1.81 | better |
| | head input, step 500 | 3.53 / 3.60 / 3.23 | 3.97 / 3.90 / 3.50 | better |
| | after block 6, step 500 | 1.74 / 1.74 / 1.69 | 1.80 / 1.81 / 1.71 | better |
| S4 full-GGN model, median \|1 − ratio\| (next batch) | steps 10–100 | 0.045 / 0.195 / 0.054 | 0.057 / 0.077 / 0.088 | mixed |
| | steps 110–500 | 0.092 / 0.067 / 0.062 | 0.069 / 0.060 / 0.062 | worse |
| S5 coupling, median (next batch) | steps 10–100 | 16.7 / 16.8 / 17.2 | 13.2 / 14.7 / 12.7 | worse |
| | steps 110–500 | 18.4 / 17.8 / 18.7 | 15.0 / 15.1 / 15.2 | worse |
| S8 logit-linearization error at matched step | size 0.1, steps 10–100 | 0.0157 / 0.0166 / 0.0143 | 0.0172 / 0.0165 / 0.0166 | mixed |
| | size 0.1, steps 110–500 | 0.0121 / 0.0110 / 0.0117 | 0.0120 / 0.0118 / 0.0119 | mixed |
| | size 1.0, steps 10–100 | 0.154 / 0.168 / 0.137 | 0.169 / 0.167 / 0.163 | mixed |
| | size 1.0, steps 110–500 | 0.114 / 0.112 / 0.113 | 0.118 / 0.115 / 0.116 | better |
| S10 \|1 − ten-step ratio\|, median | steps 20–100 | 0.58 / 0.68 / 0.58 | 0.65 / 0.63 / 0.63 | mixed |
| | steps 110–500 | 0.39 / 0.41 / 0.40 | 0.43 / 0.44 / 0.40 | mixed |

**Verdicts:**
- S3: *not shown* (3 of 4 parts better).
- S4: *not shown* (mixed, then worse).
- S5: *not shown*. Coupling is *higher* with SN in all seeds in both windows.
- S8: *not shown* (1 of 8 parts better).
- S10: *not shown*.

## 5. Descriptive measures (S6, S7, S9, S11, S12)

Medians per unit over the measurement steps. Seed values are given as a range.

| Quantity | Window | `soap` | `soap_sn` |
| --- | --- | --- | --- |
| step size in logit space (RMS ‖J_i d‖, next batch) | 110–500 | 0.84–0.96 | 0.92–1.00 |
| raw step norm ‖d‖ | 110–500 | 0.74 | 0.72 |
| full-GGN ratio (next batch) | 110–500 | 0.99–1.04 | 0.98–1.00 |
| per-tensor model \|1 − ratio\| | 110–500 | 0.85–0.96 | 0.81–1.02 |
| Hessian model \|1 − ratio\| | 110–500 | 0.044–0.066 | 0.046–0.092 |
| Gauss–Newton gap | 110–500 | 1.01–1.02 | 1.01–1.02 |
| network share of the model error, own step (next batch) | 110–500 | 0.48–0.62 | 0.62–0.74 |
| network share at matched size 0.1 / 1.0 | 110–500 | 0.92–0.93 / 0.50–0.59 | 0.92–0.95 / 0.56–0.65 |
| full-GGN ratio at matched size 0.1 / 1.0 | 110–500 | 0.998–0.999 / 0.995–1.049 | 0.998–0.999 / 0.975–1.003 |
| logit-linearization error along own step (next batch) | 110–500 | 0.106–0.112 | 0.107–0.108 |
| one-step curvature ratio | 110–500 | 0.96–0.99 | 0.96–0.98 |
| ten-step curvature ratio | 20–100 / 110–500 | 0.35–0.37 / 0.56–0.60 | 0.32–0.42 / 0.59–0.62 |
| transfer γ = next gᵀd / own gᵀd | 10–100 / 110–500 | 0.54–0.66 / 0.34–0.36 | 0.55–0.91 / 0.37–0.40 |
| curvature ratio ρ = next dᵀCd / own dᵀCd | 10–100 / 110–500 | 0.87–0.91 / 0.83–0.84 | 0.84–0.88 / 0.85–0.87 |

**Representations by depth (D),** steps 110–500, `soap` vs `soap_sn`:

| Depth | Linearization error along the step | Ten-step drift | Distance-matrix change | CKA |
| --- | --- | --- | --- | --- |
| after block 1 | 0.033–0.035 vs 0.032–0.039 | 0.21–0.22 vs 0.15–0.17 | 0.015–0.017 vs 0.008–0.009 | 0.990–0.992 vs 0.996–0.997 |
| after block 3 | 0.057–0.061 vs 0.057–0.061 | 0.29–0.32 vs 0.24–0.26 | 0.025–0.026 vs 0.018–0.021 | 0.972–0.974 vs 0.978–0.984 |
| after block 6 | 0.083–0.091 vs 0.076–0.079 | 0.37–0.39 vs 0.29–0.30 | 0.043–0.045 vs 0.029–0.033 | 0.939–0.956 vs 0.961–0.971 |
| head input | 0.112–0.117 vs 0.106–0.110 | 0.38–0.39 vs 0.35–0.36 | 0.073–0.078 vs 0.066–0.070 | 0.946–0.950 vs 0.959–0.960 |

- **Nonlinearity grows with depth** in both arms: the linearization error along the step rises from about 3% after
  block 1 to 8–9% (SOAP) or 8% (SN) after block 6, and 11% at the head input.
- **SN** lowers drift and deformation at every depth, and lowers the deep-layer linearization error slightly.

**Block-Jacobian deviation** (RMS ‖R_k − I‖, blocks 1–6, seed 1):
- `soap`: step 50 1.13 0.62 0.37 0.24 0.17 0.13; step 500 2.16 0.82 0.61 0.50 0.51 0.69.
- `soap_sn`: step 50 0.49 0.34 0.24 0.17 0.12 0.09; step 500 0.75 0.60 0.47 0.42 0.39 0.52.

Seeds 2 and 3 are within about 0.1.

**Global distortion** (head input and after block 6): no consistent difference. At step 500: 2.55–2.64 vs 2.57–2.64,
and 1.94–2.12 vs 2.08–2.14.

**S11 (prediction 3):** per-unit Spearman correlations of next-batch \|1 − ratio_full\| with the logit-linearization
error (ρ_a) and with head-input local distortion (ρ_b):

| Unit | ρ_a | ρ_b | ρ_a > 0 and ρ_a > \|ρ_b\| |
| --- | ---: | ---: | --- |
| `soap` s1 | 0.08 | −0.31 | no |
| `soap` s2 | 0.31 | 0.07 | yes |
| `soap` s3 | 0.15 | −0.28 | no |
| `soap_sn` s1 | −0.02 | −0.02 | no |
| `soap_sn` s2 | 0.26 | −0.04 | yes |
| `soap_sn` s3 | 0.39 | 0.07 | yes |

**Verdict: *not shown*** (3 of 6 units).

**S7** (head distortion vs \|1 − ratio_full\|, Spearman): −0.31 to 0.07 per unit; pooled −0.17 (`soap`), 0.02
(`soap_sn`) and −0.07 (all).

## 6. Cost

| Arm | Seconds per step | Diagnostic row | Evaluation unit |
| --- | ---: | ---: | ---: |
| `soap` | 0.22 | 8.8 s | 9.6 min |
| `soap_sn` | 0.43 | 22 s | 22.8 min |

GPU time: tuning 18 min, evaluation 97 min.

## 7. Interpretation and limitations

- **What SN does here:** it holds the blocks near identity, slows representation change and lowers shallow-layer
  distortion.
- **What it does not do here:** make the quadratic model more accurate, lower cross-tensor coupling, or keep the
  curvature closer over ten steps. SOAP trains worse with it.
- **The theory draft's separation holds in these data.** Spatial preservation, temporal drift and parameter
  nonlinearity are different quantities: SN changes the first two clearly and the third barely. The quadratic-model
  error tracks the third weakly and the first not at all.
- **SN coupling went up, not down.** With SN, more of the curvature along SOAP's step lies between tensors. A
  candidate reason, not tested: with every branch weight clipped, the loss is invariant to each weight's raw scale,
  so the curvature within each block loses those directions.
- **Limitations:**
  - **Single settings:** one bound (c = 1, which clips every branch weight), one model, three seeds and 500 steps.
    SOAP's lr was not retuned, but SN's tuned lr equals it.
  - **Own-step comparisons:** all quadratic-model comparisons except S8 run along each arm's own steps. S8 matches
    the step size in logit space, but not the direction.
  - **Possible selection effect:** the ten-step ratio may partly reflect selection of the step direction (see
    section 1). It needs a random-direction control.
  - **Noise floor:** this workload's run-to-run floor is unknown; on MNIST it was 0.002–0.008. The loss differences
    in S1 are well above it, except seed 3's EMA margin (0.009).
