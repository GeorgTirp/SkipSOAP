# Does PARSEC-L's information contain the early GGN gain? The reduced-subspace ladder and the N21 control — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, fixed before any run. No deviations.

Data: `~/skipsoap-study-data/ladder_mechanism_20261013/` (snapshots and outputs). Copies are in `results/ladder`,
`results/control` and `results/summary.json`.

**Status (2026-10-13):** completed.
- **Gate S1 passed:** the replayed SOAP and top-free states equal `cifar_ggn`'s checkpoints bitwise at steps 50 and
  100 (12 of 12).
- **Gate L0 passed in all 12 ladder states:** no rung beat the exact optimum.
- **Held-out check:** all held-out step-size selections were interior to the grid.

## 1. Summary

- **L4: directions are missing (9 of 9).** At top-free states (steps 10, 25 and 50; seeds 1–3), 56–80% of the gap
  between top-free's step and the exact per-tensor GGN step lies outside anything expressible in the span of a
  rank-16 top space plus the current momentum and gradient. The gap is measured on the damped per-tensor quadratic.
  - Even with the *exact* top-16 eigenvectors and the best coefficients, the step recovers 25–62% of the exact
    step's reduction. It exceeds 50% in only one early state (seed 1, step 50: 62%).
- **L1: the held information does not close half the gap (0 of 9).** Better coefficients in the held subspace plus
  momentum and gradient close 10–28%.
- **L2: staleness (4 of 9 ≥ 10%, *not shown*).** A fresh subspace adds 13–17% of the gap at step 10, 3–4% at step 25
  and 6–10% at step 50.
- **L3: estimation (0 of 9, *not shown*).** The exact top-16 eigenvectors add at most 3% over the probe estimate.
  The probes find the right top directions.
- **By the pre-declared rule (L4 holds, L1 does not), "close the GGN gap with PARSEC-L" is formally dropped.** Most
  of the exact per-tensor GGN step's advantage lives outside a rank-16 top space.
- **On held-out loss the picture is kinder early (L5, descriptive).** At step 10, the Galerkin step in the fresh
  subspace recovers 60–87% of the exact GGN step's *held-out* single-step advantage over top-free's step. On the
  quadratic it recovers far less. By step 25 this drops to 20–39%, and from step 50 on the exact GGN step's own
  held-out advantage is small.
- **N21 control: neither reading passes (N1: 11 of 18; N2: 8 of 18).** Along the step just taken, the curvature
  ratio ρ = dᵀC_oldd/dᵀC_nowd is 0.09–0.51. Along random directions its median is 0.39–1.01.
  - In 11 of 18 points random directions change at most half as much, so the step's large change is mostly
    selection.
  - In the other 7 (mainly at t = 60, 5 of 6 points), random directions also lose 30–60% of their curvature. Global
    sharpening contributes there.
  - The current gradient's direction lies in between (ρ 0.23–0.96).

## 2. Part L: gap decomposition (top-free states)

Shares of G = ΣR(ggn) − ΣR(topfree), pooled over tensors:

| State | Coefficients | Staleness | Estimation | Missing directions |
| --- | ---: | ---: | ---: | ---: |
| s1 / s2 / s3, t10 | 0.27 / 0.28 / 0.24 | 0.17 / 0.14 / 0.14 | 0.01 / 0.00 / 0.01 | **0.56 / 0.58 / 0.62** |
| s1 / s2 / s3, t25 | 0.19 / 0.17 / 0.17 | 0.04 / 0.04 / 0.03 | 0.01 / 0.01 / −0.00 | **0.77 / 0.78 / 0.80** |
| s1 / s2 / s3, t50 | 0.18 / 0.10 / 0.15 | 0.10 / 0.10 / 0.06 | 0.03 / 0.02 / 0.01 | **0.69 / 0.79 / 0.78** |
| s1 / s2 / s3, t100 | 0.13 / 0.11 / 0.11 | 0.06 / 0.08 / 0.08 | 0.01 / 0.02 / 0.02 | 0.79 / 0.80 / 0.79 |

**Absolute share of the exact optimum** (ΣR / ΣR*):

| Rung | Range over the 12 states |
| --- | ---: |
| SOAP's direction | 0.02–0.35 |
| top-free's step | 0.04–0.45 |
| held subspace only | 0.05–0.51 |
| held subspace + momentum + gradient | 0.20–0.55 |
| fresh subspace + momentum + gradient | 0.25–0.60 |
| exact top-16 + momentum + gradient | 0.25–0.62 |

**Held-out decrease** (step chosen on a selection batch, measured on an evaluation batch; seeds 1 / 2 / 3):

| Step | top-free | held | fresh | exact_top | ggn |
| ---: | --- | --- | --- | --- | --- |
| 10 | 0.008 / 0.050 / 0.021 | 0.051 / 0.080 / 0.034 | 0.069 / 0.102 / 0.049 | 0.067 / 0.103 / 0.050 | 0.083 / 0.110 / 0.068 |
| 25 | 0.024 / 0.027 / 0.024 | 0.036 / 0.030 / 0.025 | 0.042 / 0.034 / 0.028 | 0.043 / 0.035 / 0.028 | 0.071 / 0.057 / 0.042 |
| 50 | 0.197 / 0.045 / 0.022 | 0.193 / 0.043 / 0.025 | 0.177 / 0.046 / 0.027 | 0.178 / 0.046 / 0.027 | 0.198 / 0.055 / 0.037 |
| 100 | 0.018 / 0.004 / 0.007 | 0.018 / 0.009 / 0.014 | 0.020 / 0.013 / 0.016 | 0.021 / 0.013 / 0.017 | 0.024 / 0.020 / 0.025 |

**Readings:**
- **L1:** 0 of 9 (*not shown*).
- **L2:** 4 of 9 (*not shown*).
- **L3:** 0 of 9 (*not shown*).
- **L4:** 9 of 9, *directions are missing*. Consequence: the goal is formally dropped.
- **L5:** every selection interior.

## 3. Part N: the N21 control

ρ(u) = uᵀC(θ_{t−11})u / uᵀC(θ_{t−1})u on the next batch.

| Point | ρ(step) | ρ(gradient) | ρ(random): quartiles | Selection |
| --- | ---: | ---: | --- | --- |
| SOAP s1 t20 / t60 / t110 | 0.27 / 0.35 / 0.51 | 0.38 / 0.79 / 0.82 | 0.74–0.86 / 0.52–0.60 / 0.76–0.88 | yes / no / yes |
| SOAP s2 t20 / t60 / t110 | 0.17 / 0.39 / 0.50 | 0.44 / 0.69 / 0.84 | 0.94–1.09 / 0.63–0.76 / 0.73–0.86 | yes / no / yes |
| SOAP s3 t20 / t60 / t110 | 0.27 / 0.27 / 0.51 | 0.64 / 0.76 / 0.96 | 0.43–0.82 / 0.45–0.52 / 0.81–0.86 | no / no / yes |
| top-free s1 t20 / t60 / t110 | 0.10 / 0.37 / 0.22 | 0.23 / 0.68 / 0.66 | 0.30–0.42 / 0.65–0.84 / 0.72–0.80 | no / yes / yes |
| top-free s2 t20 / t60 / t110 | 0.14 / 0.16 / 0.27 | 0.70 / 0.55 / 0.68 | 0.79–1.29 / 0.37–0.46 / 0.71–0.84 | yes / no / yes |
| top-free s3 t20 / t60 / t110 | 0.09 / 0.20 / 0.25 | 0.44 / 0.55 / 0.62 | 0.52–0.67 / 0.51–0.54 / 0.81–0.90 | yes / no / yes |

"Selection" means the median \|1 − ρ(random)\| is below half of \|1 − ρ(step)\|.

**Readings:** N1 11 of 18 and N2 8 of 18, both *not shown*. The staleness of N21 is mostly, but not only, a property
of the step's direction. Around step 60, curvature also grows along random directions.

## 4. Interpretation and limitations

- **What is formally dropped:** PARSEC-L as "an affordable approximation of the exact per-tensor GGN step". The
  missing part is not in better coefficients, fresher probes or more accurate top directions. It is in directions
  outside the top 16. What remains is PARSEC-L as a SOAP modification that trains better (top-free, and the OT
  variant pending its control), whose mechanism is not a GGN approximation.
- **Early on, a reduced solve in the probe subspace would carry much of the GGN's single-step held-out gain** (step
  10: 60–87%). This is a narrow window: the exact GGN's own held-out advantage fades by step 50. It matches the
  earlier finding that curvature pays early.
- **Limitations:**
  - **Model:** all of this is the damped per-tensor quadratic, one step ahead, at rank 16, with γ = 1e-2.
  - **Rank:** a larger rank would move some "missing" directions into the top space. That was not measured.
  - **N21 control:** it uses 8 random directions per point.
