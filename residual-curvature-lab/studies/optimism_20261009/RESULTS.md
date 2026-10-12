# Signal/optimism test of the transfer share — results

Study `optimism_20261009`, protocol [PROTOCOL.md](PROTOCOL.md) (hash in `protocol_sha256.txt`, written
before the run). Script `scripts/optimism.py`; outputs `results/optimism.json` and
`results/optimism_tables.md`. Offline, banked states only; about two minutes on one CPU thread.

## 1. Summary

- **Gates passed.** At all six states, the recomputed SOAP-map and exact-GGN candidates reproduce gap-audit
  check 2c exactly, on both right-hand sides: own-batch and held-out decreases differ by 0.0. Construction
  gradients match the stored ones (difference 0.0).
- **Calibration: partial** by the pre-declared rule. Spearman ρ_s = 0.56 between the predicted share
  (1 − r)/(1 + r) and the measured share over 24 gradient-RHS points, with MAE 0.24. The bar for
  "supported" was ρ_s ≥ 0.6 with MAE ≤ 0.2.
- **The noise-scale picture holds for the three fixed-metric operators.** Optimism ratio r:
  - SGD < SOAP map < exact GGN in all six states (exploratory reading; the pre-declared chain also
    included EF-Newton);
  - r grows with training in every metric. For exact GGN it rises from 0.35–0.58 at step 10 to
    1.36–1.52 at step 50, where the predicted share turns negative.
- **The pre-declared four-operator ordering fails: 0 of 6 states.** Empirical-Fisher Newton has a *lower*
  r than exact GGN in every state. Its A is built from the same batch's per-example gradients, so it
  suppresses that batch's noise directions. That violates the theory's fixed-A assumption.
- **The same-batch line-search mechanism explains the realized shares well.** For the realized direction
  D, the quadratic search share 2ρ − 1 with ρ = ḡ_panelsᵀD / g_BᵀD gives Spearman 0.86 (gradient) and
  0.90 (momentum) against the measured share, with MAE 0.15 (gradient). Caveat: ρ uses the same
  panels as the measurement.
- **Momentum is not described by the expectation formula.** The theory predicts near-zero optimism for
  momentum, which is independent of the construction batch, and so shares near 1.
  - Measured momentum shares are not systematically closer to 1: median 0.42 versus 0.36 for gradient
    steps, with a far wider spread (−4.0 to 1.69).
  - Five of the twelve momentum steps at the exact-GGN-trajectory states lose held-out loss.
  - The realized alignment ratio ρ varies widely in both directions, so the share is set by realized
    alignment, not by optimism.

**Next step:** the theory's core is right for fixed metrics: more whitening means a larger noise scale
B·r, and r grows during training. A noise-corrected line search (subtracting α·tr(A⁻¹Σ̂)/B) or a step
size chosen on a second batch is the direct remedy to test, on GGN and on PARSEC-L. The expectation
formula is too coarse to predict single-step shares, and the momentum RHS is not an automatic fix.

## 2. Readings

| Pre-declared reading | Result |
| --- | --- |
| 1. Calibration: Spearman, MAE (supported ≥ 0.6 and ≤ 0.2; not supported < 0.3) | 0.56, 0.24: **partial** |
| 2. Ordering r(EF) > r(GGN) > r(SOAP) > r(SGD) | 0 of 6 states; measured shares in reverse four-way order: 3 of 6 |
| 3. Sign: points with r > 1 and negative held-out decrease | 1 of 2 (SOAP-50 exact GGN: yes; GGN-50 exact GGN: no, +0.017) |
| 4. Mechanism: Spearman of 2ρ − 1 vs measured (gradient) | 0.86 (MAE 0.15) |
| Exploratory: mechanism on momentum RHS | 0.90 |
| Exploratory: ordering SGD < SOAP map < exact GGN | 6 of 6 states |

## 3. Gradient-RHS points

B·r is the gradient-noise scale in the operator's metric; the batch is B = 512.

| State | Operator | r | B·r | Predicted | Realized 2ρ−1 | Measured | Own | Held-out |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOAP-10 | sgd | 0.09 | 45 | +0.84 | +0.41 | +0.54 | 0.0507 | +0.0274 |
| SOAP-10 | soap_map | 0.16 | 81 | +0.73 | +0.41 | +0.58 | 0.0570 | +0.0332 |
| SOAP-10 | exact_ggn | 0.35 | 178 | +0.48 | +0.40 | +0.73 | 0.1862 | +0.1353 |
| SOAP-10 | ef_newton | 0.25 | 128 | +0.60 | +0.61 | +0.81 | 0.1455 | +0.1185 |
| SOAP-25 | sgd | 0.14 | 71 | +0.76 | +0.79 | +0.56 | 0.0369 | +0.0205 |
| SOAP-25 | soap_map | 0.28 | 145 | +0.56 | +0.50 | +0.37 | 0.0706 | +0.0259 |
| SOAP-25 | exact_ggn | 0.59 | 301 | +0.26 | +0.14 | +0.34 | 0.2512 | +0.0851 |
| SOAP-25 | ef_newton | 0.28 | 145 | +0.56 | +0.34 | +0.48 | 0.2516 | +0.1200 |
| SOAP-50 | sgd | 0.25 | 128 | +0.60 | +0.20 | +0.16 | 0.0361 | +0.0057 |
| SOAP-50 | soap_map | 0.55 | 282 | +0.29 | +0.05 | +0.06 | 0.0524 | +0.0030 |
| SOAP-50 | exact_ggn | 1.52 | 778 | −0.21 | −0.35 | −0.06 | 0.1986 | −0.0123 |
| SOAP-50 | ef_newton | 0.47 | 240 | +0.36 | −0.13 | −0.12 | 0.1676 | −0.0196 |
| GGN-10 | sgd | 0.13 | 65 | +0.77 | +0.22 | +0.30 | 0.0284 | +0.0084 |
| GGN-10 | soap_map | 0.27 | 139 | +0.57 | +0.25 | +0.32 | 0.0493 | +0.0156 |
| GGN-10 | exact_ggn | 0.58 | 298 | +0.26 | +0.14 | +0.23 | 0.1698 | +0.0392 |
| GGN-10 | ef_newton | 0.38 | 195 | +0.45 | +0.34 | +0.37 | 0.1414 | +0.0521 |
| GGN-25 | sgd | 0.36 | 183 | +0.47 | +0.79 | +0.72 | 0.0171 | +0.0123 |
| GGN-25 | soap_map | 0.41 | 210 | +0.42 | +0.51 | +0.43 | 0.0406 | +0.0174 |
| GGN-25 | exact_ggn | 0.96 | 490 | +0.02 | −0.10 | +0.06 | 0.1718 | +0.0108 |
| GGN-25 | ef_newton | 0.38 | 194 | +0.45 | +0.22 | −0.02 | 0.1332 | −0.0025 |
| GGN-50 | sgd | 0.21 | 110 | +0.65 | +1.00 | +1.16 | 0.0169 | +0.0197 |
| GGN-50 | soap_map | 0.52 | 264 | +0.32 | +0.30 | +0.60 | 0.0349 | +0.0208 |
| GGN-50 | exact_ggn | 1.36 | 696 | −0.15 | −0.28 | +0.10 | 0.1714 | +0.0172 |
| GGN-50 | ef_newton | 0.45 | 231 | +0.38 | +0.12 | +0.01 | 0.1258 | +0.0014 |

Momentum-RHS rows are in `results/optimism_tables.md`. At GGN-25 the SGD momentum step was rejected
(zero step), so its share is undefined.

**Why EF-Newton breaks the ordering.** Its damping is not larger than GGN's (μ_F/μ = 0.77–0.99). Its A
contains the construction batch's own gradient outer products, so A⁻¹ suppresses exactly that batch's
noise. Its optimism is therefore lower than GGN's in five of six states. Its held-out signal from
cross-panel products is also 1.3–4.0 times its batch-based estimate. For exact GGN the two agree within
15% at the three SOAP-trajectory states and GGN-10 (ratios 0.85–1.08), but differ 2.1–2.4 times at
GGN-25 and GGN-50.

## 4. Limitations

- Six states, one workload, one realization per point.
- The theory assumes a fixed A; exact GGN and EF-Newton build A from the construction batch (GGN from
  sampled-free outputs, EF from the batch's own labels and gradients).
- Shares computed from small own-batch decreases are unstable, e.g. the SOAP-map momentum step at GGN-25
  (own 0.0016, share −4.0).
- The realized mechanism check is not independent of the measurement: it uses the same panels.

## 5. Follow-up: refined theory, step-size remedies, drift and PARSEC-L

Protocol [PROTOCOL_FOLLOWUP.md](PROTOCOL_FOLLOWUP.md) (hash in `protocol_followup_sha256.txt`, written
before the run). Script `scripts/followup.py`; outputs `results/followup.json` and
`results/followup_tables.md`.

**Gates.** All gates pass at 0.0. The first run built the SOAP-map and exact-GGN directions from
explicit matrix inverses and reproduced check 2c only to 4.6e-10, failing the 1e-12 gate. The gate was
kept; the script was changed to check 2c's Cholesky solves and rerun. The fixed SOAP momentum step
equals the stored native step to 7.6e-11, with cosine deficit ≤ 2e-15.

| Pre-declared reading | Result |
| --- | --- |
| 1. EF ceiling g_bᵀ(F_b + μ_F Ω_b)⁻¹g_b ≤ 1 | **Holds:** maximum 0.81 over all tensors and states |
| 2. Independent mechanism (fit on panels 1–2, measure on 3–4) ≥ 0.6 | **Holds:** Spearman 0.80 |
| 3. Second-batch step size beats same-batch for exact GGN (≥ 5 of 6) | **Holds:** 5 of 6; all candidates 50 of 60. Mean held-out decrease on panels 2–4: same 0.0259, second 0.0363, oracle 0.0374 (second-batch recovers 92% of the oracle's gain over same-batch). |
| 4. Optimism-corrected search beats same-batch (≥ 12 of 18) | **Holds:** 14 of 18; PARSEC-L 5 of 6 |
| 5. Drift: momentum cosine lower at GGN-25/50 than SOAP-25/50 for SGD, SOAP map and exact GGN | **Partial:** holds for SOAP map and exact GGN, not for SGD |
| 6a. r(SOAP map) < r(PARSEC-L) < r(exact GGN) (≥ 5 of 6) | **Fails:** 0 of 6; PARSEC-L has the largest r of all operators (0.41–3.33) |
| 6b. PARSEC-L fixed step keeps SOAP's momentum share (≥ 5 of 6) | **Holds:** 5 of 6. But its held-out decrease exceeds SOAP's in only 2 of 6 states. |
| 6c. Same for norm blend 1 | Keeps share 5 of 6; more held-out decrease than SOAP 2 of 6 |
| Exploratory: calibration on label-free operators only (post hoc) | Spearman 0.70, MAE 0.23 (24 points) |

**Momentum cosine to the held-out gradient** (SGD / SOAP map / exact GGN metric): SOAP-25 +0.35 /
+0.30 / +0.18 and SOAP-50 +0.25 / +0.23 / +0.32, against GGN-25 −0.45 / −0.06 / +0.12 and GGN-50 +0.54 /
+0.14 / +0.03.

**PARSEC-L at its real step** (native SOAP step size, momentum, no search):
- It keeps SOAP's protection: share above SOAP's minus 0.1 in 5 of 6 states.
- It makes less held-out progress at the four states where SOAP's own step helps.
- It wins at the two states where SOAP's native step *loses* held-out loss: GGN-10 +0.0151 vs −0.0116,
  and GGN-25 +0.0015 vs −0.0112.
- Its step keeps 79–91% of SOAP's norm, but its own-batch decrease falls 3–8× at the SOAP-trajectory
  states. The removed top-16 whitened directions carry most of the per-step progress, which is also why
  PARSEC-L's r is the largest.

**Exploratory (not pre-declared): learning-rate calibration** (`scripts/lr_calibrated.py`,
`results/lr_calibrated.json`).

*Rationale.* In SOAP's whitened coordinates its step −η z is a proximal step for ‖ζ‖²/(2η). Adding the
true curvature gives ζ = −η (I + η F0ᵀCF0)⁻¹ z, which is PARSEC-L with β = η instead of
β = 1/(μ·scale). Then κ_k = η λ_k is the edge-of-stability ratio along whitened direction k.

*Findings at the six states:*
- **Scale of β:** β = η is about 10⁴ times smaller than the damping-calibrated β.
- **κ:** κ_max is 0.03–0.66, against 1,500–7,300; no direction has κ > 1.
- **Step:** the learning-rate-calibrated step keeps 96–100% of SOAP's norm, and its held-out decrease
  is within 0.0015 of SOAP's at the four states where SOAP's momentum step helps.
- **Where SOAP hurts:** at GGN-10 the learning-rate-calibrated step helps less than the
  damping-calibrated one (gradient −0.0049 vs +0.0071, momentum +0.0044 vs +0.0151).
- **Interpretation:** SOAP's harmful steps there are not curvature overshoot by the quadratic model
  (κ < 1).
