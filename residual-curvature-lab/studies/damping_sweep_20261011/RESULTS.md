# Top-space damping sweep and offline noise-model (OU) check — results

Protocol [PROTOCOL.md](PROTOCOL.md) (SHA-256 in `logs/protocol_sha256.txt`, written before the runs). Clarifications
and one stopped launch are in [DEVIATIONS.md](DEVIATIONS.md); all three entries were written before any part-2
result existed (hashes in `logs/deviations_sha256_before_ou.txt`).

- **Sweep run:** `~/skipsoap-study-data/damping_sweep_20261011/runs/20261011T141750-48966bdf`, mirrored in
  [runs/](runs/). Summary: `results/sweep_summary.json`.
- **c = 64 units:** from `topfree_confirm_20261011` (`20261011T140034-192dd041`).
- **OU check:** `results/ou_check.json`, log `logs/ou_check.log`.

**Status (2026-10-11, 14:34Z):**
- All 13 sweep units completed, none diverged.
- The c = 64 seed-4 rerun reproduces `topfree_confirm` bitwise (maximum validation difference 0.0).
- The OU check processed all 72 states (3 arms × 24 steps). The whitening matches upstream SOAP to 1.0e-14 at
  most.

## 1. Summary

**Decision on the Kalman version (protocol section 4, fixed in advance): do not build it.**
- **S2 is *flat or undetermined*, not an interior optimum.** No c beats c = 64 in all seeds.
- **O1 is *not adequate*:** the noise model fails for every arm, in every variant computed.
- The rule says to revise the noise model before any filter. The simpler alternative, removing the top-subspace
  component, is not supported either: freezing (c = 10⁹) has the worst seed mean.

**Sweep.**
- More damping than c = 64 does not help. The per-seed comparisons are all mixed.
- The seed means rise steadily from c = 64 to freezing: 0.121, 0.123, 0.126, 0.128, 0.130.
- With the seed-0 trend 1 < 4 < 16 < 64 from the earlier studies, the best fixed c lies near 64, between 16 and
  a few hundred.
- Every c still beats SOAP + EMA (0.135) on the seed mean. Per seed, c = 64, 256 and 4096 are better in all three
  seeds; c = 1024 and 10⁹ are mixed.

**OU check.** The model predicts each arm's spread of the weights around their EMA along the exact top GGN
directions from gradient noise alone. It fails in two ways:

- **Wrong time structure.**
  - In the sharp directions (η·λ ≥ 0.1), the measured lag-10 correlation is 0.81–0.93 in all three arms.
  - The model predicts 0.42 for SOAP at 0.1 ≤ η·λ < 1, and −0.62 at η·λ ≥ 1, where noise-driven heavy-ball
    motion should oscillate.
  - So even for SOAP, whose overall spread matches (ratio 1.03 and 1.44), the measured deviations are slow, not
    noise-driven jitter.
- **Wrong size under damping, and not stationary.**
  - **Top-free:** spread 16 times the prediction at 0.1 ≤ η·λ < 1 and 110 times at η·λ ≥ 1. The model expects
    the c = 64 damping to cut the sharp-direction spread about 33-fold relative to SOAP. The measured spread is
    only about 40% smaller (summed over steps 300–480; the direction sets differ between arms).
  - **Over training:** the ratio drifts strongly in every arm. SOAP goes from 2.1–3.5 before step 100 to 0.28–0.53
    from step 300 on; top-free from 40–62 to 4.9–9.7.

**Reading (hypotheses, not tested here).** Along the exact top directions, the deviation of the weights from their
average is dominated by a slow component that the noise model lacks. Three candidates fit the data:
- drift of the trajectory, which the EMA follows with a lag of about 49 steps;
- rotation of the top directions;
- leakage: the optimizer's held subspace (rank 16, sampled-label probes, refreshed every 10 steps) only partly
  covers the exact top-16 directions of the true-label GGN, so part of each direction still gets SOAP's undamped
  momentum step.

The data do not separate these. In the most damped bins, the excess grows with the strength of the damping:
about 5× for the brake (whose predicted spread is cut at least 5-fold) and 110× for top-free (cut at least
65-fold). That fits leakage. SOAP's slow deviations fit drift.

**Consequence for the top-free mechanism.** `topfree_confirm` explained the gain as keeping the top coordinates
near the valley floor by removing their noise. This check does not support the noise-only form of that
explanation. The training gain itself is not in question; this check measures the deviation from the weight
average, not the distance from the valley floor.

## 2. Part 1: damping sweep

Validation loss at step 500, seeds 4 / 5 / 6, constant lr 1e-2:

| c | EMA d = 0.98 | Mean | Raw mean | d = 0.9 mean | d = 0.995 mean | Sharpness at 500 (seeds 4/5/6) |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 64 | 0.1292 / 0.1141 / 0.1195 | **0.1209** | 0.1418 | 0.1277 | 0.1454 | 27 / 33 / 35 |
| 256 | 0.1217 / 0.1211 / 0.1260 | 0.1229 | 0.1490 | 0.1328 | 0.1523 | 25 / 36 / 22 |
| 1024 | 0.1378 / 0.1221 / 0.1180 | 0.1260 | 0.1487 | 0.1332 | 0.1552 | 23 / 26 / 25 |
| 4096 | 0.1271 / 0.1255 / 0.1305 | 0.1277 | 0.1496 | 0.1366 | 0.1560 | 37 / 22 / 34 |
| 10⁹ (freeze) | 0.1292 / 0.1359 / 0.1255 | 0.1302 | 0.1563 | 0.1399 | 0.1566 | 56 / 112 / 50 |
| SOAP + EMA (reference) | 0.1360 / 0.1345 / 0.1346 | 0.1350 | | | | |
| Brake + EMA (reference) | 0.1313 / 0.1207 / 0.1219 | 0.1246 | | | | |

| Code | Reading | Result |
| --- | --- | --- |
| S1 | c = 256 vs 64 | mixed (better in seed 4 only) |
| S1 | c = 1024 vs 64 | mixed (better in seed 6 only) |
| S1 | c = 4096 vs 64 | mixed (better in seed 4 only) |
| S1 | c = 10⁹ vs 64 | mixed (seed 4: 0.12916 vs 0.12923; worse in 5 and 6) |
| S2 | Shape over c | **flat or undetermined** (no interior candidate; freezing is not best) |
| Gate | c = 64 seed-4 rerun | passed, difference 0.0 at every validation step |

- **Over training** (EMA d = 0.98, seed means at steps 100 / 200 / 300 / 400 / 500): c = 64 has the lowest loss at
  every step. Its values are 0.545 / 0.231 / 0.163 / 0.134 / 0.121; freezing gives 0.609 / 0.258 / 0.172 / 0.144 /
  0.130.
- **Sharpness under freezing.** With c = 10⁹ the full-network preconditioned sharpness climbs to 50–112 (seed 5
  reaches 111.6 at step 500). That exceeds the EMA-momentum limit of 78 without instability, because the
  optimizer no longer steps along the frozen directions, where this curvature sits. With finite c the sharpness
  peaks at 50–59 over the run (c = 64: 59).

## 3. Part 2: OU check (seed 4, steps 20–480, exact top-16 per tensor)

All steps. Predicted lag-10 is the residual-process value (DEVIATIONS #1); the literal ρ₁₀ of the raw process is
given for comparison.

| Arm | η·λ bin | n | Spread ratio | Lag-10 measured | Lag-10 predicted | Literal ρ₁₀ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SOAP | < 0.01 | 2533 | 2.05 | 0.94 | 0.95 | 0.99 |
| SOAP | 0.01–0.1 | 5455 | 1.94 | 0.95 | 0.86 | 0.91 |
| SOAP | 0.1–1 | 5830 | **1.03** | 0.85 | **0.42** | 0.48 |
| SOAP | ≥ 1 | 251 | **1.44** | 0.86 | **−0.62** | −0.60 |
| Brake | < 0.01 | 2245 | 1.65 | 0.94 | 0.95 | 0.99 |
| Brake | 0.01–0.1 | 4920 | 1.82 | 0.95 | 0.88 | 0.93 |
| Brake | 0.1–1 | 6370 | **1.37** | 0.85 | **0.72** | 0.74 |
| Brake | ≥ 1 | 537 | **5.02** | 0.90 | **0.54** | 0.57 |
| Top-free | < 0.01 | 1791 | 1.27 | 0.68 | 0.78 | 0.97 |
| Top-free | 0.01–0.1 | 4642 | 3.88 | 0.89 | 0.73 | 0.90 |
| Top-free | 0.1–1 | 6571 | **15.9** | 0.87 | **0.70** | 0.86 |
| Top-free | ≥ 1 | 1069 | **110** | 0.93 | **0.69** | 0.86 |

O1 criteria (bins η·λ ≥ 0.1; spread ratio in [0.5, 2], lag within ±0.3):

| Arm | 0.1–1 | ≥ 1 | Fails on |
| --- | --- | --- | --- |
| SOAP | spread ✓, lag ✗ (0.43) | spread ✓, lag ✗ (1.47) | time structure |
| Brake | spread ✓, lag ✓ (0.13) | spread ✗, lag ✗ (0.36) | sharpest bin |
| Top-free | spread ✗, lag ✓ (0.17) | spread ✗, lag ✓ (0.24) | size |

- **Verdict:** O1 is *not adequate* for every arm in all four variants: primary or literal lag prediction, all
  steps or t ≥ 100.
- **Restricted to t ≥ 100** (reported only): the spread ratios become 0.74 / 1.07 (SOAP), 1.12 / 3.75 (brake)
  and 12.4 / 78.6 (top-free), for 0.1–1 and ≥ 1 respectively. The pattern is the same.

Spread ratio over training, all directions with η·λ ≥ 0.1:

| Step | 20 | 60 | 100 | 200 | 300 | 400 | 480 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOAP | 2.12 | 2.98 | 2.34 | 0.78 | 0.46 | 0.37 | 0.40 |
| Brake | 2.84 | 3.68 | 3.62 | 1.32 | 0.64 | 0.64 | 0.57 |
| Top-free | 40.1 | 62.4 | 35.4 | 13.7 | 9.65 | 6.65 | 6.44 |

**Model checks.** Before use, the closed forms were compared with simulated scalar processes (400,000 steps): heavy
ball, brake and gradient descent, each with the EMA. Variance and EMA factor agreed within 1.7%, and lag-10
correlations within 0.003.

## 4. Decision and next step

- **Kalman version:** not built, per the pre-declared rule (S2 not interior, O1 not adequate).
- **"Remove the top-subspace component":** not adopted. S2 is not monotone to freezing, and freezing has the
  worst mean and the sharpest trajectories.
- **Standard setting stays:** lr calibration, `top_rhs='gradient'`, c = 64, plus weight EMA.
- **Revising the noise model (proposed, offline, about 10 minutes on the same checkpoints).** The checkpoints
  hold the optimizer's subspace, so two quantities can be measured directly:
  - **Capture:** the norm of each exact top direction's projection onto the optimizer's held subspace, that
    is, the share that is actually damped. This tests leakage.
  - **Distance to the valley floor:** g_k/λ_k from the analysis-batch gradient, corrected for its noise. This
    tests whether damping keeps the top coordinates near the floor, which the EMA deviation used here cannot
    show.

  Then refit the model with a leakage share and a drift term, before any filter.

## 5. Limitations

- **Part 1:** one workload (ViT width 16, depth 2, MNIST), 500 steps, seeds 4–6. These seeds were already used
  to confirm c = 64, so the sweep is descriptive (as declared). The seed spread (about ±0.007) is larger than
  the mean differences between c values (0.002–0.009).
- **Part 2:** one seed, one run per arm, 24 states each.
  - The exact directions and noise come from 512 held-out analysis examples with true labels. The optimizer's
    probes use sampled labels on its training batches.
  - The model holds F̂ and λ fixed between refreshes and assumes stationarity, which the drift over training
    shows is false.
  - The stationary EMA factor is used even at t < 100, where the bias-corrected EMA covers only part of its
    window.

## 6. Cost, isolation, suites

- **Wall time:**
  - sweep 14:17:52–14:32:17Z with three workers, 47–72 s of training per unit;
  - OU check 14:26:43–14:32:18Z in one process, after a stopped first launch (DEVIATIONS #3).
- **Isolation:** the manifests before and after are identical (3,336 files), and no new `__pycache__` was
  created. The idle `tail -f` of the queue runner was stopped after the STOP line so that the runner could exit.
- **Suites:** the repository suite passed 106 before and after; the study tests passed 23 (1 harness, 22
  PARSEC-L).
