# Does in-loop top-space damping beat weight averaging? — results

Protocol [PROTOCOL.md](PROTOCOL.md) (SHA-256 in `logs/protocol_sha256.txt`, written before the runs). External
runs `~/skipsoap-study-data/topfree_confirm_20261011/runs/`:
- tuning `20261011T135636-50234c98`;
- fixed evaluation `20261011T135638-221c0b6a`;
- tuned evaluation `20261011T140034-192dd041`.

Light mirrors are in [runs/](runs/). Summary: `runs/20261011T135638-221c0b6a/topfree_confirm_summary.json`.

**Status (2026-10-11, 14:10Z):** all 20 units completed, no divergence, all gates passed (seed-0 units reproduce
the noise-controls runs with difference 0.0).

## 1. Summary

**Verdict: in-loop damping adds beyond averaging (P1 and P2 both better).** On seeds 4–6, never used before,
the momentum-free top-space brake (`top_rhs='gradient'`, c = 64, lr 1e-2, constant) evaluated on an EMA of its
weights beats both SOAP baselines with the same averaging in every seed:

| Step 500, seeds 4 / 5 / 6 | Raw weights | EMA of weights (d = 0.98) |
| --- | --- | --- |
| **Top-free brake, c = 64** | 0.150 / 0.136 / 0.140 (mean 0.142) | **0.129 / 0.114 / 0.119 (0.121)** |
| SOAP, constant | 0.165 / 0.170 / 0.166 (0.167) | 0.136 / 0.134 / 0.135 (0.135) |
| SOAP, final decay (WSD) | 0.140 / 0.136 / 0.141 (0.139) | 0.138 / 0.132 / 0.134 (0.135) |
| Brake (momentum kept, c = 4; reference) | 0.164 / 0.154 / 0.155 (0.158) | 0.131 / 0.121 / 0.122 (0.125) |
| Top-free + WSD (exploratory) | 0.133 / 0.117 / 0.127 (0.126) | 0.129 / 0.116 / 0.124 (0.123) |

- **Size of the effect:** 0.014 lower loss than SOAP + EMA (about 10%); the smallest per-seed margin is 0.007.
- **More damping is better again.** Seed-0 tuning chose c = 64 over c = 16 (EMA loss 0.112 vs 0.125), continuing
  the trend 1 < 4 < 16 < 64 from the noise controls. c = 64 was the largest value tried, so the optimum may lie
  higher.
- **Sharper trajectories.** The top-free runs move into much sharper regions: full-network preconditioned
  sharpness 27–59 against SOAP's 10–23, still below the stability limit 78. Validation loss is lower anyway. This
  is the self-stabilization risk named in the proposal; generalization outside this validation set was not
  checked.

**Interpretation.** Removing momentum in the 16 sharpest whitened directions per tensor and damping them strongly
(effective step lr/(1 + 64·lr·λ), so lr'·λ ≤ 1/64) keeps those coordinates near the valley floor *during*
training. Averaging the weights only cleans the evaluated copy. The gain over SOAP + EMA shows that the jitter on
the hills also hurts the training trajectory itself, not only the final evaluation.

**Next step:**
- Extend the c grid (64, 256, ∞) and test whether a per-direction Kalman gain beats one fixed c.
- Run the offline Ornstein–Uhlenbeck check on the seed-4 checkpoints, saved every 10 steps for SOAP, the brake
  and the top-free brake.
- Then test on longer runs and other models.

## 2. Readings

| Code | Comparison (step 500, seeds 4–6, EMA d = 0.98 for every arm, chosen on seed 0) | Result |
| --- | --- | --- |
| P1 | Top-free + EMA vs SOAP constant + EMA | **better** (0.121 vs 0.135) |
| P2 | Top-free + EMA vs SOAP WSD + EMA | **better** (0.121 vs 0.135) |
| Reported | Top-free + EMA vs brake + EMA | better (0.121 vs 0.125) |
| Reported | Top-free raw vs SOAP raw | better (0.142 vs 0.167) |
| Reported | Top-free WSD + EMA vs SOAP WSD + EMA | better (0.123 vs 0.135) |

Seed-0 EMA losses at step 500 (d = 0.9 / 0.98 / 0.995):

| Arm | d = 0.9 | d = 0.98 | d = 0.995 |
| --- | ---: | ---: | ---: |
| SOAP constant | 0.144 | 0.136 | 0.170 |
| SOAP WSD | 0.141 | 0.137 | 0.170 |
| Brake | 0.128 | 0.121 | 0.146 |
| Top-free c = 16 | 0.135 | 0.125 | 0.144 |
| Top-free c = 64 | 0.118 | **0.112** | 0.127 |

## 3. Sharpness

lr·λ_max(P̂H), full network, steps 10 / 25 / 50 / 100 / 200 / 300 / 400 / 500:

| Arm | Seed 4 | Seed 5 | Seed 6 |
| --- | --- | --- | --- |
| SOAP | 12 16 20 16 15 13 10 12 | 30 15 21 19 15 15 12 13 | 21 15 23 14 10 17 14 12 |
| Brake | 26 25 29 22 14 17 15 18 | 32 28 37 30 24 20 22 15 | 41 18 19 29 18 23 19 23 |
| Top-free | 11 21 39 39 31 43 28 27 | 22 29 59 47 39 40 30 33 | 11 22 42 41 34 28 30 35 |

## 4. Limitations

- One workload (ViT width 16, depth 2, MNIST), 500 steps, three evaluation seeds; c chosen on seed 0 at the edge
  of a two-point grid.
- Wall-clock is not compared: the top-free brake has the brake's cost, about 2× SOAP per step on this CPU workload.
- Sharper solutions may generalize differently out of distribution; not tested.
- The OU-model check on the saved checkpoints has not been run yet.

## 5. Cost, isolation, suites

- Wall time 13:56–14:10Z with four workers.
- Isolation manifests before and after are identical (3,123 files); no new `__pycache__`.
- Repository suite 106 passed before and after; study tests (1 harness, 22 PARSEC-L) passed.
- Checkpoints: 50 per diagnostic unit, about 156 MB, external only.
