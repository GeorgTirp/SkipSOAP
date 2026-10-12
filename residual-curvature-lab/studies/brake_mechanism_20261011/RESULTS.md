# Why does the PARSEC-L brake help? — results

Protocol [PROTOCOL.md](PROTOCOL.md) (hash in `protocol_sha256.txt`, written before the run). Script
`scripts/mechanism.py`; output `results/mechanism.json`. Offline on banked data, about four minutes on one CPU
thread.

## Summary

- **Not edge of stability.** Along tuned SOAP's trajectory (lr 1e-2, gap-audit seeds 1–3), the full-network
  preconditioned sharpness lr·λ_max(P̂H) is:
  - 16–27 at step 50, 11–15 at step 200, 10–14 at step 500 (the GGN gives the same values to within about 1);
  - 13–35% of the EMA-momentum stability limit 2(1+β₁)/(1−β₁) = 78;
  - falling during training.

  By the pre-declared rule this is *intermediate*: not at the edge (≥ 58), and not ≤ 20 at every checkpoint.
  The per-tensor block maximum, which stage D logged as ηλ ≈ 4, is 5–8 times smaller than the full value.
- **Gradient noise is concentrated in the braked directions.** In SOAP's whitened coordinates, the 16 sharpest
  directions per tensor (11% of dimensions) carry 57–70% of the gradient-noise variance at every checkpoint.
- **The brake costs those directions no speed.**
  - Under SOAP's momentum (β₁ = 0.95), a direction is underdamped for lr·λ ∈ (0.013, 78). There it converges at
    √β₁ ≈ 0.975 per step whatever the learning rate is.
  - 65–77% of the braked directions stay underdamped after braking.
  - In those directions a smaller learning rate lowers only the stationary noise excess, about lr·σ²/4 per
    direction.
- **Noisy-quadratic estimate.** At frozen states, SOAP's noise-induced excess loss is 0.05–0.11, of which the
  top-16 directions carry 0.03–0.07. The brake (c = 4) removes about 0.019–0.043 of it.
- **Optimization effect.** On identical training batches, the brake's training-batch loss is lower than SOAP's
  in every seed and window. Brake minus SOAP: −0.09 to −0.13 in steps 1–50, −0.02 to −0.08 in 51–200, and
  −0.004 to −0.018 in 201–500. SOAP's own late training loss is 0.155–0.171.

**Interpretation.** The brake acts as *selective learning-rate annealing*. With heavy momentum, SOAP's sharp
whitened directions are underdamped: they equilibrate at a rate set by β₁, not by the learning rate, and their
excess loss is noise ∝ lr. The noise is concentrated exactly there. Lowering the learning rate only in those
directions removes that excess, while the flat bulk, where the learning rate does set the speed, keeps SOAP's
full step.

This is the effect a global learning-rate decay produces at the end of training, without slowing the bulk.
Edge-of-stability work describes a related loss excess from oscillation in the top directions, but here it is
not caused by instability.

The early gap (steps 1–50) is larger than the stationary noise estimate. This suggests an additional transient
effect: braking also damps the ringing of underdamped sharp directions after large early gradients. It was not
tested separately.

## Readings

| Pre-declared reading | Result |
| --- | --- |
| A. At adaptive EoS (S_Hessian ≥ 58 at steps 200 and 500, all seeds) | No |
| A. Far below (S_Hessian ≤ 20 at every checkpoint) | No (21.9 and 27.2 at step 50): *intermediate*, 13–35% of the limit |
| B. Noise concentrated (top-16 share ≥ 30% at every checkpoint) | **Yes** (57–70%) |
| C. Optimization effect (brake training loss lower in steps 201–500, all seeds) | **Yes** (−0.018, −0.015, −0.004) |

## Predictions that follow (untested)

1. A learning-rate decay schedule for SOAP removes much of the same excess, so the brake's advantage at the final
   step shrinks against decayed SOAP. Its advantage should persist at intermediate steps, and the two may add.
2. Larger batches (less noise) shrink the brake's gain; smaller batches enlarge it.
3. Removing momentum in the braked top directions only (critically damped instead of underdamped, stable because
   of the brake) should speed them up without raising their noise floor.

## Limitations

Frozen-state quadratic model with stationary noise; curvature and noise measured on a 512-example held-out
batch; true-label noise (not the probes' model Fisher); one workload; checkpoints only at steps 50, 200 and 500.
