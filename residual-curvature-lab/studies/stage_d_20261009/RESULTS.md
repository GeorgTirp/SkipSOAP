# Stage D: PARSEC-L training test — results

Study `stage_d_20261009`, protocol [PROTOCOL.md](PROTOCOL.md) (approved 10 October; SHA-256 in
`logs/protocol_sha256.txt`), deviations in [DEVIATIONS.md](DEVIATIONS.md). External runs:
`~/skipsoap-study-data/stage_d_20261009/runs/`:
- tuning `20261010T171813-415c766d`;
- tuning-independent evaluation `20261010T171814-46f283ad`;
- tuned evaluation `20261010T180135-101694d8`.

Light mirrors are in [runs/](runs/). Figure: [stage_d.png](figures/stage_d.png). Summary:
`runs/20261010T171814-46f283ad/stage_d_summary.json`.

**Status (2026-10-11):** 39 of 40 units completed with no divergence. One unit (SOAP lr 1e-2, seed 1)
stopped on the pre-declared reproduction gate (DEVIATIONS #2). An earlier launch was aborted over a logging
bug and relaunched (DEVIATIONS #1).

## 1. Summary

> **Qualification (11 October, `studies/noise_controls_20261011`):** every baseline here, including SOAP, ran at a
> constant learning rate without weight averaging. Against SOAP with a 100-step final decay (0.129) or SOAP
> evaluated on an EMA of its weights (0.127), the brake (0.149) is worse in all seeds. With decay or averaging
> added, the brake's extra effect is mixed. Its gain is noise removal, which those standard techniques already
> provide. The readings below stand as stated, for constant-lr training.


- **The PARSEC-L brake works.** Learning-rate calibration with c = 4 at SOAP's own learning rate 1e-2 beats
  tuned SOAP at step 500 in all three seeds: 0.149 ± 0.006 against 0.161 ± 0.010; per seed 0.145 / 0.156 /
  0.146 against 0.172 / 0.157 / 0.153.
  - It is also lower at step 50 in all seeds: 0.802 against 0.904.
  - Accuracy at step 500 is 95.7% against 95.2%.
  - It reaches the gap-audit targets in fewer steps: T50 117 against 142, T100 158 against 192.
  - Its step-500 loss is the lowest of every optimizer run on these seeds.
- **Early-Newton does not work as designed.** The PARSEC-L direction with its step length chosen on a 128-example
  selection batch is *mixed* against SOAP at steps 50 and 100, with and without hand-over.
  - Two seeds are much faster early (0.53 / 0.57 against 0.88 / 0.82 at step 50); seed 2 is much worse
    (1.38 against 1.01).
  - At step 500, the no-hand-over variant is worse than SOAP in all three seeds. The hand-over variant is worse
    in two of three (seed 1: 0.165 against 0.172). Means: 0.215 and 0.210 against 0.161.
  - The model-free hand-over never triggered: PARSEC-L won the selection in 75–88% of steps, but SOAP never
    won ten steps in a row.
- **Exact GGN with a selected step improves late but does not remove the collapse.** It is better than the
  gap audit's same-batch GGN at step 200 in all seeds (0.225 against 0.236) and *mixed* at step 500 (0.179
  against 0.182). It is worse early (0.397 against 0.355 at step 50).
  - Its next-batch transfer after step 50 is still under 1%.
  - It does not reach native SOAP's step-500 loss.
- **Wall-clock time:** native SOAP stays fastest to the targets on this CPU workload. With sequential costs
  from the dry run (SOAP 0.044 s/step, brake 0.087), SOAP reaches T50 in about 6 s and the brake in about
  10 s. The brake's gain is in final loss and steps, not in seconds to a fixed target.

| Seeds 1–3, mean ± sd | Loss 50 | Loss 100 | Loss 200 | Loss 500 | Acc. 500 | T50 / T100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **PARSEC-L brake** (lr-calibrated, c = 4, lr 1e-2) | 0.802 ± 0.101 | 0.385 ± 0.040 | **0.214 ± 0.016** | **0.149 ± 0.006** | **95.7%** | 117 / 158 |
| Native SOAP, lr 1e-2 (gap audit) | 0.904 ± 0.098 | 0.431 ± 0.010 | 0.236 ± 0.029 | 0.161 ± 0.010 | 95.2% | 142 / 192 |
| Tensor OT, γ 0.1 (geometry follow-up) | 0.451 ± 0.051 | 0.297 ± 0.027 | 0.219 ± 0.008 | 0.165 ± 0.006 | 95.1% | 83 / 142 |
| SOAP, lr 2e-2 / 3e-2 (this study) | 0.869 / 1.012 | 0.440 / 0.502 | 0.253 / 0.266 | 0.177 / 0.177 | — | 142 / 192, 150 / 217 |
| Exact tensor GGN, selected step (this study) | 0.397 ± 0.027 | 0.292 ± 0.018 | 0.225 ± 0.011 | 0.179 ± 0.004 | 94.5% | 83 / 142 |
| Exact tensor GGN, same-batch search (gap audit) | **0.355 ± 0.011** | **0.265 ± 0.018** | 0.236 ± 0.008 | 0.182 ± 0.002 | 94.5% | **58 / 142** |
| Original PARSEC, default / split (geometry) | 1.380 / 1.077 | 0.745 / 0.611 | 0.357 / 0.342 | 0.186 / 0.192 | 94.7 / 94.3% | 200 / 275, 192 / 300 |
| Adam, lr 1e-2 (gap audit) | 1.113 ± 0.212 | 0.650 ± 0.092 | 0.312 ± 0.040 | 0.202 ± 0.011 | 93.9% | 175 / 275 |
| Searched SOAP, tuned (gap audit) | 0.697 ± 0.064 | 0.368 ± 0.045 | 0.262 ± 0.019 | 0.210 ± 0.007 | 93.8% | 108 / 200 |
| PARSEC-L Early-Newton, no hand-over | 0.827 ± 0.479 | 0.500 ± 0.257 | 0.315 ± 0.131 | 0.215 ± 0.037 | 93.7% | 167 / 292 |
| PARSEC-L Early-Newton, hand-over | 0.837 ± 0.495 | 0.512 ± 0.316 | 0.306 ± 0.159 | 0.210 ± 0.048 | 93.8% | 183 / 283 |
| Dense empirical Fisher (gap audit) | 0.705 ± 0.170 | 0.499 ± 0.035 | 0.411 ± 0.045 | 0.306 ± 0.093 | 92.2% | 2 of 3 reach T50 |
| Full GGN (geometry) | 0.670 ± 0.083 | 0.568 ± 0.108 | 0.479 ± 0.038 | 0.330 ± 0.035 | 90.6% | 442 / — |

All rows use the same seeds, data order, batch 512 and validation set. Earlier rows come from the stored gap-audit
and geometry follow-up runs. The Early-Newton and GGN-with-selection arms also use 64,000 selection examples
(128 per step) on top of 256,000 gradient examples.

**Next step:** make learning-rate calibration (c ≈ 4) the default PARSEC-L configuration. Then test it where
the per-step cost can pay: longer runs and larger, GPU-batched models, where SOAP's own step dominates and the
probe pass is amortized. Early-Newton needs a more stable step-length rule before it is worth another run.

## 2. Pre-declared readings

Seeds 1–3; settings chosen on seed 0. Better or worse means lower or higher validation loss in all three seeds.

| Reading | Result |
| --- | --- |
| R1. Brake (selected: lr calibration, c = 4, lr 1e-2) vs tuned SOAP at step 500 | **Better** |
| R2. Early-Newton, no hand-over, vs SOAP at steps 50 / 100 | mixed / mixed |
| R2. Early-Newton, hand-over, vs SOAP at steps 50 / 100 | mixed / mixed |
| R3. GGN with selected step vs same-batch GGN at steps 200 / 500 | **better** / mixed |

**SOAP baseline.** The readings use the gap audit's native SOAP at lr 1e-2, the reference of the pre-declared
gate (DEVIATIONS #2). This study's own SOAP at lr 1e-2 matched it within 2.0e-8 (seed 2) and 2.5e-8 (seed 3),
and within 2.2e-6 for seed 1 until that unit stopped at step 300. Its seed-2 and seed-3 values (0.157 / 0.153 at
step 500) do not change any reading.

## 3. Seed-0 tuning

Step-500 validation loss on seed 0:

| Configuration | lr 1e-2 | lr 2e-2 | lr 3e-2 |
| --- | ---: | ---: | ---: |
| SOAP | 0.171 | 0.193 | 0.231 |
| Brake, lr calibration c = 1 | 0.171 | 0.167 | 0.215 |
| **Brake, lr calibration c = 4** | **0.147** | 0.180 | 0.175 |
| Brake, lr calibration c = 30 | 0.164 | 0.178 | 0.178 |
| Brake, damping calibration γ 0.01 | 0.158 | 0.168 | 0.161 |

Early-Newton on seed 0, step-100 loss averaged over the hand-over variants: momentum 0.385, gradient 0.557.
Momentum was selected.

- **The brake does not move the optimal learning rate.** Its best configuration sits at SOAP's own optimum.
- **It does stabilize larger learning rates.** At lr 3e-2 every brake beats SOAP at the same rate
  (0.161–0.215 against 0.231).
- **c = 1** is almost plain SOAP at lr 1e-2 (0.171).
- **c = 4 is best.** Removing the sharp directions more strongly is worse but still better than SOAP: c = 30
  gives 0.164 and damping calibration 0.158.

## 4. Diagnostics (exploratory)

Medians over seeds 1–3 in steps 1–50 / 51–200 / 201–500.

**Sharpest directions.** In the brake arm, ηλ_max (SOAP's step size times the largest whitened GGN
eigenvalue per tensor) is 3.8 / 4.1 / 2.6. With c = 4 the brake shrinks those directions to about
1/(1 + 4·ηλ) ≈ 1/17 of SOAP's step.

*Correction (11 October).* The first version of this report called ηλ ≈ 4 "near the heavy-ball stability limit
2(1 + β₁) ≈ 3.9" and named edge of stability as the likely mechanism. That limit applies to heavy-ball momentum
written with the raw gradient. For SOAP's EMA momentum the linear stability limit is
lr·λ < 2(1 + β₁)/(1 − β₁) = 78.
- **Measured:** the full-network preconditioned sharpness along SOAP's trajectory is 10–27, not at the edge
  ([../brake_mechanism_20261011/RESULTS.md](../brake_mechanism_20261011/RESULTS.md)).
- **What that study supports instead:** the brake removes gradient noise concentrated in underdamped sharp
  directions, whose speed does not depend on the learning rate.

**Optimism ratio r̂ and transfer.**

| Arm | r̂ (SOAP metric) | Next-batch transfer (paired) |
| --- | --- | --- |
| SOAP | 0.59 / 1.11 / 1.52 | 75% / 23% / −2% |
| Brake | 0.75 / 1.71 / 2.21 | 76% / 23% / 0.4% |
| Early-Newton | — | 55–56% / 12–15% / about 0% |
| GGN, selected step | 1.2 / 2.8 / 2.8 (own operator 5.6 / 15 / 16) | 29% / 0.8% / −0.2% |

- SOAP's own r̂ crosses 1 in steps 51–200 (median 1.11), and its transfer falls to zero after step 200, as
  the theory predicts.
- The brake's optimism ratio is higher (2.3–2.9 in its own operator) without lowering its transfer
  relative to SOAP's.
- The 128-example selection batch did not restore GGN's late transfer.

**Early-Newton.** Median selected α is 4.5–4.8 times the base step. The hand-over variant's PARSEC-L
candidate won 75–88% of selections.

## 5. Observations, hypotheses and limitations

Observations:
1. A mild, learning-rate-calibrated shrinkage of SOAP's sharpest whitened directions (c = 4) improves SOAP
   at its tuned learning rate throughout training, without any search.
2. Choosing a large step length every step on a small selection batch helps early in two of three seeds but is
   unstable and ends worse.
3. Out-of-batch step selection reduces GGN's late degradation only modestly.

Hypotheses (not tested):
- (a) The brake works by keeping SOAP's sharpest directions below the edge of stability. *Not supported* by
  the follow-up (correction above). The supported mechanism predicts that the gain grows with the learning
  rate and with gradient noise (smaller batches), and shrinks under a learning-rate decay schedule.
- (b) Early-Newton's instability comes from the noisy 128-example selection and from steps 4–5× SOAP's.
  A larger selection batch, a capped α, or selection only every k steps may fix it.

Limitations:
- One workload (ViT width 16, depth 2, MNIST), 500 steps, three seeds; settings chosen on seed 0 only.
- The reported c = 4 is the best of 12 seed-0 configurations.
- Measured wall-clock varies up to 2.5× between seeds of the same arm (core scheduling under four workers), so
  wall-clock comparisons use the sequential dry-run costs.
- Training stops at 500 steps; the brake's per-second advantage at equal compute is not measured.

## 6. Cost, gates, isolation and suites

- Wall time 17:18–18:23Z with four workers; caffeinate attached.
- Dry run (sequential s/step): SOAP 0.044, brake 0.087 (0.46 on probe steps), Early-Newton about 0.2,
  GGN with selection 3.3.
- Gates:
  - SOAP reproduction: seeds 2 and 3 passed (2.0e-8, 2.5e-8); seed 1 failed at 2.2e-6 (DEVIATIONS #2).
  - GGN action checks and solve residuals passed at all audit steps.
  - No divergence.
- Isolation manifests before (17:15Z) and after (12:24Z next day, 2,400 files): identical. No new
  `__pycache__`.
- Repository suite: 106 passed before and after. Study tests: 4 harness and 21 PARSEC-L, passed before and after.
- Light mirrors 9.6 MB.

## 7. Reproduction

From `residual-curvature-lab/`, with `export PYTHONDONTWRITEBYTECODE=1`:

```sh
S=studies/stage_d_20261009/scripts
.venv/bin/python -m pytest -q -p no:cacheprovider studies/stage_d_20261009/tests studies/parsec_l_20261009/tests
.venv/bin/python $S/train.py dryrun
T=$(.venv/bin/python $S/train.py create --phase tune); F=$(.venv/bin/python $S/train.py create --phase eval_fixed)
# queue all units of `train.py pending --run $T` and `--run $F` (queue_runner.sh, 4 workers); orchestrate.sh $T QUEUE
# finalizes the tuning run, creates eval_tuned (--tuning $T) and queues it
.venv/bin/python $S/summarize.py $T $F <eval_tuned run>
```
