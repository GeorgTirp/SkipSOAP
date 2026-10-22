# Noise scales of preconditioned steps, break time and fractional Newton along CIFAR training — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, fixed before any run. One deviation, before
launch: [DEVIATIONS.md](DEVIATIONS.md) D1. The pre-launch smoke ran twice, and the Lanczos basis layout was changed for
speed (same arithmetic, tests unchanged).

Outputs:
- `results/noise/{soap,topfree}_s{1,2,3}.json`, copies of `~/skipsoap-study-data/noise_scale_20261021/noise/`;
- `results/summary.json` / `summary.md` (readings), from `scripts/summarize.py`;
- `results/descriptive.json` / `descriptive.md` (the declared descriptive tables D), from `scripts/describe.py`.
  That script was written after the run. It reuses `summarize.py`'s theory functions and checks its pooled values
  against `summary.json` exactly.

**Status (2026-10-21):** all six trajectories (36 states) completed, in 704–762 s each (78 min in all, peak GPU
memory 3.8–4.1 GiB).
- **G1:** each replay equals its run's checkpoint parameters bitwise at steps 50, 100 and 250 (18 of 18).
- **G2:** the worst relative error of a Lanczos form against the float64 eigendecomposition is 1.2e-4 at step 10
  and 3.1e-5 at step 400, against the 1e-2 tolerance.
- **Suites:** passed before and after (17 suites, including this study's 8 tests).
- **Isolation manifest:** unchanged.
- **Lanczos sanity:** the smallest Ritz value of any A is 0.9987·δ. In exact arithmetic it is at least δ; the
  difference is float32 rounding in K.

**Conventions.**
- All forms are summed over the 39 matrix tensors (block-diagonal model). "Early" means steps 10, 25 and 50;
  "late" means steps 100, 200 and 400. Ranges are across seeds 1–3.
- A "decrease" is the one-step loss decrease on the evaluation batch. That is the second batch after the step batch,
  unaugmented. Each power uses its step size selected on the selection batch.
- ν = 0 is SOAP's preconditioned direction without momentum; ν = 1 is the exact damped per-tensor GGN step.

## 1. Summary

- **Noise scales grow about tenfold during training, and they grow with the power of the curvature.**
  - Newton's noise scale K₁ rises from 1,340–2,170 examples at step 10 to 17,300–20,400 at step 400. N2 holds:
    Spearman 0.94–1.0 in every trajectory.
  - At B = 2,048 the Newton step's noise-to-signal ratio N₁ grows from 0.63–1.01 to 8.1–9.5.
  - K₁ > K½ holds in all 36 states. K½ > K₀ fails in two, top-free seeds 1 and 3 at step 10, by 7–9%. **N1 (36
    of 36) is therefore not supported.**
- **The break comes early (N3 holds).**
  - B = 2,048 stops being enough for the full Newton power at step 25 in five trajectories and at step 50 in the
    sixth (top-free seed 2).
  - The required batch size rises from 1,380–1,770 at step 10 to about 10⁴ from step 100 on (9,170–11,900), then
    levels off.
  - The binding pair is always ¾→1.
- **The best power is interior, and it falls during training (F1 and F2 hold).**
  - It is interior in 35 of 36 states. The exception is SOAP seed 1 at step 100, where ν = 0 is best.
  - The seed-mean best power falls from 0.72 to 0.58 for top-free and from 0.67 to 0.42 for SOAP (early to late).
  - Pairwise, ¾ beats 1 in 35 of 36 states and ½ beats 0 in 35 of 36. Early, ¾ beats ½ in 14 of 18 states; late,
    ½ beats ¾ in 15 of 18.
  - The validation set picks the same best power as the evaluation batch in 31 of 36 states.
- **The theory places the optimum about one grid step too high and overstates the gains.**
  - P1 holds: 32 of 36 states are within one grid step. But the theory's fine-grid optimum lies above the measured
    best in 31 of 36 states, by 0.165 on average. It falls from 0.9 at step 10 to 0.65–0.7 at steps 200–400; the
    measured best falls from ¾ early to ½ late.
  - Medians of the predicted gain over ν = 0 against the measured ratio of decreases (ranges over the two
    optimizers):

    | Power | Phase | Predicted Ψ_ν/Ψ₀ | Measured decrease ratio |
    | --- | --- | --- | --- |
    | ½ | early | 3.3–5.6 | 1.5–2.2 |
    | ½ | late | 3.1–3.5 | 1.3–2.0 |
    | 1 | early | 6.8–12 | 0.94–1.29 |
    | 1 | late | 2.5–2.8 | 0.63–1.44 |
- **One step ahead, a fractional power still beats SOAP's direction late in training.** At steps 100–400, ν = ½
  decreases the evaluation loss 1.3× (SOAP) to 2.0× (top-free) as much as ν = 0 (medians).

## 2. Readings

| Code | Result | Values |
| --- | --- | --- |
| N1 noise scales grow with the power, 36 of 36 | **not supported** | K₁ > K½ in 36 of 36. K½ > K₀ in 34 of 36; it fails at top-free s1 t10 (270 vs 289) and s3 t10 (399 vs 437) |
| N2 noise scales grow with training | **holds** | Spearman(step, K₁) is 0.943 in two trajectories and 1.0 in four |
| N3 a break within training at B = 2,048 | **holds** | B_required(10) is 1,380–1,770 < 2,048 ≤ B_required(400) of 9,650–10,700, in 6 of 6. t_break is step 25 in five trajectories and step 50 in top-free s2 |
| F1 interior optimum, ≥ 24 of 36 | **holds** | 35 of 36. Exception: SOAP s1 t100 (ν = 0) |
| F2 the optimum falls during training | **holds** | top-free 0.72 → 0.58; SOAP 0.67 → 0.42 |
| P1 theory within one grid step, ≥ 24 of 36 | **holds** | 32 of 36. Misses: SOAP s1 t10 and s3 t10 (theory 1, measured ½), SOAP s1 t100 (¾ vs 0), SOAP s2 t400 (¾ vs ¼) |

P1 is a loose test here. The theory's grid optimum is ¾ or 1 in every state, while the measured best is mostly ½ or
¾. The bias described in section 1 is systematic.

## 3. Descriptive (D)

Per step (ranges across seeds; the best powers are listed for seeds 1/2/3):

| Arm | Step | K₁ | N₁(2,048) | Φ₁/Φ₀ | B_required | Theory ν* (fine) | Best ν, evaluation | Best ν, validation |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| top-free | 10 | 1,340–1,800 | 0.63–0.84 | 10.9–26.8 | 1,380–1,660 | 0.9 | ¾ ¾ ¾ | ¾ ¾ ¾ |
| top-free | 25 | 2,630–3,240 | 1.2–1.5 | 30–45 | 1,770–2,400 | 0.85 | ¾ ¾ ¾ | ¾ ¾ ¾ |
| top-free | 50 | 2,350–6,740 | 1.1–3.1 | 6.8–10.6 | 4,040–6,700 | 0.75–0.85 | ½ ¾ ¾ | ½ ¾ ¾ |
| top-free | 100 | 11,200–13,200 | 5.2–6.1 | 15.5–21.7 | 9,710–10,500 | 0.7 | ¾ ¾ ¾ | ¾ ¾ ¾ |
| top-free | 200 | 14,700–16,500 | 6.9–7.7 | 15.7–17.7 | 10,900–11,900 | 0.65 | ½ ½ ½ | ½ ½ ½ |
| top-free | 400 | 18,600–20,100 | 8.7–9.4 | 23.1–33.6 | 10,600–10,700 | 0.65–0.7 | ½ ½ ½ | ½ ½ ½ |
| SOAP | 10 | 1,800–2,170 | 0.84–1.01 | 11.7–16.2 | 1,520–1,770 | 0.9 | ½ ¾ ½ | ¾ ½ ½ |
| SOAP | 25 | 3,280–3,920 | 1.5–1.8 | 14.3–21.2 | 2,990–3,220 | 0.85 | ¾ ¾ ¾ | ¾ ¾ ¾ |
| SOAP | 50 | 5,730–6,690 | 2.7–3.1 | 8.6–15.9 | 5,370–6,700 | 0.75–0.8 | ½ ¾ ¾ | ¾ ¾ ¾ |
| SOAP | 100 | 9,810–11,900 | 4.6–5.5 | 9.7–14.1 | 9,170–10,200 | 0.7 | 0 ½ ½ | ½ ½ ½ |
| SOAP | 200 | 15,700–18,800 | 7.3–8.8 | 16.6–22.1 | 10,600–11,700 | 0.65 | ½ ½ ½ | ¼ ½ ½ |
| SOAP | 400 | 17,300–20,400 | 8.1–9.5 | 20.7–26.6 | 9,650–9,870 | 0.7 | ½ ¼ ½ | ½ ¼ ½ |

- **K₀ and K½:**
  - K₀ is 16–878 and not monotone; it dips at step 50 (top-free 16–234, SOAP 59–218).
  - K½ grows from 106–399 at step 10 to 1,400–2,970 at step 400.
- **Deterministic gain:** Φ₁/Φ₀, the noiseless gain of the Newton step over SOAP's direction, is 6.8–45 and has no
  monotone trend. In the theory, the growth of the noise scales, not a loss of deterministic gain, moves the optimum.
- **Crossover per pair:**
  - ¾→1 (medians over seeds): 1,590–1,650 at step 10, 9,710–11,300 from step 100 on.
  - The other pairs stay low in every state: ½→¾ at most 1,580, ¼→½ at most 238, 0→¼ at most 7.
  - So by the theory, at B = 2,048 every step up to ¾ pays in every state; only ¾→1 can fail.
- **Step sizes:** none of the 180 selected multipliers lies at the edge of the grid.
- **Decreases** (medians over the nine states of each arm and phase, for ν = 0 / ¼ / ½ / ¾ / 1):

  | Arm, phase | ν = 0 / ¼ / ½ / ¾ / 1 |
  | --- | --- |
  | top-free, early | 0.021 / 0.032 / 0.048 / 0.066 / 0.043 |
  | top-free, late | 0.013 / 0.018 / 0.019 / 0.025 / 0.019 |
  | SOAP, early | 0.020 / 0.030 / 0.048 / 0.050 / 0.026 |
  | SOAP, late | 0.030 / 0.036 / 0.038 / 0.029 / 0.018 |

  - ν = 1 beats ν = 0 in only 20 of 36 states.
  - At top-free step 25, SOAP's direction barely decreases the evaluation loss (0.0005–0.0056), so ratios to ν = 0
    there are 6.5–89.
- **By tensor kind** (medians over seeds and both optimizers, step 10 → step 400):

  | Kind | K₁ | Φ₁/Φ₀ | Theory ν* |
  | --- | --- | --- | --- |
  | query + key | 1,510 → 17,500 | 18.2 → 8.4 | 0.9 → 0.6 |
  | value + output | 1,280 → 14,700 | 10.2 → 8.6 | 0.9 → 0.65 |
  | MLP w1 | 1,920 → 20,800 | 10.7 → 46.2 | 0.9 → 0.7 |
  | MLP w2 | 1,980 → 20,300 | 9.7 → 46.6 | 0.9 → 0.7 |
  | input | 3,820 → 17,600 | 20.9 → 12.9 | 0.8 → 0.65 |
  | head | 790 → 5,680 | 5.8 → 2.8 | 0.95 → 0.6 |

  The noise scale grows 10–12× in the attention and MLP weights, 7× in the head and 4.6× in the input layer. Late,
  the MLP weights carry the largest deterministic curvature gain.

## 4. Interpretation and limitations

- **For [batch_noise.md](../../ideas/batch_noise.md) T1–T2:**
  - **What the law gets right:** noise scales grow with the power and with training, and the optimal power is
    interior and falls.
  - **Where it is too optimistic** about strong preconditioning: its optimum is about ¼ too high. It overstates the
    gain over ν = 0 by 1.5–3.8× at ν = ½ and by 2–9× at ν = 1 (ratios of the medians in section 1).
  - The law treats the reference curvature as exact and the step size as tuned in expectation. This differs from
    N20's same-batch search, where the equal-curvature form overstated the harm. With a second-batch search it
    understates the cost of strong preconditioning.
  - **Candidate causes (hypotheses, untested):**
    - noise in the curvature estimate itself (4,096 reference examples), which A^{−ν} amplifies most in the bulk,
      where the eigenvalues are least accurate;
    - step-size selection on a single batch;
    - non-quadratic effects along long steps.
- **For the break time:**
  - At B = 2,048 the full Newton power is past its break from step 25–50 on. Theory and measurement agree on this:
    ¾ beats 1 in 35 of 36 states.
  - The break is gradual. The optimum slides from ¾ to ½; it does not collapse to SOAP's direction.
  - This is consistent with rule 9 (the exact GGN step wins early only), but it does not explain top-free's lost
    gain at 2,000 steps. Top-free is not a member of this family, and one step ahead, curvature beyond SOAP still
    pays at ν = ½ late.
- **Identity (what the family is in parameter coordinates):**
  - Let M = (F0F0ᵀ)⁻¹ be the inverse of SOAP's preconditioner. Then the whitened step −A^{−ν}z, unwhitened, is
    −(M #_ν (C + δM))⁻¹ g.
  - Here M #_ν X = M^{½}(M^{−½}XM^{−½})^ν M^{½} is the weighted matrix geometric mean: the affine-invariant geodesic
    from M (ν = 0) to the damped GGN C + δM (ν = 1).
  - The best one-step preconditioner of this family late in training is therefore the geometric mean (ν = ½) of
    SOAP's metric and the damped GGN, per tensor.
- **Not shown:**
  - any training benefit (all results are one step ahead, at fixed states);
  - momentum (ν = 0 is SOAP's direction without it);
  - that the step-size search generalizes beyond the selection batch.
- **Limitations:**
  - **Model and runs:** one model and two optimizers; 500-step runs, states up to step 400.
  - **Curvature:** from 4,096 reference examples rather than the corpus, in a block-diagonal model of the 39 matrix
    tensors.
  - **Noise:** from 43 batch gradients of 1,024.
  - **Scores:** one evaluation batch per state; validation (5,000 images) is descriptive.
