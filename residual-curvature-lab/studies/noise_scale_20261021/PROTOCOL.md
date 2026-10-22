# Noise scales of preconditioned steps, break time and fractional Newton along CIFAR training — protocol

Study `noise_scale_20261021`. Written on 21 October 2026, before any run. Offline: bitwise replays of existing runs,
no new training.

**Request.** The owner approved on 21 October ("Yes please run that") the offline test of predictions 1 and 2 of
[ideas/batch_noise.md](../../ideas/batch_noise.md).
- **Break time:** do the noise scales of curvature-preconditioned steps grow during training, so that at the
  training batch size B = 2,048 stronger preconditioning stops paying at a predictable stage?
- **Fractional Newton:** is the best power of the curvature between SOAP's whitened step (ν = 0) and the exact
  per-tensor GGN step (ν = 1), and does it fall during training?

## 1. Theory under test (batch_noise.md, T1–T2)

**Setting.** In SOAP-whitened coordinates, per matrix tensor b:
- A_b = K_b + δI, with the whitened GGN block K_b and the ladder's damping δ = μ·scale;
- the step family d_ν = −A^{−ν} z, with ν ∈ [0, 1].

**Quantities.** With z the corpus gradient and Cov(g_B) the covariance of a batch-B gradient (sampling without
replacement from N = 45,000):
- Φ_ν = (zᵀA^{−ν}z)² / (2 zᵀA^{1−2ν}z), the deterministic gain;
- N_ν(B) = tr(A^{1−2ν} Cov(g_B)) / zᵀA^{1−2ν}z, the noise-to-signal ratio at batch size B;
- K_ν = N_ν(B)/(1/B − 1/N), the noise scale in examples;
- Ψ_ν(B) = Φ_ν / (1 + N_ν(B)), the expected corpus progress with a tuned scalar step.

**Pooling over tensors.** All forms are summed over the 39 matrix tensors, as a block-diagonal model.

**Crossover.** For adjacent ν₁ < ν₂ on the grid {0, ¼, ½, ¾, 1}:
- G = Φ_{ν₂}/Φ_{ν₁};
- B_crit = [1/N + (G − 1)/(K_{ν₂} − G·K_{ν₁})]⁻¹ when K_{ν₂} > G·K_{ν₁}, and 0 otherwise (the stronger power wins
  at every B);
- B_required(t) = max over the four adjacent pairs of B_crit.

The predicted break at B = 2,048 is the first measured state with B_required ≥ 2,048.

## 2. Trajectories and states

- **Runs:** `cifar_ggn_20261012`'s `soap__lr0.01__s{1,2,3}` and `topfree__lr0.03__s{1,2,3}`, replayed bitwise from
  step 0 with `ladder_mechanism`'s `Replay`.
- **States:** steps 10, 25, 50, 100, 200 and 400 (36 states).
- **Batches,** taken without consuming the replay's order or augmentation:
  - the step batch is the next training batch, augmented as training would augment it;
  - the selection and evaluation batches are the two batches after it, unaugmented, as in `ladder_mechanism`.
- **Coordinates:** whitened with the SOAP contexts at the state (copied).

## 3. Measurements per state

- **Damping:** δ = μ·scale with μ = max(1e-2·tr C/D, 1e-8), where tr C is the exact GGN trace of the step batch (as
  in `ladder_mechanism`).
- **Corpus gradient and gradient noise:**
  - **Partition:** a fixed permutation (seed 20261021) of the training positions, cut into m = 43 disjoint batches
    of b = 1,024 (44,032 examples), unaugmented.
  - **Corpus gradient:** z = the mean of the 43 whitened batch gradients.
  - **Noise covariance:** Cov(g_b) is estimated by (1/m) Σ_j (g_j − z)(g_j − z)ᵀ and scaled to B by
    (1/B − 1/N)/(1/b − 1/N), which is exact for sampling without replacement.
- **Curvature:** dense K_b from 4,096 unaugmented reference examples (`spectral_rotation`'s batches A and B).
- **Quadratic forms:** zᵀA^q z and (1/m) Σ_j ξ_jᵀA^q ξ_j for powers q ∈ {−1, −0.95, …, 1}, by batched Lanczos
  quadrature.
  - Settings: 300 steps with full reorthogonalization; float32 products, float64 tridiagonal eigendecompositions.
  - The same runs give the steps A^{−ν}z_B of the step batch, for ν ∈ {¼, ½, ¾, 1}.
- **Steps:** v_ν is the unwhitened y_ν = −A^{−ν}z_B on the matrix tensors, zero elsewhere.
  - Step size: α = α₀·2^j, j = −14…2, with α₀ = −Σ z_Bᵀy/Σ yᵀAy, the block-model optimum.
  - Selection on the selection batch; scores on the evaluation batch (primary) and on the 5,000 validation images
    (descriptive).
- **Theory values:**
  - Φ_ν, N_ν(2,048), K_ν, Ψ_ν(2,048) and B_crit on the grid;
  - the theory's best ν, both on the grid and on the fine 0.05 grid.

## 4. Readings

| Code | Reading |
| --- | --- |
| N1 | **Noise scales grow with the power:** pooled K_1 > K_½ > K_0 in every state (36 of 36). |
| N2 | **Noise scales grow with training:** Spearman(step, K_1) ≥ 0.8 in each of the 6 trajectories. |
| N3 | **A break within training at B = 2,048:** B_required(10) < 2,048 ≤ B_required(400) in each of the 6 trajectories. The first state with B_required ≥ 2,048 (t_break) is reported. |
| F1 | **Interior optimum:** the evaluation-batch-best ν (each at its selected α) is ¼, ½ or ¾ in ≥ 24 of 36 states. |
| F2 | **The optimum falls during training:** the seed-mean best ν over steps 10, 25 and 50 exceeds that over steps 100, 200 and 400, for both optimizers. |
| P1 | **Theory against measurement:** the theory's grid argmax of Ψ_ν(2,048) lies within one grid step of the evaluation-best ν in ≥ 24 of 36 states. |
| D | Descriptive: Φ_ν, G, K_ν, N_ν, B_crit per pair, the fine-grid theory optimum, validation decreases, selected multipliers, per-kind breakdowns. |

## 5. Gates

- **G0:** suites before and after, including this study's tests:
  - Lanczos quadrature and matrix functions against exact eigendecompositions (float64);
  - breakdown handling;
  - the Φ/K/Ψ formulas against direct expectations;
  - B_crit and the toy optimum against numerical searches;
  - non-consuming batch peeks against `Replay.next_batch`.
- **G1:** each replay equals the run's checkpoint parameters bitwise at steps 50, 100 and 250. A failure stops the
  study.
- **G2:**
  - **Check:** at steps 10 and 400 of every trajectory, for block 0's query, key, value and output weights, every
    Lanczos quadratic form used must agree with a float64 eigendecomposition of the same float32 matrix to a
    relative 1e-2. A failure stops the study.
  - **Tolerance:** it reflects float32 products with condition numbers up to about 10⁴ (rule 12). It guards
    against non-convergence, which would show as errors far above it.
- **Smoke:** before launch, one GPU state is run outside the study's outputs to check memory, time and G2. It is
  not a result. If G2 fails there, the Lanczos step count is raised before launch and the change is recorded in
  DEVIATIONS.md.

## 6. Cost, isolation and limitations

- **Cost:** about 2.5–3 min per state (dense curvature on 4,096 examples, 44 gradients, Lanczos) for 36 states,
  plus replays: **about 2 GPU-hours**, run detached.
- **Writes:** only to `studies/noise_scale_20261021/` and `~/skipsoap-study-data/noise_scale_20261021/`.
  `cifar_ggn` checkpoints, `ladder_mechanism` code and `spectral_rotation`'s batch definitions are read only.
- **Limitations:**
  - **Model and runs:** one model, two optimizers, 500-step runs (states up to 400).
  - **Curvature:** "corpus" curvature from 4,096 examples, in a block-diagonal per-tensor model of the matrix
    tensors only.
  - **Noise estimate:** from 43 batch gradients, so the noise forms have sampling error of order sqrt(2/(43·r_eff)).
  - **Theory:** it assumes a step size tuned in expectation and equal curvature. `transfer_law` (N20) showed that
    the equal-curvature assumption overstates the noise cost of strong preconditioning under same-batch search.
  - **Scope:** one-step scores, not training.
