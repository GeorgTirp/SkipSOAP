# Batch noise, curvature and pooling: theory and results

Living document, started 21 October 2026. It collects the lab's theory and evidence on how minibatch noise interacts
with curvature-based steps and with the estimation of curvature itself. It is separate from
[parsec.md](parsec.md), which banks the PARSEC design rules; entries there that concern batch noise (N1, N16, N20,
N24, N27, N28, N29) are summarized and cited here.

**How to maintain this file.**
- **Labels:** every item is labeled *identity* (exact algebra), *approximation*, *observation* (measured, with
  its study) or *hypothesis* (untested).
- **Evidence levels:** results carry those of `parsec.md`: [T] training, fresh seeds; [L] local, at fixed states;
  [H] hypothesis.
- **Numbers:** verify every number against its RESULTS file and cite the study. Never delete a negative result.
  Add a changelog line for every change.

## 1. Setting and notation

- **Corpus and batches:** a corpus of N examples and batches of size B. The lab's sampler is random reshuffling: a
  fresh permutation per epoch, batches cut from a continuous stream. f = B/N is the corpus fraction.
- **Corpus quantities:** g is the corpus gradient and C a curvature matrix (GGN) at fixed weights. Σ is the corpus
  covariance of per-example gradients (denominator N − 1). The batch gradient is g_B = g + ξ, with E ξ = 0 and
  Cov ξ = (1/B − 1/N) Σ under sampling without replacement.
- **Preconditioned directions:** d = −P g_B with SPD P. The power family P = C^{−ν} interpolates SGD (ν = 0) and
  Newton (ν = 1). In SOAP-whitened coordinates, SOAP's own step is ν = 0 and the exact per-tensor GGN step is ν = 1.
- **CIFAR workload:** N = 45,000 and B = 2,048, so f = 0.046 and an epoch is about 22 steps. **MNIST workload**
  (transfer-law study): a corpus of N = 5,000.

## 2. Theory

### T1. Expected corpus progress of a preconditioned batch step (identity; classical form)

With a scalar step α chosen to maximize the expected decrease of the corpus quadratic:

- D(α) = α gᵀPg − ½α² [gᵀPCPg + (1/B − 1/N) tr(PCPΣ)];
- max_α D = Ψ_P(B) = Φ_P / (1 + (1/B − 1/N) K_P);
- Φ_P = (gᵀPg)² / (2 gᵀPCPg) is the deterministic gain of the direction;
- K_P = tr(PCPΣ) / gᵀPCPg is its noise scale, in examples.

For P = C^{−ν}, PCP = C^{1−2ν}. The special cases are McCandlish et al.'s noise scales:

| ν | Step | Noise scale K |
| --- | --- | --- |
| 0 | SGD | tr(CΣ)/gᵀCg, the Hessian-weighted B_noise |
| ½ | — | tr(Σ)/‖g‖², B_simple |
| 1 | Newton | tr(C⁻¹Σ)/gᵀC⁻¹g |

The formula assumes the step size is tuned in expectation (as a scheduled learning rate is), not fitted to the
batch, and that the batch's curvature equals the corpus curvature.

**Observed calibration (R8).** With the step size selected on a second batch, the law gets the ordering and the
trend right. It is optimistic about strong preconditioning: its optimal power is about ¼ too high, and it overstates
the gain over ν = 0 by 1.5–3.8× at ν = ½.

### T2. When stronger preconditioning wins (derivation)

For ν₂ > ν₁, let G = Φ_{ν₂}/Φ_{ν₁} ≥ 1.
- If K_{ν₂} ≤ G·K_{ν₁}, the stronger direction wins at every B.
- Otherwise it wins exactly when

  B ≥ B_crit = [1/N + (G − 1)/(K_{ν₂} − G·K_{ν₁})]⁻¹.

**Break time.** At a fixed training batch size B, the predicted break time is
t_break(B) = inf{t : B < B_crit(t)}: the first training stage at which the stronger preconditioning stops paying.
It is a state-dependent boundary, not a fixed corpus fraction. The derivation is from the owner's discussion
(20–21 October), checked here; the algebra is elementary.

### T3. The optimal exponent is below Newton whenever noise sits in flat directions (exact toy)

Take C = diag(1, L), g = (1, 1) and Σ = diag(σ², 0), with s = (1/B − 1/N)σ². Then:

Ψ_ν ∝ (1 + L^{−ν})² / (1 + s + L^{1−2ν}),  so  ν* = max{0, 1 − log(1 + s)/log L}.

Any noise in the flat direction moves the optimum below Newton. The shortfall shrinks like 1/log L for
ill-conditioned problems. The relative boost of a flat direction over a sharp one is then (λ_sharp/λ_flat)^{ν*}
rather than λ_sharp/λ_flat.

### T4. Drift against noise in any averaged statistic (identity-based bound; Islamov et al. 2026, Lemma 2)

For an exponential moving average Ŝ_k = (1 − α)Ŝ_{k−1} + αX_k of noisy observations X_k of a target S(θ_k), where
the target drifts by at most δ per step and the observations have noise σ:

E‖Ŝ_k − S(θ_k)‖ ≤ (1 − α)^k·(initial error) + δ/α + σ√α.

The paper proves this for the momentum buffer of stochastic conditional gradient methods, with δ = 2Lβη (smoothness
× step length) and σ = ρσ_grad. The minimizer is α* ∝ (δ/σ)^{2/3}, with error ∝ δ^{1/3}σ^{2/3}.

**Approximation (transplant, not proved here):** the same algebra applies to curvature statistics such as SOAP's
factors, probe Grams and PARSEC's held subspace.
- δ is the curvature drift per step: the curvature's parameter-Lipschitz constant times the step length.
- σ is the per-batch curvature noise, ∝ B^{−1/2}.
- **Hypotheses:**
  - the optimal pooling window, about 1/α*, grows as noise rises relative to drift;
  - it scales as B^{−1/3} at fixed drift, and shortens as the learning rate (and so the drift) grows.

### T5. Token-budget regimes (Islamov et al. 2026, Corollary 4.1; assumptions there)

**Assumptions:** momentum stochastic conditional gradient (Scion-type) under L-smoothness in a norm, μ-KL
(‖∇f‖_* ≥ μ(f − f*)), norm equivalence ‖·‖_* ≤ ρ‖·‖₂, variance σ²_grad = σ*²/(BS), and token budget T = K·B·S
(with sequence length S = 1 for images).

ε = Õ(max{ LBS/(μ²T), (Lρ²σ*²/(μ⁴T))^{1/3}, ρσ*/(μ(T²BS)^{1/6}) }).

This gives three regimes in BS:
- **noise-dominated** (small BS): larger batches help;
- **batch-independent:** ε ∝ T^{−1/3};
- **iteration-starved** (large BS): ε grows ∝ BS.

The critical scale is BS ≍ (Tμρσ*/L)^{2/3}, with the stepsize β ∝ 1/K and the momentum α constant.

**Hypothesis (connection to curvature methods):**
- A curvature-adapted geometry lowers L/μ (the conditioning in the iteration-starved term) and can raise ρ, the
  noise amplification in the noise terms.
- Curvature should therefore pay mainly in the iteration-starved regime: short horizons, large batches, early
  training. It should stop paying once training is noise-dominated.

### T6. Random-reshuffling corrections (identities, plus one hypothesis)

Three corrections distinguish the queue from i.i.d. sampling:
- **Finite-population factor (identity):** (1/B − 1/N) replaces 1/B.
- **Anti-correlation (identity):** disjoint batches within an epoch have covariance −Var/(m − 1), with m batches
  per epoch.
- **Visit-phase bias (hypothesis):** recently consumed examples are fitted and have lower loss and curvature.

### T7. Same-batch optimism and the transfer law (N16, N20)

**Search rule.** With the step size searched on the batch's own quadratic and equal batch and corpus curvature:
- the corpus share of the batch-optimal decrease is (1 − r)/(1 + r);
- r = (1 − f)·B_noise/B, where B_noise = tr(A⁻¹Σ)/(ḡᵀA⁻¹ḡ) is the noise scale in the step's metric A⁻¹. For
  A = C this equals K₁ of T1.

**Post-hoc refinement (N20).** Share = 2γ − κ_target/κ_batch, where γ = 1/(1 + r) is the realized alignment, and
κ_target/κ_batch compares the curvature of the target and the batch along the step.

## 3. Results

**R1 — Same-batch line search overfits** (N1, rule 8; [T], MNIST). Curvature steps whose length is chosen on the
batch they were built from fit that batch. Step lengths must be chosen with a second batch.

**R2 — Noise scales grow with preconditioning strength and with training** (N16 optimism study; [L], MNIST, 6
states):
- r is ordered SGD < SOAP map < exact GGN in 6 of 6 states, and rises with training: exact GGN 0.35–0.58 at step
  10, 1.36–1.52 at step 50.
- Damping-calibrated PARSEC-L has the largest r (0.41–3.33).
- The calibration of the share law is partial (Spearman 0.56).

**R3 — Transfer law, curvature mismatch, ranking agreement** (N20 transfer-law study; [L], MNIST, N = 5,000):
- **Share law:** predicted vs measured corpus share gives Spearman 0.963 and MAE 0.182 (partial). It is accurate
  for SGD and the SOAP map, and for every metric at B ≥ 1,024.
- **Noise scales B_noise** (examples): SGD 36–210, SOAP map 68–307, exact GGN 195–1,204.
- **Overstated harm for exact GGN at small B:** predicted −0.5 to −0.9, measured about 0. Along a step fitted to a
  small batch, that batch's curvature exceeds the corpus curvature: κ_corpus/κ_batch = 0.25, 0.35, 0.50, 0.66, 0.81,
  0.92 and 0.99 at B = 64 to 4,096. The search self-limits.
- **New-data gap:** at f = 0.82 exact GGN's corpus share is 0.92–0.99, its validation share 0.57–0.95.
- **Ranking agreement** (recomputed 2026-10-21 from `results/corpus_*.json`, post hoc). For SGD, the SOAP map and
  exact GGN, the order by own-batch decrease matches the order by held-out (validation) decrease in this many of 6
  states:

  | B | 64 | 128 | 256 | 512 | 1,024 | 2,048 | 4,096 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Agreeing states (of 6) | 1 | 1 | 3 | 4 | 6 | 6 | 6 |

  Agreement saturates near the largest exact-GGN noise scale, as T1 and T2 lead one to expect.

**R4 — Curvature drift exceeds batch noise for the top subspace** (`spectral_rotation_20261020`; [L], CIFAR, raw
per-tensor GGN, 6 replayed runs):
- **Batch floor:** at fixed weights, one batch's top-16 subspace captures 0.93–0.999 of another batch's top-16
  curvature. It deepens with training.
- **Drift:** ten training steps lose more. Late in training, capture₁₆ after 10 steps is 0.84–0.88.
- **Noise/drift ratio:** R = (1 − floor capture₁₆)/(1 − capture₁₆ after 10 steps) is 0.02–0.30 (top-free) and
  0.03–0.60 (SOAP), rising over training.
- **Spectrum:** late in training it is stationary (α ≈ 1) while the eigenvectors rotate.

**R5 — Pooling and staleness, not queue position** (`queue_curvature_20261020`; [L], CIFAR, top-free, steps 50–400,
12 cases):
- **Pooling:** 8 batches beat 1 in held-out capture (+0.014–0.073, 12 of 12). The gain grows with training.
- **Staleness:** probing the same batches at their own older weights costs 0.03–0.09 capture (12 of 12).
- **Trailing pool vs fresh batch:** the realistic trailing pool (8 batches at their own weights) loses to one
  fresh batch at steps 50–100 and wins at steps 250–400. The pre-declared break-even rule (R ≥ 0.8) was set too
  high (F1 not supported).
- **Queue position:** next 8 ≈ last 8 ≈ random 8 at current weights (differences ≤ 0.007), so the T6 corrections
  are negligible for curvature at f = 0.046.
- **Visit phase in losses:** examples about to be revisited have higher loss than just-trained ones by about
  0.04–0.08 nats from step 100 on (descriptive). The declared reading V1 fails: the effect is absent at step 50.
- **Rademacher probes:** marginally better subspaces than sampled-label probes (10 of 12, by at most 0.008); steps
  are indistinguishable.
- **Probe-row Krylov:** 8 CG steps on probe-row Grams roughly double the one-step held-out decrease of the plain
  whitened step (median 2.1×).

**R6 — Probes from an independent batch** (N24, `probe_variants`; [T], CIFAR): mixed at every point (EMA 0.537 vs
0.541).

**R7 — The braked top directions are high-SNR** (I11, `basis_refresh_snr`; [L], CIFAR). Top-free's brake does not
act as a static noise filter.

**R8 — Noise scales grow tenfold, full Newton breaks early, and the best power is fractional and falls**
(`noise_scale_20261021`; [L], CIFAR, SOAP and top-free replays, seeds 1–3, steps 10–400, 36 states).
- **Setup:**
  - Whitened per-tensor A = K + δI, with K from 4,096 reference examples.
  - Noise from 43 batch gradients of 1,024.
  - Step size selected on a second batch, decrease scored on a third.
- **T1 noise scales (observation):**
  - Newton's K₁ grows from 1,340–2,170 to 17,300–20,400 examples, and N₁(2,048) from 0.63–1.01 to 8.1–9.5.
  - K₁ > K½ in 36 of 36 states and K½ > K₀ in 34 of 36; the declared 36 of 36 for both orderings is not supported.
  - K₀ is 16–878 and not monotone.
- **T2 break (observation):**
  - B_required rises from 1,380–1,770 at step 10 to about 10⁴ from step 100 on.
  - t_break(2,048) is step 25 in five trajectories and step 50 in one.
  - The binding pair is always ¾→1; the other crossovers stay at or below 1,580.
- **T3 optimum (observation):**
  - Interior in 35 of 36 states: about ¾ early and ½ late (seed means 0.72 → 0.58 top-free, 0.67 → 0.42 SOAP).
  - ¾ beats 1 in 35 of 36 states, and ½ beats 0 in 35 of 36.
  - Late, ν = ½ gives 1.3–2.0× SOAP's one-step decrease.
- **Calibration of T1–T2 (observation):**
  - The predicted optimum is within one grid step in 32 of 36 states, but above the measured one in 31 of 36, by
    0.165 on average.
  - The predicted gains are too large: Ψ_½/Ψ₀ is 3.1–5.6 predicted against decrease ratios of 1.3–2.2 measured, and
    Ψ₁/Ψ₀ is 2.5–12 against 0.63–1.44.
  - So under a second-batch step search the law is optimistic about strong preconditioning. Contrast R3: under
    same-batch search, the equal-curvature form overstated the harm.
- **Deterministic gain:** Φ₁/Φ₀ is 6.8–45, with no monotone trend. In the theory, the growth of K, not a loss of Φ,
  moves the optimum.
- **Identity:**
  - In parameter coordinates, −A^{−ν}z_B is −(M #_ν (C + δM))⁻¹ g_B, with M = P_SOAP⁻¹ and
    M #_ν X = M^{½}(M^{−½}XM^{−½})^ν M^{½}.
  - That is the affine-invariant geodesic from SOAP's metric to the damped GGN, and ν = ½ is their matrix geometric
    mean.

**External — Islamov et al. 2026** (NanoGPT 124M–1B; not reproduced here):
- gradient variance ∝ (BS)^{−λ}, with λ ≈ 0.9 (batch size) and 1.1 (sequence length);
- the three regimes of T5 observed;
- the BST rule predicts the optimal β and BS when the token budget grows;
- keeping µP-transferred hyperparameters fixed becomes suboptimal as the token budget grows.

## 4. Open predictions and tests

1. **Break time** (T2; CIFAR replays). **Tested 2026-10-21 (R8):** noise scales grow with training, and full
   Newton breaks at B = 2,048 from step 25–50.
   - Measure G and K_ν for ν ∈ {0, ¼, ½, ¾, 1} in whitened coordinates along training (steps 10–500).
   - Predict B_crit(t) and t_break(B = 2,048).
   - Still open: compare with the stage where the exact-GGN and top-free advantages vanish in training (rule 9,
     `long_run`). Top-free is not a member of the ν family.
   - Caveat (before the test): R3's curvature mismatch means the equal-curvature form may overstate the noise cost of
     strong preconditioning. Under second-batch selection, R8 found the opposite bias.
2. **Fractional Newton** (T3). **Tested 2026-10-21 (R8):** the optimum is interior in 35 of 36 states and falls
   from about ¾ to ½. This holds one step ahead only.
   - Held-out one-step decreases of whitened K^{−ν} steps at B = 2,048. Prediction: an interior optimum ν* < 1 that
     decreases during training.
   - This also answers how strongly the bulk should be boosted: by (λ₁/λ_k)^ν with ν about ½ late, not
     λ₁/λ_k.
3. **Pooling window** (T4).
   - Test whether the best curvature pooling or EMA window follows (δ/σ)^{2/3} from R4's drift and noise.
   - Check its predicted dependence on B (via subsampled batches) and on the learning rate.
4. **Geometry constants** (T5).
   - Estimate L, μ and ρ in each optimizer's norm (SOAP and top-free whitened coordinates) with the paper's recipes:
     - L: ‖g_k − g_{k−1}‖_*/‖x_k − x_{k−1}‖;
     - μ: the slope of ‖g‖_* against the loss;
     - ρ: the dual-to-Euclidean ratio of gradient noise.
   - Derive each optimizer's critical batch. Test whether larger batches (B = 8,192 with accumulation) extend the
     curvature gains.
5. **Learning rate against horizon.** In our runs the tuned lr rose with the horizon (SOAP 1e-2 at 500 steps, 3e-2
   at 1,000, at least 9e-2 at 2,000). The BST rule predicts β ∝ 1/K. Explain the discrepancy before using T5
   quantitatively. Candidates: early transients outside the μ-KL regime (the paper's fit holds only below a loss
   threshold); the coupling of lr and weight decay.
   - SOAP's weight decay is decoupled and lr-scaled (p ← p − lr·wd·p, wd = 0.05).
   - The tuned runs therefore had decay timescales 1/(lr·wd) of about 2,000, 667 and 222 steps, i.e. about 4K,
     0.7K and 0.1K. This is not a fixed fraction of the horizon, which the EMA-timescale view of Wang & Aitchison
     (cited by Islamov et al.) would suggest.
6. **Curvature-estimation noise in T1** (from R8; hypothesis; derivation for Gaussian least squares in
   [batch_noise_quantities/three_quantities.pdf](batch_noise_quantities/three_quantities.pdf), Proposition 6).
   - T1 treats A as exact. With P = Â^{−ν} built from an estimate Â, the quadratic and noise terms use the
     evaluation curvature: gᵀP C_eval P g and tr(P C_eval P Σ).
   - Errors in Â's small eigenvalues inflate P exactly where C_eval is then larger than assumed. That would lower the
     optimal power and shrink the gains, the direction of R8's miss.
   - Offline test: build Â from one reference batch, evaluate the forms with a disjoint batch's curvature, and check
     whether the predicted optimum moves down by about ¼.
7. **Training test of a falling fractional power** (R8, N29).
   - ν from about ¾ early to ½ late, against tuned SOAP, with WSD and EMA for every arm.
   - It needs A^{−ν} applied online, for example by Lanczos on probe-row Grams (R5's probe-row CG generalizes). Cost
     it first.

## 5. Sources

- McCandlish, Kaplan, Amodei et al. (2018), *An empirical model of large-batch training*: noise scales B_noise and
  B_simple (cited from memory; check before quoting).
- Islamov, Machacek, Lucchi, Silveti-Falls, Gorbunov, Cevher (2026), *On the role of batch size in stochastic
  conditional gradient methods*, arXiv:2603.21191v2. All 33 pages were read on 21 October 2026 from the PDF the
  owner supplied.
- *Kaon: Muon is not that special*: named by the owner as the source of the batch-alignment idea. Not read here;
  T1–T3 are derived independently of it.
- Lab studies: `optimism_20261009`, `transfer_law_20261011`, `probe_variants_20261013`, `basis_refresh_snr_20261013`,
  `spectral_rotation_20261020`, `queue_curvature_20261020`, `noise_scale_20261021`.

## 6. Changelog

- **2026-10-21:** created.
  - It consolidates N1, N16, N20, N24, I11, N27 and N28 and the two 20 October studies.
  - It adds the tuned-step noise-scale law (T1), the break batch size (T2), the exact toy for the optimal exponent
    (T3), the drift-against-noise bound transplanted from Islamov et al. (T4), and their token-budget regimes (T5).
  - It adds a post-hoc recomputation of ranking agreement in the transfer-law study (R3).
- **2026-10-23:** derivation note [batch_noise_quantities/three_quantities.pdf](batch_noise_quantities/three_quantities.pdf)
  added (no new study).
  - It derives the share law and the surviving share 1/(1 + r) (T1, T7), and the compute-optimal batch
    B* = √(K·B_ov).
  - It adds the curvature-noise channel: for Gaussian least squares the kept share is κ(n, D)/(1 + r), from
    inverse-Wishart moments, with a threshold near n ≈ D. This is a candidate mechanism for open test 6.
  - All synthetic checks are in `make_figure.py`.
- **2026-10-21 (evening):** R8 added from `noise_scale_20261021`.
  - Open tests 1–2 are marked tested; tests 6 (curvature-estimation noise) and 7 (training test) are added.
  - T1 is annotated with its observed optimism under second-batch selection, and the companion list now includes
    N29.
