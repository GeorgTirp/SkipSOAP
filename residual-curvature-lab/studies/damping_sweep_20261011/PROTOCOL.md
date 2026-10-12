# Top-space damping sweep and offline noise-model (OU) check — protocol

Study `damping_sweep_20261011`, approved by the owner on 11 October 2026 ("Yes, please do 1 and 2 now"). It also
gates the decision on a per-direction Kalman version (section 4). Written before any run.

## Part 1. Damping sweep (training)

- **Arm:** momentum-free top-space brake (`top_rhs='gradient'`), learning-rate calibration, lr 1e-2 constant,
  rank 16, s = 4 probes every 10 steps, norm blend 0. Settings and harness as in `topfree_confirm_20261011`.
- **Strength:** c ∈ {256, 1024, 4096, 10⁹}. c = 10⁹ is effectively freezing the top-16 directions per tensor.
  As c → ∞ the step becomes SOAP's momentum step with its top-subspace component removed.
- **Seeds:** 4, 5, 6, matching the existing c = 64 runs (`20261011T140034-192dd041`). This sweep is descriptive;
  a confirmation of its best c would need fresh seeds.
- **Gate:** c = 64 seed 4 is rerun and must reproduce the existing unit bitwise (validation difference ≤ 1e-12 at
  every step).
- **Measurements:** raw and EMA (d ∈ {0.9, 0.98, 0.995}) validation loss; sharpness at steps 10–500.
- **Pre-declared readings** (step 500, EMA d = 0.98, seeds 4–6):
  - S1: each c vs c = 64, *better* / *worse* / *mixed* (all three seeds).
  - S2: shape of the curve over c ∈ {64, 256, 1024, 4096, 10⁹}, using seed means:
    - *interior optimum* if some c < 10⁹ is better than both c = 64 and c = 10⁹ in all three seeds;
    - *monotone to freezing* if c = 10⁹ is better than c = 64 and than every c in between, in all seeds;
    - otherwise *flat* or *undetermined*.

## Part 2. Offline OU check (no training)

- **Data:** the seed-4 checkpoints of `topfree_confirm_20261011`, every 10 steps, for SOAP (constant), the brake
  (c = 4, momentum) and the top-free brake (c = 64). Each holds parameters, the SOAP state and the bias-corrected
  EMA (d = 0.98).
- **At steps t = 20, 40, …, 480:**
  - Whitened coordinates ζ = F̂⁻¹θ, with F̂ = (1−β₂ᵗ)^¼ · Q · diag(d^-½) from the checkpoint's SOAP state.
  - Exact top-16 eigenpairs (λ_k, w_k) per tensor of F̂ᵀC_bF̂, using the exact GGN blocks on the first 512
    analysis examples.
  - Per-step noise variance σ_k² along w_k, from true-label per-example gradients on the same examples divided
    by 512.
  - Deviation of the weights from their EMA along w_k: y_k(t) = w_kᵀF̂⁻¹(θ_t − θ̄_t). Lag-10 partner: y_k(t+10),
    computed with the step-t directions and whitening.
- **Model predictions** for each arm's dynamics along direction k (η = lr):
  - SOAP: heavy ball with α = η(1−β₁), β = β₁ = 0.95.
  - Brake: the same with η/(1 + 4ηλ).
  - Top-free: plain gradient descent with η' = η/(1 + 64ηλ).
  - Stationary variance: heavy ball αsσ²(1+β)/(λ(1−β)(2(1+β) − αλ)); gradient descent η'σ²/(λ(2 − η'λ)).
  - The EMA's own averaging enters through the exact factor 1 − 2(1−d)S + (1−d)(2S−1)/(1+d), with
    S = Σ_j d^j ρ_j and ρ the model autocorrelation.
  - Predicted lag-10 autocorrelation: ρ₁₀.
- **Aggregation:** by η·λ bins (< 0.01, 0.01–0.1, 0.1–1, ≥ 1).
  - Spread ratio: Σ measured y² / Σ predicted E[y²].
  - Measured lag-10 correlation, Σ y(t)y(t+10) / √(Σy(t)² · Σy(t+10)²), against the mean predicted ρ₁₀.
- **Pre-declared reading O1:** the *OU model is adequate* if, for every arm, the spread ratio in the bins with
  η·λ ≥ 0.1 lies in [0.5, 2] and the measured lag-10 correlation is within ±0.3 of the prediction. Otherwise it
  is *not adequate*.

## Part 3. Rules and cost

- Isolation as in the previous side studies: write only to `studies/damping_sweep_20261011/` and
  `~/skipsoap-study-data/damping_sweep_20261011/`; `PYTHONDONTWRITEBYTECODE=1`; at most four workers; manifests
  and suites before and after.
- Cost: 13 training units (about 0.15 CPU-h) and the offline check (72 states, about 0.15 CPU-h).

## 4. Decision on the Kalman version (fixed in advance)

- **Build** it only if S2 is *interior optimum* and O1 is *adequate*.
- If S2 is *monotone to freezing*, use the simpler rule instead: remove the top-subspace component.
- If O1 is *not adequate*, revise the noise model before any filter.
