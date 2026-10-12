# Why does the PARSEC-L brake help? — protocol (exploratory, offline)

Study `brake_mechanism_20261011`, prompted on 11 October 2026 by the owner's question: why does shrinking only
the top directions help at SOAP's tuned learning rate, and is it edge of stability (EoS)? It is offline,
uses banked data only and runs no training. Written before running `scripts/mechanism.py`.

## 1. Correction that motivates check A

Stage D's write-up called ηλ_max ≈ 4 "about the heavy-ball stability limit 2(1+β₁) ≈ 3.9". That limit applies
to heavy-ball momentum written with the raw gradient. SOAP uses Adam-style EMA momentum, m ← β₁m + (1−β₁)g with
step −lr·P̂·m̂. That equals heavy ball with α = lr(1−β₁) and β = β₁, so linear stability needs
lr·λ(P̂H) < 2(1+β₁)/(1−β₁) = 78 for β₁ = 0.95. This is the adaptive edge of stability of Cohen et al. (2022),
38/lr for β₁ = 0.9. The per-tensor value ≈ 4 is far below it, but the full-network value can be larger.

## 2. Data

- Gap-audit native SOAP at lr 1e-2 (`20261007T143302-baba2664`): checkpoints at steps 50, 200 and 500 for
  seeds 1–3. Each holds model, SOAP state and the context of the last step (bases, denominators, bias
  correction).
- Curvature batch: the first 512 analysis examples (held out from training).
- Stage-D brake and SOAP training logs, which use identical training batches per step and seed.

## 3. Checks and pre-declared readings

**A. Edge of stability.**
- S = lr·λ_max(P̂M), where P̂ = Q diag(√(1−β₂ᵗ)/d) Qᵀ is the bias-corrected preconditioner of the last step and
  M is either the full Hessian or the full GGN (Lanczos, 30 iterations, full reorthogonalization). Also
  reported: the per-tensor GGN block maximum, comparable to stage D's logged ηλ_max.
- *At the adaptive EoS* if S_Hessian ≥ 58 (0.75 × 78) at steps 200 and 500 in all seeds.
- *Far below* if S_Hessian ≤ 20 at every checkpoint.
- Otherwise *intermediate*.

**B. Noise concentration and a noisy-quadratic estimate.** In whitened coordinates ζ (θ = F̂ζ, F̂F̂ᵀ = P̂),
SOAP is momentum SGD with step lr. For each of the top-16 directions per tensor (exact eigenpairs (λ_k, w_k)
of F̂ᵀC_bF̂):
- σ_k² is the per-step gradient-noise variance along w_k, from true-label per-example gradients on the
  curvature batch divided by B = 512.
- Stationary noise excess of EMA momentum: e_k ≈ lr·σ_k²/4. The brake divides lr by (1 + 4·lr·λ_k).
- Under SOAP's momentum (β₁ = 0.95), a direction is underdamped for lr·λ ∈ (0.0128, 78). There its
  convergence rate per step is √β₁ ≈ 0.975 whatever lr is, so the brake costs it no speed.
- Reported: the share of total whitened noise variance in the top-16 directions (against their 11% share
  of dimensions), the predicted excess removed by the brake, and the fraction of top directions that stay
  underdamped under the brake.
- *Noise concentrated* if that share ≥ 30% at every checkpoint.

**C. Optimization or generalization.**
- The stage-D brake and SOAP (gap audit; same data order) see identical training batches.
- Paired difference of the training-batch loss before each step, brake minus SOAP, averaged over steps
  1–50, 51–200 and 201–500 per seed.
- *Optimization effect* if the difference is negative in steps 201–500 in all three seeds.

All three checks are exploratory; no training decision depends on them.
