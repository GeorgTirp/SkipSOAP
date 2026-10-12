# Signal/optimism test of the transfer share — protocol

Study `optimism_20261009`, requested 9 October 2026: can the signal/optimism theory of how much of a
step's same-batch gain transfers be tested on banked data? Offline only; no training.

## 1. Theory under test

A step Δ = −α A⁻¹ g_B is chosen on minibatch B (size B = 512). Write g_B = ḡ + ξ with Cov(ξ) = Σ/B.
To second order:

- signal S = ḡᵀA⁻¹ḡ;
- optimism O = tr(A⁻¹Σ)/B;
- r = O/S.

With the step length chosen by line search on the same batch, the predicted share of the own-batch
decrease that appears on held-out data is (1 − r)/(1 + r). With a small fixed step it is 1/(1 + r).
B·r is the gradient-noise scale measured in the metric A⁻¹. The theory predicts that more whitening
metrics have larger r.

## 2. Data (banked, read only)

- Six saved states: `runs/20261005T143029-9b582ed5`, searched SOAP and exact-GGN trajectories at steps
  10/25/50. Each holds:
  - parameters, SOAP contexts, the construction-batch gradient and momentum;
  - per-tensor GGN and empirical-Fisher blocks on the construction batch (512 analysis examples);
  - four held-out panels of 512 examples.
- Gap-audit check 2c (`~/skipsoap-study-data/gap_audit_20261007/runs/20261007T123608-7f39721a`): searched
  candidates with own-batch (`actual_decrease`) and held-out (`independent_mean_decrease`) decreases.

## 3. Operators, candidates and quantities

Per-tensor operators A, damping as in check 2c (γ = 0.01, exact trace, normalized inverse SOAP metric Ω):

| Name | A_b | Source of the measured share |
| --- | --- | --- |
| `soap_map` | Ω_b (check 2c "identity") | Recomputed; must reproduce check 2c |
| `exact_ggn` | C_b + μΩ_b | Recomputed; must reproduce check 2c |
| `sgd` | I | New: same search and measurement |
| `ef_newton` | F_b + μ_F Ω_b, μ_F = max(.01 tr F / D, 1e-8) | New: same search and measurement |

Each candidate is searched on the construction batch with `search_candidate` (native-SOAP radius) and
measured with `measurement` (own-batch and mean held-out decrease). Measured share = held-out / own.

Quantities per state and operator:
- Σ̂_b = B/(B − 1) · (F_b − g_b g_bᵀ);
- O = tr(A⁻¹Σ̂)/B;
- S estimated without bias from cross-products of the four panel gradients, S = mean_{p<q} g_pᵀA⁻¹g_q;
- r = O/S; predicted search share (1 − r)/(1 + r), and −1 if S ≤ 0.
- Realized mechanism check per candidate direction D: ρ = ḡ_panelsᵀD / g_BᵀD, with quadratic
  line-search share 2ρ − 1.

## 4. Gates

- Construction gradient equals the stored gradient (≤ 1e-12).
- `soap_map` and `exact_ggn`, gradient and momentum: own and held-out decreases equal check 2c (≤ 1e-12).
- No state mutation.

## 5. Pre-declared readings (24 gradient-RHS points: 6 states × 4 operators)

1. **Calibration:** Spearman ρ_s between predicted (1 − r)/(1 + r) and measured share, and the mean
   absolute error (MAE).
   - *Supported* if ρ_s ≥ 0.6 and MAE ≤ 0.2;
   - *not supported* if ρ_s < 0.3;
   - otherwise *partial*.
2. **Ordering:** in how many of the six states r(ef_newton) > r(exact_ggn) > r(soap_map) > r(sgd), and in
   how many the measured shares are ordered the other way.
3. **Sign:** among points with r > 1, the fraction with a negative held-out decrease.
4. **Mechanism (realized):** Spearman ρ_s between 2ρ − 1 and the measured share.

Exploratory: the same quantities for the momentum RHS. The theory predicts optimism ≈ 0 there, because
momentum is independent of the construction batch.

## 6. Limitations known in advance

Six states, one workload. Each measured share is a single realization: one construction batch, one
search. The theory is second order with a fixed A, but `exact_ggn` and `ef_newton` build A from the same
batch. The panels estimate ḡ with their own noise.
