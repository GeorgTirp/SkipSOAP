# Transfer law and computable step sizes — protocol

Study `transfer_law_20261011`, requested by the owner on 11 October 2026 ("lets give this line of reasoning and
theory a quick chance"). Offline only: banked states, no training. Written before any run.

The proposal relates the optimism study's transfer share to the alignment γ of "Muon is Not That Special"
(arXiv 2605.11181). That relation is not checked here: the paper was not read, and nothing is downloaded. This
study tests the theory itself.

## 1. Theory under test

A step D = −A⁻¹g_B is built on a batch of size B, with g_B = ḡ + ξ.
- **Quantities:** signal S = ḡᵀA⁻¹ḡ; optimism O = tr(A⁻¹Σ)/B; r = O/S; noise scale B_noise = tr(A⁻¹Σ)/S, the
  gradient-noise scale in the metric A.
- **Same-batch search share:** with the step length chosen on the batch, the held-out share is
  (1 − r)/(1 + r) = 2γ − 1, where γ = ⟨ḡ, D⟩/⟨g_B, D⟩.
- **Finite corpus of N examples, batch B = fN drawn without replacement:**
  - transfer to the corpus: r_train = (1 − f)·B_noise/B;
  - transfer to new data: r_pop = B_noise/B, with no (1 − f) factor (a gap of tr(A⁻¹Σ)/N remains at f = 1);
  - positive corpus transfer requires f > f* = B_noise/(N + B_noise).
- **Computable step size:** α* = γ̂/λ̂, from statistics only.
- **Wiener step:** per direction k of the curvature, the Newton step shrunk by SNR_k/(1 + SNR_k), with
  SNR_k = B·s_k/σ_k².

## 2. Data (read only)

- **Six banked states:** `runs/20261005T143029-9b582ed5`, the searched-SOAP and exact-GGN trajectories at steps
  10, 25 and 50. Each has a 512-example construction batch and four 512-example panels, all disjoint and drawn
  from the analysis split.
- **Corpus:** the 5,000-example analysis split, N = 5,000.
- **New data:** the 5,000-example validation split. The official MNIST test split is not accessed.

## 3. Part A: transfer law on a finite corpus (tests 1 and 2)

- **Fixed operators,** independent of the subsample:
  - `sgd`: A = I;
  - `soap_map`: A = Ω, the state's normalized inverse SOAP metric;
  - `exact_ggn_N`: A = C_N + μ_N Ω. C_N is the exact per-tensor GGN on the whole corpus, and
    μ_N = max(0.01 tr C_N / P, floor).
- **Subsamples:**
  - sizes B ∈ {64, 128, 256, 512, 1024, 2048, 4096}, so f = B/N runs from 0.013 to 0.82;
  - 8 subsets per size, drawn without replacement (generator seed 20261011 + state index), shared by the three
    operators.
- **Step:** D = −A⁻¹g_B, with the lab's same-batch search (`search_candidate`, native radius).
- **Measured** per subset: decrease on the subset (own), on the corpus and on the validation split. Shares are
  ratios of means over the 8 subsets.
- **Exact corpus quantities:** ḡ_N; Σ_N per tensor (normalized by N − 1); S_N = ḡ_NᵀA⁻¹ḡ_N; O = tr(A⁻¹Σ_N)/B;
  B_noise = tr(A⁻¹Σ_N)/S_N.
- **Predictions:**
  - **Corpus share:** (1 − r_train)/(1 + r_train), with r_train = (1 − f)·tr(A⁻¹Σ_N)/(B·S_N).
  - **Validation share,** conditional on the realized validation gradient:
    (2S_vN − S_N − (1 − f)O)/(S_N + (1 − f)O), with S_vN = ḡ_valᵀA⁻¹ḡ_N. Also reported: the population form
    with r_pop = B_noise/B.
  - **Threshold:** f* = B_noise/(N + B_noise).
- **Realized mechanism** (reported): γ = ḡ_NᵀA⁻¹g_B / g_BᵀA⁻¹g_B, and 2γ − 1.
- **Boundary cases:** a search whose best grid point is the largest one (2¹⁶ native radii) raises in the lab's
  code. Such a subset is recorded and excluded from the means, and the count is reported.

## 4. Part B: computable step sizes (test 3)

- **Setting:** each state's construction batch (B = 512), the four label-free operators of the optimism
  follow-up (`sgd`, `soap_map`, `exact_ggn`, `parsec_l`), and the gradient RHS. Directions are built exactly as
  in `studies/optimism_20261009/scripts/followup.py`.
- **Held-out measure:** the mean decrease on panels 2–4. Panel 1 supplies the second-batch statistics and the
  second-batch search.
- **Reference policies,** recomputed: `same` (search on the construction batch), `second` (search on panel 1)
  and `oracle` (search on panels 2–4).

New policies, which evaluate no losses at all:

| Policy | Role | Rule |
| --- | --- | --- |
| `gamma_x` | primary | α = γ̂·q/κ̂₁, where q = g_BᵀA⁻¹g_B, γ̂ = Ŝ/(Ŝ + Ô), Ŝ = g₁ᵀA⁻¹g_B (panel-1 gradient), Ô = tr(A⁻¹Σ̂_B)/B (construction per-example covariance from the banked empirical-Fisher blocks), κ̂₁ = DᵀC₁D (exact GGN of panel 1 along D). No step if Ŝ ≤ 0. |
| `gamma_b` | reported | Construction batch only: α = (q − Ô)₊/κ̂_B, with κ̂_B = DᵀC_BD. |
| `slope_x` | reported | α = Ŝ₊/κ̂₁, the quadratic-model form of second-batch selection. |
| `wiener` | primary, `exact_ggn` only | In each tensor, take the eigenbasis of the batch GGN whitened by the metric: Ω_b = LLᵀ and L⁻¹C_bL⁻ᵀ = Σλ_k w_k w_kᵀ. With h = WᵀL⁻¹g_B, the step is −L⁻ᵀ Σ_k φ_k h_k/(λ_k + μ) w_k, a full step with no search. φ_k = ŝ_k/(ŝ_k + σ̂_k²/B), ŝ_k = (h_k² − σ̂_k²/B)₊, σ̂_k² = w_kᵀL⁻¹Σ̂_bL⁻ᵀw_k. |
| `wiener_x` | reported | As `wiener`, with ŝ_k = (h_k·h₁,k)₊ from the panel-1 gradient. |
| `wiener_search` | reported | The `wiener` direction with same-batch search. |
| `exact_ggn_xc` | reported | `exact_ggn` with panel 1's curvature, A = C₁ + μΩ, same-batch search. Its share on panels 2–4 is compared with `exact_ggn`'s (curvature part of the optimism). |

## 5. Pre-declared readings

**Part A** (126 points: 6 states × 3 operators × 7 sizes):
- **A1, corpus calibration:** Spearman ρ_s and mean absolute error (MAE) between predicted and measured corpus
  shares. *Supported* if ρ_s ≥ 0.8 and MAE ≤ 0.15; *not supported* if ρ_s < 0.5; otherwise *partial*.
- **A2, 1/B law:** take r_meas = (1 − s)/(1 + s) for points whose measured and predicted shares both lie in
  (−0.9, 0.9). For each (state, operator) pair with at least 3 such points, fit the least-squares slope of
  log r_meas on log r_pred. *Holds* if the median slope is in [0.8, 1.25] and the median ratio r_meas/r_pred
  over all such points is in [0.5, 2]; *partial* if one of the two holds; otherwise *fails*.
- **A3, new data:** Spearman and MAE between the conditional prediction and the measured validation shares,
  with A1's thresholds.

**Part B:**
- **B1, computable step:** the mean held-out decrease of `gamma_x` over the 24 points is at least 0.9 times that
  of `second`.
- **B2, Wiener step:** the mean held-out decrease of `wiener` over the six `exact_ggn` points is at least 0.9
  times that of `second`.

**Reported:**
- per-operator and per-state values;
- the number of points where each new policy beats `second`;
- all reported variants;
- f* and B_noise.

## 6. Gates

- **Gradients:** the mean per-example gradient over the construction batch equals the banked gradient (relative
  ≤ 1e-12).
- **Part B references:** the recomputed `same`, `second` and `oracle` own and held-out decreases equal
  `studies/optimism_20261009/results/followup.json` (≤ 1e-12).
- **Wiener form:** with φ ≡ 1, the eigenbasis form equals `exact_ggn`'s direction (relative ≤ 1e-8).
- **No state mutation.**

## 7. Rules and cost

- **Isolation:** write only to `studies/transfer_law_20261011/`; `PYTHONDONTWRITEBYTECODE=1`; at most four
  processes; isolation manifests and suites before and after.
- **Cost:** Part A about 6–7 CPU-minutes per state, about 15 minutes with four workers. Part B a few minutes.

## 8. Known limitations

- One workload, six states, 8 subsets per size.
- The theory is second order and assumes equal curvature along D on the batch and on the target.
- In Part A, A is fixed (built on the whole corpus). A batch-built curvature, as in training, adds a curvature
  optimism, which only Part B's `exact_ggn_xc` probes.
- N = 5,000, so f* and the new-data gap refer to this corpus, not to the 50,000-example training split.
