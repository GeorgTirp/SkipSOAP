# SOAP bases from the probe GGN, triggered refresh, and noise in the braked directions — protocol

Study `basis_refresh_snr_20261013`. Written on 13 October 2026, before any run. The owner asked for runs of three ideas
(12–13 October):
- **N9:** SOAP bases from the probe GGN instead of GGᵀ, "as an extra ladder rung".
- **I10, I15a and N8:** adaptive or triggered SOAP refresh. "Offline first: does a basis-rotation or ρ signal fire
  when the curvature has gone stale?"
- **I11:** noise from the same probes. "It could test whether top-free's braked directions are the noisy ones."

**Everything is offline**, at states of the CIFAR runs. There is no training comparison; states are replayed
deterministically where needed.

## 1. Questions

- **A (N9).** At SOAP states, is a Kronecker eigenbasis built from per-example sampled-label gradients (the probe
  GGN) a better basis for a diagonal curvature model than SOAP's own basis? Better means: within each tensor, a step
  closer to the exact per-tensor GGN step. And does the deployable version beat SOAP's map? The deployable version
  is the probe basis with the probe-estimated diagonal.
- **B (I10, I15a, N8).** Along SOAP trajectories with refresh switched off, how fast does refreshing *now* become
  worth it? Do three cheap signals track that benefit, and do they beat the trivial signal "steps since the last
  refresh"? The three signals are:
  - SOAP's factor rotation (I15a);
  - the rise of the probe off-diagonal ratio ρ (N8);
  - the spread of the probe's top directions in the held basis (I10).
- **C (I11).** At top-free PARSEC-L states, are the directions it brakes noisier than other directions? Is the brake
  strength ordered by noise? And how does its per-direction response compare with the static signal-to-noise
  optimum α* = SNR/(1 + SNR)? That optimum comes from the theory draft the owner shared on 12 October.

## 2. Workload and states

- **Workload:** that of `cifar_ggn_20261012` and `cifar_sn_soap_20261012`.
  - Data: CIFAR-10, 45k/5k split of the training set; the test split is never accessed.
  - Model: M1d6 ViT, 308,042 parameters.
  - Training: batch 2048 in microbatches of 512, augmentation crop + flip, GPU float32 with deterministic
    algorithms.
- **SOAP states (A, B):** the `soap` arm (lr 1e-2, betas 0.95/0.95, refresh every 10 steps, weight decay 0.05),
  seeds 1–3. They are replayed deterministically with `cifar_sn_soap_20261012`'s runner. Its `soap` path reproduced
  `cifar_ggn`'s bitwise (G1 there).
  - **Gate S1:** replayed parameters equal the stored checkpoints of `cifar_ggn`'s evaluation run
    `20261012T110529-d747f9fc` bitwise at steps 50, 100 and 250.
  - States saved: after steps 10, 11, 25, 50, 51, 100, 250 and 251, with the optimizer state and the data-order and
    augmentation-generator states.
- **Top-free states (C):** the stored checkpoints of `cifar_ggn`'s `topfree` units (lr 3e-2, c = 64, rank 16,
  `top_rhs='gradient'`), seeds 1–3, at steps 50, 100 and 250, with the PARSEC-L state.
- **SOAP's refresh timing:** upstream SOAP refreshes its bases at the end of optimizer steps 11, 21, … (its internal
  counter skips step 1). The state after step 10 therefore holds the step-1 eigenbasis. States 11, 51 and 251 come
  right after a refresh.
- **Preconditioned modes:** SOAP preconditions both modes of every matrix (`max_precond_dim` 10000), but not the
  1-D tensors (`precondition_1d=False`, so Adam). Every basis variant below changes only the modes SOAP
  preconditions; 1-D tensors keep the identity.

## 3. Shared quantities

At a state θ with the batch (x, y) of the next step (augmented, the run's own order):
- **Gradient:** g, mean cross-entropy, true labels.
- **Exact GGN block:** C_bb for each tensor b.
- **Damping:**
  - μ = max(γ tr C / D, 1e-8) with γ = 1e-2;
  - the metric is Ω = scale·P⁻¹, the normalized inverse metric of the state's own SOAP preconditioner
    P = Q diag(1/den) Qᵀ, with den = √v + ε (`SOAPMetric`).
  - These are the definitions used by `cifar_ggn`'s GGN arm.
- **Damped per-tensor quadratic:** q_b(d) = g_bᵀd + ½ dᵀ(C_bb + μΩ_b)d.
  - **Ray reduction** of a direction: R_b(d) = (g_bᵀd)² / (2 dᵀ(C_bb + μΩ_b)d) if g_bᵀd < 0, else 0. This is the
    best reduction along that direction, with the scale chosen per tensor, so it measures within-tensor direction
    quality.
  - **Exact per-tensor GGN step:** δ_b = −(C_bb + μΩ_b)⁻¹g_b, solved in SOAP-whitened coordinates
    (`cifar_ggn`'s `ggn.tensor_ggn_step`, whitened residual gate 1e-4, no action audit). Its reduction is
    R*_b = −½ g_bᵀδ_b.
- **Probes:** s = 1 sampled label per example on the same batch (`micro.lean_probes`, as top-free uses),
  2,048 rows G_j per tensor.

## 4. Part A: the basis ladder (N9)

**States:** SOAP seeds 1–3 at steps 10, 25, 50, 100 and 250 (15 states).

**Bases** for each matrix tensor (m × n); the exact-diagonal operator in basis B = B_L ⊗ B_R is
d = −B diag(1/(c + s)) Bᵀ g, with c = diag(BᵀC_bbB) and s = μ·scale·diag(BᵀP⁻¹B).

| Name | Basis | Diagonal |
| --- | --- | --- |
| `soap_map` | SOAP's held basis Q | SOAP's own 1/den (its actual preconditioner) applied to g; direction only |
| `soap_exact` | SOAP's held basis Q | exact |
| `soapeigh_exact` | exact eigenvectors of SOAP's current factors GG (no staleness, no single power step) | exact |
| `probe_exact` | **N9:** eigenvectors of the probe factors S_L = Σ_j G_jG_jᵀ, S_R = Σ_j G_jᵀG_j | exact |
| `probe_probe` | **N9, deployable:** the probe basis | probe estimate Σ_j (B_LᵀG_jB_R)² plus the same damping s |
| `kron_exact` | **ceiling:** eigenvectors of the nearest Kronecker product A ⊗ B to C_bb (Van Loan; 30 power iterations) | exact |
| `ggn` | exact per-tensor GGN step δ | — |

**Metrics**, computed per tensor and pooled over tensors:
- **Headroom share:** H_X = (Σ_b R_b(d_X) − Σ_b R_b(d_soap_map)) / (Σ_b R*_b − Σ_b R_b(d_soap_map)), with ray
  reductions. It is the fraction of the exact GGN's per-tensor advantage over SOAP's map that a candidate recovers.
  The lab's earlier 46% / 21% numbers are of this kind.
- **Absolute share:** Σ_b R_b(d_X) / Σ_b R*_b.
- **Transfer:** for each candidate, the global step d_X is scaled by α₀ (the ray-optimal scalar on the summed damped
  quadratic) times 2^k, k ∈ {−3, …, 3}.
  - k is chosen on a *selection* batch: the next 2,048 examples in the order, unaugmented.
  - The decrease L(θ) − L(θ + α d_X) is then measured on an *evaluation* batch: the 2,048 after those,
    unaugmented.
  - It is reported relative to `ggn` under the same rule.

**Gate A0 (consistency):** no candidate may beat the exact optimum on the same quadratic: R_b(d_X) ≤ R*_b (1 + 1e-3)
+ 1e-12 for every tensor and candidate. A violation means the dense block and the whitened solve disagree; it stops
the part.

**Readings:**

| Code | Reading |
| --- | --- |
| A1 | **N9 basis:** pooled H of `probe_exact` vs `soap_exact`, at steps 10, 25 and 50. *The probe basis is better* if `probe_exact` is higher in all 3 seeds at all three steps (9 of 9). |
| A2 | **N9 deployable:** pooled H of `probe_probe` vs 0 (that is, vs `soap_map`). *Deployable N9 beats SOAP's map* if positive in 9 of 9 at steps 10, 25 and 50. |
| A3 | **Transfer:** evaluation-batch decrease of `probe_exact` vs `soap_exact`, and of `probe_probe` vs `soap_map`. *Better* if higher in 9 of 9 at steps 10, 25 and 50. |
| A4 | Descriptive: every candidate's H and absolute share at all five steps; `soapeigh_exact` vs `soap_exact` (cost of SOAP's stale or inexact basis); `kron_exact` (ceiling of any Kronecker basis with an exact diagonal); breakdown by tensor type (attention q/k/v/o, w1, w2, embedding and head). |

## 5. Part B: is a refresh worth it now, and do the signals know? (I10, I15a, N8)

**Windows:** from the SOAP states right after a refresh (steps 11, 51 and 251; seeds 1–3; 9 windows), continue
SOAP for 30 steps with refreshes switched off. The data are the run's own batches. The basis age a is the number
of steps since that refresh, 1–30. Ages 1–10 lie on the real SOAP trajectory; from age 11 on, SOAP would have
refreshed. (Corrected before any run: the previous version said 1–9 and 10. A refresh at the end of step s + 10
first changes the step at age 11.)

**At each window state θ_t:**
- **Candidate next steps:** the *held* step (SOAP's step t+1 with its held basis) and the *refreshed* step (the same
  step after applying SOAP's own refresh now). The refresh is upstream's `get_orthogonal_matrix_QR` with the
  momentum carried over, exactly as at a scheduled refresh. Both use the real gradient of batch t+1 and are taken
  with weight decay 0. The trajectory continues with the held step and the real weight decay.
- **Ground truth:**
  - per tensor, the relative refresh benefit (R_b(refreshed) − R_b(held)) / max(R_b(refreshed), R_b(held)), with
    the ray reductions of section 3, from one per-example-Jacobian pass;
  - globally (secondary), the actual loss decrease of each candidate step at its real size on the *next* batch,
    unaugmented.
- **Signals,** per tensor, computed from information available at θ_t:
  - **age** (the trivial control);
  - **I15a factor rotation:** ‖offdiag(QᵀL Q)‖_F / ‖QᵀL Q‖_F for SOAP's current factors L ∈ GG in the held basis Q,
    pooled over the preconditioned sides;
  - **N8 ρ rise:** the same ratio for the probe factors S_L, S_R in the held basis, minus its value at the window
    start;
  - **I10 spread:** for the top r = 8 eigenvectors v of each probe factor, the mean of 1 − max_i (q_iᵀv)², pooled
    over sides. It is 0 when every probe top direction coincides with one held basis vector.

**Readings:**

| Code | Reading |
| --- | --- |
| B1 | **Staleness grows:** per window, the Spearman correlation of age with the mean relative benefit over matrix tensors (30 steps). *Refresh benefit grows with age* if positive in 9 of 9 windows. The median benefit at ages 1–5, 6–10, 11–20 and 21–30 is reported. |
| B2 | **Signals track the benefit:** per window, the Spearman correlation of each signal with the per-tensor relative benefit, pooled over matrix tensors × 30 steps. A signal *fires when stale* if this correlation is ≥ 0.3 in at least 7 of 9 windows *and* exceeds age's correlation in at least 7 of 9 windows. |
| B3 | Descriptive: the cross-sectional Spearman correlation (across tensors, within a step; median over steps) of each signal with the benefit; the global next-batch benefit by age; the signals' time courses. |

## 6. Part C: are the braked directions the noisy ones? (I11)

**States:** top-free seeds 1–3 at steps 50, 100 and 250 (9 states; lr in its constant phase).

**Held brake directions:**
- From PARSEC-L's state: the held factor `wt` (stored in rotated coordinates), carried to the current bases
  (`transport`) and divided by the current √den.
- With β = c·lr·√(1 − β₂^t) for the next step, V = √β·heldᵀ, as in PARSEC-L's `_shrink`.
- The SVD Vᵀ = U_eff diag(√κ) W gives the orthonormal whitened brake directions u_k and the brakes κ_k exactly as
  `_shrink` applies them: factor 1/(1 + κ_k) along u_k. The curvature is λ_k = κ_k/β.
- **Control directions:** r = 16 random whitened unit directions per tensor (seeded), orthogonalized against U_eff.

**Gradient statistics:**
- **Sample:** per-example true-label gradients g_i for 8,192 training examples (a seeded random subset, freshly
  augmented with a seeded generator), whitened as z_i = Qᵀg_i / √den.
- **Per direction u:** the sample mean ḡ_u and variance σ_u² of uᵀz_i; the signal G_u² = ḡ_u² − σ_u²/n (bias
  corrected, floored at 0).
- **SNR** = G_u² / (τσ_u²), with τ = 1/2048. That is the batch-mean noise; with fresh augmentation, the
  finite-corpus correction (≤ 4.6%) is not applied.
- **Static optimum:** α* = SNR/(1 + SNR).

**Per-direction responses** (in units of the whitened gradient), compared with the static optimum α*/λ_k:
- top-free: η/(1 + κ_k), with η = lr·√(1 − β₂^t);
- SOAP without the brake: η.
- **Efficiency** e = response·λ_k/α*, so e = 1 means statically SNR-optimal.

**Readings:**

| Code | Reading |
| --- | --- |
| C1 | **Braked directions are noisier:** per state, the fraction of tensors (those with held directions) whose median SNR over u_k is below the median SNR over their control directions. *The braked directions are the noisy ones* if this fraction is ≥ 0.75 in all 9 states. |
| C2 | **Brake strength follows noise:** per state, the Spearman correlation of κ_k with SNR_k over all held directions of all tensors. *Stronger brakes sit on noisier directions* if ≤ −0.3 in all 9 states. |
| C3 | Descriptive: the distributions of SNR, α* and e for top-free and for SOAP, by state; the median \|log e\| of top-free vs SOAP (which is closer to the static optimum); the fraction of the gradient's population energy and of its noise energy that lies in span(U_eff), against the dimension fraction r/count. |

## 7. Gates and hygiene

- **G0:** suites before and after: the repository (106), PARSEC-L (22), `cifar_ggn` (11), `cifar_sn_soap` (12) and
  this study's tests.
- **S1:** bitwise replay (section 2).
- **A0:** consistency (section 4).
- **Writes:** only to `studies/basis_refresh_snr_20261013/` and `~/skipsoap-study-data/basis_refresh_snr_20261013/`.
  Earlier studies are imported or read, never modified.
- **Process:** `PYTHONDONTWRITEBYTECODE=1`; pytest `-p no:cacheprovider`; isolation manifests before and after; one
  GPU process at a time.
- **Budget (estimate before the dry run):** replays about 5 min; A about 20 min; B about 1 h; C about 10 min.

## 8. Limitations known in advance

- **All offline, one step ahead.** None of these shows a training gain; a positive reading motivates a training
  test.
- **N9's exact-diagonal rungs are oracles.** Only `probe_probe` is deployable as measured. A training version would
  also have to carry SOAP's Adam second moments in the new basis, which is not measured here.
- **Part B looks only one step ahead.** It switches refresh off to reach ages beyond 10. It uses SOAP's single
  power-iteration refresh, not an exact one.
- **Part C is static.** The static optimum ignores momentum and multi-step dynamics. It is the static version of
  the shrinkage argument by design (the dynamic noise model failed, N19).
- **Part C's curvature is the held probe estimate** λ_k on an earlier batch, not the population curvature.
