# Curvature along the data queue: visit phase, pooled sources and steps — protocol

Study `queue_curvature_20261020`. Written on 20 October 2026, before any run. Offline: bitwise replays of existing
runs, no new training.

**Request.** On 20 October the owner asked to queue the three cheap tests of the queue theory:
1. the floor-vs-drift ratio from `spectral_rotation_20261020`;
2. per-example loss and GGN trace against the time since each example's last visit, measured on replays;
3. offline steps built from current-batch, next-n, last-n and random-n curvature, scored on the next n batches' loss
   and on held-out loss.

The owner also pasted the remark that the clean pooled sketch uses Rademacher probe vectors. A Rademacher-probe
source is therefore included as a descriptive comparison.

## 1. Theory being tested

**Sampler.** The lab's sampler is random reshuffling: a fresh permutation per epoch, cut into batches of B = 2,048
from a continuous stream. With N = 45,000 an epoch is about 22 steps, and every example is consumed exactly once per
epoch.

**Decomposition.** The curvature the optimizer meets s steps ahead splits into:
- the full-data curvature now;
- drift from parameter movement;
- selection noise from which examples are in the batch.

**Where the queue differs from i.i.d. sampling:**
- the finite-population correction, with exact zero-sum noise over an epoch;
- a weak anti-correlation between disjoint batches;
- a **visit-phase** effect: examples consumed recently are fitted, so they have lower loss and curvature than
  examples about to be revisited;
- order-dependent second-order terms.

Pooling n batches trades noise (about 1/n) against drift. All estimates below are at fixed weights except the
realistic trailing pool `last8own` and the target `T_queue`.

## 2. Trajectories and states

- **Runs:** `cifar_ggn_20261012`'s `topfree__lr0.03__s{1,2,3}`, replayed from step 0 with
  `ladder_mechanism_20261013`'s `Replay`.
- **Bookkeeping:** batch positions are taken with the same calls as `Replay.next_batch` and recorded.
- **States:** t ∈ {50, 100, 250, 400}, all in the constant-lr phase. For state t the replay continues to t + 8 so
  that the next eight batches and the weights at which they are consumed are known.
- **Notation:** B_s is the batch consumed at step s, applied to θ_{s−1} to produce θ_s.
- **Images:** all curvature and loss measurements use **unaugmented** images. Training applied crops and flips at
  consumption, so a "visit" saw an augmented copy.
- **Coordinates:** SOAP-whitened at θ_t, with contexts copied at t. The gradient for all steps is that of B_{t+1}
  at θ_t.

## 3. Part V: visit phase

At θ_t, for all 45,000 training examples and the 5,000 validation examples:
- the per-example cross-entropy loss with the true label;
- the per-example GGN trace estimate ‖r_i‖², where r_i = J_iᵀ(p_i − e_y), y is sampled from p_i (one draw), and
  E‖r_i‖² = tr(J_iᵀ H_i J_i).

**Age:** the age of a training example is t − (the last step that consumed it). The **placebo age** uses an
independent `DataOrder` (seed 777000 + seed) advanced in lockstep: the same sampler, not used for training.

**Per case (seed × state):**
- D_loss = mean loss of examples with age ≥ 20, minus that of examples with age ≤ 1;
- D_trace is the same for the trace estimate;
- D′ are the same differences under placebo ages.

| Code | Reading |
| --- | --- |
| V1 | **Visit phase in loss:** D_loss > 0 in all 12 cases, and min D_loss > max \|D′_loss\| over the 12 cases. |
| V2 | **Visit phase in curvature:** the same for D_trace. |
| V3 | Descriptive: mean loss and trace per age (0–50), real and placebo; the batch about to be consumed against the batch just consumed; the validation means. |

## 4. Part S: curvature sources

**Sources.** Probe rows are one sample per example, weighted so that ZZᵀ is the per-example-average GGN estimate,
and whitened at θ_t.

| Source | Rows |
| --- | --- |
| `cur` | B_{t+1} at θ_t (PARSEC's choice: the step's batch) |
| `next8` | B_{t+1}, …, B_{t+8} at θ_t (the coming queue) |
| `last8` | B_{t−7}, …, B_t at θ_t (just consumed, current weights) |
| `last8own` | B_{t−s} at θ_{t−s−1}, s = 0…7 (realistic trailing pool, stale weights) |
| `rand8` | 8 × 2,048 uniformly random training positions (an independent `DataOrder` per state) at θ_t (i.i.d. control) |
| `rand8_rad` | `rand8`'s positions with Rademacher output probes r_i = J_iᵀ S_i ξ_i, where S_i = diag(√p_i) − p_i√p_iᵀ (so S_iS_iᵀ = H_i) and ξ_i ∈ {±1}^10 (descriptive) |
| `identity` | no curvature: the whitened gradient step (step comparison only) |

**Per source and matrix tensor (the 39 two-dimensional tensors with both sides above 1):**
- the top-64 left singular vectors U of Z (randomized SVD with oversampling 16 and 2 power steps, PARSEC's
  settings);
- the step y: 8 CG steps from 0 on (ZZᵀ + δI) y = −z_g, where z_g is the whitened gradient of B_{t+1}.
  - δ = μ·scale, as in `ladder_mechanism`: μ = max(1e-2·tr C/D, 1e-8), with tr C the exact GGN trace of B_{t+1} at
    θ_t.

**Targets** (independent probe draws, whitened at θ_t):
- `T_queue`: B_{t+s} at θ_{t+s−1}, s = 1…8. This is the curvature the next eight steps actually meet.
- `T_held`: validation positions 1000–4999 at θ_t.

**Capture fraction:** f_X(T, k) = ‖Z_Tᵀ U_{X,k}‖²_F / ‖Z_T‖²_F, for k ∈ {16, 64}. Each case uses the median over
matrix tensors.

**Steps:**
- **Direction:** v_X is y_X unwhitened on the matrix tensors and zero elsewhere.
- **Step size:** α = α₀·2^j, j = −14…2, chosen on validation positions 0–999 (`V_sel`). α₀ = 1 for CG steps (their
  natural scale) and α₀ = η_t for `identity`.
- **Scores:** the decrease of the mean loss on the queue set B_{t+1…t+8}, and on validation positions 1000–4999,
  both at the selected α.

| Code | Reading (12 cases: 3 seeds × 4 states) |
| --- | --- |
| S1 | **Queue in-sample:** f_next8(T_queue, 64) > f_rand8(T_queue, 64) in ≥ 10 of 12. |
| S2 | **Visit phase in curvature estimates:** f_rand8(T_held, 64) > f_last8(T_held, 64) in ≥ 10 of 12. |
| S3 | **Staleness of the trailing pool:** f_last8(T_queue, 64) > f_last8own(T_queue, 64) in ≥ 10 of 12. |
| S4 | **Pooling:** f_rand8(T_held, 64) > f_cur(T_held, 64) in ≥ 10 of 12. |
| S5 | **Queue objective:** `next8`'s queue-set decrease > `rand8`'s in ≥ 10 of 12. |
| S6 | **Held-out:** `next8`'s held-out decrease > `rand8`'s in ≥ 10 of 12. The theory predicts *not supported*: the queue is exchangeable for held-out data. |
| S7 | Descriptive: every source's captures for k = 16 and 64 on both targets; step decreases at the selected and the oracle α; interior choices; `rand8_rad` vs `rand8`; `identity`. |

## 5. Part F: floor against drift

From `spectral_rotation_20261020` (top-free, anchors 50, 100, 250, 400, per seed):

R = (1 − floor capture_16) / (1 − capture_16(t → t+10)).

This compares batch noise with ten steps of drift, both in raw coordinates.
- **Model:** a trailing 8-batch pool (ages 0–7) beats a single fresh batch when its 1/8 noise reduction exceeds its
  drift cost. With capture loss linear to quadratic in age, and the floor holding about twice a single estimate's
  noise, break-even is R ≈ 0.3–0.8.
- **Predictions:** R ≥ 0.8 ⇒ f_last8own(T_held, 16) ≥ f_cur(T_held, 16); R ≤ 0.3 ⇒ the reverse; otherwise no
  prediction.

| Code | Reading |
| --- | --- |
| F1 | Holds if at least 4 of 12 cases carry a prediction and every predicted case agrees; *undetermined* if fewer than 4 carry one; otherwise not supported. |

## 6. Gates, cost, isolation and limitations

- **G0:** suites before and after, including this study's tests:
  - sampled-label rows against `micro.lean_probes`;
  - the Rademacher factor identity S Sᵀ = H, and its rows against explicit Jacobians;
  - CG with Gram products against a dense solve;
  - capture identities;
  - age bookkeeping.
- **G1:** each replay equals the run's checkpoint parameters bitwise at steps 50, 100 and 250; a failure stops the
  study.
- **Cost:** about 50 restricted probe passes per state and tensor group, plus the scoring; **about 1.5 GPU-hours**.
  It runs after `spectral_rotation_20261020`.
- **Writes:** only to `studies/queue_curvature_20261020/` and `~/skipsoap-study-data/queue_curvature_20261020/`.
  Read only: `cifar_ggn` checkpoints, `ladder_mechanism` code, `spectral_rotation`'s summary.
- **Limitations:**
  - **Model:** one model, one optimizer (top-free), states 50–400 of 500-step runs.
  - **Augmentation:** measurements use unaugmented images, while visits used augmented ones.
  - **Probes:** one probe draw per example, so per-example traces are noisy (averaged within age bins).
  - **Step:** the step affects matrix tensors only.
  - **Overlap:** random sets overlap the queue sets by chance, as sampling does.
