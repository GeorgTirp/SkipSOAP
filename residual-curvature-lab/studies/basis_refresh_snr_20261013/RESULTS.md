# SOAP bases from the probe GGN, triggered refresh, and noise in the braked directions — results

Protocol: [PROTOCOL.md](PROTOCOL.md). The SHA-256 of the version run is in `logs/protocol_sha256.txt`; one sentence was
corrected before any run. Deviations: [DEVIATIONS.md](DEVIATIONS.md).
- D1: a loader bug crashed Part C before it wrote output; it was fixed and Part C was rerun.
- D2: Part A's transfer grid never bracketed, so a post-hoc check with an extended grid was run, labeled as such.

Everything is offline, at states of the CIFAR-10 depth-6 ViT runs (`cifar_ggn_20261012`, `cifar_sn_soap_20261012`):
- SOAP seeds 1–3, replayed bitwise. Gate S1 passed 9 of 9 at steps 50, 100 and 250.
- Top-free PARSEC-L seeds 1–3, from the stored checkpoints.

Outputs: `results/{ladder,ladder_posthoc,refresh,snr}/`, `results/summary.json`. Full copies are in
`~/skipsoap-study-data/basis_refresh_snr_20261013/`.

**Status (2026-10-13):** all 15 ladder states, 9 refresh windows and 9 top-free states completed.
- Gate A0 passed in every state: no candidate beat the exact per-tensor optimum.
- On the dense blocks, the exact GGN step matches its own optimum to 1.4e-4 or better.

## 1. Summary

- **Headline: N9 holds offline.** The probe-GGN basis gives a markedly better diagonal curvature model on the
  per-tensor quadratic in every state (pre-declared, 9 of 9).
  - **Held out, early in training (post hoc, D2):** with step sizes chosen on a separate batch, the deployable
    probe-basis step beats SOAP's map in 9 of 9 states at steps 10–50. At steps 10–25 it recovers 31–73% of the
    exact GGN step's held-out advantage.
  - **Later:** at steps 100–250 it does not (2 of 6), and neither, mostly, does the exact GGN step.
  - **The other two ideas came out negative or reversed.** No refresh trigger beats the clock. Top-free brakes its
    cleanest directions, not its noisiest.

- **N9: the probe-GGN basis is a much better SOAP basis on the per-tensor quadratic (A1, A2: 9 of 9).**
  - The candidates are diagonal curvature models in a Kronecker basis, scored on the damped per-tensor quadratic.
  - The eigenbasis of the probe factors S_L = Σ G_jG_jᵀ, S_R = Σ G_jᵀG_j (2,048 sampled-label per-example
    gradients) recovers 12–18% of the exact GGN's advantage over SOAP's map at steps 10–50, and 9–14% at steps 100
    and 250.
  - SOAP's own basis with the *exact* diagonal recovers 2–5%. The eigenbasis of SOAP's own current factors recovers
    4–10%.
  - The deployable version (probe basis, probe-estimated diagonal) is indistinguishable from the exact-diagonal
    version (to 0.01).
  - The gain is largest for the head (47–77%) and the attention query and key weights (19–35%). It is smallest for
    the MLP and value weights (8–14%). 1-D tensors are not rotated by any variant.
- **But a diagonal in any Kronecker basis recovers little of the per-tensor GGN step.** The absolute share is 22–33%
  for the probe basis, against 9–19% for SOAP's map. Most of the exact GGN's per-tensor advantage is not diagonal
  in a Kronecker basis.
- **The nearest-Kronecker "ceiling" is not a ceiling.** The eigenbasis of the Frobenius-nearest Kronecker product
  to the exact block recovers 6–13%, *less* than the probe basis, for every matrix type. Frobenius-nearest is not
  the best basis for a diagonal inverse.
- **Transfer (A3, pre-declared): not shown, and not measured at useful step sizes.** On held-out data every
  candidate's selection hit the smallest step of the pre-declared grid, and 99 of 105 held-out "decreases" were
  increases (D2).
  - **Post hoc, with an extended grid:** all selections are interior. The deployable probe-basis step beats SOAP's
    map in 9 of 9 states at steps 10–50, and the exact-diagonal version beats SOAP's basis in 8 of 9 (section 2).
- **Refresh (B1: 9 of 9): staleness is real and starts at once.** The per-tensor benefit of refreshing SOAP's basis
  now grows with basis age:
  - median relative benefit 0.08 at ages 1–5, 0.16 at 6–10, 0.19 at 11–20 and 0.22 at 21–30;
  - on the next batch, the refreshed step decreases the loss by 0.000–0.006 more at ages 1–5 and 0.004–0.017 more
    at ages 11–30 (medians per age).
  - Even a basis refreshed one step ago gains from another power-iteration step: SOAP's single QR step is far from
    its own factors' eigenbasis. Part A says the same: SOAP's basis recovers 2–5%, its factors' eigenbasis 4–10%.
- **No trigger signal beats the clock (B2: none passes).**
  - The pooled Spearman correlation with the per-tensor refresh benefit is 0.20–0.40 for age itself.
  - SOAP's factor rotation (I15a) is close: ≥ 0.3 in 6 of 9 windows, and above age in 5 of 9.
  - The probe ρ rise (N8): 5 and 3 of 9. The probe-direction spread (I10): 0 and 2 of 9.
  - Across tensors at a fixed step, the signals rank the tensors that need a refresh weakly (median Spearman about
    0–0.4).
- **Top-free brakes its cleanest, most informative directions, not its noisiest (C1, C2: 0 of 9, both reversed).**
  - **Cleaner than random:** in 61–88% of tensors per state, the held directions have a *higher* median SNR than
    random control directions in the same whitened space. Among weight matrices, the held directions are noisier in
    only 3–21% of tensors.
  - **Brake strength rises with SNR:** the correlation is positive (Spearman 0.14–0.27), not negative.
  - **Where the signal sits:** in the weight matrices, the 16 held directions are 0.4% of each tensor's dimensions.
    They carry 29–57% of the population gradient's whitened signal energy and 8–32% of its noise energy.
  - **Far below the static optimum:** top-free's response along them is about 1/50–1/85 of the static
    SNR-optimal response (median log₁₀ efficiency −1.7 to −1.9). SOAP's unbraked response is closer, at
    10^−0.4 to 10^−1.4.
  - The static noise-shrinkage explanation of top-free's gain is therefore refuted.

## 2. Part A: the basis ladder (N9)

Pooled headroom share H (fraction of the exact per-tensor GGN's ray-reduction advantage over SOAP's map), seeds
1 / 2 / 3:

| Step | `soap_exact` | `soapeigh_exact` | `probe_exact` | `probe_probe` | `kron_exact` |
| ---: | --- | --- | --- | --- | --- |
| 10 | 0.02 / 0.02 / 0.02 | 0.07 / 0.07 / 0.04 | **0.16 / 0.17 / 0.17** | 0.16 / 0.17 / 0.17 | 0.11 / 0.13 / 0.12 |
| 25 | 0.03 / 0.03 / 0.03 | 0.06 / 0.05 / 0.08 | **0.18 / 0.15 / 0.18** | 0.18 / 0.15 / 0.18 | 0.12 / 0.10 / 0.12 |
| 50 | 0.02 / 0.02 / 0.03 | 0.08 / 0.04 / 0.06 | **0.14 / 0.12 / 0.15** | 0.14 / 0.12 / 0.15 | 0.10 / 0.09 / 0.11 |
| 100 | 0.03 / 0.03 / 0.04 | 0.06 / 0.07 / 0.09 | **0.10 / 0.09 / 0.13** | 0.10 / 0.10 / 0.13 | 0.07 / 0.06 / 0.09 |
| 250 | 0.05 / 0.04 / 0.05 | 0.09 / 0.09 / 0.10 | **0.13 / 0.14 / 0.14** | 0.13 / 0.14 / 0.14 | 0.09 / 0.10 / 0.10 |

Absolute share (Σ R / Σ R*): SOAP's map 0.09–0.19, `soap_exact` 0.12–0.21, `probe_exact` 0.22–0.33, exact GGN 1.

**By tensor type** (H, mean over seeds; step 10 / 50 / 250):

| Type | `soap_exact` | `soapeigh_exact` | `probe_exact` | `kron_exact` |
| --- | --- | --- | --- | --- |
| head | 0.06 / 0.05 / 0.13 | 0.11 / 0.12 / 0.21 | **0.77 / 0.62 / 0.47** | 0.74 / 0.58 / 0.45 |
| attention query | 0.02 / 0.03 / 0.10 | 0.09 / 0.13 / 0.16 | **0.35 / 0.29 / 0.24** | 0.30 / 0.26 / 0.23 |
| attention key | 0.02 / 0.04 / 0.09 | 0.08 / 0.10 / 0.13 | **0.24 / 0.21 / 0.19** | 0.21 / 0.15 / 0.16 |
| attention output | 0.01 / 0.02 / 0.06 | 0.06 / 0.08 / 0.12 | **0.22 / 0.19 / 0.18** | 0.15 / 0.13 / 0.12 |
| attention value | 0.01 / 0.02 / 0.04 | 0.05 / 0.05 / 0.09 | **0.12 / 0.11 / 0.10** | 0.04 / 0.05 / 0.04 |
| MLP w1 | 0.01 / 0.01 / 0.03 | 0.03 / 0.03 / 0.06 | **0.12 / 0.08 / 0.09** | 0.06 / 0.05 / 0.05 |
| MLP w2 | 0.01 / 0.01 / 0.03 | 0.04 / 0.04 / 0.08 | **0.14 / 0.11 / 0.12** | 0.09 / 0.06 / 0.09 |
| embedding | 0.05 / 0.03 / 0.07 | 0.08 / 0.04 / 0.08 | **0.15 / 0.08 / 0.08** | 0.13 / 0.06 / 0.08 |
| 1-D (not rotated) | 0.09 / 0.07 / 0.03 | same | same | same |

**Readings:**
- **A1 (probe basis vs SOAP basis, exact diagonal, steps 10–50): 9 of 9. *The probe basis is better.***
- **A2 (deployable probe basis + probe diagonal vs SOAP's map): 9 of 9. *Deployable N9 beats SOAP's map.***
- **A3 (transfer): 0 of 9 for both comparisons, *not shown*.** It is uninformative: all 105 selections sat at the
  grid's lower edge (D2).

### Post-hoc transfer check (D2; no verdict)

Same states, candidates, selection batch and evaluation batch; step α₀·2^k with k ∈ {−14, …, 0} chosen on the
selection batch. The quadratic results reproduced the first run bitwise in all 15 states, and every one of the 105
selections is interior to the grid.

Evaluation-batch loss decrease (seeds 1 / 2 / 3):

| Step | SOAP map | `soap_exact` | `probe_probe` (deployable N9) | exact GGN | N9's share of GGN's advantage over the map |
| ---: | --- | --- | --- | --- | --- |
| 10 | 0.020 / 0.068 / 0.048 | 0.021 / 0.072 / 0.045 | **0.034 / 0.088 / 0.066** | 0.063 / 0.096 / 0.080 | 0.32 / 0.73 / 0.57 |
| 25 | 0.019 / 0.007 / 0.017 | 0.033 / 0.015 / 0.022 | **0.034 / 0.022 / 0.031** | 0.067 / 0.053 / 0.050 | 0.31 / 0.32 / 0.41 |
| 50 | 0.047 / 0.034 / 0.014 | 0.050 / 0.037 / 0.022 | **0.049 / 0.047 / 0.023** | 0.049 / 0.040 / 0.038 | 1.15 / 2.31 / 0.39 |
| 100 | 0.051 / 0.034 / 0.022 | 0.049 / 0.040 / 0.022 | 0.041 / 0.036 / 0.024 | 0.046 / 0.047 / 0.030 | — / 0.16 / 0.26 |
| 250 | 0.023 / 0.039 / 0.015 | 0.023 / 0.043 / 0.014 | 0.023 / 0.037 / 0.013 | 0.025 / 0.036 / 0.016 | — |

**Counts at steps 10–50:**
- `probe_exact` > `soap_exact`: 8 of 9 (the exception is seed 1 at step 50);
- `probe_probe` > `soap_map`: 9 of 9.

**At steps 100 and 250:** `probe_probe` > `soap_map` in 2 of 6.

**Where the exact GGN step stands:** its own held-out advantage over the map is clear at steps 10 and 25, small at
step 50 (in seed 1, 0.0489 vs 0.0474), and absent or mixed at 100 and 250 (better in 4 of 6). This matches the lab's
earlier finding that curvature pays early.

**Chosen step sizes:**
- SOAP's map and `soap_exact`: k = −4 to −6.
- The probe-basis steps: −5 to −7.
- The exact GGN step: −7 to −8, that is 1/128 to 1/256 of the step size suggested by the summed per-tensor quadratic.
  Cross-tensor coupling and nonlinearity together make that quadratic overstate the useful step by about two orders
  of magnitude.

**Post-hoc caveat:** the grid was extended after the pre-declared one failed to bracket. The comparison itself (same
batches and rule, every candidate on the same grid) was defined before this run.

## 3. Part B: refresh benefit and trigger signals (I10, I15a, N8)

Nine 30-step SOAP windows with refresh switched off, starting right after a refresh (steps 11, 51 and 251).

**B1 (staleness grows): 9 of 9 windows.** The Spearman correlation of age with the mean relative benefit is
0.24–0.57.

| Ages | 1–5 | 6–10 | 11–20 | 21–30 |
| --- | ---: | ---: | ---: | ---: |
| median relative per-tensor benefit of refreshing now | 0.08 | 0.16 | 0.19 | 0.22 |

Next-batch loss decrease, refreshed minus held step (median over windows): 0.0001–0.006 at ages 1–5, 0.004–0.007 at
6–10, and 0.004–0.017 at 11–30.

**B2 (signals):** pooled Spearman correlation with the per-tensor benefit, per window (s1 t11, s1 t51, s1 t251,
s2 t11, …, s3 t251).

| Signal | Values | ≥ 0.3 | Beats age | Verdict |
| --- | --- | ---: | ---: | --- |
| age (control) | 0.27 0.40 0.20 0.25 0.36 0.39 0.22 0.35 0.31 | — | — | — |
| I15a factor rotation | 0.17 0.37 0.36 0.23 0.35 0.53 0.25 0.44 0.44 | 6 | 5 | not shown |
| N8 probe ρ rise | 0.25 0.31 0.31 0.18 0.13 0.37 0.22 0.31 0.33 | 5 | 3 | not shown |
| I10 spread | 0.27 0.20 0.21 0.25 0.18 0.28 0.19 0.19 0.14 | 0 | 2 | not shown |
| probe ρ level (descriptive) | 0.26 0.19 0.32 0.21 0.07 0.39 0.16 0.21 0.31 | — | — | — |

- **Late windows:** factor rotation is the strongest signal in the windows starting at step 251 (0.36, 0.53, 0.44).
- **Ranking tensors:** across tensors at a fixed step, the signals rank which tensors gain from a refresh only
  weakly (median Spearman −0.11 to 0.41).

## 4. Part C: signal-to-noise of top-free's braked directions (I11)

Nine top-free states (seeds 1–3, steps 50, 100, 250). SNR = G²/(τσ²) per direction, τ = 1/2048, over 8,192 per-example
gradients.

| State | Tensors with top SNR < control SNR | Spearman(κ, SNR) | Top SNR quartiles | Control SNR quartiles | Median log₁₀ efficiency, top-free / SOAP |
| --- | ---: | ---: | --- | --- | --- |
| s1 t50 | 0.39 | +0.23 | 2.0 / 13.2 / 44.0 | 1.8 / 10.5 / 33.8 | −1.93 / −1.41 |
| s1 t100 | 0.29 | +0.21 | 0.07 / 1.42 / 5.42 | 0 / 0.72 / 2.92 | −1.77 / −0.76 |
| s1 t250 | 0.12 | +0.19 | 0.04 / 1.06 / 3.61 | 0 / 0.29 / 1.44 | −1.69 / −0.37 |
| s2 t50 | 0.39 | +0.18 | 0.36 / 3.20 / 9.90 | 0.18 / 2.34 / 7.96 | −1.89 / −1.44 |
| s2 t100 | 0.29 | +0.19 | 0.04 / 1.25 / 4.26 | 0 / 0.53 / 2.41 | −1.76 / −0.79 |
| s2 t250 | 0.15 | +0.18 | 0.11 / 1.87 / 6.08 | 0 / 0.61 / 2.43 | −1.75 / −0.43 |
| s3 t50 | 0.36 | +0.14 | 0.32 / 2.87 / 10.6 | 0.17 / 1.87 / 7.39 | −1.89 / −1.43 |
| s3 t100 | 0.27 | +0.19 | 0.13 / 1.81 / 6.66 | 0 / 0.80 / 3.33 | −1.78 / −0.72 |
| s3 t250 | 0.14 | +0.27 | 0.09 / 1.57 / 5.75 | 0 / 0.51 / 2.13 | −1.74 / −0.45 |

**Verdicts:**
- **C1: 0 of 9 (*not shown*; the fraction is 0.12–0.39, far below 0.75).** The braked directions are cleaner than
  random directions.
- **C2: 0 of 9 (*not shown*; the correlation is positive).**

**Signal and noise energy in the held span** (medians over tensors):
- **Weight matrices** (39 tensors; the held span is 0.39% of the dimensions):
  - signal share 0.50–0.57 at step 50, 0.42–0.52 at step 100, 0.29–0.36 at step 250;
  - noise share 0.29–0.32, 0.15–0.20 and 0.08–0.10.
- **1-D tensors and the CLS token** (the held span is 25% of the dimensions): signal 0.84–0.95, noise 0.69–0.90.

**Efficiency.** e = response × λ / α* compares each response with the static SNR-optimal one:
- top-free's response along its held directions is 10^−1.7 to 10^−1.9 of it;
- SOAP's unbraked response is 10^−0.4 to 10^−1.4 of it;
- 9–23% of held directions have no detectable signal (α* = 0).

## 5. Interpretation and limitations

- **N9 is worth a training test.**
  - **Quality:** the probe basis gives a markedly better diagonal curvature model than SOAP's basis, in every
    state, and its cheap probe diagonal is as good as the exact one. The gain concentrates in the head and the
    attention query and key weights.
  - **Cost:** the probe basis costs the same s = 1 probe pass top-free already makes.
  - **Held out:** post hoc, its single-step advantage carries over to held-out loss early in training (steps
    10–50), where the GGN gap is.
  - **Open:** whether the advantage survives as a training method. That means carrying Adam's second moments in
    the new basis, with refreshes and noise; this study does not measure it.
- **The per-tensor GGN's advantage over SOAP lies mostly outside any Kronecker-diagonal model.** Even the best
  basis recovers under 20% of it. Closing the early gap needs non-diagonal per-tensor structure, such as
  PARSEC-L's low-rank top space, or coupling across tensors.
- **SOAP's basis is chronically behind its own statistics.** A single power-iteration step per 10 steps leaves the
  basis well short of its factors' eigenbasis (Part A: 2–5% against 4–10%), and the benefit of refreshing grows
  from the first step on (B1).
  - The trivial schedule is as good a trigger as any tested signal.
  - The cheaper improvement may be *more* refresh work (more power steps or more frequent QR) rather than a
    trigger. That is a cost question, not a signal question.
- **Top-free's gain is not static noise filtering.** It brakes its highest-SNR, most signal-dense directions, to
  about 1/60 of the static optimum.
  - Together with the earlier finding that most gradient noise also lies there (brake-mechanism study; here
    8–32% of the noise in 0.4% of the dimensions), the remaining candidates are dynamic:
    - momentum-free tracking of a moving signal;
    - stability or curvature drift (N21);
    - an implicit, sharpness-dependent annealing.
  - The static SNR model would have the brake *loosened*, not tightened, in these directions.
- **Limitations:**
  - **One step ahead:** all three parts are offline and look one step ahead.
  - **Part B's refresh:** it uses SOAP's single QR refresh.
  - **Part C's assumptions:** a static quadratic, the held probe curvature from an earlier batch, and τ = 1/2048
    without the finite-corpus correction.
  - **Part A's quadratic:** it is per tensor, and coupling between tensors is ignored. That is also why its global
    step scale was far too large (D2).
