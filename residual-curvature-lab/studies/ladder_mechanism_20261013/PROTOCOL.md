# Does PARSEC-L's information contain the early GGN gain? The reduced-subspace ladder and the N21 control — protocol

Study `ladder_mechanism_20261013`. Written on 13 October 2026, before any run.

**Request.** The owner asked for this on 13 October: "lets do 3". Item 3 of the previous recommendation was: "Run the
offline ladder test (held against fresh subspace, plus the N21 random-direction control). Its answer decides whether
the early gap is reachable at all with PARSEC-L's information. If it isn't, formally drop 'close the GGN gap' as a
goal."

Everything is offline, at replayed CIFAR states.

## 1. Questions

1. **Ladder.** At top-free PARSEC-L states early in training, how much of the exact per-tensor GGN step's advantage
   over top-free's own step can be recovered by each of these?
   - (a) better coefficients inside the subspace PARSEC-L already holds, together with its current momentum and
     gradient;
   - (b) a freshly extracted probe subspace instead of the held one;
   - (c) the exact top eigenvectors instead of the probe estimate;
   - (d) the remainder: directions outside all of these.

   And does the recovery carry over to held-out loss?
2. **N21 control.** Along the step just taken, the GGN curvature ten steps earlier was only 32–62% of its current
   value. Is that a property of the step's direction (selection), or does the curvature change along every
   direction?

## 2. States

The CIFAR workload of `cifar_ggn_20261012` (M1d6 ViT, batch 2048, GPU float32). Replayed deterministically with
`cifar_ggn`'s runner:
- **top-free** (lr 3e-2, c = 64, rank 16), seeds 1–3: snapshots after steps 10, 25, 50 and 100 for the ladder;
- **SOAP** (lr 1e-2) and **top-free**, seeds 1–3: snapshots after steps 9, 19, 20, 49, 59, 60, 99, 109 and 110 for
  the N21 control. For t ∈ {20, 60, 110} that gives θ_{t−11}, θ_{t−1} and θ_t.

**Gate S1:** the replayed parameters equal `cifar_ggn`'s stored checkpoints (evaluation run
`20261012T110529-d747f9fc`) bitwise at steps 50 and 100.

## 3. Part L: the ladder

All quantities are per tensor b, in the SOAP-whitened coordinates of the state's own SOAP (F0 = Q diag(den^-½)):
- K_b = F0ᵀC_bbF0, the exact whitened GGN block (`ggn.accumulate` with whitening, as `cifar_ggn`'s GGN step);
- damping δ = μ·scale, with μ = γ tr C / D and γ = 1e-2;
- z_g = F0ᵀg_b, the whitened current gradient of the next batch;
- z_m, the whitened momentum update;
- the damped quadratic q(y) = z_gᵀy + ½ yᵀ(K_b + δI)y.

The reference is y* = −(K_b + δI)⁻¹z_g, with reduction R* = ½ z_gᵀ(K_b + δI)⁻¹z_g.

**Candidates** (whitened steps). For any basis B, the Galerkin step is y_B = −B[Bᵀ(K_b + δI)B]⁻¹Bᵀz_g, the best step
in span(B) on the quadratic. Each candidate is scored by its ray reduction R (scale chosen per tensor).

| Rung | Candidate |
| --- | --- |
| 0 | `topfree`: top-free's actual next step (optimizer copy, weight decay 0), whitened |
| 1 | `held`: Galerkin in B = orth[U_held, P⊥z_m, P⊥z_g]. U_held holds PARSEC-L's held brake directions (reconstructed as in `basis_refresh_snr`, Part C), and P⊥ projects out span(U_held). z_m = F0ᵀm is the whitened momentum after this step's update, m = β₁m_old + (1 − β₁)g. F0 is the state's before the step throughout. |
| 2 | `fresh`: the same with U_fresh, the top 16 directions extracted now from s = 1 probes of the next batch (PARSEC-L's own randomized SVD, 2 power steps) |
| 3 | `exact_top`: the same with the exact top 16 eigenvectors of K_b (subspace iteration on the dense block) |
| 4 | `ggn`: y* (R*) |

**Gap decomposition.** Pooled over tensors, the gap G = ΣR* − ΣR(topfree) splits exactly into four shares:
- coefficients: ΣR(held) − ΣR(topfree);
- staleness: ΣR(fresh) − ΣR(held);
- estimation: ΣR(exact_top) − ΣR(fresh);
- missing directions: ΣR* − ΣR(exact_top).

Each is reported as a fraction of G. A share can be negative where a "better" rung's subspace happens to serve the
gradient worse.

**Held-out check.** For each candidate's global step:
- the scale α₀ (ray-optimal on the summed per-tensor quadratic) is multiplied by 2^k, k ∈ {−14, …, 2}, and k is
  chosen on a selection batch (the next 2,048 examples, unaugmented);
- the decrease is measured on an evaluation batch (the 2,048 after those);
- whether each chosen k is interior to the grid is reported.

**Gate L0:** no candidate's ray reduction may exceed R* by more than 1e-3 relative, per tensor.

**Readings** (top-free seeds 1–3 at steps 10, 25 and 50; step 100 is descriptive):

| Code | Reading |
| --- | --- |
| L1 | **Information present:** `held` closes ≥ 50% of G (pooled) in all 9 states. |
| L2 | **Staleness matters:** the staleness share is ≥ 10% of G in all 9 states. |
| L3 | **Estimation matters:** the estimation share is ≥ 10% of G in all 9 states. |
| L4 | **Directions are missing:** the missing-directions share is ≥ 50% of G in all 9 states. If L4 holds and L1 does not, the early GGN gain lies mostly outside what any rank-16 PARSEC-L top space with momentum and gradient can express, and the goal "close the GGN gap with PARSEC-L" is formally dropped. |
| L5 | **Held out (descriptive):** decreases of `held`, `fresh` and `exact_top` vs `topfree` and `ggn`, with the interior check. |

## 4. Part N: the N21 control

At t ∈ {20, 60, 110}, for SOAP and top-free, seeds 1–3 (18 points):
- d is the step just taken, θ_t − θ_{t−1};
- C_t is the GGN at θ_{t−1};
- C_old is the GGN at θ_{t−11};
- both are evaluated on the next batch, unaugmented, as in `cifar_sn_soap`.

**Directions:**
- the step d;
- the current gradient g (of batch t, at θ_{t−1});
- 8 random directions r = F0ξ with ξ ~ N(0, I), drawn in the state's own SOAP-whitened coordinates and scaled to
  ‖d‖ (seeded).

For each direction u, the ratio is ρ(u) = uᵀC_old u / uᵀC_t u (forward mode; `nonlinear.directional_ggn` of
`cifar_sn_soap`).

**Readings:**

| Code | Reading |
| --- | --- |
| N1 | **Selection:** the median over the 8 random directions of \|1 − ρ(r)\| is below half of \|1 − ρ(d)\| at all 18 points. If so, the staleness of N21 is mostly a property of the step's direction. |
| N2 | **Global change:** the median ρ(r) is below 0.7 at all 18 points, meaning curvature grows along random directions too. |
| N3 | Descriptive: ρ(d), ρ(g) and the quartiles of ρ(r) per point. |

## 5. Gates, cost, isolation

- **G0:** suites before and after (the repository, PARSEC-L, `cifar_ggn`, `cifar_sn_soap`, `basis_refresh_snr`,
  `n9_training`, `probe_variants`, and this study's tests). The tests cover: Galerkin steps against dense solves;
  the gap decomposition telescoping; subspace iteration against eigh; the reconstruction of the held directions
  against PARSEC-L's logged brakes; the random directions' whitening; and the replay equality.
- **S1, L0:** as above.
- **Cost:** replays about 5 min; ladder about 1 min per state (12 states); control about 2 min. **About 0.5
  GPU-hour.**
- **Writes:** only to `studies/ladder_mechanism_20261013/` and `~/skipsoap-study-data/ladder_mechanism_20261013/`.
- **Limitations:**
  - **One step ahead, quadratic model.** The ladder measures the damped per-tensor quadratic; cross-tensor
    coupling is outside its scope.
  - **One rank:** rank 16 only.
  - **What the fresh rung does not cover:** it uses this batch's probes, so it removes staleness but not the
    sampling noise of the probes.
