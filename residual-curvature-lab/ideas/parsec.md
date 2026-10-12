# PARSEC: insights, design rules and idea status

Living document, started 8 October 2026. It banks what the lab has *learned* about PARSEC
(PARtial-Spectrum Eigen-Correction) and the status of every idea. The original proposals, with their
derivations and kill criteria, are in [PARSEC_ideas.md](PARSEC_ideas.md); this file does not replace
them.

**How to maintain this file.**

- Update it after every study that touches PARSEC: the design rules, the idea-status table and the
  changelog.
- Every claim names its source study. Every claim carries one evidence level:
  - **T**: training result, fresh seeds 1–3, tuned baselines;
  - **L**: local, fixed-state result (one step at saved states);
  - **H**: hypothesis, not tested.
- Never delete a negative result or a killed idea; mark it and say why.
- A new idea gets the next free ID and a first test and kill criterion before anyone builds it.

Workload behind all numbers unless stated: ViT width 16, 2 blocks, MNIST 28x28, batch 512, CPU
float64. Training rows: seeds 1–3, 500 consumed steps, each arm tuned on seed 0.

## 1. What PARSEC-L is (current form)

Per tensor b, with SOAP's frozen factor F0_b (SOAP's preconditioner is P_b = F0_b F0_bᵀ), damping
μ = max(γ tr C / D, 1e-8), β = 1/(μ · scale) and SOAP's effective momentum u:

    Δ_b = −η F0_b [ I − Σ_{k≤r} (κ_k / (1 + κ_k)) w_k w_kᵀ ] F0_bᵀ u_b

(w_k, κ_k) are the top eigenpairs of the whitened GGN block K_C,b = β F0_bᵀ C_bb F0_b. This is SOAP's
native step, shrunk along the r directions where the curvature is large compared with SOAP's own
metric. With r = 0 it is plain SOAP. With exact eigenpairs it equals the capacity screen's
damping-calibrated `teacher_r` core.

**Damping sweep and noise-model check (2026-10-11, [T] sweep, [L] OU check): no Kalman version; c = 64 stays.**
- **Sweep** (seeds 4–6, EMA d = 0.98, step 500): more damping than c = 64 does not help.
  - Every c vs c = 64 is mixed.
  - Seed means rise from c = 64 to freezing: 0.121, 0.123, 0.126, 0.128, 0.130.
  - Freezing the top directions (c = 10⁹) is worst by mean, and its sharpness climbs to 112.
  - The best fixed c is near 64.
- **OU check** (seed-4 checkpoints): the gradient-noise model of the deviation from the weight EMA along the exact
  top GGN directions is not adequate for any arm.
  - Measured deviations are slow (lag-10 correlation 0.81–0.93; SOAP's prediction is 0.42, or −0.62 in the
    sharpest bin).
  - Under top-free they are 16–110 times the damped-noise prediction.
  - The model is not stationary over training.
  - Candidates for the missing slow component: drift, rotating directions, or leakage of the exact directions
    out of the optimizer's held subspace.
- **Decision** by the pre-declared rule: no Kalman filter; revise the noise model first. The noise-only
  explanation of the top-free gain is not supported, though the gain itself stands.
([../studies/damping_sweep_20261011/RESULTS.md](../studies/damping_sweep_20261011/RESULTS.md))

**Top-free confirmation (2026-10-11, [T], fresh seeds 4–6): in-loop damping of the top directions beats weight
averaging.** The momentum-free top-space brake (`top_rhs='gradient'`, c = 64, lr 1e-2) + EMA reaches 0.121, against
SOAP + EMA 0.135 and SOAP + final decay + EMA 0.135. It is better in all seeds; both pre-declared readings passed.
More damping keeps helping (c = 64 > 16 > 4 > 1); the optimum is not yet found. *(Damping sweep: no gain beyond
c = 64; see above.)* Its trajectories are sharper
(27–59 vs SOAP's 10–23, limit 78). **Current best PARSEC-L: lr calibration, `top_rhs='gradient'`, c = 64, plus
weight EMA.** ([../studies/topfree_confirm_20261011/RESULTS.md](../studies/topfree_confirm_20261011/RESULTS.md))

**Noise controls (2026-10-11, [T]): the brake's gain is noise removal and does not survive standard practice.**
With a 100-step final decay (WSD) SOAP reaches 0.129, and on an EMA of its weights 0.127. Both beat the constant-lr
brake (0.149) in all seeds. With decay or averaging added, the brake's extra effect is mixed (0.130 vs 0.129,
0.123 vs 0.127). The statistics effect, edge of stability and early transient explanations are not supported.
Remaining candidate: the momentum-free top space (`top_rhs='gradient'`, c = 16) beat the brake at step 500 in all
seeds (0.140), and with EMA reached 0.119, the lowest here; not yet tested against SOAP + WSD or SOAP + EMA.
**Rule from now on: every optimizer comparison includes decay and weight averaging for all arms.**
([../studies/noise_controls_20261011/RESULTS.md](../studies/noise_controls_20261011/RESULTS.md))

**Stage D result (2026-10-11, [T], seeds 1–3, 500 steps):** with learning-rate calibration c = 4 at lr 1e-2,
`norm_blend = 0` and no search, PARSEC-L beats tuned SOAP at step 500 in all seeds (0.149 vs 0.161). It
reaches the targets in fewer steps (T50 117 vs 142) but is slower in CPU wall-clock (about 2× SOAP's cost per
step). This "brake" is the first PARSEC variant to beat tuned SOAP
([../studies/stage_d_20261009/RESULTS.md](../studies/stage_d_20261009/RESULTS.md)). Recommended spec v2,
pending the owner's decision on defaults: `calibration='lr', strength=4, norm_blend=0`. The current code
default is `norm_blend = 1`, which stage D did not test.

**Working spec v1 (implemented 2026-10-09 in [../studies/parsec_l_20261009/](../studies/parsec_l_20261009/README.md);
not yet tested in training):**

- Curvature: s = 4 model-sampled labels per example, as ordinary backward passes. Randomized SVD of
  √β F0_bᵀ U_bᵀ with 2 power steps, per tensor, re-extracted at each refresh.
- Rank r = 16 (8 is the cheaper fallback).
- **Standard since 2026-10-09 (owner decision, one-seed smoke evidence):** `norm_blend = 1` (the step keeps
  SOAP's global norm) and residual-branch scaling 1/depth (`with_branch_scaling`, exponent 1). Both have
  an off-setting (0) that stage D must keep as an ablation. Evidence against them so far:
  - norm blend 1 slightly hurt at step 100 in the smoke run (0.356 vs 0.343);
  - offline it did not raise PARSEC-L's held-out progress (2 of 6 states, as without it);
  - branch scaling hurt at depth 2 (0.384 vs 0.343).
- μ from the probe trace, tr Ĉ = ‖U‖²_F.
- One tuned learning rate, SOAP's momentum, no line search on the training batch.

The implementation (`PARSECL`, a torch optimizer on upstream SOAP) passes 11 tests. Without curvature
it is bitwise upstream SOAP. At full rank its step solves the lab's damped per-tensor system in SOAP's
metric to a 1.1e-12 residual. Truncated, it equals the exact top-r shrinkage. Tracking, ρ-triggered
per-side refresh and the stop rule are switches, off by default.

Smoke run (one seed, untuned, lr 1e-2, not an evaluation): validation loss 0.72 / 0.34 at steps 50 / 100
versus native SOAP's 0.88 / 0.44, at 0.085 versus 0.041 s per step. Ten steps after a refresh, the held
subspace's invariance residual is 0.5–0.86.

## 2. Design rules supported by evidence

1. **Correct SOAP; don't replace it.** [T, gap audit and geometry follow-up] Tuned native SOAP is the bar
   at step 500 (0.161). Tuned Adam (0.202), tuned PARSEC default (0.186) and tuned PARSEC split (0.192)
   stay behind it. The default branch feeds a transformed gradient into Adam, whose normalization partly
   undoes the transform. The tuned split branch is effectively Adam: its top-space step averages 0.67% of
   the complement step.
2. **Per-tensor scope, not global and not full.** [T, geometry follow-up] At matched damping, metric and
   search, full (cross-tensor) GGN is worse than per-tensor GGN at steps 50, 200 and 500 in all seeds
   (0.330 versus 0.182 at step 500). That holds although the cross-tensor blocks carry 54–92% of the
   curvature energy. Full OT is better than per-tensor OT only at step 50. Caveat: damping was tuned for
   the per-tensor versions.
3. **Shrink the SOAP step; never invert or add a separate Newton step.** [T for the failure modes, L for
   the remedy] The split branch's Newton step diverged at lr_top ≥ 0.1 and was negligible at stable
   sizes. Exact inversion gains early, then mostly fits each minibatch (rule 8). The shrinkage form can
   only shorten SOAP's step, and does so exactly along directions where SOAP's step is too long for the
   curvature. That is a direction-wise answer to edge-of-stability concerns, without per-layer learning
   rates from λ_max.
4. **Calibrate against damping, in SOAP-whitened coordinates.** [L, capacity screen] The same rank-16
   correction recovers 56.1% of local headroom with damping calibration and 15.4% with trace calibration
   (six saved states). The scale of the uncorrected part matters more than extra rank.
5. **Precondition SOAP's momentum; one learning rate, one momentum.** [T for momentum, L for the pairing]
   Removing SOAP's momentum ruins training (0.495 versus 0.238 for default searched SOAP at step 500, gap
   audit 2b). Exact GGN applied to the momentum is poor locally (0.032 mean decrease, against 0.030 for the
   SOAP map on the gradient). PARSEC-L applied to the momentum is as
   good as on the gradient (about 63% of exact-GGN headroom, PARSEC offline).
6. **Sampled-label probes are a sufficient curvature source; true labels are not.** [L, PARSEC offline;
   T for the negative]
   - s = 4 sampled labels keep 97.8% of exact PARSEC-L's local gain (r = 16, momentum, 72 states);
     every setting keeps 93–104%, even s = 1.
   - True-label empirical Fisher trained worse than tuned searched SOAP (0.306 versus 0.210 at step 500,
     gap audit 2a).
   - A probe row is a plain backprop gradient with a sampled label, so s backward passes give every
     tensor's probes at once (study test).
7. **Re-extract at each refresh; don't track the subspace.** [L, PARSEC offline] One power step per
   update tracks worse than a fresh probe extraction at r = 4, 8 and 16 (median sin²Θ 0.088 / 0.081 /
   0.077 versus 0.044 / 0.055 / 0.073). The whitened top subspaces rotate about 0.05–0.06 per step.
8. **No line search on the batch being fitted.** [T for the effect: post-hoc analysis of the stored
   seed 1–3 training logs, 8 October; not in a RESULTS file. H for the remedy] Share of each step's loss
   drop on its own batch that shows up on the next batch:

   | Steps | 2–50 | 51–200 | 201–499 |
   | --- | ---: | ---: | ---: |
   | Native SOAP | 76% | 24% | 3% |
   | Searched SOAP, tuned | 57% | 7% | 0.7% |
   | Per-tensor GGN (γ 0.001) | 20% | 0.7% | 0.1% |
   | Per-tensor OT (γ 0.1) | 33% | 1.5% | 0.2% |

   After step 50, curvature steps chosen by search on the same minibatch mostly fit that minibatch.
   Early, GGN's 20% is still the most real progress per step of any arm. Remedy (untested, N1): a fixed
   tuned learning rate or a step size chosen on another batch. Heavier damping also helps late: OT at
   γ = 0.1 transfers more than GGN and ended best among curvature methods (0.165).

   Remedy tested offline [L, optimism follow-up, six states], against same-batch search:
   - a step size chosen on a second 512-example batch gives a higher held-out decrease for 50 of 60
     searched candidates;
   - mean held-out decrease 0.0363 vs 0.0259, recovering 92% of the oracle's gain over same-batch
     search;
   - subtracting the estimated optimism α·tr(A⁻¹Σ̂)/B wins in 14 of 18 label-free cases;
   - PARSEC-L with same-batch search had the worst late held-out decrease (−0.035 at SOAP-50).

   Any step-length adaptation in PARSEC-L must use another batch.

   Loss-free step sizes tested [L, transfer-law study, six states]. A step α = γ̂·q/κ̂ is set from noise
   statistics and the GGN curvature along the step, with no loss evaluations:
   - overall it reaches 52% of second-batch selection, below even same-batch search (72%);
   - at steps 25 and 50 it reaches 96% and beats same-batch search in 14 of 16 points;
   - at step 10 it overshoots (8%), because the GGN quadratic underestimates the curvature along the step by up
     to 2.5×;
   - a full Newton or Wiener step without search is catastrophic (held-out loss −3.5 to −20.7).

   Keep loss evaluations on a second batch for the step length. Noise statistics can correct a searched step,
   not replace the search.
9. **Curvature pays per step early, not per second or late.** [T] Exact GGN reaches the step-50 GGN
   target in 58 steps (native SOAP 142) but needs 350 s against SOAP's 15 s. Tuned searched SOAP's loss
   is 1.8–2.1 times GGN's at step 50 and 1.09–1.20 times at step 500. PARSEC-L must
   keep its overhead within a small multiple of SOAP's 0.11 s per step.
10. **Rank around 16, with diminishing returns.** [L] Share of exact-GGN headroom: r = 8 about 57–61%,
    r = 16 about 63%. In PARSEC default training, a larger rank and more frequent refreshes helped
    (seed-0 tuning: rank 8 0.2126, 16 0.1826, 32 0.1738).

## 3. Reference numbers

| Seeds 1–3 | Loss 50 | Loss 500 | s/step (measured, 4 workers) |
| --- | ---: | ---: | ---: |
| Native SOAP, lr 1e-2 | 0.904 | **0.161** | 0.11 |
| OT per-tensor, γ 0.1 | 0.451 | 0.165 | 6.6 |
| Exact per-tensor GGN, γ 0.001 | **0.355** | 0.182 | 6.5 |
| PARSEC default (rank 32, refresh 10, lr 3e-3) | 1.380 | 0.186 | 0.83 |
| PARSEC split (rank 16, refresh 100, lr 1e-2) | 1.077 | 0.192 | 0.16 |
| Adam, lr 1e-2 | 1.113 | 0.202 | 0.09 |
| Searched SOAP, tuned | 0.697 | 0.210 | 1.37 |
| Full OT / full GGN | 0.386 / 0.670 | 0.276 / 0.330 | 9.0 / 9.3 |
| Dense empirical Fisher | 0.705 | 0.306 | 1.86 |
| SOAP map without momentum | 1.474 | 0.495 | 1.32 |

Local share of exact per-tensor GGN headroom over the SOAP map (gradient RHS):

| Family | Six main ViT states (2c) | 69 states across ViT architectures (2c + partial 4b) | Source |
| --- | ---: | ---: | --- |
| PARSEC-L exact, r = 16 | 56.1% | 63.0% | capacity screen; PARSEC offline |
| PARSEC-L probes, s = 4, r = 16 | — | 64.4% | PARSEC offline |
| Full-GGN CG, 5 matvecs | 67.1% | 68–114% by architecture | gap audit 2c/4b |
| Diagonal in nearest-Kronecker eigenbasis | 46.0% | 36–75% by architecture | gap audit 2c/4b |
| Kronecker sums, rank 1–3 | 15–19% | — | gap audit 2c |
| Exact diagonal in SOAP's basis | 21.0% | 9–13% by architecture | gap audit 2c/4b |

Gap-audit 4b stopped at 21 of 22 units on its own solver-agreement assertion, so its ranges are partial.

## 4. Idea status

IDs I1–I15 and SN-1 refer to [PARSEC_ideas.md](PARSEC_ideas.md); N-IDs are added here.

| ID | Idea | Status | Evidence / next test |
| --- | --- | --- | --- |
| I1 | Per-example sampled-label probes as the curvature source | **Supported locally** (statistics) | I2's result covers the statistics. The training arm (dense MC-GGN, s ∈ {1, 4}) has not run; it is optional now that PARSEC-L itself can be tested. |
| I2 | Per-tensor extraction in SOAP-whitened coordinates | **Passed** (97.8% ≥ 70%) | PARSEC offline B1 |
| I3 | Persistent one-step tracking instead of refreshes | **Killed** | PARSEC offline B2: tracking worse than fresh extraction for every r. Revisit only with several power steps per update. |
| I4 | SOAP bulk, not Adam bulk | **Adopted** (rule 1) | Tuned original PARSEC branches stay behind native SOAP |
| I5 | One lr, one momentum; precondition the momentum | **Adopted** (rule 5) | Two-lr variants: lr_top ≥ 0.1 diverged |
| I6 | Fix split-branch leakage | **Moot** under PARSEC-L | Measured leakage: up to 36% of the complement step lies in the top space |
| I7 | Early-phase schedule (raise μ or stop refreshing) | **Open**, motivated by rules 8 and 9; implemented as `probe_until` | Smoke (1 seed): hand-over at 30 or 50 made step 100 worse (0.375 / 0.399 vs 0.343). Test in stage D: hand-over N ∈ {100, 200, ∞} |
| I8 | Per-layer μ from probe traces; evidence-based damping | **Open** | Probe trace error 2.8% at s = 4 (PARSEC offline) |
| I9 | Step-size safety (λ_max cap, spectral trust region) | **Partly built in** (shrinkage only) | Test only if stage D shows instability |
| I10 | Adaptive SOAP basis refresh | **Open** | Refresh every step: better at step 50, worse at 500 (seed-0 tuning) |
| I11 | Gradient-noise scale and NI-SOAP noise from the same probes | **Open**, low priority | — |
| I12 | Reduced cross-layer coupling | **Parked for the optimizer** (rule 2) | Keep for UQ only |
| I13 / SN-1 | SN as architecture-side conditioning | **S2 not confirmed** (partial test); **no optimizer effect at matched settings** [T] | Stream-writer Kronecker error not consistently lower (ViT 5 of 6 pairs). GGN and SOAP train the same with and without SN at c = 0.95 (see below the table). The full SN-1 test (paper recipe, c ∈ {0.5, 1, 3}) has not run. |
| I14 | Laplace posterior as a byproduct | **Open** | — |
| I15 | Cheaper SOAP from shared, forward-computable statistics | **Open** | Zero-cost first check on saved states (cross-layer off-diagonal energy) not yet run |
| N1 | Same-batch line search makes curvature steps fit the minibatch | **Established** [T] (rule 8) | Test: GGN or PARSEC-L with the step size chosen on a different batch |
| N2 | Full GGN rank ≤ B(K−1) = 4,608 < 5,738 parameters | Fact | Bears on any full-matrix variant at batch 512 |
| N3 | Tight SN keeps residual branches alive and lowers early sharpness | **Observed** [L, exploratory] | MLP under searched SOAP, step 200: without SN every branch is dead (0% active GELU units); at c = 6 almost every branch is dead (0–41% per block); at c ≤ 2 34–62% stay active. ViT at c = 0.95: top GGN eigenvalue 74 versus 179 at step 10; diagonal and empirical-Fisher inverse errors lower in all six pairs. Untested in PARSEC training. |
| N4 | Check branch activity before reading late curvature diagnostics | Practice | Dead branches make Kronecker errors trivially 0 |
| N5 | SOAP as two half-steps: z = m̃ ⊘ √d, shrink z along w_k, unwhiten | **Same operator as section 1** | Algebraically the PARSEC-L step with λ = μ · scale. Only an implementation layout; nothing new to test. |
| N6 | Warm-started tracker step every k steps (W ← orth(Y Yᵀ W)) | **Conflicts with I3**; implemented as `refresh='track'` (with Rayleigh–Ritz) | One power step every step already tracked worse than fresh extraction (B2). Every k steps the subspace moves k times as far. Use re-extraction (randomized SVD, 2 power steps), optionally seeded with the old W (untested). The whitened coordinates also move every step as ṽ changes, not only at basis refreshes. |
| N7 | Rank-one probe columns y_j = (Q_Lᵀδ_j)(Q_Rᵀa_j)ᵀ ⊘ √d; rank-one storage of w_k | **Wrong for transformers as stated** | A per-example gradient of a token-shared weight is Σ_t δ_t a_tᵀ, rank up to T, so each probe column is dense or token-factored (the offline study used dense per-example gradients). Storing w_k as u_k v_kᵀ is an extra approximation; untested. |
| N8 | Per-side SOAP refresh when ρ = ‖offdiag(S)‖/‖S‖ of the probe partial traces has *risen* by τ | **Open** (extends I10, I15); implemented as `soap_refresh='triggered'/'both'`, `rho_threshold` | Validate offline first: does high ρ predict a better step from a refreshed basis, and does it fire more often early? Use S = Σ_j G_j G_jᵀ with dense G_j for token-shared weights. |
| N9 | Build SOAP's bases from the probe GGN (S_L, S_R) instead of GGᵀ | **Open** | Hint only: the exact GGN diagonal in a nearest-Kronecker basis recovered 46% versus 21% in SOAP's basis (gap audit 2c). That was an inverse oracle, not an Adam second moment in that basis. |
| N10 | Tracker residual e = ‖K̂W − W diag κ‖/‖K̂W‖ as a re-extraction trigger | **Open**, cheap; implemented as `residual_threshold`, always logged | In the supplied source, `innovation_residual` is only a state field (`eigentools_excerpt.py`); `parsec_pasted.py` never sets or reads it. |
| N11 | Correction energy c = 1 − ‖z′‖²/‖z‖²: stop probing when it stays low | **Open** (automates I7); implemented as `stop_energy`, c logged every step | Log c in stage D first; decide the rule afterwards. |
| N12 | Rescale PARSEC-L's step globally to SOAP's step norm | **Open** | Lets SOAP's tuned lr carry over; no longer "only shrinks". Never per layer (allocation matters). Implemented as `norm_blend` (0 = off). Smoke: PARSEC-L's step is only about 12% shorter than SOAP's on average; a = 1 helped at step 50 (0.672 vs 0.717) and slightly hurt at 100. |
| N13 | Fixed depth branch scaling s = α/depth^p (residual h + s·branch) as architecture-side conditioning instead of SN | **Open**; implemented as `with_branch_scaling(config, exponent, alpha)`, exponent 0 = plain; block-Jacobian instrument `block_jacobian_deviation` implemented. Smoke (1 seed, PARSEC-L, 100 steps): plain ViT blocks are far from identity (first block RMS ‖R−I‖ 1.7 at init, 2.5–3.0 after training); at depth 4, 1/depth scaling keeps all blocks near 0.6 and gave the best loss (0.319 vs 0.396 plain); at depth 2 it did not help | Targets the block Jacobians R_k = I + s·DF_k that the transport argument needs. Weight SN does not bound LayerNorm or attention, and c = 0.95 per layer bounds a GELU branch only by about 1.02. |
| N14 | Depth test: depth {2, 4, 8}, ViT and MLP, width 16, plain vs branch scaling (SN as reference), with per-block ‖DF_k‖ from random JVP probes | **Proposed** | Prediction: Jacobian deviation, Kronecker error and the SOAP–GGN gap grow with depth in plain networks and branch scaling flattens it. Kill: Jacobian deviation stays small with depth even when plain. Rationale (other assistant; paper table not verified here): depth, not training length, correlates with worse distance preservation. |
| N16 | Signal/optimism theory of transfer: share = (1 − r)/(1 + r) under same-batch search, r = tr(A⁻¹Σ)/(B ḡᵀA⁻¹ḡ) (B·r = noise scale in the optimizer's metric) | **Partly supported** [L, optimism study] | Banked states, 24 points: calibration partial (Spearman 0.56, MAE 0.24); r ordered SGD < SOAP map < exact GGN in 6/6 states and rising with training (exact GGN 0.35–0.58 at step 10, 1.36–1.52 at step 50); EF-Newton breaks the predicted ordering (its A is built from the batch's own gradients); the realized line-search mechanism explains shares (Spearman 0.86); momentum shares are not predicted by the expectation. Next: noise-corrected line search or second-batch step size (N1). |
| N16 follow-up | Refined theory and PARSEC-L under it | [L, optimism follow-up] | EF ceiling holds (≤ 0.81, so batch-built preconditioners self-normalize). The mechanism holds when measured independently (Spearman 0.80). Label-free calibration (post hoc): Spearman 0.70, MAE 0.23. Momentum staleness is higher on the large-step GGN trajectory (SOAP-map and GGN metrics, not SGD's). **PARSEC-L (damping-calibrated, γ 0.01) has the largest r of all operators (0.41–3.33).** Its κ of 1,500–7,300 removes the top-16 whitened directions, which carry most per-step progress at these step sizes. At SOAP's real step it keeps 79–91% of SOAP's norm and SOAP's transfer share (5 of 6 states), but beats SOAP's held-out decrease only where SOAP's own step hurts (2 of 6). |
| N17 | Learning-rate calibration β = c·η_t instead of β = 1/(μ·scale): SOAP's step is a proximal step for ‖ζ‖²/(2η) in whitened coordinates, so adding curvature gives κ_k = c·η·λ_k, the edge-of-stability ratio; no γ | **Supported in training** [T, stage D]: c = 4 at lr 1e-2 beats tuned SOAP at step 500 in all seeds. In the seed-0 grid, c = 1 ≈ SOAP and c = 30 / damping are in between; every brake beats SOAP at lr 3e-2, but the optimum stays at lr 1e-2. Per tensor, SOAP's sharpest directions run at ηλ_max ≈ 4. The full-network value is 10–27, far below the EMA-momentum stability limit of 78, so this is not edge of stability. Mechanism (brake_mechanism study): gradient noise is concentrated in underdamped sharp directions (57–70% in 11% of dimensions), and the brake lowers the noise floor there without slowing them. | At the six states κ_max = 0.03–0.66 (no overshoot by the quadratic model). The step keeps 96–100% of SOAP's norm and its held-out decrease is within 0.0015 of SOAP's where SOAP helps. It helps less than damping calibration at GGN-10, where SOAP's harm is not curvature overshoot. Needs a stage-D comparison: calibration ∈ {damping γ 0.01, lr c ∈ {1, 4}}. |
| N15 | Spectral-norm trust region on the curvature step (I9) to get SN's early batch-fit reduction directly | **Proposed** | SN raised GGN's steps-1–50 transfer from 19% to 32% by shrinking the own-batch decrease; a step-level bound would act on the same same-batch fitting (N1) without changing the architecture. |
| N18 | Momentum-free top space (`top_rhs='gradient'`): inside the held subspace, replace SOAP's momentum by the gradient, then brake with κ = c·η·λ | **Supported in training** [T, topfree_confirm; damping sweep] | Fresh seeds 4–6: c = 64 + EMA 0.121 vs SOAP + EMA 0.135 and SOAP + WSD + EMA 0.135, better in all seeds. Sweep: c ∈ {256, 1024, 4096, 10⁹} all mixed vs 64, and seed means get worse with c; freezing is worst. The mechanism is open: the noise-only story failed the OU check (N19). |
| N19 | Per-direction Kalman gain for the top coordinates, from an OU noise model with σ_k² and λ_k per direction | **Not built** [L, damping sweep] | Gate failed: S2 flat or undetermined, O1 not adequate (measured deviations slow, 16–110× the damped-noise prediction under top-free, non-stationary). Next: measure capture of the exact top directions by the held subspace, and distance to the valley floor g_k/λ_k, on the saved checkpoints; refit the model with leakage and drift. |
| N20 | Transfer law for batch-fitted steps with the finite-corpus correction r = (1 − f)·B_noise/B, the new-data gap r_pop = B_noise/B, and statistics-only step sizes (γ̂/λ̂, Wiener) | **Partly supported; step rule fails** [L, transfer-law study] | Corpus share: Spearman 0.96, MAE 0.18 (partial). Accurate for SGD and the SOAP map (MAE 0.07 / 0.11) and for every metric at B ≥ 1024. Overstates harm for exact GGN at small B (predicted −0.5 to −0.9, measured about 0). The noise part is exact; the miss is curvature: along a batch-fitted step the batch's own curvature is up to 4× the corpus curvature (κ ratio 0.25 at B = 64), so the batch search self-limits. The new-data gap is real (exact GGN at f = 0.82: corpus 0.92–0.99, new data 0.57–0.95). B_noise for exact GGN is 195–1,204 examples. Loss-free γ̂/λ̂ step: 52% of second-batch selection (fails), 96% late, 8% early. Wiener full step: catastrophic; gate failed, so B2 is not evaluated. The proposal's "curvature optimism" is not supported: another batch's curvature changes the share by at most 0.06. |

**Does SN help optimizers? Not at matched settings (first direct test).** [T, SN x GGN study]
Per-tensor GGN (γ 0.001, same-batch search) and native SOAP (lr 1e-2), each with and without SN at
c = 0.95, main ViT, seeds 1–3, 250 steps, settings tuned without SN:

- All nine readings are mixed: SN versus no SN at steps 50, 100 and 250 for each optimizer, and the
  SOAP/GGN loss ratio. Step-250 means: GGN 0.212 versus 0.213; SOAP 0.213 versus 0.221.
- GGN's advantage is unchanged: T50 in 58 steps versus SOAP's 142, with and without SN.
- SN raises GGN's next-batch transfer in steps 1–50 from 19% to 32% in every seed. It does so by
  shrinking the own-batch decrease (0.21 to 0.14 per update); real next-batch progress is unchanged
  (0.040 versus 0.044 ± 0.005). After step 50, GGN updates make no measurable next-batch progress
  with or without SN.
- In raw parameters SN halves the GGN trace and doubles the raw step norm and spectral norms.

Earlier indirect signals (SN curvature study, searched SOAP, 2 seeds, 200 steps): early sharpness
halves (top GGN eigenvalue 74 versus 179 at step 10); exact GGN's local headroom over SOAP at step 10
shrinks (0.118 versus 0.198); diagonal and empirical-Fisher estimates approximate the damped inverse
better in all six states; searched SOAP's next-batch transfer in steps 51–199 is 14% versus 8%
(post-hoc); validation loss at step 200 is about the same (0.274 versus 0.265).

Still open: SN after retuning γ or lr; longer runs, where c = 2 activates; SN with PARSEC-L; SN's
effect on Laplace or OOD quality, which these in-distribution runs do not measure.

## 5. Parked or rejected

- **Global Lanczos ranks** (original PARSEC): only match per-tensor GGN at step 500 and are slowest
  early; a global basis spends its rank on a few sharp tensors.
- **Split Newton top step with its own learning rate:** unstable at useful sizes, negligible at stable
  ones.
- **Cross-tensor coupling in the optimizer** (rule 2); **true-label empirical Fisher** (rule 6).
- **Kronecker sums fitted in Frobenius norm** (15–19%); **allocation-only variants** (local gains did not
  survive training).
- **Learned GNN predictor:** its gate failed; revisit only if a cheap family needs predicting. PARSEC-L
  probes compute the operator directly.
- **One-step subspace tracking** (I3, killed).

## 6. Open questions and next tests (in order)

**Current next test (2026-10-11):** revise the noise model offline on the `topfree_confirm` seed-4 checkpoints.
Measure how much of each exact top direction the held subspace captures (leakage) and the distance to the valley
floor g_k/λ_k (drift), before any filter (N19). Then run c = 64 top-free on longer horizons and other models. The
list below predates stage D and is kept for its remaining items.

1. **Stage D: PARSEC-L training** (spec v1) against tuned native SOAP and per-tensor GGN, seeds 1–3, 500
   steps, steps and seconds to target. Tune lr and γ on seed 0. Ablations: r ∈ {8, 16}, refresh interval,
   I7 hand-over. This is decisive, because local gains have repeatedly failed to survive training
   (allocation, empirical Fisher, full GGN).
2. **Refresh interval:** how fast does a k-step-old extraction lose its local value? This can be measured
   offline along a saved every-step trajectory. The smoke run's held-subspace residual (0.5–0.86 after ten
   steps) suggests probe_every 10 may be too long.
3. **Cost:** time s = 4 backward passes plus the per-tensor SVDs against SOAP's step. The parallel
   speedup across tensors needs GPU batching.
4. **N1 remedy:** step size from a held-out batch for curvature methods.
5. **SN, remaining:** retuned or longer runs and SN with PARSEC-L (matched settings showed no effect;
   section 4); **I15 zero-cost check;** the full **SN-1** test with the paper's recipe; **longer
   horizons** (the evidence here ends at 500 steps).

## 7. Sources

- Gap audit (checks 1–4): [../studies/gap_audit_20261007/RESULTS.md](../studies/gap_audit_20261007/RESULTS.md)
- Geometry follow-up (tuned PARSEC, OT, full versus per-tensor): [../studies/geometry_followup_20261008/RESULTS.md](../studies/geometry_followup_20261008/RESULTS.md)
- SN curvature (A–C): [../studies/sn_curvature_20261008/RESULTS.md](../studies/sn_curvature_20261008/RESULTS.md)
- PARSEC offline (I2, I3, SN-1 partial): [../studies/parsec_offline_20261008/RESULTS.md](../studies/parsec_offline_20261008/RESULTS.md)
- SN x GGN training (SN with GGN and SOAP, 250 steps): [../studies/sn_ggn_20261008/RESULTS.md](../studies/sn_ggn_20261008/RESULTS.md)
- Capacity screen: [../docs/learnable_capacity_results_20261007.md](../docs/learnable_capacity_results_20261007.md)
- Signal/optimism test (N16): [../studies/optimism_20261009/RESULTS.md](../studies/optimism_20261009/RESULTS.md)
- PARSEC-L implementation and smoke run: [../studies/parsec_l_20261009/README.md](../studies/parsec_l_20261009/README.md)
- Original source: `references/user_optimizers/parsec_pasted.py` (`pns_eigenadam`)

## 8. Changelog

- **2026-10-08** Created. Banked rules 1–10, the reference tables and the idea status after the gap
  audit, geometry follow-up, SN curvature and PARSEC offline studies. I2 passed, I3 killed, S2 not
  confirmed, N1–N4 added. Spec v1 proposed for stage D.
- **2026-10-08** Added N5–N12 from the written-out "SOAP half-step + shrinkage" algorithm (refresh
  signals ρ, e, c). Started `studies/sn_ggn_20261008` (per-tensor GGN and native SOAP with and without SN,
  250 steps).
- **2026-10-08** SN x GGN study complete: all readings mixed; SN raises GGN's early next-batch transfer
  (19% to 32%) without changing real progress. Section 4 SN answer and I13 updated.
- **2026-10-09** Implemented PARSEC-L as a separate PyTorch optimizer (`studies/parsec_l_20261009`), with
  extract/track/exact refresh, ρ-triggered per-side SOAP refresh, residual trigger and stop rule as
  switches, plus depth branch scaling (N13). 11 tests pass; one-seed smoke run added. N13–N15 added.
- **2026-10-09** PARSEC-L gained `probe_until` (I7), `norm_blend` (N12), batched extraction ('auto':
  CPU per tensor) and an optional ρ signal; `block_jacobian_deviation` added for N13/N14. 15 tests pass;
  one-seed smoke ablations added. N16 (signal/optimism theory) tested on banked states: partly supported.
- **2026-10-09** Optimism follow-up: second-batch step size validated offline (50 of 60, 92% of oracle); PARSEC-L's damping calibration found to remove its most productive directions at trajectory step sizes (N16 follow-up); learning-rate calibration proposed (N17). Norm blend 1 and 1/depth branch scaling made standard settings by owner decision, with their off-settings kept as required ablations.
- **2026-10-09** PARSEC-L gained `calibration='lr'` (N17), `rhs='gradient'`, step length chosen on a selection
  batch with a model-free hand-over to SOAP, and the true-label optimism diagnostic; 21 tests pass. Stage-D
  protocol drafted (`studies/stage_d_20261009/PROTOCOL.md`), awaiting approval; runs deferred by the owner.
- **2026-10-11** Stage D complete. The brake (lr calibration, c = 4) beats tuned SOAP at step 500 in all seeds
  (N17 supported). Early-Newton (selection-batch step) was mixed and unstable; its model-free hand-over never
  triggered. GGN with a selected step was better at step 200 and mixed at 500 (N1 remedy, partial). Spec v2
  proposed.
- **2026-10-11** Brake mechanism checked offline: not edge of stability (full preconditioned sharpness 10–27
  vs limit 78; an earlier edge-of-stability claim was an error and is corrected). Noise is concentrated in the
  braked directions (57–70%), and the brake lowers training loss on identical batches in all seeds. Mechanism:
  selective annealing of underdamped sharp directions.
- **2026-10-11** Noise controls: SOAP + WSD decay or + weight EMA beats the constant brake in all seeds. The
  brake adds nothing consistent on top (mixed). Mechanism N (noise) supported; S, E and T not. Stage-D claim
  qualified. Momentum-free top space (c = 16) is the remaining PARSEC-L candidate.
- **2026-10-11** Top-free confirmation on fresh seeds 4–6: momentum-free top space with c = 64 + EMA beats SOAP +
  EMA and SOAP + WSD + EMA in all seeds (0.121 vs 0.135). In-loop damping adds beyond averaging. Next: larger c,
  per-direction Kalman gain, OU check on the saved checkpoints, longer runs and other models.
- **2026-10-11** Damping sweep and OU check (`studies/damping_sweep_20261011`). c ∈ {256, 1024, 4096, 10⁹} are all
  mixed against c = 64, with seed means worsening towards freezing; c = 64 stays. The OU noise model is not
  adequate for any arm (slow deviations; 16–110× the prediction under top-free; non-stationary). By the
  pre-declared rule, no Kalman version is built. N18 and N19 added.
- **2026-10-11** Transfer-law test (`studies/transfer_law_20261011`, offline, six states).
  - A1–A3 partial: the law ranks transfer (Spearman 0.96) but overstates small-batch harm for curvature steps,
    because the batch curvature along its own step is inflated.
  - B1 fails: the loss-free γ̂/λ̂ step reaches 52% of second-batch selection (96% late, 8% early).
  - B2 not evaluated: the Wiener gate failed at a 2e-8 forward difference on a block with condition number 1e10;
    the full Wiener step was catastrophic as a diagnostic.
  - A scale bug for the SOAP map was found, fixed and rerun.
  - N20 added; rule 8 extended. Repository `.gitignore` extended for the GPU move (separate from the study).
