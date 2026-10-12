# PARSEC idea bank

Status: idea bank, started 8 October 2026. Nothing in this file is a selected protocol or a
result. Every idea names the evidence it rests on, a first test and a kill criterion. An idea that
gets implemented gets its own prospective protocol (see `studies/`). Numbers quoted here come from
the linked reports; check the reports before citing them.

## 0. Name

The original JAX code is `pns_eigenadam` ("PN-S EigenAdam", `references/user_optimizers/`): top-k
global GGN eigenpairs from Lanczos, a "partial Newton-like preconditioner in a global eigenbasis",
then Adam. The expansion of PARSEC was never written down. Candidates:

- **PARtial SEcond-order Curvature**: closest to the original intent.
- **PARtial-Spectrum Eigen-Correction**: describes what the method does in both the global and
  the layerwise form. A cheap bulk preconditioner is corrected in a partial (top-k) spectrum.
- **PARallel Second-order Eigen-Correction**: fits only the layer-parallel variant.

Working proposal: PARtial-Spectrum Eigen-Correction. The layerwise variant below is **PARSEC-L**.

## 1. What we know

Workload for every row: ViT width 16, 2 blocks, MNIST 28x28, batch 512, CPU float64. Fresh seeds
1–3, 500 consumed steps, each arm tuned on seed 0. Validation loss is the mean over seeds.
Sources: `studies/gap_audit_20261007/RESULTS.md` and `studies/geometry_followup_20261008/RESULTS.md`.

| Arm | Loss 50 | Loss 500 | Measured s/step (4 workers) |
| --- | ---: | ---: | ---: |
| Native SOAP, lr 1e-2 | 0.904 | **0.161** | 0.11 |
| OT tensor blocks, gamma .1 | 0.451 | 0.165 | 6.6 |
| Exact tensor GGN, gamma .001 | **0.355** | 0.182 | 6.5 |
| PARSEC default (rank 32, refresh 10, lr 3e-3) | 1.380 | 0.186 | 0.83 |
| PARSEC split (rank 16, refresh 100, lr 1e-2) | 1.077 | 0.192 | 0.16 |
| Adam, lr 1e-2 | 1.113 | 0.202 | 0.09 |
| Searched SOAP, tuned | 0.697 | 0.210 | 1.37 |
| Full OT / full GGN (cross-tensor blocks included) | 0.386 / 0.670 | 0.276 / 0.330 | 9.0 / 9.3 |

These findings constrain the design:

1. **The bar is tuned native SOAP, not Adam.** Tuned PARSEC beats tuned Adam and tuned searched
   SOAP at step 500, but not native SOAP.
2. **Curvature pays early and per step, not per second.** GGN reaches the T50 target in 58 steps;
   native SOAP needs 142 steps but only 15 seconds.
3. **Per-tensor (block) curvature beats full curvature in training,** with damping, metric and
   search held fixed, even though the off-diagonal blocks carry 54–92% of ‖C‖²_F. Layerwise is the
   right scope. Untested caveat: full gamma was not retuned, and at batch 512 the full GGN has rank
   at most 4,608, below the 5,738 parameters.
4. **The tuned split branch is Adam in disguise.** The top-space Newton step averages 0.67% of the
   complement step, and lr_top >= 0.1 diverged. Adam's coordinatewise normalization puts up to 36%
   of the complement step back into the top space.
5. **The default branch transforms Adam's input,** and Adam's normalization partly undoes the
   transform: fixed positive coordinate scalings cancel exactly.
6. **Calibrating the complement matters.** The teacher rank-16 core recovers 15.4% of local headroom
   under trace calibration but 56.1% under damping calibration
   (`docs/learnable_capacity_results_20261007.md`).
7. **The RHS matters.** Removing SOAP's momentum ruins training (0.495 at step 500). The best local
   operator result is teacher rank 16 applied to SOAP momentum (0.046187, versus 0.045883 for exact
   GGN on the gradient).
8. **The probe type matters.**
   - Model-sampled labels, 4 per example, give 0.132 / 0.078 local decrease, versus 0.135 / 0.085
     for exact GGN (SOAP states 10 / 25; `docs/jacobian_information_results_20261005.md`).
   - True-label empirical Fisher trains badly: 0.306 at step 500 (gap audit 2a).
   - Global probe sketches (summed over examples) failed in the early MLP pilots.
9. **Basis staleness cuts both ways.** On seed 0, refreshing SOAP's basis every step was better at
   step 50 and worse at step 500.
10. **Local capacity results:**
    - Matrix-free full-GGN PCG with 5 matvecs recovers 67% (68–114% across nine ViTs); it was not
      tested in training.
    - The diagonal in the best Kronecker basis recovers 46% (36–75% across architectures).
    - Kronecker sums recover 15–19%.
    - The SOAP-basis diagonal recovers 9–21%.

## 2. Target design: PARSEC-L

Per tensor b, let F0_b be SOAP's actual factor (P_b = F0_b F0_bᵀ is SOAP's preconditioner) and μ
the trace-based damping. Use damping calibration, β = 1/(μ · scale), so that in SOAP-factor
coordinates

    K_b = I + K_C,b,   K_C,b = F0_bᵀ C_bb F0_b / (μ · scale).

Let (w_k, κ_k), k ≤ r_b, be the top eigenpairs of K_C,b. Let u be SOAP's effective momentum,
defined so that −F0 F0ᵀ u is SOAP's native step (the lab's `momentum_rhs`; one EMA buffer).
The update is

    Δ_b = −η F0_b [ I − Σ_k (κ_k / (1 + κ_k)) w_k w_kᵀ ] F0_bᵀ u_b

This is the native SOAP step, shrunk in the r_b whitened directions where the GGN is large relative
to SOAP's metric. It is exactly the capacity screen's damping-calibrated teacher_r family, made
deployable by replacing exact blocks with probes. The physical-coordinate equivalent is the
Woodbury form (μΩ_b + V_b Λ_b V_bᵀ)⁻¹ with V_b = F0_b⁻ᵀ W_b.

Properties:

- one learning rate and one momentum;
- the curvature only ever shrinks a SOAP step, so the unmodified SOAP step is the r = 0 limit;
- a per-layer r × r cost;
- no global Lanczos.

## 3. Ideas

Each idea lists its evidence, a first test and a kill criterion (the bar that would end it).

### I1. Layer-separable curvature from per-example sampled-label probes

**What.** For s draws per example, ξ ~ categorical(p) one-hot minus p, so that E[ξξᵀ] = H. Run s
batched backward passes, keeping each layer's activations a_i and probe backprop signals δ_ij. For
a dense layer, Ĉ_bb = U_b U_bᵀ with columns vec(δ_ij a_iᵀ); for shared weights, sum over tokens.
Per-layer matvecs are then two local matmuls with no network pass, so every layer can iterate
independently and in parallel.

**Why.** A VJP splits by layer for free; a JVP does not. Per-layer Lanczos on the network itself
would cost L JVPs plus L VJPs per iteration. Sampled labels (not true labels, and per example, not
summed) matched exact GGN locally (fact 8).

**First test.** A dense MC-GGN training arm, s ∈ {1, 4}, full tensor blocks, otherwise identical to
`ggn_exact_trace`.

**Kill if** it closes less than 75% of the step-50 gap between tuned searched SOAP and tensor GGN.

### I2. Per-layer extraction in SOAP-whitened coordinates

**What.** Use a randomized SVD of F0_bᵀ U_b (q = 1–2 power steps), batched across layers of equal
shape on GPU.

**Why.** This targets the directions where SOAP is wrong (fact 6).

**First test.** Offline, on the six saved states and the 66 family states: probe-estimated
teacher_r (s ∈ {1, 4}, r ∈ {4, 8, 16}) against exact teacher_r. Measure local recovery and
principal angles.

**Kill if** s = 4 reaches less than 70% of exact teacher_r's recovery.

### I3. Persistent one-step tracking instead of refreshes

**What.** Keep W_b across steps and apply one block power or Oja step per update from the current
probe factor. This is the same trick spectral normalization uses for σ_max (one power step per
update, refined at inference).

**Hooks.** The original state already has `rotation_diff`, `innovation_residual`, `active_k` and
`last_refresh_step`; these fields are ready for adaptive refresh and rank.

**Why.** It removes refresh spikes, the sequential Lanczos loop and the staleness problem
(fact 9) in one go.

**First test.** Along a trajectory saved at every step for about 30 steps, measure subspace drift
per step and the tracking error of one-step tracking against exact extraction.

**Kill if** the tracking error exceeds the drift-free extraction error for r <= 16.

### I4. SOAP bulk, not Adam bulk

Covered by the design in section 2. This removes facts 4 and 5 by construction.

**First test.** PARSEC-L against PARSEC default and split, all tuned, against tuned native SOAP.

### I5. One learning rate and one momentum; precondition the momentum

**Evidence.** Facts 4, 6 and 7.

**Ablations.** β1 for the top space ∈ {0, β1}, since top directions often have higher
signal-to-noise. Also gradient versus momentum as the RHS.

**Kill** two learning rates unless a tuned two-rate variant beats the single-rate one in all seeds.

### I6. If a split variant is kept, fix the leakage

Apply (I − V Vᵀ) after Adam, not before; alternatively, run Adam in rotated coordinates. This is
cheap and checkable with the existing diagnostic (fact 4: 36% leakage).

### I7. Early-phase schedule

**What.** Let curvature act mainly early: raise μ over time, or stop refreshing after N steps, so
the method converges to SOAP.

**Evidence.** Facts 1 and 2. OT at heavy damping (gamma .1) was worse early but better late.

**Test.** Add a hand-over step N ∈ {100, 200, ∞} to PARSEC-L.

### I8. Damping

- Per-layer μ_b from probe traces, since tr Ĉ_bb = ‖U_b‖²_F.
- Optionally, evidence-based damping: μ plays the role of a prior precision, and the Laplace
  evidence of the same low-rank model gives it a criterion instead of a grid.

### I9. Step-size safety

- Cap the per-layer step with probe λ_max.
- Put a spectral-norm trust region on the curvature component (see the SN section): ‖ΔW_b‖₂ ≤ τ_b.

**Target.** The divergence of large Newton steps (fact 4).

### I10. Adaptive SOAP basis refresh

Refresh SOAP's eigenbases when principal angles between them and the probe top-r subspace exceed a
threshold. Fact 9 suggests a fixed frequency is wrong both early and late.

### I11. Noise from the same probes

- True-label and sampled-label signals from the same passes give the gradient-noise scale, usable
  for lr or batch schedules.
- They also give a correctly calibrated top-space noise for NI-SOAP, whose realized noise was
  99.9% fallback.

### I12. Reduced cross-layer coupling

With shared probes, VᵀĈV of size Σ r_b is available for free. Fact 3 says not to use it in the
optimizer; it is useful for the UQ byproduct (I14).

### I13. Spectral normalization as architecture-side conditioning

See section 4.

### I14. UQ byproduct (not a primary goal)

(V_b, Λ_b) is a running layerwise Laplace posterior. Block-diagonal Laplace double-counts
constraints along cross-tensor symmetry directions (the toy example f = q·k gives predictive
variance 2 instead of 1). I12 partially repairs this.

### I15. Cheaper SOAP from shared, forward-computable statistics

**What SOAP pays for.** For an m×n matrix with refresh every f steps (pinned source:
`third_party/soap/soap.py`):

- every step: factor accumulation GGᵀ and GᵀG plus two-sided rotations, O(mn(m+n)); these are
  matmuls and GPU-friendly;
- every f steps: one power-iteration step plus QR, O(m³ + n³), which is sequential and
  latency-bound (eigh only at initialization);
- memory: 2(m² + n²) for factors and bases.

In a standard transformer block (Q, K, V, O as d×d; MLP as d×4d and 4d×d):

- the six residual-stream sides (d) are half of the factors, but only ≈4% of refresh FLOPs and
  ≈14% of factor memory;
- the two MLP hidden sides (4d) are ≈93% of refresh FLOPs.

Ideas, ordered by expected saving:

**a. Adaptive refresh (with I10).**

- Refresh only when the basis has actually rotated. Detect this from the off-diagonal energy of
  Qᵀ L Q (a matmul) or from the change produced by one persistent power step.
- Evidence (seed-0 tuning, searched SOAP, betas .9/.99, loss at step 500): refresh 10 gives .2020,
  refresh 100 gives .2069, refresh 1 gives .2262; refresh 1 was best at step 50.
- So most refresh work can go if the schedule is front-loaded. SN (S5) should make this easier.

**b. Hidden-side reduction.**

- Use one-sided SOAP on the 4d sides; the pinned SOAP already skips any side larger than
  `max_precond_dim`.
- If needed, add a PARSEC-L rank-r correction on that side.
- Free lab test: `max_precond_dim = 16` leaves the hidden-32 sides unrotated.

**c. Shared stream bases.**

- Readers: Q, K, V (and w1) read the same LayerNorm output. The noise part of SOAP's E[GᵀG] is
  (1/B) E[‖δ‖² a aᵀ], which is ∝ E[a aᵀ] when ‖δ‖² is independent of a. The mean-gradient part
  ḠᵀḠ stays layer-specific, so sharing is approximate even within a block.
- Across depth, S1 keeps stream covariances close, so one reader basis per block, or per network.
- Writers (O, w2): the noise part (1/B) E[‖a‖² δδᵀ] shares δ_k ≈ T_kᵀ δ_n (S2), so one
  writer-side basis.
- Saves the stream-side eigendecompositions and about 14% of factor memory per block.

**d. Forward-only statistics for the stream sides.**

- Readers: a token-subsampled activation covariance.
- Writers: the head residual r = p − onehot(y) and the final-norm Jacobian are known after the
  forward pass. W_headᵀ E[‖a‖² r rᵀ] W_head (SOAP-style, true labels) or W_headᵀ E[H] W_head
  (GGN-style, sampled labels at the head) needs no backward pass, and is valid to O(ε) under SN.
- These bases do not depend on the backward pass, so they can be refreshed asynchronously, off the
  critical path.
- Caveat: in the ViT only the CLS path reaches the head directly. Decoder LMs, which have a head at
  every position, fit S2 better.

**Zero-cost first check.** On saved SOAP states, measure how diagonal one layer's factor is in
another layer's basis: the off-diagonal energy of Q_aᵀ L_b Q_a. Do this for readers within a block,
readers across blocks and writers across blocks. Then repeat with SN (part of SN-1).

**Kill c/d** if the cross-layer off-diagonal energy exceeds that of the same layer's own basis from
10 steps earlier. Sharing would then cost more than staleness.

## 4. Spectral normalization: why it might help PARSEC

**Source.** `UDBench_ICLR2027.pdf`, Section 5 and Appendix D.4, residual MLPs for regression.

- SN is a hard clip on every dense layer of the backbone, W ← W · min(1, c / σmax(W)), with
  c ∈ [0.5, 3] and σmax from persistent power iteration (one step per update).
- SN raised local distance preservation in all eight methods (pooled +0.21, versus +0.01 for strong
  weight decay).
- For Laplace and FSP-Laplace, the gain in uncertainty quality is mediated by preservation: the SN
  indicator vanishes once preservation is in the model.
- Deeper plain networks preserve distances worse (ρ_S = −0.57). When tuned with SN, networks picked
  deeper backbones and had lower error (NMSE ratio 0.38–0.48 for Laplace and FSP).

**Theory sketch.** Notation follows `docs/mathematical_model.md`: residual stream
h_{k+1} = h_k + F_k(h_k), R_k = I + ∂F_k/∂h_k, M_k = R_kᵀ M_{k+1} R_k, C_kk = B_kᵀ M_{k+1} B_k.
Let L_k = Lip(F_k); SN with constant c on both dense layers gives L_k ≤ c² · Lip(σ).

- **S1. Near-identity transport.** ‖R_k − I‖ ≤ L_k, so per example ‖T_{k→n} − I‖ ≤
  Π_j (1 + L_j) − 1 =: ε_k. Then ‖M_{k+1} − M_n‖ ≤ ((1 + ε_k)² − 1)‖M_n‖. The bi-Lipschitz bound
  (1 − L)^m ‖x − y‖ ≤ ‖Φ(x) − Φ(y)‖ ≤ (1 + L)^m ‖x − y‖ is the global form of the same statement,
  and it is what the UQ paper measured as distance preservation.
- **S2. A shared output factor gives Kronecker structure.** For a dense layer that writes into the
  stream (the second layer of a branch):
  C_kk = E_i[(a_i a_iᵀ) ⊗ M_{n,i}] + O(ε_k).
  Every stream writer then shares one output-side factor, the head metric. The remaining distance
  from a single Kronecker product is E[(aaᵀ) ⊗ M_n] − E[aaᵀ] ⊗ E[M_n], i.e. the dependence between
  activations and per-example prediction confidence.
  - SN removes the depth-transport part of the Kronecker error, but not the output-heterogeneity
    part.
  - First-in-branch layers keep an example-dependent factor D_kᵀ M D_k (through σ′), so SN does not
    help them in this way.
  - Prediction: the SOAP–GGN gap shrinks for stream writers specifically.
- **S3. No collapse of the input factor.** This concerns layers that read the stream (first in a
  branch, and the head), whose input a_k is the stream state up to LayerNorm. For these,
  2 tr Cov(a_k) = E‖a(x) − a(y)‖² ≥ (1 − L)^{2k} E‖e(x) − e(y)‖², so total activation variance
  cannot collapse. S2 helps stream writers through the output factor; S3 helps stream readers
  through the input factor. With S1, the spectrum
  of Cov(a_k) stays close to that of the embedding. The Kronecker input factor stays well
  conditioned, and the preconditioner is less dominated by damping.
- **S4. Bounded sharpness.** λ_max(C) ≤ E‖J_i‖² · λ_max(H_i), with λ_max(H_i) ≤ 1/2 for softmax
  cross-entropy (Gershgorin), and ‖J_i‖ ≤ ‖W_head‖ Π(1 + L_j) ‖a‖. This is an a priori step cap.
- **S5. Slow subspace drift (heuristic).** Davis–Kahan gives sin Θ(V_t, V_{t+k}) ≤
  ‖C_{t+k} − C_t‖ / gap. With spectrally bounded weights, ‖ΔC‖ per step is easier to bound; the
  eigengap is not controlled. If it holds, it supports one-step tracking (I3) and longer refresh
  intervals.
- **S6. A function-space trust region.** A finite perturbation of the stream at depth k moves the
  final stream by between (1 − L)^{n−k} and (1 + L)^{n−k} times its size, and
  ‖ΔW_k a_k‖ ≤ ‖ΔW_k‖₂ ‖a_k‖. A per-layer spectral-norm cap on the curvature component is therefore
  a calibrated function-space trust region (I9). This is the modular-norm argument made
  quantitative by SN.
- **S7. Side effect.** When the clip is active, W · c / σmax(W) is scale invariant. That creates a
  radial flat direction; the gradient is orthogonal to W, and norm growth acts as an implicit
  learning-rate decay. Log clip activity, and compare against a soft penalty.

**Caveats.**

- **ViT.** Non-CLS tokens reach the head only through attention, so S1–S2 apply cleanly to residual
  MLPs (the UQ paper's backbones and the lab's MLP) and to the CLS path.
- **Attention is not globally Lipschitz.** Dot-product attention has no global Lipschitz bound
  (Kim et al., ICML 2021), and pre-LN's constant depends on token variance.
- **Classification.** Classification benefits from within-class collapse, so distance preservation
  can cost accuracy there. The UQ evidence is for regression.

**Test SN-1.**

- Models: the residual MLP and the ViT, with c ∈ {plain, 3, 1, 0.5} using the UQ paper's recipe.
- Offline, at saved states: the paper's pres_loc on penultimate or CLS features; the
  rank-1 Kronecker (`ks_1`) error and `kron_basis_diag` recovery, separately for stream writers and
  first-in-branch layers; SOAP–GGN local headroom; subspace drift.
- Training: native SOAP, tensor GGN and PARSEC-L, each with and without SN.
- Predictions:
  - P1: the stream-writer Kronecker error falls as c falls;
  - P2: the SOAP–GGN gap shrinks;
  - P3: drift slows;
  - P4: with a spectral cap, a larger stable curvature step is possible.
- **Kill if** pres_loc rises but the stream-writer Kronecker error does not fall (S2 absent).

## 5. Parked or rejected, with reasons

- **Cross-tensor coupling in the optimizer:** it hurt training (fact 3). Keep it only for UQ.
- **True-label empirical Fisher as the curvature source:** it trained badly (fact 8).
- **Kronecker sums fitted in Frobenius norm:** 15–19% local recovery.
- **Allocation-only variants:** local gains did not survive training (`docs/mini_vit_results_20261005.md`).
- **Learned GNN predictor of the operator:** gate failed. No cheap family reached 60% with oracle
  parameters. Revisit only if I1–I2 give a family with enough capacity.
- **Global Lanczos with refresh 100:** the split arm was Adam for its first 100 steps.

## 6. Suggested order

Stages A, B and C can run in parallel; D depends on their outcomes.

- **A.** I1 dense MC-GGN training arm: decides whether probes are a valid curvature source.
- **B.** I2 and I3 offline, on existing saved states plus one 30-step every-step trajectory.
- **C.** SN-1 offline part.
- **D.** PARSEC-L training against tuned native SOAP and tensor GGN, with the I5, I7, I8 and I9
  ablations; seeds 1–3, 500 steps; report steps and seconds to target.
