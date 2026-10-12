# PARSEC-L offline tests: probe extraction (I2), one-step tracking (I3), SN stream writers (SN-1)

Study ID `parsec_offline_20261008`. Selected by the project owner on 8 October 2026 ("Offline first:
B + C") from `ideas/PARSEC_ideas.md` (SHA-256 `cd3958f9…`). This file is the prospective protocol,
written at 2026-10-08T16:25Z before any code or computation of this study; departures go to
`DEVIATIONS.md`. No training comparison is run: stages A (MC-GGN training arm) and D (PARSEC-L
training) are not part of this study.

## Isolation and conventions

As in the earlier side studies. Write only to this folder and `$PARSEC_OFFLINE_DATA` (default
`~/skipsoap-study-data/parsec_offline_20261008/`); read/import only from the main lab and earlier study
outputs; never modify the main lab's protected paths, `ideas/` or other `studies/` folders;
`PYTHONDONTWRITEBYTECODE=1`; pytest with `-p no:cacheprovider`; no git writes, installs, downloads or
official MNIST test data. Isolation manifest written first (16:22:12Z, 1,818 files) and compared at the
end. At most four single-threaded processes (two if diagnostics contend as in the SN study). CPU
float64, deterministic algorithms. Numerical tolerance 1e-8; checks use residual, action or energy
criteria. Code copied from earlier studies is recorded with its source SHA-256. `HANDOFF.md` is not
edited.

## The PARSEC-L operator (idea bank, section 2)

Per tensor b, with SOAP's frozen factor F0_b (P_b = F0_b F0_bᵀ), damping μ = max(.01 tr C / D, 1e-8)
(exact trace) and β = 1/(μ · SOAPMetric.scale):

    K_C,b = β F0_bᵀ C_bb F0_b,   P_r = β F0_b [I − Σ_{k≤r} (κ_k/(1+κ_k)) w_k w_kᵀ] F0_bᵀ,

with (w_k, κ_k) the top-r eigenpairs of K_C,b. With exact eigenpairs this is the capacity screen's
damping-calibrated `teacher_r` core (`optim.capacity.fit_core(..., 'teacher', r)`; ranking by
population gain equals ranking by κ when K = I + K_C ⪰ I). Candidates are −P_r v for v = current
gradient g or effective SOAP momentum m (`momentum_rhs`). All tensors (including 1-D) take the
operator, as in the screen.

## B1 (I2): probe-estimated versus exact teacher_r

**States.** The six main states (`runs/20261005T143029-9b582ed5/{searched_soap,exact_tensor_ggn}-{0010,0025,0050}`)
and the 66 architecture-family states (`~/skipsoap-study-data/gap_audit_20261007/runs/20261007T125950-63edd820/*/state-*`;
nine ViTs, searched-SOAP trajectories). Artifact checksums verified first.

**Probes.** For each example i of the 512-example construction batch and draw j ≤ s, a label
ξ_ij ~ categorical(p_i) (fixed generator, seed 303 + 7919·step + 1000·j-block), probe gradient
g̃_ij = J_iᵀ(p_i − onehot(ξ_ij)) from the exact per-example Jacobian, U = rows g̃_ij / √(B s), so
E[UᵀU] = C (block-wise). s ∈ {1, 4}; the s = 1 draws are the first draw of the s = 4 set.

**Extraction.** Z_b = √β F0_bᵀ U_bᵀ (n_b x Bs). Randomized SVD with oversampling 8 and q = 2 power steps
(QR re-orthonormalization; fixed Gaussian test matrix): top-r left singular vectors Ŵ_b and κ̂ = σ².
r ∈ {4, 8, 16} (min with n_b). The probe operator uses Ŵ, κ̂ in the same core; μ and β stay exact, so
the comparison isolates subspace and eigenvalue estimation. The probe trace's relative error
(tr Ĉ versus tr C) is reported as a diagnostic.

**Scoring.** As in the gap audit's 2c/4b: `search_candidate` (zero plus 2^[-16..16] native-radius
multipliers, 18 golden refinements) on construction loss, mean independent decrease on the four analysis
panels (`measurement`). Per state and RHS: SOAP map (identity core), exact teacher_r (r = 4/8/16), probe
teacher_r (s = 1/4 x r = 4/8/16). Also per tensor the principal angles between Ŵ_b and the exact top-r
subspace (energy-weighted mean of ‖sin Θ‖_F²/r over tensors) and the relative error of κ̂_1.

**Gate (≤ 1e-9; > 1e-6 stops the study).** Exact teacher_r for r = 4/8/16 on both RHS at the six main
states reproduces the capacity screen's damping-calibration rows (`runs/20261007T093903-8d7d86f1`).

**Pre-declared reading (idea bank kill criterion).** R(s, r, RHS) = (D̄_probe − D̄_SOAP-map) /
(D̄_exact-teacher − D̄_SOAP-map), means over all 72 states (undefined if the denominator ≤ 1e-4).
Primary: s = 4, r = 16, momentum RHS (PARSEC-L's RHS). **Kill I2 if R < 0.70.** Report every (s, r,
RHS) and the main/family subsets separately.

## B2 (I3): subspace drift and one-step tracking

**Trajectory.** Main ViT, searched SOAP with default settings, seed 0, data order seed 102, steps
1–50; every state from step 20 to 50 is analysed (31 states). Gate: parameters at steps 25 and 50 equal
the gap-audit 4a `b2_w16__s0` states bitwise (≤ 1e-12).

**Per state t.** Exact tensor blocks (`streamed_blocks`), SOAP contexts and factor F0_t, μ_t, K_C,t,
exact top-r subspaces W_t (r ∈ {4, 8, 16}); probes as in B1 (s = 4, a new draw per step).

**Transport.** Subspaces are compared in physical coordinates: W in whitened coordinates of step t maps
to V = F0_t^{-T} W; transported into step t+1 coordinates as orth(F0_{t+1}ᵀ V). This handles SOAP's
basis refreshes (every 10 steps).

**Quantities** (energy-weighted mean over tensors of ‖sin Θ‖_F²/r):
- drift(t) = angle(transport(W_t), W_{t+1});
- exact tracking: W̃_{t+1} = orth(K_C,t+1 · transport(W̃_t)), W̃_20 = W_20; error = angle(W̃_t, W_t);
- probe tracking: the same with the probe matrix Ẑ_{t+1}Ẑ_{t+1}ᵀ (s = 4) in place of K_C,t+1;
- fresh probe extraction error: angle(Ŵ_t, W_t), randomized SVD from scratch (B1's procedure).

**Pre-declared reading (kill criterion).** For each r ≤ 16, compare the median over t = 21..50 of the
probe-tracking error with the median fresh-extraction error. **Kill I3 if probe tracking is worse for
any r ∈ {4, 8, 16}.** Drift and exact tracking are reported for interpretation.

## C (SN-1, partial): stream writers versus first-in-branch layers

**States.** The 48 states of `~/skipsoap-study-data/sn_curvature_20261008/runs/20261008T130821-44610eee`
(ViT and MLP; SN off and c ∈ {0.95, 2, 6}; seeds 0, 1; steps 10, 50, 200). This uses that study's SN
recipe (exact σ, residual-branch weights only), not the UQ paper's (all dense layers, c ∈ {0.5, 1, 3},
power iteration); it is a partial test of prediction P1.

**Groups.** Stream writers: ViT `attention.output.weight` and `w2.weight`; MLP `w2.weight`.
First-in-branch: ViT `attention.{query,key,value}.weight` and `w1.weight`; MLP `w1.weight`. Input
projection and head reported separately.

**Quantities.** Exact tensor GGN blocks recomputed at the saved parameters (action checks against
`GGN.action` at 1e-8). Primary: effective-weight parameterization (the dense layers' own GGN, the object
of S2); secondary: raw parameterization. Per group, the GGN-energy-weighted relative Frobenius error of
the nearest Kronecker product (rank 1, `ks_1`) and of `kron_basis_diag`.

**Gate.** Recomputed effective/raw blocks for no-SN ViT seed 0 at steps 10 and 50 equal the stored 4a
`b2_w16__s0` blocks (relative Frobenius ≤ 1e-12).

**Pre-declared reading.** For c = 0.95 (SN active in every state) versus no SN, matched (seed, step),
six pairs per model: *consistently lower* / *consistently higher* / *mixed*, separately for stream
writers and first-in-branch layers. **S2 supported** if the stream-writer ks_1 error is consistently
lower and the first-in-branch error is not; **S2 absent** if the stream-writer error is not consistently
lower while local distance distortion is lower (SN study, step-10 states). For c = 2 and 6, only pairs
where SN is active are reported (descriptive).

## Budget and stop conditions

Budget 3 CPU-h. Stop on a gate failing by more than 1e-6, a numerical check failing at 1e-8, a failing
baseline suite, an exceeded budget, or a required change outside this folder or its data directory.

## Outputs

`README.md`, `RESULTS.md`, `DEVIATIONS.md`, `HANDOFF_PROPOSED_ENTRY.md`, light mirrors (json/md/png/yaml).
