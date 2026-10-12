# Prospective decisions and numerical audit notes

2026-10-01 — First delivery implements Stage 0 and prepares Stage 1; later stages are
explicitly deferred. The user-selected plan fixes the pilot settings and forbids
automatic MNIST/sweep/large-model training. No novelty or positive-result gate is assumed.

2026-10-01 — Installed/tested PyTorch is 2.14.1. Its matching official 2.14 torch.func
documentation was inspected. Functional calls, jacrev, jvp, vjp and a Hessian of the
linearized-output loss are exercised locally. Float64 initialization now explicitly
sets and restores default dtype so the caller's dtype cannot change seeded weights.

2026-10-01 — Upstream SOAP internally rounds basis calculations to float32. Preserve
upstream source and updates. Define the positive map using those actual saved bases;
invert using axiswise linear solves, and compute trace normalization using inverse-row
norms. Do not replace the native basis with a reorthogonalized one.

2026-10-01 — An early SOAP-context test produced a Woodbury residual of 1.2864e-8,
above the declared 1e-8 solve threshold. Add at most four refinement corrections using
the independent matrix-free residual. Keep the solve threshold unchanged and fail
the run if it is still exceeded. Record correction count. Ill-conditioned damped
systems can have small backward error but larger solution disagreement: report the
dense damped condition and use `max(1e-9,100*eps*condition)` for that forward-error
audit only. Algebraic identities keep 1e-9 relative / 1e-11 absolute tests. CPU float32
has a separate 1e-4 solve threshold and is not certified by float64 results.

2026-10-01 — Synthetic teacher data shares a fixed teacher between disjoint training,
construction and independent examples. MNIST uses deterministic area downsampling,
50k/5k/5k split, and reserves official test data. Synthetic smoke is three consumed
SOAP iterations followed by a snapshot; it is not a training-performance study.

2026-10-01 — The optional training prototype uses current shared-probe trace for
Algorithm A damping, explicitly distinguished from exact snapshot trace. Practical
statistic reuse, training hyperparameter comparison and message convergence studies
require a later prospectively selected protocol, not automatic execution here.

2026-10-02 — The user selected exact tensor GGN versus SOAP at construction batches
256 then 512. The protocol was recorded in large_batch_oracle_protocol_20261002.md
before the run. Use streamed full within-tensor blocks, direct Cholesky solves,
the existing normalized inverse-SOAP damping grid, and a common radius grid
2^[-16,...,16] plus 18 golden-section evaluations. The tensor-workspace budget is
768 MiB, distinct from measured total process RSS. Positive boundary selections
fail the search gate. Existing batch-64 SOAP states isolate construction-batch size.

2026-10-02 — Only batch 512 met the recorded primary-gamma independent-panel gate.
The subsequent protocol in exact_training_followup_protocol_20261002.md was
written before its 50-iteration paired training run. Retain gamma 0.01; do not
choose gamma 0.1 using its better independent-panel results. Compare native SOAP,
searched SOAP and exact tensor GGN with the same initialization and example order.
Validation measures all selected checkpoints without tuning or early stopping.

2026-10-05 — Resumed the downstream-curvature experiment specified before execution
in downstream_allocation_protocol_20261002.md. The new candidate retains SOAP's
bases and uses a probe-estimated GGN diagonal in those coordinates. The exact
coordinate-diagonal oracle separates compression from finite-probe error. Tensor
norm transfers and reduced nonnegative allocation distinguish direction, allocation,
alignment and global step-size effects. No rank selection uses candidate losses.
The prior independent panels are reused, so these remain diagnostic comparisons,
not a fresh confirmatory evaluation or a training superiority claim.

2026-10-05 — The user selected a small ViT and SOAP allocation derived from the
blockwise quadratic. The pre-execution mini_vit_protocol_20261005.md fixes 50
paired batch-512 iterations and shared-state references. Full allocation solves
the nonnegative amplitude QP with exact reduced GGN; the componentwise formula
applies only to the diagonal model. Training damping uses eight independent
Rademacher trace probes; snapshots use exact trace. The optional exact tensor-GGN
training arm requires both independent-panel opportunity and a 300-second cost
estimate. The ViT changes input resolution as well as architecture. The generic
GGN path supports its attention/CLS/final norm; the explicit MLP residual-transport
path is not certified for it. Adding inactive ViT defaults does not invalidate
earlier MLP checkpoints; all active configuration differences still fail loading.

2026-10-05 — The user clarified that the suspected missing "residual mapping" is
the full parameter-to-output-error Jacobian, not skip-branch interference. Select
the six-state protocol in jacobian_information_protocol_20261005.md, with four
previously unused analysis panels and no new training. Cross dense empirical
Fisher/GGN targets with exact diagonals in SOAP's actual bases; retain raw and
trace-matched F, current-gradient/effective-momentum RHS, norm transfers and
model-sampled-label covariances. The sampled target is GGN in expectation, but
true-label empirical Fisher is not. The dense diagnostic reuses exact Jacobians
and does not measure a practical VJP-only optimizer. Report all states, including
those where current-gradient GGN has no independent-loss advantage.

2026-10-06 — The user selected a transport-metric run and supplied PARSEC and
NI-SOAP code. The user explicitly confirmed PARSEC's default global inverse-root
gradient transformation followed by AdamW, not its separate Newton/Adam branch.
The prospective transport_parsec_noise_protocol_20261006.md selects eight paired
50-step ViT trajectories at batch 512. Native/searched PARSEC have matched Adam
controls; NI-SOAP has an alpha-zero control and uses native steps to retain its
specified noise law. The graph-Wasserstein and exact tensor-GGN arms share exact-
trace damping, tensor scope and global search. The graph has equal class costs
and arithmetic probability mobility; this is a dynamic tangent metric, not a
static Kantorovich distance on weight matrices.

The attached sources and required local helper definitions are archived with
hashes. Torch float64 ports retain the selected finite equations, with disclosed
differences from JAX PRNG and lower-precision Lanczos storage. PARSEC refresh is
10 rather than the source default 100 so it activates within this pilot; its
rank-16, 32-iteration Ritz approximation has residuals reported separately from
exact tensor-solve checks. NI-SOAP's source claim of energy matching is not assumed:
the matrix expectation is alpha^2 lr/B times the product of cached factor traces,
while fallback noise is isotropic. Both contributions are measured without
retuning or normalizing after observing outcomes. All 92 tests passed before
launching runs/20261006T100557-f1efb932.

2026-10-06 — After the eight-arm pilot, the user explicitly requested line search
for both supplied optimizers. PARSEC already had matched searched/native runs;
ni_soap_lr_search_protocol_20261006.md prospectively selected two NI-SOAP search
arms. Search actual learning rate on `a*d + sqrt(a)*n`, holding one noise draw
and one moment update fixed. This differs from scaling the whole noisy direction.
Loss selection changes the conditional noise law; preserve that distinction and
the original native results. Both arms completed in runs/20261006T153126-f597adda
after the 96-test suite passed. Final noisy/noiseless losses were .973104/.867103;
all 100 searches and both checkpoint replays passed. No further sweep was launched.

2026-10-07 — The user authorized the learnable-optimizer sequence experiment by
experiment. Start with the capacity protocol, retaining scalar, diagonal and
rank-4/8/16 SPD subspace corrections in actual SOAP-factor coordinates. Teacher
directions and cheap state/coordinate bases are separate. Fit analytic population
energy rather than fitting a single teacher gradient; evaluate independent RHS,
global ray errors, independent nonlinear decrease and full-vs-block quadratics.
Expensive oracle coefficients are not called a deployable optimizer. Learning
and three-seed training remain contingent on the selected representation gate.

The first attempt (20261007T092113-f1ad8fe0) stopped at a Euclidean solution
comparison affected by conditioning. The second (20261007T092734-02f0a4d9) exposed
roundoff from materializing the exact inverse on random RHS. Both are retained.
The protocol documents Cholesky factor-coordinate reference applications, cached
teacher-energy/physical-equation/backward checks at unchanged 1e-8 tolerance,
and provenance-preserving reuse of fully verified states. An analytically
motivated damping-calibration control was added after first-state observations;
the original trace calibration remains in the screen. This amendment is explicit,
not presented as a fully prospective replication. No rank or gate was retuned.

The third attempt (20261007T093220-108e6ea8) completed the six trace-calibrated
states and stopped at damping/SOAP-10/state-4 on a 3.09e-8 gradient-forward
comparison. The core remained SPD. The next runner compares full factorized and
dense operator matrices in relative Frobenius norm at 1e-8, logging the potentially
condition-amplified gradient discrepancy separately. Independent exact-reference
energy, equation and backward checks remain. The protocol preserves this audit
amendment and all three failures. Reused states retain their original stricter
forward comparisons, with nested origin/source verification.

2026-10-07 — Capacity screen completed in runs/20261007T093903-8d7d86f1, with
324 bracketed candidate searches, 156 fitted operators, verified nested source
archives and 103 passing tests. No family passes the unchanged development gate:
state rank 4/16 recovers 37.3%/41.6% of aggregate current-gradient local headroom;
diagonal 21.0%; damping teacher rank 16 56.1%. Do not proceed to predictor learning
or held-out training under this protocol. The teacher rank-16/momentum result
(.046187 mean decrease versus .045883 exact/current-gradient) is explicitly an
exploratory cross-RHS lead, not a post-hoc replacement gate. Its expensive basis
and 17.747 model-vector storage remain unresolved. Preserve early successes and
late negative states; neither a GNN nor all low-rank families have been ruled out.

2026-10-07/08 — The user selected matched OT/GGN construction and capacity tests.
The attached critique correctly identified the absent diagonal-plus-subspace
control. Add sequential diagonal plus rank-3/16 teacher/state corrections, with
the adjusted-coordinate isotropic objective and oracle-diagonal dependency stated
explicitly. Keep the prior failed gate intact. The new protocol uses common
physical/factor RHS, both gradient and momentum, and compares nonlinear quality
separately from inverse error. Gradient/history-weighted fitting is deferred;
no predictor or new optimization trajectory is selected by this follow-up.

On resume, the user pointed to Claude Code's isolated gap-audit study. Its check 3
had already executed our exact runner/protocol in external run
20261007T124647-1d8487a2. Reuse it rather than launch duplicate computations.
The main review verifies 48 source archives, current implementation, six parent
arrays, twelve operator artifacts, paired RHS and 420 search records. Independently
replay 120 stored proposals: maximum loss discrepancy 4.44e-16. The study is
unchanged; review artifacts are in runs/20261008_ot_ggn_capacity_review.

Combined-state rank 3 recovers 57.0% of OT's own gap versus 46.4% for GGN; absolute
mean decreases .032685/.031652, memory 4.063 model vectors. OT's smaller teacher
gap explains part of the difference. No 80% current-gradient shortlist is met.
The study's additional predeclared 10-percentage-point comparative reading is
reported separately from our shortlist. Do not promote it to proof of uniformly
better inverse approximation, general OT superiority or a learned graph success.
GGN's categorical output Hessian is also a graph Laplacian (edges p_i*p_j); the
output-class graph and a tensor-message graph are different constructions.

Integrate the broader study's reported three-seed, 500-step context into the
handoff, with attribution: the GGN gap is mainly early/per step, tuned native
SOAP wins final loss/runtime, and dense EF/momentum-free SOAP local gains do not
transfer to training. The separate partial check 4b is not repaired or rerun.
