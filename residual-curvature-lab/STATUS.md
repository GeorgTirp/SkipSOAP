# Delivery status — 2026-10-08

The requested first delivery is implemented in this independent repository.
Following explicit user authorization, MNIST was downloaded and the seed-0 local
snapshot pilot plus two paired sketch training arms were completed. No replication
sweep, cluster submission, publishing or language-model run occurred.

On 2 October, explicit user authorization selected exact tensor GGN comparisons at
construction batch 256 followed by 512. Six fixed-state measurements completed
with streamed exact blocks and adequately bracketed common direction searches.
Only batch 512 passed the prespecified independent-panel opportunity gate, so a
50-iteration paired training followup compared native SOAP, searched SOAP and
exact tensor GGN at that batch size. All runs and negative measurements are saved.

| Stage | Status |
| --- | --- |
| 0: foundation, reference algebra and synthetic smoke | Implemented and tested |
| 1: downsampled MNIST fixed-state pilot | Completed batch-64 pilot plus exact tensor comparisons at construction batches 256/512 and states 10/50/100; numerical and search gates passed |
| 2: approximation development | Shared-probe pair, explicit memory-safe residual VJP transport, SOAP-coordinate curvature diagonal and allocation controls audited; initial rank/structure targets failed; practical statistic reuse deferred |
| 3: short training comparisons | Paired MLP pilots at batches 64/512 and a five-arm mini ViT allocation/exact tensor pilot at batch 512 completed; replication and tuning deferred |
| 4: tiny decoder | Deferred; CLI configurations explicitly fail with status |
| 5: large-model readiness | Interface/storage design documented; no framework integration or readiness claim |

Completed components include a complete tensor registry and block grouping, all three
GGN routes, exact diagonal reference, SOAP-preconditioned CG, cone/signed-span/ray
allocation, explicit residual transport, shared categorical VJPs and microbatch
accumulation, nonlinear policies, stored arrays and provenance, and plots reconstructed
from stored measurements. The official SOAP source and license are pinned and unchanged.

The current suite has **106 passing tests**. It checks three seeded residual models,
gradient/mean normalization, linearized Hessians, PSD with signed interactions,
transport and messages (including nonzero initialization), solves and matched
diagonals, NNLS KKT/order, SOAP state/updates through refreshes, frozen-map inverse,
microbatches, analysis non-mutation, monitored training equality, rejected-step
statistics, actual checkpoint restoration, config/resource rejection and a CPU
float32 operator/solve audit. Full CUDA and float32 training replay are not certified.
PyTorch emits a non-failing upstream `torch.jit.script` deprecation warning during
functional differentiation. No numerical test is skipped or marked expected-failure.
New tests cover streamed tensor blocks with uneven microbatches, SOAP-metric direct
solves versus the output-factor reference, memory guards, broader search boundaries,
and exact-training proposal replay with one statistics update and no state mutation.

Synthetic measurements and executed commands are in [docs/verification.md](docs/verification.md).
The initial float64 smoke has 211 parameters, 16 stored tensors and at most eight
GGN directions on its construction batch. It does not establish optimizer superiority.
Some common-grid selections hit the maximum radius; inspect that boundary before
interpreting direction rankings as unconstrained nonlinear optima. SOAP inverse
metrics can be poorly conditioned at early states; solve residuals and conditioning
are logged, and a failed tolerance still aborts the exact-reference report.

Open work: calibrated downstream-statistic reuse (Algorithm B); systematically
measured rank/sweep/cost studies; per-method incremental memory profiling beyond
process RSS and workspace estimates; further nonlinear-search sensitivity; paired
training seeds, batch sizes and common tuning budgets; decoder token/masking/replay
audits; JAX/distributed adapters and realistic runtime estimates. No result is
silently substituted for these missing experiments.

Local results and exact run locations are in [docs/local_mnist_results_20261001.md](docs/local_mnist_results_20261001.md).
On 5,000 reserved validation examples, final accuracy was 88.98% for SOAP, 57.54%
for sketch diagonal and 53.72% for sketch coupled. Both sketch arms deteriorated
after iteration 50 despite passing minibatch Armijo searches. These are retained
negative feasibility results, not replicated optimizer rankings.

The larger-batch results are in
[docs/local_large_batch_results_20261002.md](docs/local_large_batch_results_20261002.md).
At batch 256 the primary exact tensor direction transferred worse than searched
SOAP at all three states. At batch 512 it transferred better at two of three.
The followup's final validation losses were 0.750541 native SOAP, 0.491103 searched
SOAP and 0.417581 exact tensor GGN; accuracies were 83.12%, 89.04% and 86.92%.
Exact tensor GGN took about 44 times searched SOAP's measured training-step time.
This is limited evidence for a loss-based opportunity, with an accuracy/cost tradeoff,
under one seed and the stated damping/search settings. It does not establish that
a cheap downstream-metric approximation recovers this benefit, or reproduce the LLM
paper. The separately selected mini ViT pilot is described below; no tuning or
replication sweep was launched.

On 5 October the resumed downstream-curvature/allocation experiment completed
60 fixed-state candidates at batch 512, states 10/50/100. Shared stagewise probe
transport matches direct full-network VJPs to 6.8e-16. The new tests cover projected
skip width changes, masked/uneven microbatches, exact factors, all registered
tensors, nonorthogonal SOAP-coordinate actions/solves and state non-mutation.
An independent code review found and fixed matrix-RHS diagonal broadcasting before
the dataset run. Early metric dense-solve comparisons use the declared normwise
1e-8 tolerance plus equation residual rather than elementwise errors near zero.

No probe rank in 1/4/8/16 passed both predeclared approximation targets in both
repeats. Even exact curvature diagonalization in SOAP's bases has tensor-action
errors 0.7655/0.7951/0.5129: probe noise is not the only limitation. Allocation
optimized for SOAP's own directions improves independent loss at the first two
states but hurts at the third; copying GGN's shares offers only small gains.
All searches were bracketed, all numerical/state checks passed, and no new
training followed this diagnostic. See
[docs/downstream_allocation_results_20261005.md](docs/downstream_allocation_results_20261005.md).

The user-selected mini ViT and KKT allocation pilot completed in
`runs/20261005T135920-d7ce29b2`, under
[the prospective protocol](docs/mini_vit_protocol_20261005.md). The implementation
includes explicit attention, shared patch projection, complete tensor registration,
exact reduced GGN via directional JVPs, nonnegative full/diagonal amplitudes and
candidate-independent trace estimates. New tests check ViT normalization and
streamed blocks, exact reduced curvature, KKT conditions including negative
alignment, proposal replay/non-mutation and legacy MLP checkpoint compatibility.
The exact tensor-GGN training arm passed its snapshot opportunity and cost gates.
Final validation losses were 1.354356 native SOAP, .731623 searched SOAP, .748057
diagonal allocation, .816645 full allocation and .325714 exact tensor GGN.
Exact GGN accuracy was 90.30% versus searched SOAP 77.40%, at 4.76x measured
training-step time. Full allocation won the three shared-state independent-panel
comparisons against searched SOAP but did not sustain a training advantage.
All five final checkpoints reproduced validation metrics; 36 source files and
all snapshot array checksums verified. Maximum snapshot tensor-action error was
1.15e-14; tensor-solve residual across training/snapshots was 5.43e-14; maximum
allocation KKT residual was 4.44e-16. All 211 searched
proposals passed the boundary gate. See
[the mini ViT report](docs/mini_vit_results_20261005.md). Start with
[HANDOFF.md](HANDOFF.md) for the experiment ledger and current research context.

The subsequent user-selected output-Jacobian diagnostic completed 102 candidates
at six saved ViT states in `runs/20261005T143029-9b582ed5`. It compares dense GGN,
dense per-example empirical Fisher, exact SOAP-coordinate diagonals, effective
momentum/current-gradient RHS choices, tensor-norm transfers and model-sampled
label covariances with 1/4/16 draws in two repeats. Four fresh analysis panels were
reserved for this diagnostic. All numerical, non-mutation, solve and search checks
passed; the six new tests cover exact targets/sampled matrices, uneven microbatches,
trace controls, actual nonorthogonal SOAP bases and momentum-ray reconstruction.
Dense empirical Fisher captures much of the early local GGN advantage, while
diagonal compression loses it; later states show different target/momentum behavior.
No empirical-Fisher training or skip-specific ablation was run. See
[the information report](docs/jacobian_information_results_20261005.md).

The user-selected transport/PARSEC/NI-SOAP pilot and NI learning-rate-search
extension are complete. Torch float64 ports archive supplied JAX sources and
local helpers, with disclosed precision/PRNG differences. PARSEC uses the
user-confirmed inverse-root precondition-then-AdamW branch, rank-16 approximate
global GGN Ritz vectors and refresh 10. Searched loss 1.007097 improves on searched
Adam 1.239108, but remains above earlier searched SOAP .731623. The maximum Ritz
residual .160455 is retained. Tensor graph-Wasserstein loss .381630 did not beat
the exact-trace tensor-GGN control .327676.

NI-SOAP preserves cached-factor matrix noise and isotropic fallback noise. Native
alpha-one/zero losses are 1.406900/1.365508; searched losses .973104/.867103.
Search follows eta-linear drift and sqrt(eta)-scaled noise with one fixed draw.
Selection changes its conditional stochastic law; this is a searched variant.
Native realized noise is 99.94% fallback energy. All ten final checkpoints reproduce
validation and the next proposal exactly. The 21 added tests cover transport
gauge/PSD/actions, PARSEC projection/Ritz checks, noise covariance/zero-noise
controls, replay and NI learning-rate paths. Source/parent hashes, data pairing
and numerical/search gates passed. See
[the main report](docs/transport_parsec_noise_results_20261006.md) and
[the search report](docs/ni_soap_lr_search_results_20261006.md).

The learnable inverse-capacity screen completed in `runs/20261007T093903-8d7d86f1`:
six saved ViT states, two calibrations, 324 bracketed global searches and 156
fitted operator states. State-basis rank 4/16 recovers 37.3%/41.6% of aggregate
current-gradient local headroom; exact diagonal fitting 21.0%; damping-calibrated
teacher rank 16 56.1%. No eligible family passes all quality/cost/storage gates,
so contingent predictor learning and held-out training were not launched.
An exploratory teacher-rank-16/momentum combination has mean decrease .046187
versus .045883 for exact/current-gradient, but uses expensive teacher geometry
and 17.747 additional model vectors. No cheap learned optimizer is established.

Seven tests cover actual nonorthogonal SOAP factors, structured/dense actions,
population fitting, rank selection/clipping, global ray errors and damping
normalization. All 103 tests passed after the run. The report verifies 42 source
files, six parent arrays, 12 operator artifacts, nested resume provenance and
non-mutation. Maximum teacher equation residual is 1.59e-14; maximum completed
representation comparison is 3.19e-9. Three interrupted attempts and amendments
to condition-sensitive numerical checks are preserved at unchanged 1e-8 tolerance.
See the [capacity report](docs/learnable_capacity_results_20261007.md) and
[protocol](docs/learnable_capacity_protocol_20261007.md). No run is active.

The user-selected matched OT/GGN follow-up is complete through the isolated
gap-audit study's first execution of our prepared runner. Run
`~/skipsoap-study-data/gap_audit_20261007/runs/20261007T124647-1d8487a2` contains
420 searched steps and 204 fitted operators. Combined diagonal + state rank 3
recovers 57.0% of OT's own local advantage versus 46.4% for GGN, at 4.063 extra
model vectors. The absolute difference is .001033, and OT's reference gap is
smaller. No 80% current-gradient shortlist is reached. The new diagonal-subspace
implementation and Fisher-output graph identity have three additional tests.

On 8 October the main-lab review verified 48 source archives against current
implementation, parent arrays, operator hashes, mirrors, paired RHS and searches;
120 stored proposals replayed to maximum loss discrepancy 4.44e-16. The main
suite passed 106 tests; the study suite separately passed 12. No duplicate run,
new training or predictor fitting was launched. See the
[verified report](docs/ot_ggn_capacity_results_20261008.md) and
[complete broader study](studies/gap_audit_20261007/RESULTS.md).

The study's three-seed 500-step training results revise the broader context:
GGN's large early per-step advantage persists after baseline tuning, but shrinks
at 500 steps; tuned native SOAP has lower final loss and much faster time to
target. Dense EF and momentum-free SOAP do not preserve local diagnostic gains.
The separate architecture capacity study is explicitly partial (21/22 units),
and no predictor was trained. Those training checkpoints were not independently
replayed by the main review; their settings and limits are recorded in the study.
