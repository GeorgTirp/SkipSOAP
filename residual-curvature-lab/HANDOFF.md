# Research handoff

Last updated: 8 October 2026. Start here when resuming this repository.

## Current question and active work

Can useful downstream GGN information improve SOAP at acceptable cost, through
better directions inside tensors, better allocation across tensors, or both?
The question is broader than recovering cross-tensor coupling. Near-full performance
of layerwise GGN does not rule out improving SOAP's within-tensor approximation.

**Active work:** no run is active. The user-selected matched OT/GGN teacher
construction and compression comparison is complete. Claude Code executed our
existing protocol as gap-audit check 3 in external run `20261007T124647-1d8487a2`.
On resume, we verified 48 archived sources, unchanged implementation, six parent
arrays, twelve operator artifacts, identical paired RHS and all 420 searches;
120 stored proposals were independently replayed, maximum loss discrepancy
4.44e-16. We reused this execution without launching a duplicate. Main suite:
106 passed; isolated study tests: 12 passed. See the
[verified OT/GGN report](docs/ot_ggn_capacity_results_20261008.md),
[study report](studies/gap_audit_20261007/RESULTS.md) and
[protocol](docs/ot_ggn_capacity_protocol_20261007.md).

The best affordable tested family is diagonal plus three state-derived directions:
57.0% of OT's own local advantage versus 46.4% of GGN's. Absolute mean decreases
are .032685/.031652; storage 4.063 model vectors. OT's smaller reference advantage
explains part of the recovery difference; on a common GGN denominator these are
50.3%/46.4%. No family meets the 80% current-gradient shortlist. Diagonal and
coefficients are still teacher-fitted, and sequential fitting changes whitening;
no cheap learned optimizer or gradient/history-weighted fitting result is established.
The previous learned-predictor and training gates remain closed.

**Broader study context:** on fresh seeds 1–3, tuned searched SOAP/GGN losses are
.697/.355 at step 50 and .210/.182 at step 500. Tuned native SOAP ends at .161 and
reaches the study's targets much faster. The large GGN advantage is mainly early
and per step, under this workload. Dense empirical-Fisher training and removing
SOAP momentum do not sustain prior local gains. These training results are from
the study, not checkpoint-replayed by this review; tuning horizons and concurrent
timing differ. Check 4b remains partial (21/22 units), and its predictor stage was
skipped. No repair/rerun of that separate study was performed.

**Previous experiment:** the user-authorized experiment 1 capacity
screen completed in `runs/20261007T093903-8d7d86f1`: six saved ViT states, two
calibrations, 324 searched candidate steps and 156 fitted operator states. No
eligible family passed the declared gate, so contingent predictor learning
(experiment 2) and paired training (experiment 3) were not launched. See the
[results](docs/learnable_capacity_results_20261007.md) and
[protocol](docs/learnable_capacity_protocol_20261007.md).

Cheap state bases with oracle coefficients recover 37.3% (rank 4) and 41.6%
(rank 16) of aggregate current-gradient local headroom; exact diagonal fitting
recovers 21.0%. Damping-calibrated teacher rank 8 recovers 96.6% at SOAP state 10,
but teacher rank 16 recovers only 56.1% across all six states, versus 15.4% with
original trace calibration. Late negative results remain included. An exploratory
teacher-rank-16/momentum combination has mean decrease .046187 versus .045883
for exact/current-gradient, but uses expensive geometry and 17.747 additional
model vectors; it did not pass the predeclared gate. This is no training result.

Three interrupted attempts are retained: `20261007T092113-f1ad8fe0`,
`20261007T092734-02f0a4d9`, `20261007T093220-108e6ea8`. The protocol explains
condition-sensitive forward comparisons, Cholesky applications, operator-matrix
checks and the added damping calibration. Numerical tolerance remains 1e-8;
all completed checks pass. Fully completed states were reused with verified
nested source provenance. All 103 tests pass after completion.

**Previous completed experiments:** the eight-arm transport/PARSEC/NI-SOAP pilot,
`runs/20261006T100557-f1efb932`, and the user-requested NI-SOAP learning-rate-search
extension, `runs/20261006T153126-f597adda`. All use the paired ViT, MNIST batch 512
and 50 consumed steps. Searched PARSEC improved over searched Adam (validation
loss 1.007097 versus 1.239108), while native PARSEC gave a smaller gain. Searched
NI-SOAP with noise finished at .973104 versus .867103 without noise. Native noisy/
noiseless losses were 1.406900/1.365508. Tensor graph-Wasserstein finished at
.381630, compared with the exact-trace tensor-GGN control .327676. All runs,
checkpoints and searches passed verification; the suite then had 96 tests.
Those experiments are complete. See the [main report](docs/transport_parsec_noise_results_20261006.md)
and [search extension](docs/ni_soap_lr_search_results_20261006.md).

**Earlier information diagnostic:** an output-Jacobian information diagnostic at six
saved ViT states (both searched-SOAP and exact-GGN trajectories, 10/25/50), using
four previously unused analysis panels. All 102 candidate searches and numerical
checks passed. Dense per-example gradient covariance recovers most of the early
SOAP-state GGN advantage, and exceeds GGN at the middle state; either matrix's
diagonal in SOAP coordinates loses much of that gain. GGN directions retain the
early gain with SOAP tensor norms. The effect is state dependent and momentum
interacts strongly with the preconditioner. This does not uniquely attribute the
50-step training gap or establish empirical-Fisher training performance. The user
clarified that "residual mapping" means the full parameter-to-output Jacobian;
no skip-branch derivative ablation was selected or run. See
[the information report](docs/jacobian_information_results_20261005.md) and
[protocol](docs/jacobian_information_protocol_20261005.md).

**Earlier five-arm training:** the small ViT on MNIST, batch 512, with five
paired 50-iteration trajectories. Exact tensor GGN reached validation loss .325714
and accuracy 90.30%, versus searched SOAP .731623 and 77.40%, at 4.76x its measured
training-step time. Diagonal/full allocation finished at .748057/.816645: neither
improved final training outcomes over searched SOAP. Full allocation improved all
three shared-state independent-panel means, so local allocation headroom did not
translate into a training win. All final checkpoints reproduced their metrics;
source/array checksums and numerical gates passed. That five-arm experiment is complete.
See [the report](docs/mini_vit_results_20261005.md) and
[the prospective protocol](docs/mini_vit_protocol_20261005.md).

## Document map

| File | Purpose |
| --- | --- |
| [HANDOFF.md](HANDOFF.md) | Current question, findings, run index, limitations and next actions |
| [STATUS.md](STATUS.md) | Implementation stages and numerical validation status |
| [docs/decisions.md](docs/decisions.md) | Protocol and numerical decisions, with dates |
| `docs/*protocol*.md` | Settings and spending/interpretation gates recorded before an experiment |
| `docs/*results*.md` | Measurements and conclusions from completed experiments |
| `runs/<id>/` | Resolved config, exact data IDs, raw results, checkpoints and source archives |

After an experiment, update this handoff, STATUS and the report. Record negative
results and failed checks. Do not replace a prospective protocol after looking at
results; document any change as a new decision/run. Final conclusions must identify
which of the direction, allocation, scalar-search and training comparisons support them.

## Experiment ledger

| Question / experiment | Finding | Artifacts |
| --- | --- | --- |
| Matched OT/GGN compression, six states, first execution by gap-audit study | Combined diagonal + state rank 3: OT recovery 57.0%, GGN 46.4%; absolute mean decrease .032685/.031652. No 80% current-gradient shortlist. Main-lab review independently replays 120 selected proposals to 4.44e-16. | External `~/skipsoap-study-data/gap_audit_20261007/runs/20261007T124647-1d8487a2`; [verified report](docs/ot_ggn_capacity_results_20261008.md), [study](studies/gap_audit_20261007/RESULTS.md) |
| Gap audit: tuned baselines, fresh seeds 1–3, 500-step training | GGN beats tuned searched SOAP early (.355 vs .697 at 50); gap shrinks (.182 vs .210 at 500). Tuned native SOAP .161 at 500, with much lower time to target. Dense EF and momentum-free SOAP do not reproduce local gains. | Study runs `20261007T121557-dfef403c`, `20261007T122018-998da6a3`, `20261007T143302-baba2664`; [report](studies/gap_audit_20261007/RESULTS.md) |
| Gap audit: richer structured capacity and architecture family | Kronecker-basis diagonal recovers 46% on original six states; full-GGN PCG with 5/10 matvecs 67%/133% locally. Nine-ViT study has headroom, but capacity check stopped at 21/22 units and predictor stage skipped. These are oracle/local results, not cheap optimizer training wins. | Study runs `20261007T123608-7f39721a`, `20261007T125950-63edd820`, `20261007T143410-b579ef62`; [report and limitations](studies/gap_audit_20261007/RESULTS.md) |
| Learnable inverse capacity, six saved ViT states, two calibrations | State rank 4/16 recovers 37.3%/41.6% of aggregate current-gradient local headroom; diagonal 21.0%; damping teacher rank 16 56.1%. No affordable family passes the declared gate. Teacher rank 16 on momentum is promising but exploratory and expensive. | `runs/20261007T093903-8d7d86f1`; [report](docs/learnable_capacity_results_20261007.md), [protocol](docs/learnable_capacity_protocol_20261007.md) |
| Initial MLP, batch 64, SOAP snapshots at 10/50/100 | Exact algebra passed. Narrow scalar searches hit their upper limit; this did not establish absent GGN headroom. | `runs/20261001T080647-fd8e50e5`; [report](docs/local_mnist_results_20261001.md) |
| Batch-64 rank-8 diagonal/coupled sketch training, 100 consumed steps | Final validation accuracy: SOAP 88.98%, sketch diagonal 57.54%, coupled 53.72%. Low-rank arms deteriorated; accurate residual identities did not imply accurate sketches or successful training. | Comparison `runs/20261001T081542-3836b346`; diagonal `runs/20261001T081543-323a92f8`; coupled `runs/20261001T081556-39824518` |
| Exact tensor GGN at construction batches 256 then 512, same saved MLP states | At primary gamma .01, independent decrease was worse than searched SOAP at all three batch-256 states; better at two of three batch-512 states. All broader searches bracketed; exact-reference checks passed. | `runs/20261002T121124-74f10aae`; [report](docs/local_large_batch_results_20261002.md) |
| Paired batch-512 MLP training, 50 consumed steps | Final validation loss: native SOAP .750541, searched SOAP .491103, exact tensor GGN .417581. Accuracy: 83.12%, 89.04%, 86.92%. Exact GGN used ~44x searched-SOAP step time. | `runs/20261002T121635-0dac3337`; [protocol](docs/exact_training_followup_protocol_20261002.md) |
| Downstream probes and allocation controls, batch-512 MLP frozen states | Explicit residual VJPs matched direct factors to 6.8e-16. No rank 1/4/8/16 passed the declared approximation targets. Exact curvature diagonal in SOAP bases still had action errors .7655/.7951/.5129. Copying GGN shares gave small gains; conditional allocation helped early/middle, hurt late. | `runs/20261005T074622-d81830d0`; [report](docs/downstream_allocation_results_20261005.md) |
| Mini ViT and KKT allocation, batch 512, 50 iterations | Validation losses: native SOAP 1.354356, searched SOAP .731623, diagonal allocation .748057, full allocation .816645, exact tensor GGN .325714. Exact GGN accuracy 90.30% versus searched SOAP 77.40%; measured step time 219.33s versus 46.04s. All 5 final checkpoints verified. | `runs/20261005T135920-d7ce29b2`; [report](docs/mini_vit_results_20261005.md), [protocol](docs/mini_vit_protocol_20261005.md), [config](configs/mini_vit_allocation.yaml), [runner](scripts/run_mini_vit_allocation.py) |
| Output-Jacobian information versus representation, six ViT states | Fresh-panel decreases at SOAP states 10/25: GGN .135327/.085051; dense empirical Fisher .119131/.119290; GGN diagonal in SOAP basis .044453/.040127. Dense F needs only per-example gradients. GGN directions at SOAP norms preserve gains. Later states and momentum swaps prevent a universal single-cause conclusion. | `runs/20261005T143029-9b582ed5`; [report](docs/jacobian_information_results_20261005.md), [protocol](docs/jacobian_information_protocol_20261005.md) |
| Transport, PARSEC and native NI-SOAP, eight paired 50-step ViT arms | Searched PARSEC loss 1.007097 versus Adam 1.239108. Native noisy NI-SOAP 1.406900 versus alpha-zero 1.365508. Graph-Wasserstein .381630 versus exact-trace GGN .327676. | `runs/20261006T100557-f1efb932`; [report](docs/transport_parsec_noise_results_20261006.md), [protocol](docs/transport_parsec_noise_protocol_20261006.md) |
| NI-SOAP learning-rate search, two paired trajectories | Noisy NI-SOAP .973104 / 66.46% versus zero-noise .867103 / 70.78%. Search improves both native variants, but noise still hurts. | `runs/20261006T153126-f597adda`; [report](docs/ni_soap_lr_search_results_20261006.md), [protocol](docs/ni_soap_lr_search_protocol_20261006.md) |

## Insights to preserve

1. **Global size, tensor shares and tensor directions are distinct.** Write
   `Delta_b = R sqrt(w_b) u_b`. A global ray search changes R while preserving that
   step's shares and directions. NI-SOAP learning-rate search instead follows a
   curved drift/noise path. None of our completed searches was per-layer.
   Across diverged MLP trajectories, searched SOAP closed 77.9% of the final loss
   gap from native SOAP to exact tensor GGN. That is descriptive, not a universal
   causal attribution; native lr=.003 was not retuned.
2. **Allocation must suit the actual directions.** A namesake/fallback learning-rate
   ratio controls one aggregate group share, not all per-tensor shares. Muon's
   fan-out scaling does not equal SOAP's tensor scaling. Matching shares or exposure
   alone does not preserve gradient alignment. The user's specific failed tuning
   run was not identifiable in the PDF; its path/ID remains needed for diagnosis.
3. **The clean convex variables are nonnegative amplitudes.** With unit SOAP tensor
   directions E, set phi=-E^T g and H=E^T(C+mu Omega)E. Minimize
   `a^T H a/2 - phi^T a` over a>=0. KKT: a>=0, Ha-phi>=0 and componentwise
   complementarity. Diagonal H gives `a_b=[phi_b]_+/H_bb`; normalize a^2 to get
   shares. Full coupling requires the QP, not that diagonal formula. Cross terms
   change with allocation. An exact fixed-norm equality is not generally a convex
   constraint in a; use free amplitudes plus global search or a convex norm budget.
4. **Large cross curvature is not the same as useful coupling.** For
   `C=I+rho 11^T, g=1`, full and diagonal inverse updates have the same direction,
   although cross terms can dominate curvature. A global search corrects their
   scale difference. Our original MLP cross fractions were ~69–77%; the paper's
   ~90% fractions likewise describe current directions, not oracle gain.
5. **Residual-path and parameter-block interactions differ.** Exact tensor GGN
   retains all downstream residual paths inside C_bb while dropping C_bc for b!=c.
   Ordinary backprop already sees those paths through the error gradient; curvature
   probes carry additional operator information.
6. **The failed downstream candidate is a specific compression.** It used the
   diagonal of Q^T C_bb Q in SOAP's bases and the current gradient. It discarded
   within-tensor correlations and did not preserve SOAP's momentum proposal.
   The frozen -Pg control exposes the momentum difference. Exact diagonal errors
   show that more probes alone cannot fix the representation. Do not summarize
   this as 'downstream information cannot help SOAP'.
7. **The external GGN paper is motivation, not a reproduced result.** Its LLM/data,
   warmup, tuning, very-large-batch and iterative-solve settings differ. Exact GGN
   improved our MLP loss modestly and showed a larger final loss/accuracy advantage
   in the ViT pilot. The ViT also changes input resolution and other workload
   details, so this does not isolate an architectural cause. Source:
   [The Potential of Second-Order Optimization for LLMs](https://arxiv.org/html/2510.09378v2).
8. **A correct local KKT optimum need not improve a training trajectory.** On the
   ViT's searched-SOAP states, full allocation beat searched SOAP's independent
   decrease at 10/25/50 (.127626/.056756/.001395 versus
   .048118/.015684/.000558). Full allocation's own trajectory was ahead at 10 and
   25 but behind at 50. Its KKT residual remained below 4.5e-16. The discrepancy
   is not a failed QP solve; its cause is unresolved. Exact GGN at state 50 reduced
   construction loss by .198640 yet increased independent loss by .004763.
   Construction fit, transfer, momentum evolution and multi-step behavior remain
   separate questions. These are conditional amplitudes, not universal ideal shares.
9. **Full output-Jacobian information is not necessary for every local oracle gain.**
   Dense per-example empirical Fisher retains all within-tensor correlations but
   sees one true-label output-gradient direction per example. It captures most of
   the early GGN gain on the SOAP trajectory; raw and trace-matched results agree
   closely. Exact C/F diagonals in the same SOAP bases lose much of that early
   benefit. This supports preserving richer within-tensor geometry. It does not
   equate empirical Fisher to SOAP or prove a training win. On later GGN-trajectory
   states, exact GGN beats dense empirical Fisher, while current-gradient SOAP and
   diagonal-GGN controls are also strong. The target, representation and history
   interact. SOAP's EMA lag and inverse versus inverse-square-root scaling were
   not separately identified.
10. **The RHS cannot be treated as an incidental implementation detail.** At SOAP
    states 10/25, GGN on the current gradient gives .135327/.085051 independent
    decrease, but GGN on effective SOAP momentum gives only .035343/.033845. At
    state 50 the sign reverses: -.012308 versus +.045150. On the GGN trajectory,
    momentum can again hurt. This is not a universal recommendation to remove it.
    Model-sampled-label gradient covariances have GGN as their expectation; the
    1/4/16-draw diagnostics retain all tensor correlations and are not cheap SOAP
    diagonal implementations. The exact-Jacobian diagnostic does not benchmark
    the cost of a VJP-only approximation.
11. **PARSEC transforms the input to Adam.** The user confirmed the default
    `(I-V.T V)g + V.T diag((lambda+delta)^(-1/2))Vg`, followed by AdamW in parameter
    coordinates. The complement is unchanged. This does not impose that same
    metric on the final Adam displacement: fixed positive coordinate scaling
    cancels in Adam when epsilon is negligible, while dense mixing and changing
    transforms can matter. The pilot uses rank 16, 32 Krylov iterations and refresh
    10 instead of the source default 100. Maximum relative Ritz residual was
    .160455; it is an approximate eigenspace. Searched PARSEC improves Adam but
    remains behind earlier searched SOAP (.731623), under untuned settings.
12. **NI-SOAP noise is dominated by fallback parameters here.** Native noise energy
    was 53.19% of drift energy; 99.94% of realized noise went to isotropic fallback
    parameters. Its Kronecker expectation uses the product of cached factor traces,
    not automatically true SGD noise energy. Noise worsened both native and searched
    outcomes. Search uses `Delta(a)=a*d+sqrt(a)*n` with one frozen draw; alpha zero
    reduces to ordinary ray search. Loss-based selection changes the conditional
    noise distribution. Searched noise/drift energy was 18.06%, still 99.99% fallback.
    Use the matched alpha-zero port, not official SOAP, to isolate added noise.
13. **Transport is a declared geometry, not a unique parameter norm.** This pilot
    pulls back a dynamic graph-Wasserstein tangent metric on output probabilities,
    with equal class costs and arithmetic edge mobility. At uniform predictions
    it equals GGN/K; trace-scaled damping and search give the same ray when the
    damping floor is inactive. Its .381630 final loss did not beat the matched
    exact-trace GGN .327676. No learned class geometry or fixed Wasserstein norm
    on signed weights was tested. Both use exact tensor blocks, so this does not
    establish a cheap SOAP approximation.
14. **A useful correction needs more than accurate coefficients on an arbitrary
    cheap basis.** Even teacher-fitted coefficients on the tested state bases
    recover only 37–42% of aggregate local headroom. Teacher directions nearly
    close the early SOAP-state gap but fail uniformly across states. Complement
    calibration and the gradient/momentum RHS materially change outcomes. The
    diagonal core improves physical-probe ray error by 93.2% while recovering
    only 21.0% of local loss headroom; physical Gaussian probes alone are an
    inadequate optimization surrogate. No claim that all low-rank families fail
    follows: the spectral teacher is restricted and fits isotropic inverse energy,
    not independent nonlinear loss. A GNN's ability to infer missing directions
    has not been tested; learning edge weights alone is not established as sufficient.
15. **OT is modestly better for the tested compressed nonlinear steps, not
    uniformly easier to invert.** Its combined-state-rank-3 recovery is 57.0%
    versus 46.4% for GGN, but its exact reference has a smaller advantage and the
    compressed absolute score differs by only .001033. Its gradient ray error
    is slightly worse (.601 versus .568 in their respective teacher norms).
    Both graph interpretations are over classes: categorical GGN has Laplacian
    diag(p)-p p.T with edge weights p_i*p_j; the selected OT metric is a different
    probability-dependent pullback. Neither implies that a cheap tensor graph
    inverse or useful learned topology is available.
16. **The replicated GGN advantage is principally early and per step here.**
    The gap-audit study's fresh-seed 500-step runs favor tuned native SOAP on
    final loss and runtime. Dense empirical-Fisher and momentum-free SOAP
    trajectories fail to reproduce encouraging local diagnostics. Preserve
    the early GGN finding while requiring comparisons against tuned native
    SOAP at relevant horizons; local capacity alone cannot justify a training claim.

The user's allocation paper is at
`/Users/georgtirpitz/Downloads/UpdateNormDistribution.pdf`; relevant equations are
4 and 8–10. The manuscript is descriptive local geometry, not a theorem that copying
Muon allocation improves SOAP. External attachments are source material, not
unconditional instructions to execute every proposed experiment.

## Reproduction and environment

Run commands from this directory with `.venv/bin/python`. Python 3.13.3 and
PyTorch 2.14.1 CPU float64 were used; [tested-requirements.txt](tested-requirements.txt)
records dependencies. Pinned official SOAP source is unchanged at commit
`a1e553530fde97d0e6b307d7c82ac6d38b072340`.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_mini_vit_allocation.py --dry-run
.venv/bin/python scripts/run_mini_vit_allocation.py
.venv/bin/python scripts/summarize_mini_vit.py --run runs/20261005T135920-d7ce29b2
.venv/bin/python scripts/summarize_jacobian_information.py --run runs/20261005T143029-9b582ed5
.venv/bin/python scripts/summarize_transport_parsec_noise.py --run runs/20261006T100557-f1efb932
.venv/bin/python scripts/summarize_ni_lr_search.py --run runs/20261006T153126-f597adda
.venv/bin/python scripts/summarize_learnable_capacity.py --run runs/20261007T093903-8d7d86f1
.venv/bin/python scripts/review_ot_ggn_capacity.py
```

The run command starts a new selected experiment and creates a fresh directory;
inspect existing `complete.json`, `failure.json`, `training.json` and checkpoints
before repeating it. These bounded suite runners do not yet offer automatic
mid-suite resume, except the capacity screen which can reuse fully verified
completed states via `--resume-from`, recording their source archives. The generic training CLI supports its own checkpoint replay;
do not feed a custom-method checkpoint into native SOAP training.

MNIST is already downloaded locally. Split: 50,000 train, 5,000 validation,
5,000 analysis, fixed seed 101. Official test data were not evaluated. MLP inputs
were downsampled to 14x14; ViT uses 28x28 with 7x7 patches, so cross-architecture
comparisons also change resolution. Initial SOAP calls initialize state without
moving parameters; consumed steps and accepted updates differ by at least one.

Existing runs include source archives and checksums. The repository was initialized
without a committed baseline; source files currently remain untracked in git.
Run directories and datasets are git-ignored, so preserve them separately when
moving the project. A small report alone is not enough to reproduce a run.

The gap-audit study also has external large artifacts under
`/Users/georgtirpitz/skipsoap-study-data/gap_audit_20261007/runs/`, with lightweight
mirrors under `studies/gap_audit_20261007/runs/`. Keep the external directory when
moving the project. The main review artifacts live in
`runs/20261008_ot_ggn_capacity_review/`; these contain analysis and replay results,
not a new experiment execution. Study files and frozen protocols were not changed.

## Boundaries and next decisions

- Keep numerical residual tolerance 1e-8 in CPU float64; explicitly distinguish
  exact blocks, exact reduced projections, sketches and unconverged solves.
- Use the same gradient, scope, damping and scalar budget in fixed-state ablations.
  Training arms diverge in states and therefore are not causal decompositions.
- Search is one bounded global scalar search, not exact nonlinear minimization.
  The wider policy has 34 grid points including zero and 18 refinement evaluations.
- Independent panels do not select damping/rank; reused panels are diagnostic,
  not fresh confirmatory data. Single seeds and short runs do not establish rankings.
- Training step time excludes loading, validation and checkpoint/plot writes.
  Workspace estimates are not measured process RSS or deployment overhead.
- ViT uses generic parameter-space GGN/JVP code. Existing explicit residual-stream
  transport assumes an MLP terminal head and is not certified for the ViT's CLS
  selection and final norm. Do not silently reuse that path.
- The ViT pilot, six-state information diagnostic and transport/PARSEC/NI-SOAP
  trials (including searched NI-SOAP), plus both inverse-capacity screens, are complete; the current suite has 106
  passing tests and final artifacts passed verification.
  Replication/common tuning remain necessary before a ranking.
- Exact tensor GGN now offers clear headroom in this pilot, while the tested
  allocation-only variants did not close the final training gap. Next candidates
  should preserve useful within-tensor correlations lost by the SOAP-coordinate
  diagonal. The added transport and PARSEC trials test other geometries, but
  neither establishes a practical downstream-statistic SOAP improvement. No
  replication, hyperparameter sweep or additional run is active.
- If pursuing allocation, first diagnose it on its own saved trajectory. A
  prospectively selected convex amplitude-norm budget could test whether limiting
  displacement helps transfer. Neither that hypothesis nor a causal explanation
  for the final allocation loss is established by these results.
- The gap-audit study has now performed dense empirical-Fisher training and the
  current-gradient SOAP-map control. Both perform worse than tuned searched SOAP
  under the study's settings; dense EF damping was not independently tuned. Their
  earlier positive local diagnostics did not establish a trajectory benefit.
  The requested matched OT/GGN comparison is complete. Any further predictor,
  objective-weighting or trajectory experiment needs its own selected protocol.
