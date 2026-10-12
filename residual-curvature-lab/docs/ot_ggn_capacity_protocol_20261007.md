# Matched OT/GGN capacity comparison — 7 October 2026

The user selected construction of both teachers at identical saved states and
comparison of their affordable representations, including gradient and momentum.
This is a new bounded diagnostic. It does not reopen the previous failed gate,
train predictors, learn graph topology, or launch optimization trajectories.

Use all six saved ViT states from runs/20261005T143029-9b582ed5 (searched SOAP
and exact GGN at 10/25/50), MNIST batch 512, all 40 tensors/5738 parameters,
one CPU thread and float64. Reuse their construction batches and four independent
analysis panels; these panels are development data used before. No state is
dropped for negative independent teacher advantage. There are no OT-trajectory
states in this comparison; do not infer preservation of its earlier training gain.

## Matched teachers

GGN uses cached exact C_bb, checked with new matrix-free actions. OT uses new
streamed exact blocks M_bb=mean J_p,b.T L(p)^dagger J_p,b. L is the existing
complete unit-conductance graph over output classes, with arithmetic mobility
(p_i+p_j)/2. Check OT blocks against independent probability JVP/VJP actions.
The graph's nodes are classes, not parameter tensors. Cross-tensor metric blocks
are discarded for both teacher steps. A graph-Laplacian output metric does not
automatically provide cheap parameter-space inverse actions or a tensor GNN.

For each target A_b=target_bb+mu Omega_b, use the SAME normalized inverse frozen
SOAP Omega, gradient/momentum, gamma=.01, exact trace rule and damping floor.
mu=max(.01 trace(target)/5738,1e-8) therefore differs numerically between metrics.
Set beta=1/(mu*SOAPMetric.scale) and K_b=beta F0_b.T A_b F0_b. This declared
damping calibration makes the identity complement exact for damping alone.
Original trace calibration remains reported in the previous experiment; it is
not rerun in this bounded follow-up.

## Representations and fitting

Retain all 13 previous families: identity, scalar, diagonal, exact, plus teacher,
coordinate and state bases at ranks 4/8/16. Their equations and bounds [1e-6,1e6]
are unchanged. Fitting uses analytic isotropic inverse energy in SOAP-factor
coordinates. Teacher-selected spectral bases are restricted references, not
optimal bases for independent loss. Cheap bases still receive oracle coefficients.

Add the combined family identified as missing in the supplied critique:

    P = beta F0 D^(1/2) [I+U(T-I)U.T] D^(1/2) F0.T.

First fix D=clip(1/diag(K)); form Ktilde=D^(1/2) K D^(1/2). Fit the existing
teacher/state subspace rule to Ktilde, at ranks 3 and 16. State inputs are
D^(1/2) z_g, D^(1/2) z_m and effective denominator denominator/D. T is the bounded
inverse of U.T Ktilde U. This is a sequential diagonal-then-residual fit, with an
isotropic population objective in the ADJUSTED factor coordinates. It changes
the RHS weighting relative to the original factor coordinates; it is not a joint
optimum of D,U,T or an isolated test of representation with identical weighting.
The objective remains analytic and independent of realized evaluation probes;
no gradient/history-weighted fitting experiment is being claimed.

The diagonal is an expensive teacher diagonal in this screen. Thus even the
combined state basis is cheap only conditional on D being available. Count
n+n*r+r*r numbers per tensor (effective r can be smaller). Rank 3 can fit the
4.25-model-vector budget; rank 16 is a capacity reference. This corrects the
previous absence of a combined family without retrospectively changing its result.

## Evaluation and practical interpretation

17 families x two RHS + native SOAP =35 candidate steps per state/target;
6 states x 2 targets =420 global searches and 204 fitted operator states.
Use the existing 52-evaluation global construction-loss search, including zero,
and all four independent panels. No per-layer search. Save full loss-GGN
quadratic, full selected-target metric quadratic and block-teacher quadratic
separately; OT is a metric, not the cross-entropy Hessian.

Evaluate raw and one-global-scalar A-energy errors on g, momentum, 16 Gaussian
physical RHS and 16 RHS F0^-T z. Use identical seeds and actual RHS for OT and
GGN; unlike the previous experiment, factor probes omit beta so their physical
vectors and cross-state pooling weights do not change just due to calibration.
Record pooled errors and equal-state means. Assess each candidate against its
own exact teacher, and show absolute independent losses to compare teachers.
Do not compare a candidate's GGN error against another's OT error as if the
underlying norm were identical.

Report all-state and per-state local recovery separately for g and momentum,
relative to the matching SOAP RHS control and exact SAME-RHS target. Ratios are
undefined if the aggregate reference advantage is <=1e-4. Also show cross-RHS
absolute scores without calling them same-RHS recovery. A development shortlist
requires >=80% aggregate local recovery, storage<=4.25 model vectors, median
application<=2x baseline and passed numerical/search audits. Probe accuracy is
reported as a separate diagnostic, not silently equated to nonlinear quality.
Teacher-selected bases are not deployable candidates. No shortlist automatically
authorizes training; the user's present selection is only steps 1 and 2.

Record streaming teacher cost, fitting and basis cost, 9-repeat median application,
storage and process peak RSS. GGN blocks are cached; zero new production time is
NOT a GGN-vs-OT production speed comparison. All inverse/dense checks use the
previous repaired operator-Frobenius/energy/equation/backward tests at 1e-8.
Preserve failures, full source/test/protocol/attachment hashes, arrays and searches.

## Scope of the supplied critique

The critique correctly notes diminishing returns for state rank, complement
scaling, and state-dependent teacher transfer. Its combined-family suggestion is
included above. Its separate optimizer-RHS-weighted objective needs disjoint
fitting/evaluation batches and a new protocol; it is deferred here. Neither the
previous failure nor this comparison tests a learned graph preconditioner.

For graph interpretation, also verify the elementary identity that a complete
class graph with edge weights p_i*p_j has Laplacian diag(p)-p p.T: categorical
GGN already contains a graph-Laplacian output factor. OT changes how output
geometry is used; the mere presence of a graph does not establish superiority.
