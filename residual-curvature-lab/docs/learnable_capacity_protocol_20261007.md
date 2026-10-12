# Experiment 1: affordable inverse representation capacity

Selected by the user on 7 October, before execution. Experiments 2 (information
and prediction) and 3 (paired training/cost) are contingent on this capacity
screen. No GNN or further training is launched by this protocol.

Use all six saved states from runs/20261005T143029-9b582ed5, SOAP/GGN trajectories
at 10/25/50, their exact tensor GGN blocks, gradients, effective momentum RHS,
SOAP contexts, construction IDs and four independent 512-example panels. These
panels were used before: they are development diagnostics, not fresh confirmation.
Verify archived array hashes, exact solves/actions, source and split provenance.
All computations use one CPU thread and float64; numerical tolerance is 1e-8.

## Fixed target and normalization

For every registered tensor, A_b=C_bb+mu Omega_b, with the previous exact-trace
mu=max(.01 trace(C)/5738,1e-8) and normalized inverse-SOAP Omega. If
P0_b=Q_b diag(1/denominator_b) Q_b.T, use its actual factor
F0_b=Q_b diag(denominator_b^(-1/2)), retaining nonorthogonal saved Qs.
Normalize by a SINGLE global beta=D/sum_b trace(F0_b.T A_b F0_b), and set
F_b=sqrt(beta) F0_b, K_b=F_b.T A_b F_b. This removes irrelevant units while
preserving the baseline SOAP ray. The exact inverse is F_b K_b^-1 F_b.T.
Unscaled errors are relative to A^-1, with beta included, not rescaled post hoc.

Candidate inverse cores M_b in P_b=F_b M_b F_b.T:

- baseline I; scalar s_b I; coordinate diagonal diag(t_b);
- M_b=I+U_b(T_b-I)U_b.T with ranks 4/8/16, U_b.T U_b=I;
- exact K_b^-1 (unrestricted reference).

All fitted scalar/diagonal/subspace eigenvalues are clipped to [1e-6,1e6], fixed
here. Record clipping. This permits increases and decreases from baseline.
The fitting objective is the exact expected teacher energy error over isotropic
SOAP-factor-coordinate RHS: E_z ||(M-K^-1)z||_K^2, z~N(0,I). It uses the operator,
not any realized training/test RHS. Scalar optimum is n/trace(K); diagonal optimum
is 1/diag(K); for fixed U, T=(U.T K U)^-1, clipped spectrally.

Teacher U uses K eigenvectors, ranked by the reduction in this population
objective from changing 1 to clipped 1/lambda. This is an explicitly restricted
spectral oracle, not a claimed global optimum over every possible rank-r SPD
correction. It never uses the teacher's current-gradient solution as a basis.

Two cheap basis rules, with oracle T supplied to isolate direction availability:

- coordinate: select r SOAP-factor coordinates by z_g^2/||z_g||^2 +
  z_m^2/||z_m||^2, where z_g=F.T g and z_m=F.T momentum;
- state span: span columns t^j*z_g and t^j*z_m for j=0..7, taking the first r
  columns before rank-revealing orthogonalization. Here t is centered log SOAP
  denominator divided by its maximum absolute value, so |t|<=1. Use SVD with
  relative singular cutoff 1e-10; effective rank may be smaller. No curvature
  action, inverse teacher direction, gradient history or new probe selects U.

These cheap-basis candidates still receive expensive teacher coefficients: they
are capacity diagnostics, not deployable optimizers. Dense basis storage is n*r
plus r*r per tensor; coordinate bases store r indices plus r*r coefficients.
Log QR/SVD construction cost and model-sized-vector equivalents.

## Scores and controls

Evaluate every operator on current g, effective SOAP momentum, 16 new isotropic
physical-coordinate Gaussian RHS, and 16 new isotropic factor-coordinate RHS
r_b=F_b^-T z_b. Local seeded generators do not alter global RNG. Report per-RHS
e_A, pooled energy error, and errors after one nonnegative GLOBAL scalar per RHS.
No per-tensor scalar adjustment is allowed at evaluation. Fitting uses the exact
population expectation above, independent of the realized evaluation probes.

For g and momentum proposals, use the same zero + 2^[-16..16] radius search and
18 golden evaluations on construction loss, then evaluate all four panels. Save
all trials, signed outcomes, tensor shares and full coupled-GGN quadratic scores.
Score the block-teacher quadratic separately. Include native SOAP momentum ray
and current-gradient SOAP map controls. Exact tensor inverse is not called the
optimum of the full coupled quadratic or independent nonlinear loss.

Record offline teacher construction/fitting, cheap basis construction, repeated
operator-application timing, stored parameter count, and process peak RSS during
construction. Existing dense teacher storage is NOT a deployment memory estimate.
A separate cheap factorized application must agree with dense operator references.
Do not infer large-model/GPU costs from this small CPU experiment.

## Progression gate fixed before results

All states enter the aggregate. Define local recovery using mean independent
decrease D: (D_candidate-D_SOAP_current_g)/(D_exact-D_SOAP_current_g), provided
the denominator exceeds 1e-4. Report original SOAP-momentum comparisons too and
every state, including states where exact GGN hurts. No state is excluded for
having a negative teacher advantage.

A cheap-basis/scalar/diagonal family is eligible for experiment 2 only if:
1. current-gradient aggregate local recovery >= .8;
2. pooled held-out physical AND factor RHS errors after global ray correction
   each improve by >=80% relative to the corresponding baseline-map error;
3. fitted representation storage <=4.25 extra model-sized float64 vectors (four
   basis vectors plus a small coefficient allowance), and
   measured application time <=2x the matching baseline application;
4. all numerical, PSD, non-mutation and search checks pass.

These are development engineering gates, not statistical significance claims.
Ranks 8/16 can diagnose representational requirements even if memory disqualifies
them. If no family passes, finish and report experiment 1; do not train predictors
or run held-out training on a representation that failed the selected gate.
If a family passes, freeze one or two cheapest qualifying forms and write the
experiment-2 protocol before training predictors. The held-out seed evaluation
and practical search policy remain untouched by this screen.

## Recorded implementation amendment after the interrupted first attempt

Retain runs/20261007T092113-f1ad8fe0. It completed the first state, then failed an
overly strict Euclidean solution-comparison check at SOAP step 25. Generic
physical-coordinate solve versus archived Cholesky differed by 1.36e-8; the
independently reconstructed inverse had gradient equation residual 3.53e-14 for
the worst tensor and global teacher-energy discrepancy 1.69e-11. Do not relabel
that run complete. The repaired reference uses Cholesky in factor coordinates,
checks agreement to the cached teacher in A-energy, checks the gradient equation
residual and normwise backward errors for all RHS, all at the SAME 1e-8 tolerance.
Direct factorized/dense comparisons remain unchanged. Report this numerical
distinction rather than increasing a tolerance to accept an ill-conditioned solve.

The completed first state's low-rank results also expose a scale ambiguity: a
fixed identity complement depends on the chosen baseline calibration, even though
the underlying SOAP ray does not. Retain the original trace-calibrated screen and
ADD a damping-calibrated control, beta=1/(mu*metric.scale). This makes beta*P0 the
exact inverse of the damping term mu*Omega, so K=I+beta*F0.T*C*F0 (up to roundoff).
The complement is now correct when data curvature is zero. No rank, clipping
bound, probe, panel, loss metric or progression threshold changes. This is an
analytic normalization control motivated after preliminary observations, not a
fully prospective replication. Both calibrations run all six states and are
reported separately; either must pass the entire unchanged gate to proceed.
No candidate is selected from the failed attempt. The original normalization
results must not be silently discarded if the new one performs better.

The second attempt, runs/20261007T092734-02f0a4d9, completed all three SOAP states
but stopped on exact-reference random-RHS accuracy at GGN state 10. The reference
had explicitly materialized K^-1 before multiplication, unlike the Cholesky-solve
target. Replace only the exact reference application with Cholesky solves and
the same structured factor transforms; all candidate fits remain unchanged.
Independent cached-gradient energy agreement, original physical equation checks,
normwise backward checks and factorized-vs-dense checks still apply at 1e-8.
For the exact reference the dense comparison uses teacher energy, not an
ill-conditioned Euclidean solution bound. Preserve the failed attempt and reuse
only its three fully verified completed states with per-state source provenance.

The third attempt, runs/20261007T093220-108e6ea8, finished all six trace states
(three reused), then stopped at damping/SOAP-10/state-4. The core remained SPD
(minimum eigenvalue 1.70e-4), but factorized-versus-dense gradient application had
relative Euclidean discrepancy 3.09e-8 for the input projection. This compares
two differently ordered floating-point applications of a conditioned operator.
For new states, compare the full factorized and dense operator matrices in
relative Frobenius norm at the unchanged 1e-8 tolerance; retain gradient forward
discrepancies as diagnostics, without claiming a condition-independent forward
error bound. Independent teacher equation, energy, backward-error and PSD checks
remain. Candidate arithmetic and scientific gates are unchanged. Preserve the
third attempt and reuse its six completed states with nested source provenance.
