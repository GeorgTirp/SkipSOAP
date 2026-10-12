# Frozen-state downstream curvature and allocation experiment

Selected in response to the user's request to try passing downstream residual
information into SOAP and relate that intervention to their update-allocation
paper. This protocol precedes this experiment's measurements. Use batch 512,
the existing fixed states 10/50/100 and their four independent panels from
`runs/20261002T121124-74f10aae`. Keep all parameters, tensor partition,
normalized inverse-SOAP damping, primary gamma 0.01, exact trace and existing
52-evaluation construction-only scalar search. No new training is selected here.

Construct a new full-rank structured candidate in frozen SOAP coordinates Q:
estimate c=diag(Q^T C_bb Q) using output-Hessian probes transported backward through
the terminal head and actual residual Jacobians. Use the same probe through skip
and residual paths. Define C_hat=Q^-T diag(c) Q^-1 and solve the true-gradient
quadratic with the unchanged mu Omega. With Omega=scale Q^-T diag(d) Q^-1,
the coordinate step is -(Q^T g)/(c+mu*scale*d). This changes the curvature scaling;
it is a local candidate, not an unchanged SOAP optimizer. It retains SOAP bases
and does not claim to recover the full within-tensor GGN.

Compute an exact-coordinate-diagonal reference from saved exact C_bb blocks. This
separates information lost by diagonal compression from stochastic probe error.
The probe estimator and the existing low-rank Algorithm A share the same output
curvature probes; transport is an explicit efficient evaluation of the same
chain rule, not a different GGN identity.

Use nested ranks 1,4,8,16 with two independent repeats and seed derived from the
saved configuration and state. Estimate each rank's diagonal independently of its
candidate loss. On four independent standard-normal parameter vectors per state,
measure relative RMS action error against exact tensor GGN, separately from the
relative error of the coordinate diagonal. A rank is eligible only if BOTH repeats
have diagonal error <=0.25 and tensor-action error <=0.50. Choose the smallest
eligible rank; otherwise report insufficient rank/structure. These are engineering
accuracy targets, not guarantees of optimization improvement. Never choose rank,
damping or a direction on the independent loss panel.

Compare searched SOAP, frozen -Pg, exact tensor GGN, and these diagnostic controls:

- SOAP unit tensor directions with exact GGN squared-norm shares;
- exact GGN unit directions with SOAP squared-norm shares;
- SOAP directions with diagonal and full reduced-GGN nonnegative allocation;
- exact curvature diagonal in SOAP coordinates;
- downstream-probe coordinate-diagonal candidates at every declared rank/repeat;
- rank-16 matched low-rank tensor-diagonal and coupled candidates for both repeats.

The allocation controls retain signed gradient alignment. With columns E containing
unit SOAP tensor directions, solve max_{a>=0} phi^T a - (a^T H a)/2, where
phi=-E^T g and H=E^T(C+mu Omega)E. Also use diag(H) for the matched uncoupled
allocation control. These controls optimize allocation for SOAP's own directions;
copying GGN shares is a diagnostic, not a claim that those shares are optimal for
SOAP. No Muon optimizer is implemented in this lab, so this does not reproduce
the user's transformer tuning run or copy its Muon allocation.

Save parameters, factors, contexts, actions, ranks/seeds, registry and batch IDs,
all search trials, selected and raw diagnostics, independent decreases, source
hashes, exact-reference provenance, action/solve checks, cost and process RSS.
The declared numerical tolerance is 1e-8; transport must agree with direct VJPs
and microbatch accumulation before a dataset experiment. Keep the existing
768-MiB tensor-workspace budget and report its distinction from process RSS.
