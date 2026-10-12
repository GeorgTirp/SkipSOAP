# Matched OT/GGN capacity: verified results — 8 October 2026

The user-authorized comparison is complete. Claude Code executed the existing main-lab protocol in the isolated gap-audit study on 7 October, before this agent resumed. This review reuses that execution and independently verifies it; it does not launch duplicate experiments or change the study.

External run: `/Users/georgtirpitz/skipsoap-study-data/gap_audit_20261007/runs/20261007T124647-1d8487a2`. [Study report](../studies/gap_audit_20261007/RESULTS.md); [prospective main-lab protocol](ot_ggn_capacity_protocol_20261007.md). Six saved seed-0 ViT states (SOAP/GGN trajectories at 10/25/50), MNIST batch 512, all 40 tensors, float64, one thread, target-specific exact-trace gamma .01 damping and the same global 52-evaluation search. Four reused analysis panels are development data. No OT-trajectory states or new optimization trajectories are included in this capacity comparison.

## Findings

The strongest affordable current-gradient family is a teacher-fitted diagonal plus three state-derived directions. It recovers 57.0% of OT’s own local teacher advantage versus 46.4% for GGN. Storage is 4.063 model-sized vectors; neither target reaches the 80% current-gradient shortlist. These coefficients and the diagonal still require expensive teacher information; the family is not a deployable learned optimizer.

OT has a modest nonlinear-score advantage for the state and combined-state families. For combined rank 3, the absolute mean decrease is .032685 versus .031652 for GGN (difference .001033). OT’s exact teacher has less aggregate headroom: .023469 versus .026569. With the common GGN gap as denominator, the combined candidates recover 50.3% (OT) versus 46.4% (GGN). Thus the own-teacher 10.5-percentage-point difference partly reflects a smaller target gap.

The study’s prospectively recorded comparative reading is met: at least two of {diagonal, state_4, combined_state_3} gain >=10 percentage points of own-teacher recovery and do not decrease their absolute score. This is a thin development result, not broad evidence that OT is a better optimization target. Inverse accuracy does not improve: combined-rank-3 pooled gradient ray error is .601 under OT versus .568 under GGN, each measured in its own teacher energy.

The combined family addresses the attachment’s missing control: separate diagonal and identity-complement low-rank models were not the combined model. The new control improves over state rank 4 at a slightly smaller storage cost. It is a sequential fit in diagonally adjusted coordinates, so its isotropic fitting distribution also changes. No joint optimum or isolated representation effect is claimed. The separate gradient/history-weighted fitting experiment remains unrun.

## All current-gradient candidates

| Family | GGN mean decrease | OT mean decrease | GGN own recovery | OT own recovery | Stored vectors | OT apply / SOAP map |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.019314 | 0.019314 | 0.0% | 0.0% | 0.000 | 1.00 |
| scalar | 0.023094 | 0.023137 | 14.2% | 16.3% | 0.007 | 1.00 |
| diagonal | 0.024883 | 0.024801 | 21.0% | 23.4% | 1.000 | 1.05 |
| exact | 0.045883 | 0.042783 | 100.0% | 100.0% | 399.945 | 3.31 |
| teacher_4 | 0.015566 | 0.016565 | -14.1% | -11.7% | 4.112 | 1.16 |
| teacher_8 | 0.025576 | 0.025441 | 23.6% | 26.1% | 8.446 | 1.15 |
| teacher_16 | 0.034217 | 0.032745 | 56.1% | 57.2% | 17.747 | 1.17 |
| coordinate_4 | 0.019873 | 0.019876 | 2.1% | 2.4% | 0.139 | 1.30 |
| coordinate_8 | 0.021015 | 0.021026 | 6.4% | 7.3% | 0.502 | 1.32 |
| coordinate_16 | 0.020295 | 0.020317 | 3.7% | 4.3% | 1.868 | 1.32 |
| state_4 | 0.029232 | 0.030520 | 37.3% | 47.7% | 4.112 | 1.14 |
| state_8 | 0.030201 | 0.031496 | 41.0% | 51.9% | 8.446 | 1.15 |
| state_16 | 0.030355 | 0.031644 | 41.6% | 52.5% | 17.747 | 1.17 |
| combined_teacher_3 | 0.026742 | 0.026516 | 28.0% | 30.7% | 4.063 | 1.27 |
| combined_teacher_16 | 0.025540 | 0.026239 | 23.4% | 29.5% | 18.747 | 1.31 |
| combined_state_3 | 0.031652 | 0.032685 | 46.4% | 57.0% | 4.063 | 1.26 |
| combined_state_16 | 0.031067 | 0.032201 | 44.2% | 54.9% | 18.747 | 1.29 |

Application timing excludes teacher fitting, basis construction, prediction and line search; it is not end-to-end optimizer overhead. Original process timing was collected under the study’s concurrent workload. Fitting/basis times and errors are retained in the review JSON and original artifacts.

## Momentum and graph interpretation

Exact OT on momentum has mean independent decrease .012390, slightly below the SOAP-momentum map .012656. Its aggregate recovery denominator is negative, so OT momentum recovery is undefined. The rank-16 teacher correction on momentum gives .047773 (OT) and .046187 (GGN), but uses 17.747 model vectors and expensive teacher-selected directions. These absolute local scores do not establish a training win. Same-RHS GGN momentum ratios have a small .003260 denominator and should not be promoted as practical success.

The OT graph is over output classes. Its metric is the pullback `J_p.T L(p)^dagger J_p`, with zero-sum probability tangents. A graph over parameter tensors would be a different construction; the tested tensor teachers discard cross-tensor blocks. The output graph does not remove the Jacobian or the cost of the parameter-space inverse.

Categorical GGN also has an exact graph interpretation: `H=diag(p)-p p.T` is the Laplacian of a complete class graph with edge weights `p_i p_j`, and `C=J_z.T H J_z`. OT uses `J_z.T H L(p)^dagger H J_z`. This identity is covered by the new unit test. Graph structure is meaningful for both targets; OT’s explicit Laplacian alone does not establish a cheaper or better tensor GNN.

## Broader gap-audit context

The separate study reports three fresh-seed, 500-step comparisons: GGN loss .355 at step 50 versus .697 for tuned searched SOAP; at step 500, .182 versus .210. Tuned native SOAP ends at .161 and reaches the study’s targets much faster. GGN and SOAP were tuned at different horizons, and concurrent load affects timings. This strengthens evidence for an early per-step GGN advantage while weakening the case for a general runtime advantage.

The study also reports that dense empirical-Fisher training and removing SOAP momentum do not reproduce earlier local gains. Its check 4b remains partial (21/22 units); that interrupted study was not repaired or rerun here. Broader training outcomes are attributed to the study, not independently checkpoint-replayed by this review. See its report and deviations for full settings and limits.

## Independent verification

Verified 48 archived source hashes, current implementation agreement, six parent arrays, twelve operator artifacts, mirrored JSON, all 420 search records and positive cores. Identical physical/factor/gradient/momentum RHS were confirmed across OT and GGN. Independently applied stored operators and selected scalars for identity, exact, state_4, combined_state_3 and teacher_16, both RHS, all states/targets: 120 selected proposals replayed. Maximum loss discrepancy: 4.441e-16. No search or fitting was repeated.

[Verification JSON](../runs/20261008_ot_ggn_capacity_review/verification.json); [all aggregate results](../runs/20261008_ot_ggn_capacity_review/summary.json). The main suite has 106 passing tests; the separate study suite has 12 passing tests.

The next decision remains how to obtain a more useful affordable representation and fitting signal. Neither target currently supplies an 80%-recovery cheap candidate on these states. Predictor training and new OT trajectories were not part of the selected steps 1 and 2 and were not launched.
