# Learnable optimizer: representation capacity — 7 October 2026

Run: `runs/20261007T093903-8d7d86f1`. The [protocol](learnable_capacity_protocol_20261007.md) records the initial selection and disclosed numerical/calibration amendments. Six saved seed-0 ViT states (SOAP and GGN trajectories at 10/25/50), batch 512, CPU float64, one thread. All 40 parameter tensors participate. The four independent analysis panels are reused development data, not fresh confirmation. No training or learned predictor was run.

**Progression to experiment 2: not selected: no eligible representation passed all gates.

## Findings and interpretation

A cheap rank-4 state basis with exact teacher coefficients recovers 37.3% of aggregate current-gradient local headroom. Rank 16 reaches 41.6%, with 17.747 additional model-sized vectors. Exact scalar and coordinate-diagonal fitting recover 14.2% and 21.0%. Under this inverse-fitting objective, these bases miss the 80% target even with teacher coefficients. Application is inexpensive here; quality is already insufficient before adding prediction/construction cost.

With damping calibration, teacher rank 8 gives .131890 independent decrease at SOAP state 10 versus .033191 for current-gradient SOAP and .135327 for exact GGN: 96.6% of that state's gap. Teacher rank 16 gives .076923 at SOAP state 25 versus exact .085051, but hurts at SOAP 50 and GGN 25/50. Across all six states it recovers 56.1%. Original trace-calibrated teacher rank 16 recovers only 15.4%. The baseline outside the learned span matters; global line search cannot repair an incorrect relative complement scale.

There is a promising exploratory momentum interaction: damping-calibrated teacher rank 16 applied to effective SOAP momentum gives mean decrease .046187, compared with .015916 for exact GGN on that same momentum and .045883 for exact GGN on the current gradient. This is a cross-RHS diagnostic, not the prespecified current-gradient progression test or a training result. It uses an expensive teacher-selected basis and exceeds the memory budget. It suggests that the useful correction can depend on the gradient history; closer inverse approximation alone does not guarantee better nonlinear transfer.

Probe accuracy and optimization quality also differ. The diagonal core cuts physical-probe ray error by 93.2%, yet closes only 21.0% of the local loss gap and cuts factor-probe error by only 29.8% under damping calibration. Random physical probes emphasize a different part of the inverse than actual gradients. Both probe distributions and gradient/momentum scores are needed. Factor RHS include beta, so cross-state energy pooling weights change between calibrations.

These results motivate finding useful within-tensor directions from affordable state/history or curvature measurements, with their complement scale and RHS choice treated explicitly. They do not establish that a GNN can predict the directions, that cross-tensor graph coupling is required, or that all rank-16 representations fail: the teacher basis is a restricted spectral construction fitted to an isotropic inverse objective, not an oracle for independent loss. Under the declared gate, experiments 2 and 3 remain unrun.

## What is fitted

The damped tensor target is A_b=C_bb+mu Omega_b with gamma .01 and exact trace. For the actual SOAP factor F0, P0=F0 F0.T. Set F=sqrt(beta)F0 and K=F.T A F. Every candidate is P=F M F.T. Scalar/diagonal cores and bounded subspace cores `M=I+U(T-I)U.T` are fitted using the analytic expected teacher energy over isotropic factor-coordinate RHS. The fit does not use realized evaluation probes. Bounds are [1e-6,1e6]. For fixed U, T is the bounded inverse of U.T K U.

Teacher bases select eigenvectors of K by population-error improvement. They are restricted spectral oracles, not a proof of the globally best rank-r SPD correction. Coordinate bases select entries from current gradient/momentum energy. State bases orthogonalize current gradient/momentum multiplied by powers of normalized log SOAP denominators. Both cheap basis rules still receive teacher coefficients; neither is a deployable optimizer. Coordinate bases store indices, while state/teacher bases store dense vectors.

## Aggregate development results

Independent decrease is averaged over all six states; positive is better. Local recovery is (candidate-current-gradient SOAP)/(exact-current-gradient SOAP), with the same global construction-only search. No negative-advantage state is omitted. Inverse errors pool A-energy numerators/denominators over 16 held-out complete parameter RHS per state, after one global nonnegative scalar per RHS. They are distinct from the nonlinear search and from current-gradient-only errors.


### trace calibration

| Core | Mean independent decrease | Local recovery | Physical probe ray error | Factor probe ray error | Stored model vectors | Apply / baseline | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| identity | 0.019314 | 0.0% | 5.622e-03 | 9.055e-01 | 0.000 | 1.00 | False |
| scalar | 0.023094 | 14.2% | 3.011e-01 | 8.291e-01 | 0.007 | 1.00 | False |
| diagonal | 0.024883 | 21.0% | 3.846e-04 | 6.704e-01 | 1.000 | 1.06 | False |
| exact | 0.045883 | 100.0% | 3.780e-19 | 1.389e-21 | 399.945 | 2.06 | False |
| teacher_4 | 0.022590 | 12.3% | 7.347e-01 | 8.381e-01 | 4.112 | 1.13 | False |
| teacher_8 | 0.023013 | 13.9% | 5.810e-01 | 8.064e-01 | 8.446 | 1.17 | False |
| teacher_16 | 0.023397 | 15.4% | 2.395e-01 | 7.663e-01 | 17.747 | 1.18 | False |
| coordinate_4 | 0.020442 | 4.2% | 7.384e-01 | 8.923e-01 | 0.139 | 1.31 | False |
| coordinate_8 | 0.022068 | 10.4% | 5.926e-01 | 8.776e-01 | 0.502 | 1.31 | False |
| coordinate_16 | 0.022390 | 11.6% | 2.911e-01 | 8.392e-01 | 1.868 | 1.34 | False |
| state_4 | 0.029232 | 37.3% | 7.430e-01 | 8.840e-01 | 4.112 | 1.15 | False |
| state_8 | 0.030201 | 41.0% | 5.813e-01 | 8.711e-01 | 8.446 | 1.17 | False |
| state_16 | 0.030355 | 41.6% | 2.767e-01 | 8.304e-01 | 17.747 | 1.18 | False |

### damping calibration

| Core | Mean independent decrease | Local recovery | Physical probe ray error | Factor probe ray error | Stored model vectors | Apply / baseline | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| identity | 0.019314 | 0.0% | 5.622e-03 | 8.829e-01 | 0.000 | 1.00 | False |
| scalar | 0.023094 | 14.2% | 3.011e-01 | 7.885e-01 | 0.007 | 0.99 | False |
| diagonal | 0.024883 | 21.0% | 3.846e-04 | 6.197e-01 | 1.000 | 1.02 | False |
| exact | 0.045883 | 100.0% | 6.678e-19 | 2.060e-21 | 399.945 | 2.85 | False |
| teacher_4 | 0.015566 | -14.1% | 3.603e-03 | 8.043e-01 | 4.112 | 1.13 | False |
| teacher_8 | 0.025576 | 23.6% | 2.792e-03 | 7.585e-01 | 8.446 | 1.14 | False |
| teacher_16 | 0.034217 | 56.1% | 1.916e-03 | 6.996e-01 | 17.747 | 1.14 | False |
| coordinate_4 | 0.019873 | 2.1% | 5.546e-03 | 8.734e-01 | 0.139 | 1.28 | False |
| coordinate_8 | 0.021015 | 6.4% | 5.493e-03 | 8.668e-01 | 0.502 | 1.31 | False |
| coordinate_16 | 0.020295 | 3.7% | 5.354e-03 | 8.523e-01 | 1.868 | 1.28 | False |
| state_4 | 0.029232 | 37.3% | 5.103e-03 | 8.638e-01 | 4.112 | 1.12 | False |
| state_8 | 0.030201 | 41.0% | 4.275e-02 | 8.594e-01 | 8.446 | 1.14 | False |
| state_16 | 0.030355 | 41.6% | 8.307e-02 | 8.500e-01 | 17.747 | 1.14 | False |

Eligibility requires >=80% local headroom recovery, >=80% reduction of both pooled probe ray errors relative to baseline, <=4.25 additional model-sized vectors including coefficients, median application ratio <=2, and passed numerical/search checks. Exact and teacher-selected bases are references, not deployable candidates. These thresholds are development engineering gates, not statistical significance tests.

## Unscaled versus ray accuracy

The following values use actual gradients and the same block teacher energy, pooled across states. Raw values retain scalar calibration; ray values permit one scalar for the whole parameter vector. Full coupled quadratic and nonlinear independent decreases are saved separately in each candidate row.

| Calibration | Core | Gradient raw error | Gradient ray error | Momentum ray error |
| --- | --- | ---: | ---: | ---: |
| trace | identity | 3.744e+02 | 9.014e-01 | 9.703e-01 |
| trace | scalar | 2.026e+02 | 8.276e-01 | 9.389e-01 |
| trace | diagonal | 1.194e+02 | 7.675e-01 | 9.073e-01 |
| trace | exact | 3.820e-25 | 3.820e-25 | 1.678e-26 |
| trace | teacher_4 | 1.778e+01 | 7.630e-01 | 9.204e-01 |
| trace | teacher_8 | 1.075e+01 | 7.205e-01 | 9.026e-01 |
| trace | teacher_16 | 9.487e+00 | 7.171e-01 | 8.868e-01 |
| trace | coordinate_4 | 2.047e+02 | 8.858e-01 | 9.568e-01 |
| trace | coordinate_8 | 1.394e+02 | 8.740e-01 | 9.455e-01 |
| trace | coordinate_16 | 9.193e+01 | 8.595e-01 | 9.208e-01 |
| trace | state_4 | 5.856e-01 | 5.856e-01 | 8.672e-01 |
| trace | state_8 | 5.328e-01 | 5.328e-01 | 8.323e-01 |
| trace | state_16 | 4.633e-01 | 4.633e-01 | 7.896e-01 |
| damping | identity | 2.199e+05 | 9.014e-01 | 9.703e-01 |
| damping | scalar | 2.026e+02 | 8.276e-01 | 9.389e-01 |
| damping | diagonal | 1.194e+02 | 7.675e-01 | 9.073e-01 |
| damping | exact | 0.000e+00 | 1.153e-30 | 3.486e-29 |
| damping | teacher_4 | 6.006e+03 | 8.377e-01 | 9.243e-01 |
| damping | teacher_8 | 1.736e+03 | 8.223e-01 | 9.016e-01 |
| damping | teacher_16 | 6.983e+02 | 8.239e-01 | 8.645e-01 |
| damping | coordinate_4 | 1.264e+05 | 8.962e-01 | 9.645e-01 |
| damping | coordinate_8 | 9.215e+04 | 8.941e-01 | 9.624e-01 |
| damping | coordinate_16 | 6.293e+04 | 8.962e-01 | 9.591e-01 |
| damping | state_4 | 5.856e-01 | 5.856e-01 | 8.672e-01 |
| damping | state_8 | 5.328e-01 | 5.328e-01 | 8.323e-01 |
| damping | state_16 | 4.633e-01 | 4.633e-01 | 7.896e-01 |

## Per-state independent decreases (damping calibration, current gradient)

| Core | SOAP 10 | SOAP 25 | SOAP 50 | GGN 10 | GGN 25 | GGN 50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.033191 | 0.025898 | 0.002988 | 0.015570 | 0.017390 | 0.020846 |
| scalar | 0.035967 | 0.036048 | 0.001670 | 0.024005 | 0.018505 | 0.022370 |
| diagonal | 0.044453 | 0.040127 | -0.001335 | 0.027310 | 0.018176 | 0.020568 |
| exact | 0.135327 | 0.085051 | -0.012308 | 0.039168 | 0.010830 | 0.017227 |
| teacher_4 | 0.073975 | 0.032435 | -0.023878 | 0.014378 | 0.001969 | -0.005485 |
| teacher_8 | 0.131890 | 0.050766 | -0.035664 | 0.026594 | -0.004510 | -0.015622 |
| teacher_16 | 0.132421 | 0.076923 | -0.037068 | 0.048152 | -0.006730 | -0.008396 |
| coordinate_4 | 0.035122 | 0.032956 | 0.003075 | 0.015873 | 0.017863 | 0.014347 |
| coordinate_8 | 0.036062 | 0.033627 | 0.002897 | 0.018051 | 0.020898 | 0.014557 |
| coordinate_16 | 0.037331 | 0.032031 | -0.004123 | 0.022370 | 0.017873 | 0.016286 |
| state_4 | 0.064352 | 0.037960 | -0.002202 | 0.034085 | 0.019327 | 0.021871 |
| state_8 | 0.071603 | 0.040136 | -0.004145 | 0.032491 | 0.018580 | 0.022542 |
| state_16 | 0.071684 | 0.041320 | -0.005508 | 0.032986 | 0.018318 | 0.023330 |

## Momentum controls

Mean independent decrease across the same six states, damping calibration. Both columns use the same search budget. Native SOAP includes its recorded momentum proposal; the matrix-map controls apply the fitted operator to the effective momentum RHS. The teacher-fit objective did not fit either realized RHS.

| Method | Current gradient | Effective momentum |
| --- | ---: | ---: |
| native_soap | — | 0.012656 |
| identity | 0.019314 | 0.012656 |
| diagonal | 0.024883 | 0.023514 |
| teacher_16 | 0.034217 | 0.046187 |
| state_16 | 0.030355 | 0.021486 |
| exact | 0.045883 | 0.015916 |

## Interrupted attempts and calibration amendment

The first attempt, `runs/20261007T092113-f1ad8fe0`, finished one state and stopped at a Euclidean solution-comparison check on SOAP state 25. The discrepancy was 1.36e-8, with much smaller equation and teacher-energy errors. The repaired reference uses factor-coordinate Cholesky and retains 1e-8 checks on cached teacher energy agreement, gradient equation residual, all-RHS normwise backward error and independent factor/operator actions. The failed artifact is preserved, not counted as a successful run.

The second attempt, `runs/20261007T092734-02f0a4d9`, completed the three SOAP states but stopped when explicit inverse-matrix multiplication disagreed with the Cholesky reference on random RHS at GGN state 10. The final runner uses Cholesky applications for the exact reference and retains the independent cached-gradient, physical-equation, backward-error and factor checks. Only fully verified completed states were reused, with their original source archive and per-state origin recorded. Candidate fitting equations did not change.

The third attempt, `runs/20261007T093220-108e6ea8`, completed all six trace states, then stopped on damping/SOAP-10/state-4: gradient forward discrepancy was 3.09e-8 despite a positive core minimum eigenvalue 1.70e-4. New states compare the full independently materialized factorized/dense operators in relative Frobenius norm at 1e-8. Gradient forward discrepancies remain logged separately, since conditioning can amplify them. Reused trace states retain their original gradient-forward (exact: energy) comparisons; the verification records both audit kinds and verifies nested source archives. No candidate formula or scientific gate changed.

The original trace calibration beta=D/sum trace(F0.T A F0) is retained. The first-state observations exposed the fixed identity complement's dependence on calibration. The added beta=1/(mu*metric.scale) makes the baseline the exact inverse of damping alone; its K equals I plus whitened data curvature. This analytic control was added after preliminary observations and is labelled accordingly. Ranks, bounds, RHS, panels and gates were not retuned.

## Cost, verification and limits

The 103-test suite passed before the restart and again after completion. 42 source files and six parent arrays verified. All 324 searches were bracketed or explicitly rejected; 0 were rejected. Twelve state/calibration artifacts passed checksums and state/RNG non-mutation. Numerical maxima:

| Check | Maximum |
| --- | ---: |
| max_factor_error | 6.295e-16 |
| max_action_error | 1.134e-14 |
| cached_teacher_energy_error | 1.785e-11 |
| teacher_gradient_residual | 1.592e-14 |
| teacher_max_backward_error | 1.284e-16 |
| max_dense_application_error | 3.191e-09 |

Peak process RSS during construction: 1022.83 MiB. This includes dataset, dense teachers, interpreter and library memory and is a producer-process high-water mark (including resumed states). It is not incremental student memory. Representation storage counts and basis/teacher fitting timings are saved separately. Application timings use nine repetitions and exclude teacher fitting, feature prediction and search; they do not establish end-to-end optimizer speed.

| Damping-calibrated core | Basis construction ms | Total teacher fit ms | Application ms |
| --- | ---: | ---: | ---: |
| identity | 0.000 | 0.030 | 1.050 |
| scalar | 0.000 | 0.508 | 1.031 |
| diagonal | 0.000 | 0.479 | 1.081 |
| exact | 0.000 | 13.365 | 3.010 |
| teacher_4 | 144.656 | 147.712 | 1.194 |
| teacher_8 | 143.213 | 145.936 | 1.204 |
| teacher_16 | 145.172 | 148.043 | 1.195 |
| coordinate_4 | 1.777 | 3.910 | 1.344 |
| coordinate_8 | 1.837 | 4.190 | 1.357 |
| coordinate_16 | 1.970 | 4.949 | 1.366 |
| state_4 | 3.730 | 5.404 | 1.189 |
| state_8 | 4.785 | 6.729 | 1.201 |
| state_16 | 6.796 | 9.492 | 1.204 |

Median cached-teacher reconstruction/audit time: 2.651 seconds per state/calibration. This includes independent GGN actions and dense coordinate transforms; it excludes the original Jacobian/teacher production, model loading and searches. Teacher fitting includes basis construction; these two columns must not be summed.

Full cached teacher matrices are used only for this offline capacity screen. Three future seeds would test trajectory transfer, not architecture transfer. A failed family does not rule out richer factorizations, a better subspace selection objective, different feature bases or larger affordable structures. No conditional predictor or paired training result should be inferred from this screen.

Rebuild this report and figures:

```sh
.venv/bin/python scripts/summarize_learnable_capacity.py --run runs/20261007T093903-8d7d86f1
```

![Capacity screen](../runs/20261007T093903-8d7d86f1/comparison.png)
