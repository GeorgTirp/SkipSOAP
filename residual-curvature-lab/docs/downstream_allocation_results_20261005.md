# Downstream curvature and update allocation — 5 October 2026

Run: `runs/20261005T074622-d81830d0`. This is a fixed-state experiment on 14x14 MNIST, at batch 512 and saved SOAP states 10/50/100. It does not launch or claim a new training result. All candidates use the same normalized inverse-SOAP damping with gamma=.01 and the same broad global scalar search. Each scalar is selected only on construction loss. The four independent panels are measured without retuning. The original states have batch-64 training history.

## Independent loss decrease

Positive values mean improvement. Each deterministic entry averages four independent panels. The displayed rank-16 entry also averages the two prespecified probe repeats; it is a diagnostic, not a rank selected on losses. All ranks and repeats are stored in all_candidates.json.

| Candidate | State 10 | State 50 | State 100 |
| --- | --- | --- | --- |
| SOAP + global search | 0.443676 | 0.207040 | 0.003344 |
| Frozen -Pg + search | 0.153842 | 0.033115 | 0.002658 |
| Exact tensor GGN | 0.424910 | 0.261146 | 0.006875 |
| SOAP directions, GGN shares | 0.448104 | 0.211483 | 0.004353 |
| GGN directions, SOAP shares | 0.415786 | 0.256856 | -0.000048 |
| SOAP + diagonal allocation | 0.434911 | 0.257870 | 0.004645 |
| SOAP + full reduced allocation | 0.477765 | 0.260822 | -0.002473 |
| Exact curvature diagonal in SOAP basis | 0.368996 | 0.125197 | 0.004292 |
| Probe curvature diagonal, rank 16 (mean of two repeats) | 0.367124 | 0.119568 | 0.005339 |

Copying the oracle tensor shares onto SOAP directions makes only a small change here. At the middle state, both GGN directions at SOAP shares and optimized allocation along SOAP directions recover much of the exact tensor reference improvement. These are two different ways to improve the joint update; copying one method's shares is not equivalent to optimizing shares for another method. At the late state full reduced allocation increases independent loss, despite its construction-batch objective. There is no uniform allocation win across these states.

The panels are independent of each candidate construction batch, but are reused from the earlier oracle experiment. They are diagnostic evaluation data, not a fresh confirmation set.

## Downstream transport and approximation gates

The implementation transports shared output-Hessian probes through the terminal head and actual residual-module VJPs. Identity/projected skip paths, normalization derivatives, input/head tensors and all biases are included. On the selected architecture it reproduces the existing direct whole-network probe factors; the estimator itself is not new. The new structured candidate keeps SOAP bases and uses curvature-derived coordinate scaling.

Let Q be the actual saved SOAP basis, d its denominator, and c the estimated diag(Q^T C_bb Q). The candidate solves with C_hat=Q^-T diag(c)Q^-1 and Omega=scale Q^-T diag(d)Q^-1. Its coordinate update is -(Q^T g)/(c+mu*scale*d). This uses the true gradient and does not preserve SOAP's momentum proposal; frozen -Pg is retained to expose that distinction.

| State | Exact coordinate-compression action error | Rank-16 diagonal errors, repeats 0/1 | Rank-16 action errors, repeats 0/1 | Eligible rank |
| --- | --- | --- | --- | --- |
| 10 | 0.7655 | 0.2980/0.2459 | 0.7667/0.7936 | None |
| 50 | 0.7951 | 0.4982/0.4506 | 1.0615/1.0746 | None |
| 100 | 0.5129 | 0.5096/0.2005 | 0.8821/0.5192 | None |

No declared rank passed both error targets in both repeats. More probes alone cannot remove the discrepancy between the exact coordinate-diagonal model and the exact tensor GGN. The probes improve over frozen -Pg at the early and middle states, but the structured candidate does not beat momentum-based searched SOAP there. This local result does not disprove the value of downstream curvature; it rejects this particular diagonal compression as an accurate tensor approximation.

The next candidate should preserve useful within-tensor correlations or use the downstream measurements to optimize a much smaller allocation problem. Neither a larger probe rank nor a training run was selected after seeing these results.

## Numerical checks and cost

Maximum transported/direct factor relative error: 6.796e-16; maximum new candidate solve residual: 2.474e-11. All numerical tolerances are 1e-8. The 55-test suite passed before execution, including exact dense/operator/probe agreement, nonorthogonal SOAP-coordinate solves, projected skips, microbatch normalization and state non-mutation. No model, context or Torch RNG mutation occurred in the run.

All 60 candidate comparisons and their search trials are retained. Snapshot work took 5.800 seconds. Peak process RSS was 852.7 MiB. The separately budgeted tensor workspace was below 768 MiB. Probe counters refer to stage pullbacks, not complete backward passes. Cost includes oracle diagnostics and is not an optimizer-overhead estimate.

## Relation to the update-norm paper

For tensor b, write Delta_b=R sqrt(w_b) u_b. R is total update length, w_b is its squared-norm share and u_b is its unit direction. With phi_b=-g_b^T u_b and K_bc=u_b^T C_bc u_c, the quadratic decrease for amplitudes a is phi^T a - a^T K a/2. The reduced allocation experiment maximizes this expression with damping and nonnegative a; it accounts for alignment as well as exposure.

A single namesake/fallback rate ratio fixes at most the aggregate group share. It cannot generally match Muon's fan-out-dependent allocation among matrices, change SOAP's tensor directions, or fix the total step length. The PDF is descriptive and does not identify the failed matched-ratio tuning run; diagnosing that specific run requires its measurements.

A large cross term does not imply a large benefit from changing direction. For C=I+rho 11^T and g=1, full and block-diagonal inverse updates lie on the same ray even when off-diagonal terms dominate d^T C d. A global scalar search can correct that scale difference. This is a mathematical counterexample, not a claim that our model has exactly that geometry. Also distinguish residual-path interactions inside each C_bb from interactions between parameter tensors C_bc.

![Fixed-state comparison](/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Uni/Eigene Projekte/SkipSOAP/residual-curvature-lab/runs/20261005T074622-d81830d0/comparison.png)
