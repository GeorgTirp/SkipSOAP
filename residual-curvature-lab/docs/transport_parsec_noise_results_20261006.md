# Transport, PARSEC and NI-SOAP — 6 October 2026

Run: `runs/20261006T100557-f1efb932`. The [prospective protocol](transport_parsec_noise_protocol_20261006.md) selected eight paired trajectories on the 5,738-parameter ViT, MNIST batch 512, seed 0, 50 consumed iterations, CPU float64 and one thread. Validation uses 5,000 reserved examples; no official test data or hyperparameter sweep was used. The user confirmed the default PARSEC precondition-then-AdamW branch.

## Final measurements

| Method | Validation loss | Accuracy | Training seconds | Accepted updates |
| --- | ---: | ---: | ---: | ---: |
| Adam native | 1.288008 | 57.84% | 3.90 | 50 |
| Adam + search | 1.239108 | 57.78% | 48.21 | 50 |
| PARSEC native | 1.234958 | 60.28% | 14.66 | 50 |
| PARSEC + search | 1.007097 | 64.88% | 54.32 | 50 |
| NI-SOAP, alpha=0 | 1.365508 | 60.90% | 3.82 | 50 |
| NI-SOAP, alpha=1 | 1.406900 | 58.36% | 3.93 | 50 |
| Exact tensor GGN + search | 0.327676 | 90.20% | 193.69 | 49 |
| Tensor graph-Wasserstein + search | 0.381630 | 88.44% | 196.41 | 49 |

Time includes gradient, proposal, curvature refresh and global search where used. It excludes data loading, validation, writes, replay and separately timed numerical action audits. Every searched step uses one global scalar with the same 52-evaluation budget; none is a per-layer search. NI-SOAP uses native steps to preserve its noise law. All hyperparameters stayed as prospectively declared.

For context, the earlier official-SOAP run used identical model source, split, initial loss and batch IDs. It is not the alpha-zero NI-SOAP control; the supplied implementation has different initialization and epsilon semantics.

| Earlier official SOAP | Validation loss | Accuracy |
| --- | ---: | ---: |
| native_soap | 1.354356 | 62.02% |
| searched_soap | 0.731623 | 77.40% |

Within each selected pair, the relative reduction in final validation loss was (negative means higher loss):

| Candidate versus control | Relative loss reduction |
| --- | ---: |
| PARSEC native versus Adam native | +4.12% |
| PARSEC + search versus Adam + search | +18.72% |
| NI-SOAP, alpha=1 versus NI-SOAP, alpha=0 | -3.03% |
| Tensor graph-Wasserstein + search versus Exact tensor GGN + search | -16.47% |

## PARSEC as implemented

With orthonormal global Ritz vectors in rows of V, transform the current gradient as `g_tilde = (I - V.T V) g + V.T diag((lambda + delta)^(-1/2)) V g`, then run ordinary AdamW on g_tilde in the original parameter coordinates. The untracked complement is unchanged. Moments persist across curvature refreshes. Weight decay is zero in this experiment. The selected branch is not the separate Newton/Adam split.

Rank 16, 32 Krylov iterations, GGN refresh at 10/20/30/40/50, damping 1e-4, lr=.003, Adam betas .9/.999, eps=1e-8. Both native and searched variants match their Adam controls exactly through step 9. Refresh was shortened from the supplied default 100 so curvature activates during this 50-step pilot. Float64 full reorthogonalization replaces the helper's lower-precision Lanczos storage; these are approximate Ritz vectors, not exact top eigenspaces.

Preconditioning the input to Adam is different from preconditioning its final update. For a fixed positive coordinate scaling and negligible epsilon, Adam scales its first moment by c and its second moment by c^2, cancelling that input scaling. PARSEC can still change the direction through dense coordinate mixing and time-varying transforms. Its equations alone do not guarantee the inverse-root geometry survives Adam unchanged.

| Arm | Refresh | Largest eigenvalue | Smallest retained | Max relative Ritz residual |
| --- | ---: | ---: | ---: | ---: |
| PARSEC native | 10 | 70.478 | 0.52921 | 7.799e-06 |
| PARSEC native | 20 | 307.13 | 1.4742 | 4.158e-04 |
| PARSEC native | 30 | 393.89 | 1.9026 | 5.325e-02 |
| PARSEC native | 40 | 414.57 | 2.2703 | 4.092e-03 |
| PARSEC native | 50 | 579.81 | 3.6169 | 3.923e-03 |
| PARSEC + search | 10 | 101.76 | 0.63463 | 1.605e-01 |
| PARSEC + search | 20 | 109.01 | 1.0273 | 2.682e-02 |
| PARSEC + search | 30 | 108.6 | 1.1911 | 3.809e-02 |
| PARSEC + search | 40 | 82.967 | 1.1727 | 8.123e-03 |
| PARSEC + search | 50 | 85.712 | 1.6814 | 7.070e-02 |

## NI-SOAP as implemented

The supplied SOAP-like drift receives matrix noise `alpha sqrt(lr/B) L_cached^(1/2) Z R_cached^(1/2)` and isotropic fallback noise `alpha sqrt(lr/B) z`. Gradients, moments and factor statistics do not directly ingest the noise. Cached bases/eigenvalues are used before the current statistics update; the first matrix call only initializes statistics, while fallback parameters update immediately. Linear weights take the matrix route; embeddings, normalization and biases take fallback. All routes are saved.

Settings: alpha 0/1, lr=.003, B=512, betas .95/.95, factor beta .95, QR refresh 10, eps=1e-8, weight decay zero. Ports preserve finite update equations in float64, with local Torch random generators; they are not bitwise JAX reproductions.

Across the 50 alpha-one steps, sums of squared increments were:

| Quantity | Measured sum |
| --- | ---: |
| drift_energy | 0.38595898 |
| noise_energy | 0.20529151 |
| expected_noise_energy | 0.20462366 |
| realized_matrix_noise_energy | 0.0001200699 |
| realized_fallback_noise_energy | 0.20517144 |
| drift_noise_cross_term | 0.0013828987 |
| update_energy | 0.59263339 |

Noise/drift energy ratio: 53.1900%. Fallback share of realized noise: 99.9415%. These sums are per-step update energies, not squared net displacement. The matrix noise expectation is alpha^2 lr/B tr(L_cached)tr(R_cached); the fallback expectation is alpha^2 lr/B times its parameter count. This does not automatically equal SGD gradient-noise energy, despite the source comment. No additional normalization was silently introduced.

## Transport geometry and the step decomposition

Normalized steepest descent minimizes <g,d> over a unit norm ball. For matrix spectral norm, G=U S V.T gives the idealized direction -U V.T; momentum and practical Muon scaling are separate. See [Modular Duality](https://proceedings.mlr.press/v267/bernstein25a.html).

For categorical probabilities p, use a complete graph with equal class conductances and edge mobility (p_i+p_j)/2. Its Laplacian L(p) defines the zero-sum tangent norm `||s||_p^2 = s.T L(p)^dagger s`. Pulling it back through predictions gives `M = mean J_p.T L(p)^dagger J_p`. The damped steepest step has direction `-(M + mu Omega)^(-1) g`. For radius R, divide by `sqrt(g.T (M + mu Omega)^(-1) g)` and multiply by R. For the tensor-amplitude decomposition d=E a, its squared norm is a.T E.T M E a. This is the same constrained-step framework; the chosen metric need not equal loss curvature. See [Li and Montufar](https://arxiv.org/html/1803.07033v5).

This run uses exact within-tensor blocks of that dynamic graph-Wasserstein pullback. It is not static discrete Kantorovich W2 or a canonical fixed norm on signed weights. Equal class costs avoid inventing ordinal geometry for digits. The comparison uses exact tensor GGN, the same normalized inverse shadow-SOAP damping metric, mu=max(.01 trace(M)/5738,1e-8), the current gradient and the same global search. The earlier GGN training estimated trace with eight probes; this control uses exact trace to avoid conflating the trace estimator with the geometry.

At uniform probabilities over K classes, this graph metric is exactly GGN/K. With trace-scaled damping and scalar search, the resulting rays therefore agree (when the damping floor is inactive). Away from uniform predictions the metrics can differ. This limits what equal-cost categorical transport adds near initialization; it does not test a learned or feature-informed ground geometry.

## Verification and limits

All 92 tests passed before training. 46 archived source hashes verified; all eight final checkpoints reproduce validation and the uncommitted next proposal exactly. Optimizer, model, data-order and global RNG non-mutation passed. All 198 searches were bracketed or rejected; 0 were rejected. Maximum action error 1.247e-14; solve residual 6.937e-14. PARSEC Ritz errors above describe approximation quality and are not exact-solve residuals.

This is one short, untuned seed. Native and searched arms have different evaluation costs; PARSEC uses a global low-rank preconditioner whereas the exact metrics use tensor blocks. The 50 consumed batches include method-specific initialization: curvature arms make 49 updates. No result establishes a general optimizer ranking, transport ground-geometry optimum, or a causal explanation of the prior SOAP/GGN gap.

Raw validation, per-step histories (including all search trials), noise contributions, Ritz diagnostics, settings, source archives and checkpoints remain in the run. Rebuild with:

```sh
.venv/bin/python scripts/summarize_transport_parsec_noise.py --run runs/20261006T100557-f1efb932
```

![Comparison](../runs/20261006T100557-f1efb932/comparison.png)
