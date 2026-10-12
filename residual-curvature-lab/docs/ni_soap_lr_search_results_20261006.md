# NI-SOAP learning-rate search — 6 October 2026

Extension run: `runs/20261006T153126-f597adda`; parent: `runs/20261006T100557-f1efb932`. The user requested search for both supplied algorithms. PARSEC already had it; this extension adds NI-SOAP with and without noise. The [prospective protocol](ni_soap_lr_search_protocol_20261006.md) was saved before execution. All arms below use the same 5,738-parameter ViT, MNIST batch 512, seed 0, 50 consumed iterations and 5,000 validation examples.

| Method | Validation loss | Accuracy | Training seconds |
| --- | ---: | ---: | ---: |
| Adam native | 1.288008 | 57.84% | 3.90 |
| Adam + search | 1.239108 | 57.78% | 48.21 |
| PARSEC native | 1.234958 | 60.28% | 14.66 |
| PARSEC + search | 1.007097 | 64.88% | 54.32 |
| NI-SOAP, alpha=0 | 1.365508 | 60.90% | 3.82 |
| NI-SOAP, alpha=1 | 1.406900 | 58.36% | 3.93 |
| Exact tensor GGN + search | 0.327676 | 90.20% | 193.69 |
| Tensor graph-Wasserstein + search | 0.381630 | 88.44% | 196.41 |
| NI-SOAP alpha=0 + LR search | 0.867103 | 70.78% | 42.25 |
| NI-SOAP alpha=1 + LR search | 0.973104 | 66.46% | 44.45 |

The earlier official searched SOAP achieved loss .731623 and accuracy 77.40% on this paired workload. It differs from the supplied NI-SOAP alpha-zero drift, including first-call and epsilon semantics; use alpha zero as the matched noise control.

## Search definition

Decompose one native lr=.003 update as drift d plus noise n. For learning-rate multiplier a, search `Delta(a)=a*d+sqrt(a)*n`, so `eta=.003*a`. Each candidate holds the true gradient, moments and noise draw fixed. Only the chosen displacement is committed; optimizer statistics advance once, including when zero is selected. The grid is zero plus 2^[-16..16], with 18 golden-section refinements for an interior positive grid winner. This is a single global search, not per-layer search.

The alpha-zero path is the ordinary update ray. The noisy path preserves the source formula at each fixed learning rate, with drift proportional to eta and noise proportional to sqrt(eta); rescaling the entire noisy step would not. However, selecting eta using the noisy candidate loss changes its conditional noise distribution. These are searched variants, not unchanged native stochastic dynamics. Alpha and covariance structure are still untuned.

| Searched NI arm | Min eta | Median eta | Max eta | Rejections | Noise/drift energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| NI-SOAP alpha=0 + LR search | 0.00460171 | 0.0206102 | 0.0570667 | 0 | 0.000% |
| NI-SOAP alpha=1 + LR search | 0.0033718 | 0.0204965 | 0.0589612 | 0 | 18.055% |

The noisy searched arm's final loss relative to the zero-noise searched control is 1.122247; values above one favor zero noise. Fallback tensors received 99.9869% of realized noise energy. Energy ratios sum squared per-step increments, not squared total displacement. Pre-selection expected-energy formulas saved per step are not unconditional expectations after loss-based selection.

## Verification and interpretation

All 96 tests passed before the extension (and the 12 proposal/path tests passed after the final metadata change). 47 source archive files and all parent analysis artifact hashes verified. Both final checkpoints reproduce validation and the next uncommitted proposal exactly. Model, optimizer, data-order and RNG non-mutation passed. All 100 searches were bracketed or explicitly rejected. Maximum noise/drift energy-closure error: 4.594e-16.

Training seconds include proposals and searches, excluding validation, file writes and replay. Learning-rate search tests whether adaptive global step size changes these trajectories. It does not prove all remaining differences are curvature effects: momentum, noise geometry, per-tensor scaling and approximation still differ. This is one short untuned seed, not a replicated ranking. No official test data were evaluated.

The parent [transport/PARSEC/NI-SOAP report](transport_parsec_noise_results_20261006.md) contains algorithm equations, native noise accounting, Ritz residuals and the transport derivation. Rebuild this extension report with:

```sh
.venv/bin/python scripts/summarize_ni_lr_search.py --run runs/20261006T153126-f597adda
```

![Comparison](../runs/20261006T153126-f597adda/comparison.png)
