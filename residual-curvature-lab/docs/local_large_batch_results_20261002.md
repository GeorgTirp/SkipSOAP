# Local MNIST larger-batch oracle results — 2 October 2026

Fixed-state run: `runs/20261002T121124-74f10aae`. Batch 256 completed before batch 512. The model is the existing 14x14 residual MLP: 6,634 parameters in 22 exact tensor blocks. All states and SOAP moments come from the original batch-64 trajectory. Each candidate uses the true construction gradient and a cloned SOAP proposal. Damping is mu=max(gamma*trace(C)/D, 1e-8) with normalized inverse-SOAP metric.

The table reports independent-panel mean loss decreases, with positive values indicating improvement. SOAP and GGN directions use the same broad construction-only line search; the panel never selects a scalar or damping.

| Batch | State | Native SOAP | Searched SOAP | GGN .001 | GGN .01 (primary) | GGN .1 |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 10 | 0.038199 | 0.421193 | 0.225562 | 0.329589 | 0.419395 |
| 256 | 50 | 0.022547 | 0.200348 | 0.089616 | 0.161529 | 0.184468 |
| 256 | 100 | 0.002221 | 0.003544 | -0.017722 | -0.017796 | -0.029143 |
| 512 | 10 | 0.038637 | 0.443676 | 0.250387 | 0.424910 | 0.603769 |
| 512 | 50 | 0.023528 | 0.207040 | 0.190654 | 0.261146 | 0.288680 |
| 512 | 100 | 0.002018 | 0.003344 | -0.001348 | 0.006875 | 0.001803 |

At gamma=.01, GGN improves construction loss more than searched SOAP at all six states. At batch 256 it transfers worse on the independent mean at all three states. At batch 512 it transfers better at two of three states, with a small positive mean difference. The gamma=.1 sensitivity result is stronger at the early and middle batch-512 states, but it was not selected for training using the panel. The four panels and three states are not independent training seeds and these results are not a significance test.

| Batch | Primary GGN minus searched SOAP, mean over states | States with positive difference | Short-training gate |
| --- | --- | --- | --- |
| 256 | -0.050588 | 0/3 | not passed |
| 512 | +0.012957 | 2/3 | passed |

All action and solve checks passed 1e-8: maximum action error 4.356e-15; maximum combined solve residual 8.329e-14. All 30 direction searches selected an interior grid bracket and used 52 evaluations. No positive search boundary limited these comparisons. Non-mutation checks passed for all six snapshots.

Measured snapshot work: 14.849 seconds total; process-lifetime peak RSS 1135.5 MiB. The conservative tensor-workspace estimate was 636.8 MiB against a declared 768 MiB budget. RSS includes the dataset, Python and numerical libraries. Saved artifacts include the exact blocks, directions, optimizer contexts, example IDs, source archives and checksums.

![Independent loss decreases](/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Uni/Eigene Projekte/SkipSOAP/residual-curvature-lab/runs/20261002T121124-74f10aae/independent_decreases.png)

## Paired batch-512 training followup

Training run: `runs/20261002T121635-0dac3337`. The predeclared snapshot gate passed only at batch 512. All three arms started from identical seed-0 random weights and consumed the same 50 batches (25,600 examples; the first batch initializes statistics without a parameter update). Validation uses all 5,000 reserved examples at steps 0, 10, 25 and 50, without tuning or early stopping. The official test set was not evaluated. The exact arm uses the predeclared gamma=.01.

| Method | Iteration | Validation loss | Accuracy | Training-step seconds |
| --- | --- | --- | --- | --- |
| Native SOAP | 10 | 2.044920 | 47.04% | 0.089 |
| Native SOAP | 25 | 1.538456 | 63.40% | 0.207 |
| Native SOAP | 50 | 0.750541 | 83.12% | 0.398 |
| Line-searched SOAP | 10 | 0.902813 | 69.14% | 0.417 |
| Line-searched SOAP | 25 | 0.453638 | 86.78% | 1.100 |
| Line-searched SOAP | 50 | 0.491103 | 89.04% | 2.439 |
| Exact tensor GGN | 10 | 0.547983 | 83.60% | 19.259 |
| Exact tensor GGN | 25 | 0.463954 | 85.38% | 51.189 |
| Exact tensor GGN | 50 | 0.417581 | 86.92% | 106.705 |

At iteration 50, exact GGN validation loss minus searched SOAP is -0.073522; minus native SOAP is -0.332959. Its measured training-step cost is 43.8x searched SOAP and 268.4x native SOAP.

These are untuned single-seed short trajectories. Training-step time includes gradients, curvature, solves and search, but excludes dataset loading, validation, checkpoint writing and plotting. No equal-wall-clock speed advantage or replicated ranking follows from equal-iteration comparisons. This exact tensor reference uses SOAP-metric damping and is not a reproduction of the paper.

![Paired training](/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Uni/Eigene Projekte/SkipSOAP/residual-curvature-lab/runs/20261002T121635-0dac3337/comparison.png)
