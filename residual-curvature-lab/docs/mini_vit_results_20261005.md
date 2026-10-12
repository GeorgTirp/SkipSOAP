# Mini ViT and SOAP tensor allocation — 5 October 2026

Run: `runs/20261005T135920-d7ce29b2`. The [prospective protocol](mini_vit_protocol_20261005.md) selected this 5,738-parameter ViT before execution: two pre-LayerNorm attention/MLP blocks, width 16, two heads, 16 image patches plus CLS, 28x28 MNIST. All arms use seed 0, batch 512, identical initial weights and example order, CPU float64 and one thread. There is no augmentation or dropout.

## Paired training

These are measurements after 50 consumed iterations, including the first SOAP statistics-only call. Validation uses the reserved 5,000 examples. Time sums training steps, including proposal, curvature and search; it excludes dataset loading, validation, checkpoint writes and fixed-state diagnostics.

| Method | Validation loss | Accuracy | Training seconds | Accepted updates |
| --- | ---: | ---: | ---: | ---: |
| Native SOAP | 1.354356 | 62.02% | 4.23 | 49 |
| SOAP + global search | 0.731623 | 77.40% | 46.04 | 49 |
| SOAP + diagonal allocation | 0.748057 | 75.62% | 164.10 | 49 |
| SOAP + full allocation | 0.816645 | 72.88% | 159.50 | 49 |
| Exact tensor GGN + search | 0.325714 | 90.30% | 219.33 | 49 |

Exact tensor GGN's final validation loss is 0.445 times searched SOAP's, at 4.76 times the measured training-step time. This is an observed advantage in this pilot, not a replicated or equally tuned optimizer ranking.

Both allocation variants finish with higher validation loss than searched SOAP. The shared-state allocation comparisons therefore did not establish a training improvement.

Full allocation was ahead at iterations 10 and 25, but the advantage reversed by iteration 50.

Native SOAP uses lr=.003. Every searched arm uses one **global** scalar, with the same 52-evaluation policy: zero and native-radius multipliers 2^[-16..16], then 18 golden refinements. Allocation changes nonnegative tensor amplitudes before that scalar search; it does not perform independent nonlinear searches per layer. All settings remained as prospectively declared.

## Shared-state direction comparisons

Positive values below mean independent cross-entropy decrease, averaged across four disjoint 512-example panels. Each candidate uses the same 512-example construction batch, current gradient, SOAP context, exact-trace damping and global-search budget at that state. The states come from searched SOAP; they are not the separate allocation trajectories.

| Candidate | State 10 | State 25 | State 50 |
| --- | ---: | ---: | ---: |
| SOAP + global search | 0.048118 | 0.015684 | 0.000558 |
| Frozen -Pg + search | 0.040975 | 0.015984 | -0.000084 |
| SOAP + diagonal allocation | 0.050339 | 0.018513 | -0.001882 |
| SOAP + full allocation | 0.127626 | 0.056756 | 0.001395 |
| Exact tensor GGN + search | 0.131513 | 0.074808 | -0.004763 |

At state 50, exact tensor GGN reduces construction loss by 0.198640 but its mean independent decrease is -0.004763. This directly separates fitting the construction batch from transfer. It does not prove the cause of the allocation training outcome.

The exact-GGN training gate selected the optional arm: **True**. Independent differences versus searched SOAP were [0.083394, 0.059125, -0.005321]; the first-snapshot estimate for 49 updates was 211.72 seconds against a 300-second budget. Opportunity and numerical/search gates were True and True, respectively.

## What the allocation computes

E contains unit SOAP tensor directions. K=E^T C E is computed by exact directional JVPs, including cross-tensor curvature; no full parameter GGN is assembled for allocation. With phi=-E^T g and H=K+mu E^T Omega E, full allocation minimizes a^T H a/2-phi^T a over a>=0. Its KKT conditions are a>=0, Ha-phi>=0 and componentwise complementarity. Diagonal allocation uses a=[phi]_+/diag(H); only this diagonal model has that componentwise solution. Shares are a_b^2/sum(a^2). Negative-alignment directions remain eligible in the full problem.

Training uses mu=max(.01*estimated_trace(C)/5738,1e-8), with eight candidate-independent Rademacher JVP probes per step and the normalized inverse-SOAP metric. The trace estimate is approximate; the projected K is exact up to numerical precision. Snapshots use exact trace for every candidate. SOAP moments advance exactly once per consumed gradient. Exact tensor GGN uses the current gradient; allocation preserves SOAP momentum directions.

The following diagnostic evaluates all allocation rays in the same full damped quadratic. Each ray gets its optimal nonnegative quadratic scalar, so a poor unscaled diagonal step does not by itself count as evidence for useful coupling. These predicted decreases are construction-batch model values, distinct from the independent nonlinear decreases above.

| State | Native SOAP ray | Diagonal-allocation ray | Full-allocation optimum | Active full tensors |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.046360 | 0.051323 | 0.126202 | 8 |
| 25 | 0.092020 | 0.106835 | 0.192462 | 18 |
| 50 | 0.053959 | 0.045480 | 0.075595 | 12 |

## Verification and artifacts

The source archive has 36 verified files. All snapshot array checksums, paired data orders and final checkpoint restoration/evaluation checks passed. Maximum snapshot tensor-action error: 1.150e-14; tensor-solve residual across training and snapshots: 5.433e-14; allocation KKT residual across training and snapshots: 4.441e-16. All 211 searched proposals were bracketed or explicitly rejected; none selected a positive search boundary. The 69-test suite passed before the dataset run.

Estimated tensor workspace: 71.35 MiB (budget 768 MiB). Peak process RSS over the suite: 1056.61 MiB. RSS is cumulative process high-water memory, not per-method incremental overhead. Each initialized allocation step computes up to 40 directional and eight trace JVPs. Both allocation candidates are calculated and audited inside each allocation arm, so these timings describe the research runner rather than a streamlined diagonal implementation.

See `validation.json`, per-arm `training.json`, `snapshot_candidates.json`, `snapshot_diagnostics.json`, `exact_training_gate.json`, `verification.json`, `snapshot-*/arrays.pt`, and saved checkpoints/source archives in the run. The comparison plot can be rebuilt with:

```sh
.venv/bin/python scripts/summarize_mini_vit.py --run runs/20261005T135920-d7ce29b2
```

## Interpretation limits

This is one seed and a short, untuned pilot. Tensor means all 40 registered parameter tensors, including biases, normalization, embeddings and attention matrices; it is not the full GGN or necessarily the same partition as an external layerwise-GGN experiment. Optimizing the construction-batch quadratic does not guarantee nonlinear or independent-batch improvement. Fixed-state comparisons isolate a local proposal change; diverged training trajectories do not causally decompose final gains. Panels are reused across states and are diagnostic. No official test examples were evaluated.

The KKT optimum is conditional on the current SOAP directions, damping and construction batch. It is not a universal ideal allocation or a reason to copy Muon shares. Further work should first check the allocation method on its own trajectory and replicate the paired outcome. A prospectively selected convex amplitude-norm budget could test whether limiting displacement improves transfer; it was not tested in this pilot.

The prior MLP used 14x14 inputs while this ViT uses 28x28. A difference in outcomes therefore cannot be attributed solely to architecture. The existing explicit MLP residual-transport implementation was not reused for CLS/final-norm transport; this experiment uses generic parameter-space GGN operators. It does not validate a cheap downstream-statistic SOAP variant.

![Mini ViT comparison](../runs/20261005T135920-d7ce29b2/comparison.png)
