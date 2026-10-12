# What information separates SOAP from tensor GGN?

Run: `runs/20261005T143029-9b582ed5`. The [prospective protocol](jacobian_information_protocol_20261005.md) selected six saved ViT states, batch 512, gamma .01, identical normalized inverse-SOAP damping and global scalar searches. The forward network and ordinary training gradient are unchanged. The four evaluation panels (2,048 examples) were unused in earlier experiments and are disjoint from construction/training/validation. They are reused across these six states; they do not select settings. This is a local intervention, not another training comparison.

## Measurements

Each entry is the independent mean loss decrease; positive is better. Model-sampled-label entries average two prespecified repeats, retaining both repeats and every panel in raw data. Every row gets the same global scalar-search budget, selected only on construction loss.

### States from searched-SOAP training

| Candidate | Step 10 | Step 25 | Step 50 |
| --- | ---: | ---: | ---: |
| SOAP momentum + frozen map | 0.039564 | 0.021822 | 0.000640 |
| SOAP frozen map + current gradient | 0.033191 | 0.025898 | 0.002988 |
| Dense GGN + current gradient | 0.135327 | 0.085051 | -0.012308 |
| Dense GGN + effective SOAP momentum | 0.035343 | 0.033845 | 0.045150 |
| GGN diagonal in SOAP basis + gradient | 0.044453 | 0.040127 | -0.001335 |
| GGN diagonal in SOAP basis + momentum | 0.049824 | 0.068798 | 0.003981 |
| Dense empirical Fisher, raw | 0.118798 | 0.119649 | -0.019685 |
| Dense empirical Fisher, trace matched | 0.119131 | 0.119290 | -0.019770 |
| Empirical Fisher diagonal in SOAP basis, trace matched | 0.045515 | 0.040075 | -0.000053 |
| GGN directions / SOAP tensor norms | 0.134313 | 0.095388 | -0.013415 |
| SOAP directions / GGN tensor norms | 0.037197 | 0.028638 | 0.001767 |
| Model-sampled labels, 1 per example | 0.127194 | 0.063904 | -0.027518 |
| Model-sampled labels, 4 per example | 0.132185 | 0.078245 | -0.018115 |
| Model-sampled labels, 16 per example | 0.138024 | 0.083622 | -0.014684 |

### States from exact tensor-GGN training

| Candidate | Step 10 | Step 25 | Step 50 |
| --- | ---: | ---: | ---: |
| SOAP momentum + frozen map | 0.014364 | -0.006423 | 0.005969 |
| SOAP frozen map + current gradient | 0.015570 | 0.017390 | 0.020846 |
| Dense GGN + current gradient | 0.039168 | 0.010830 | 0.017227 |
| Dense GGN + effective SOAP momentum | 0.037787 | -0.020162 | -0.036464 |
| GGN diagonal in SOAP basis + gradient | 0.027310 | 0.018176 | 0.020568 |
| GGN diagonal in SOAP basis + momentum | 0.041757 | -0.014042 | -0.009236 |
| Dense empirical Fisher, raw | 0.051524 | -0.000708 | 0.002117 |
| Dense empirical Fisher, trace matched | 0.050672 | -0.002222 | 0.001457 |
| Empirical Fisher diagonal in SOAP basis, trace matched | 0.028788 | 0.018261 | 0.019684 |
| GGN directions / SOAP tensor norms | 0.037359 | 0.016480 | 0.017375 |
| SOAP directions / GGN tensor norms | 0.015688 | -0.011400 | 0.001416 |
| Model-sampled labels, 1 per example | 0.032015 | 0.005406 | -0.001342 |
| Model-sampled labels, 4 per example | 0.039066 | 0.009713 | 0.008339 |
| Model-sampled labels, 16 per example | 0.038551 | 0.010707 | 0.012063 |

## Observed interpretation

The strongest simple explanation is not supported uniformly: access to every output-Jacobian direction is not necessary to recover most of the early local gain. At the searched-SOAP step-10 state, dense trace-matched empirical Fisher gives 0.119131 independent decrease versus GGN 0.135327; at step 25 these are 0.119290 and 0.085051. The raw empirical-Fisher values are very similar, so oracle trace matching does not account for these early improvements. This control uses per-example ordinary gradients and a dense tensor covariance, not SOAP's usual aggregate-gradient statistic.

At those same two states, compressing either curvature source to a diagonal in SOAP's basis removes much of the gain. Conversely, GGN tensor directions keep the early advantage when assigned SOAP tensor norms. These interventions implicate within-tensor geometry more strongly than simply copying a different allocation or global update length.

This is state dependent. On searched-SOAP step 50, current-gradient GGN and empirical Fisher both increase independent loss. GGN with effective SOAP momentum improves it. On the GGN trajectory at steps 25/50, current-gradient GGN beats dense empirical Fisher, but the frozen SOAP map with the current gradient and the GGN diagonal are also strong. Changing momentum alone is therefore not a universal explanation either; its interaction with the preconditioner and the trajectory matters.

Model-sampled output directions approach the exact-GGN local measurements at early/middle states as the number of draws rises. At the late GGN state, even 16 draws retains a noticeable gap. This shows potential for extra output probes, without establishing a cheap representation or a successful training algorithm. All sampled controls still use dense tensor matrices.

A direct next test of the training-gap explanation would compare paired dense empirical-Fisher training with GGN under the same damping/search/RHS protocol, alongside a current-gradient SOAP-map control. It has not been launched here. These controls also do not separately identify EMA lag, basis refresh, inverse versus inverse-square-root scaling, or a new factorization.

## What each intervention identifies

Write C_bb = mean J_ib^T H_i J_ib, with H_i=diag(p_i)-p_i p_i^T. The dense empirical Fisher is F_bb=mean g_ib g_ib^T, where g_ib=J_ib^T(p_i-onehot(y_i)). It can be obtained from per-example ordinary backprop; it does not require observing the Jacobian in all output directions. It nevertheless retains all within-tensor parameter correlations. It is richer than SOAP's batch-gradient statistics and is not an implementation of SOAP.

The dense C versus dense F intervention changes the output-information target while holding the true gradient, dense representation, damping, partition and scalar budget fixed. Raw F and per-tensor trace-matched F are both retained. The trace-matched control adds oracle trace information; do not count it as a practical estimator. F is not an unbiased GGN estimate for fixed observed labels. This intervention changes output weighting/rank together, and cannot distinguish those finer mechanisms by itself.

The dense-versus-diagonal intervention keeps the curvature source and actual SOAP basis fixed. The diagonal model is Q^-T diag(diag(Q^T C_bb Q)) Q^-1, or its F counterpart, with physical tensor traces matched to C_bb. Thus a dense advantage at fixed source demonstrates the value of correlations discarded by this representation. It does not prove that a different factorization or basis could not retain those correlations cheaply.

The RHS intervention reconstructs an effective momentum vector from -P^-1 delta_SOAP, normalized to the gradient norm. Applying -P returns the native SOAP ray, verified using the actual, slightly nonorthogonal saved bases. Comparing GGN on this RHS with GGN on the current gradient measures sensitivity to that change. SOAP without momentum remains a separate control. The interaction means one cannot assign independent percentages of the training gap to momentum and curvature.

For labels sampled from the current model, E[(p-onehot(label))(p-onehot(label))^T]=diag(p)-pp^T. Their per-example gradient outer products therefore estimate C. Nested counts 1/4/16, with two fixed repeats, test extra output directions without changing the target or dense tensor representation. The diagnostic reuses the full Jacobian to construct these matrices; measured time is not the cost of a VJP-only estimator or a practical optimizer.

Norm transfers preserve each source tensor direction and assign the other method's complete tensor norms before global search. They separate the local value of within-tensor directions from tensor allocation. They do not prescribe universal Muon or GGN shares.

## Numerical checks and reproducibility

All 75 tests passed before execution. Across six states, maximum exact tensor-action error was 1.150e-14, mean-gradient error 1.558e-15, native momentum-ray reconstruction error 6.837e-11, and tensor-solve residual 1.218e-13. Model, optimizer and Torch RNG were preserved. Source archives, all six parent checkpoints and saved arrays passed checksum verification.

All 102 searches were bracketed or explicitly rejected; 0 selected rejection. Snapshot work took 151.71 seconds. Maximum estimated tensor workspace was 193.91 MiB; cumulative process peak RSS was 945.53 MiB. Exact/sample information is recomputed on the same construction batch; no damping or rank was selected on panels.

Raw candidates, scalar trials and panels are in `all_candidates.json`; discrepancies and trace scales are in `all_diagnostics.json`. Each state saves parameters, optimizer context, exact C/F blocks, sampled labels and candidate directions in `arrays.pt`. `contrasts.json` reports paired panel differences and descriptive standard errors, not independent training-replicate uncertainty. Rebuild this report with:

```sh
.venv/bin/python scripts/summarize_jacobian_information.py --run runs/20261005T143029-9b582ed5
```

## Scope of a conclusion

A dense empirical-Fisher win would show that observing every output-Jacobian direction is not necessary for that local improvement. A dense C/F advantage over both SOAP-coordinate diagonals would support preserving within-tensor correlations. Momentum and direction/norm controls further restrict the explanation. These are falsifiable local mechanisms, not a unique causal decomposition of the previous 50-step training gap. A dense empirical Fisher training arm was not run. Six related states and two probe repeats are not optimizer replication; there is no conclusion about skip-specific information.

Primary context: [SOAP](https://arxiv.org/abs/2409.11321) and [the full/layerwise GGN study](https://arxiv.org/abs/2510.09378).

![Information comparison](../runs/20261005T143029-9b582ed5/comparison.png)
