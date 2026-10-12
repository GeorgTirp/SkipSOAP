# Geometry follow-up — results

Study `geometry_followup_20261008`, protocol [PROTOCOL.md](PROTOCOL.md). Large artifacts:
`~/skipsoap-study-data/geometry_followup_20261008/runs/`. Light mirrors: [runs/](runs/).

**Status (2026-10-08, 13:03Z):** complete. All gates passed, all units completed, no divergence in evaluation.

## 1. Summary

- **PARSEC, tuned (both branches):** better than tuned Adam and tuned searched SOAP at step 500 in all seeds (.186 default, .192 split versus .202 and .210), *mixed* versus exact tensor GGN (.182), worse early. Tuned native SOAP (.161) remains lowest. The tuned split branch is effectively Adam on the complement: its top-space Newton step averages 0.7% of the complement step, and larger Newton steps diverge.
- **OT (graph-Wasserstein, tensor blocks), tuned:** *worse* than tensor GGN at step 50 but *better* at 200 and 500 in all seeds (.165 versus .182). Each arm uses its own tuned damping (OT gamma .1, GGN .001), so this is not a pure geometry comparison.
- **Full versus block curvature (matched damping, metric, search):** full GGN is *worse* than block GGN at 50/200/500 in all seeds (.330 versus .182 at 500). Full OT is *better* than tensor OT at step 50 but *worse* at 200/500. Cross-tensor blocks carry 54–92% of the full matrices' energy, yet using them hurts training here.

| Seeds 1–3, step 500 | Loss | Measured s/step |
| --- | ---: | ---: |
| native SOAP, lr 1e-2 (gap audit) | 0.161 ± 0.010 | 0.11 |
| OT tensor, gamma .1 | 0.165 ± 0.006 | 6.6 |
| exact tensor GGN, gamma .001 (gap audit) | 0.182 ± 0.002 | 6.5 |
| PARSEC default, lr 3e-3, rank 32 | 0.186 ± 0.010 | 0.83 |
| PARSEC split, lr 1e-2, refresh 100 | 0.192 ± 0.013 | 0.16 |
| full OT / full GGN | 0.276 ± 0.007 / 0.330 ± 0.035 | 9.0 / 9.3 |

**Next step:** per-tensor (block) curvature is the right scope here; adding the cross-tensor blocks does not help training. That supports the layerwise direction (per-tensor top-k with a SOAP bulk), which should be screened offline first.

## 2. Baseline suite, protocol hash and cost

- Baseline suite before any run (2026-10-08T09:16Z): `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
  → **106 passed**, 1 warning ([log](logs/baseline_suite.log)). Study tests: 6 passed.
- Protocol SHA-256: `abc5998f95241e838bff6c1556aa69d74ca734d815753f84c452c91c54b56ad3`.
- Cost dry run `20261008T091935-6d9ca4b0` (sequential, one thread): PARSEC 0.07 s per step plus a Lanczos
  refresh of 1.1/2.2/4.3 s at rank 8/16/32; OT tensor 3.67 s; full GGN 4.65 s; full OT 4.81 s per step.
  With the 1.6x slowdown observed under four workers, the projection was 11.9 CPU-h against an 18 CPU-h
  budget ([logs/cost_projection.json](logs/cost_projection.json)); no fallback. **Actual: 13.5 CPU-h** (gate 0.12, tuning 2.52,
  `ggn_full` evaluation 3.91, remaining evaluation 6.98; [logs/cost_actual.json](logs/cost_actual.json)), within budget. The machine did not sleep.

## 3. Reproduction gates (passed, bitwise)

Run `20261008T092039-cdeb20f3` ([gate.json](runs/20261008T092039-cdeb20f3/gate.json)). Differences at
steps 0/10/25/50 are exactly 0.0 for all three: PARSEC default branch (lr .003, refresh 10, rank 16) and
OT tensor (gamma .01) against the 6 October pilot, and this study's copied harness (`ggn_exact_trace`,
gamma .001, seed 1) against the gap-audit evaluation unit. A study test also shows the study's OT step at
gamma .01 is bitwise identical to the repository branch.

## 4. Tuning (seed 0 only; validation data used for selection)

Run `20261008T092438-2c2b66dd` ([selection](runs/20261008T092438-2c2b66dd/selection.json),
[results](runs/20261008T092438-2c2b66dd/tuning_results.json)). Batch 512, all tensors, validation loss on
the reserved 5,000 examples at the end of the horizon (500 steps; OT 200 steps). Stages were advanced
automatically by `scripts/orchestrate.sh`, which only sequences the decisions `train.py` makes under the
protocol's rules. These are tuning results, not rankings.

| Arm | Stage | Setting | Final loss |
| --- | --- | --- | ---: |
| parsec_default | A | lr 3e-4 / 1e-3 / **3e-3** / 1e-2 / 3e-2, refresh 10, rank 16 | .5273 / .2413 / **.1826** / .1893 / .2738 |
| parsec_default | A | same lrs, refresh 100 | .6247 / .2871 / .2114 / .2017 / .3296 |
| parsec_default | B | lr 3e-3, refresh 10, rank 8 / **32** | .2126 / **.1738** |
| parsec_split | A | lr_perp = lr_top = 3e-4 / 1e-3 / 3e-3 / **1e-2** / 3e-2, refresh 10, rank 16 | .8603 / .2959 / .2752 / **.2657** / diverged |
| parsec_split | B | lr_perp 1e-2, lr_top 1e-1 / 1 | diverged / diverged |
| parsec_split | C | lr_perp = lr_top = 1e-2, **refresh 100** | **.2334** |
| ot_tensor | A | gamma .001 / .01 / **.1** (200 steps) | .3577 / .2294 / **.2268** |
| ot_tensor | edge extension | gamma 1 | .3843 |

Selected: `parsec_default` lr 3e-3, refresh 10, rank 32; `parsec_split` lr_perp = lr_top = 1e-2,
refresh 100, rank 16; `ot_tensor` gamma .1 (also used by `ot_full`). `ggn_full` uses the gap audit's GGN
gamma .001 without tuning (matched ablation). Rank 32 is the largest rank tried; the protocol extends
only lr and gamma grids. For context (seed 0, gap-audit tuning): tuned Adam .1826, tuned native SOAP
.1713, tuned searched SOAP .2020 at 500 steps; GGN .2071 at 200 steps.

**Observations from tuning (seed 0).** Larger Newton steps in PARSEC's split branch diverge
(lr_top ≥ 0.1, and lr 3e-2); the stable split configurations take small top-space steps and refresh
curvature rarely. In the default branch, more frequent refreshes and a larger rank help.

## 5. Evaluation (fresh seeds 1–3, 500 steps)

Runs `20261008T092440-b7620df1` (`ggn_full`) and `20261008T103332-f9ee97b7` (the other four arms);
[tables with per-seed appendix](runs/20261008T103332-f9ee97b7/followup_tables.md),
[summary](runs/20261008T103332-f9ee97b7/followup_summary.json), figure
[followup_validation.png](figures/followup_validation.png). Baselines: gap-audit evaluation
`20261007T143302-baba2664` (same seeds, data order, harness and validation schedule). Workload: ViT width
16, 2 blocks, MNIST 28x28, batch 512, all 40 tensors, CPU float64; curvature arms use exact-trace damping
with the normalized inverse shadow-SOAP metric, the current gradient and one global ray search per step
on the training minibatch; PARSEC arms take native steps. All 15 units completed. Dense-solve residuals
≤ 2.0e-13; full-matrix and block action checks ≤ 1.2e-14; all 4,491 searches bracketed, none rejected.

| Arm | Setting | Loss 50 | Loss 100 | Loss 200 | Loss 500 | Acc. 500 | Measured s/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ggn_exact_trace (gap audit) | gamma .001 | **0.3550 ± 0.0114** | **0.2654 ± 0.0183** | 0.2355 ± 0.0077 | 0.1824 ± 0.0021 | 94.45 ± 0.27% | 6.47 |
| ggn_full | gamma .001 | 0.6698 ± 0.0830 | 0.5679 ± 0.1079 | 0.4785 ± 0.0378 | 0.3303 ± 0.0352 | 90.61 ± 0.73% | 9.31 |
| ot_tensor | gamma .1 | 0.4511 ± 0.0505 | 0.2965 ± 0.0265 | **0.2190 ± 0.0084** | 0.1654 ± 0.0056 | 95.10 ± 0.10% | 6.59 |
| ot_full | gamma .1 | 0.3858 ± 0.0073 | 0.3605 ± 0.0185 | 0.3237 ± 0.0180 | 0.2755 ± 0.0070 | 91.35 ± 0.09% | 9.02 |
| searched_soap_tuned (gap audit) | betas .9/.99, freq 10 | 0.6973 ± 0.0644 | 0.3676 ± 0.0453 | 0.2617 ± 0.0185 | 0.2098 ± 0.0074 | 93.76 ± 0.24% | 1.37 |
| native_soap (gap audit) | lr 1e-2 | 0.9041 ± 0.0980 | 0.4308 ± 0.0104 | 0.2361 ± 0.0294 | **0.1607 ± 0.0097** | **95.21 ± 0.47%** | 0.11 |
| adam_native (gap audit) | lr 1e-2 | 1.1126 ± 0.2119 | 0.6504 ± 0.0922 | 0.3119 ± 0.0396 | 0.2019 ± 0.0110 | 93.90 ± 0.21% | 0.09 |
| parsec_default | lr 3e-3, refresh 10, rank 32 | 1.3804 ± 0.3296 | 0.7454 ± 0.1992 | 0.3570 ± 0.0699 | 0.1856 ± 0.0100 | 94.66 ± 0.22% | 0.83 |
| parsec_split | lr_perp = lr_top = 1e-2, refresh 100, rank 16 | 1.0766 ± 0.2645 | 0.6110 ± 0.1252 | 0.3420 ± 0.0208 | 0.1916 ± 0.0130 | 94.29 ± 0.51% | 0.16 |

Seed-mean losses at steps 10/25: ggn_full 1.477/0.723 versus block GGN 1.129/0.548; ot_full 0.768/0.456
versus ot_tensor 1.293/0.769.

**Pre-declared readings** (better/worse = lower/higher validation loss in all three seeds):

| Comparison | Step 50 | Step 200 | Step 500 |
| --- | --- | --- | --- |
| parsec_default vs adam_native | worse | mixed | **better** |
| parsec_split vs adam_native | mixed | worse | **better** |
| parsec_default vs searched_soap_tuned | worse | worse | **better** |
| parsec_split vs searched_soap_tuned | worse | worse | **better** |
| parsec_default vs ggn_exact_trace | worse | worse | mixed |
| parsec_split vs ggn_exact_trace | worse | worse | mixed |
| ot_tensor vs ggn_exact_trace | worse | **better** | **better** |
| ggn_full vs ggn_exact_trace | **worse** | **worse** | **worse** |
| ot_full vs ot_tensor | **better** | worse | worse |

**Targets** (gap audit T50 = 0.355037, T100 = 0.265401; seed means; measured seconds with four
concurrent workers, sequential estimates from the dry runs):

| Arm | T50 steps | T50 measured s | T50 sequential s | T100 steps | T100 measured s | T100 sequential s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ggn_exact_trace | 58.3 | 349.9 | 225.6 | 141.7 | 918.5 | 553.3 |
| ggn_full | 441.7 | 4057.4 | 2050.2 | not reached | — | — |
| ot_tensor | 83.3 | 603.7 | 301.9 | 141.7 | 1043.3 | 515.8 |
| ot_full | 133.3 | 1206.9 | 636.2 | not reached | — | — |
| searched_soap_tuned | 108.3 | 156.1 | 94.4 | 200.0 | 285.1 | 174.9 |
| native_soap | 141.7 | **15.0** | **11.3** | 191.7 | **20.3** | **15.3** |
| adam_native | 175.0 | 16.2 | 12.0 | 275.0 | 24.9 | 18.9 |
| parsec_default | 200.0 | 167.3 | 99.3 | 275.0 | 230.1 | 136.7 |
| parsec_split | 191.7 | 27.6 | 16.8 | 300.0 | 45.0 | 26.4 |

**Diagnostics.**
- PARSEC default: 150 refreshes over the three seeds; relative Ritz residuals median 9.3e-4, maximum
  0.158; leading Ritz eigenvalue 12.6–847.
- PARSEC split (refresh 100, 15 refreshes): the top-space Newton step averages 0.67% of the complement
  step's norm. Up to 36% of the complement step lies in the top space, because Adam's coordinatewise
  normalization does not preserve orthogonality to V.
- Full matrices: off-diagonal tensor blocks carry 54–91% (GGN) and 74–92% (OT) of ‖C‖_F² at the audited
  steps.

**Observations and limitations.**
1. *PARSEC.* Tuned, both branches end better than tuned Adam and tuned searched SOAP, indistinguishable
   from exact tensor GGN at step 500 by the three-seed rule, and behind tuned native SOAP. Early they are
   the slowest arms. In wall-clock the split branch is cheap (0.16 s/step) but no faster to the targets
   than native SOAP or Adam. Its tuned form is close to "Adam on the gradient minus its top-space
   component", which suggests the separate Newton correction is not doing the work here.
2. *OT versus GGN.* Tensor OT starts slower but ends lower than tensor GGN in every seed. Its tuned
   damping (gamma .1) is 100x GGN's (.001), and GGN was tuned at a 200-step horizon (as was OT). The
   comparison is between two tuned methods, not between geometries at matched damping. At matched
   gamma .1 in seed-0 tuning, the two were close at step 200 (OT .2268, GGN .2206).
3. *Full versus block.* At matched damping and search, adding the cross-tensor blocks hurts GGN
   throughout and helps OT only for the first ~50 steps. The gamma values were tuned for the block
   versions; full matrices may need different damping. Large cross-tensor energy is again not useful
   coupling.
4. Three seeds, one workload, batch 512; tuning on seed 0 with validation data; measured time includes
   contention from four concurrent workers (the protocol's sequential estimates are given separately).

## 6. Deviations, issues in shared code

No departures from PROTOCOL.md ([DEVIATIONS.md](DEVIATIONS.md) is empty). No shared-code defect found.
Observation: the repository's `runs/` gained `20261008_ot_ggn_capacity_review` at 08:38Z, before this
study began; it was not created by this study, which writes only to its folder and
`~/skipsoap-study-data/geometry_followup_20261008/`.

## 7. Isolation audit and final suite

- Isolation manifests (`isolation_manifest_before.txt` at 09:14Z, `isolation_manifest_after.txt` at
  13:02Z, 1,157 files covering the main lab and `studies/gap_audit_20261007/`): **identical**.
  No new `__pycache__` outside `.venv`; no study run directory in the repository `runs/`.
- Final suite: **106 passed** ([log](logs/final_suite.log)); study tests: **6 passed**
  ([log](logs/final_study_tests.log)). Light mirrors 15.3 MB. Protocol hash unchanged.

## 8. Reproduction commands

From `residual-curvature-lab/`, with `export PYTHONDONTWRITEBYTECODE=1`:

```sh
P=studies/geometry_followup_20261008/scripts
.venv/bin/python -m pytest -q -p no:cacheprovider studies/geometry_followup_20261008/tests
.venv/bin/python $P/dryrun_cost.py
.venv/bin/python $P/train.py create --phase gate        # then work each unit (queue_runner.sh), finalize --run R
.venv/bin/python $P/train.py create --phase tune        # work units; sh $P/orchestrate.sh <tune-run> advances stages
.venv/bin/python $P/train.py create --phase eval --arms ggn_full
.venv/bin/python $P/train.py create --phase eval --arms parsec_default,parsec_split,ot_tensor,ot_full --tuning <tune-run>
.venv/bin/python $P/summarize_followup.py --eval <eval-run> <ggn-full-run> --tune <tune-run> --dryrun <dryrun>
```
