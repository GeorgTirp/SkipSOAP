# Gap audit study — results

Study `gap_audit_20261007`, protocol [AGENT_INSTRUCTIONS.md](AGENT_INSTRUCTIONS.md).
Large artifacts: `~/skipsoap-study-data/gap_audit_20261007/runs/`. Light mirrors: [runs/](runs/).

**Status (2026-10-07, 16:50Z, final):** checks 1, 2a/2b, 2c, 3 and 4a complete; 4b **stopped** at 21 of 22 units (DEVIATIONS #11); 4c skipped (gate failed).

## 1. Summary

- **Check 1, C (mixed):** on fresh seeds 1–3 with tuned baselines, GGN's ~2x step-50 advantage over tuned searched SOAP replicates (ratios 1.79–2.05) but shrinks to 1.09–1.20 by step 500; tuned native SOAP ends lowest, and GGN never wins on wall-clock.
- **Check 2a, no; 2b:** dense empirical-Fisher training does not close the gap (closure −2% at step 50); SOAP's map without momentum trains far worse than searched SOAP.
- **Check 2c, insufficient:** the best cheap oracle family (diagonal in a Kronecker eigenbasis) recovers 46.0% of local exact-GGN headroom; Kronecker sums 15–19%; 5-matvec full-GGN PCG 67%.
- **Check 3, "OT is easier to compress"** by the pre-declared reading, with thin margins (+10.4/+10.5 pp, ~.001 absolute) partly due to OT's smaller headroom.
- **Check 4:** exact-GGN headroom exists in all nine ViTs; Kronecker diagonal 36–75% (main ViT 36%); 4b stopped at 21/22 units; 4c skipped (gate failed).

| Seeds 1–3, batch 512 (tuned on seed 0) | Loss 50 | Loss 500 | Measured seconds to T50 |
| --- | ---: | ---: | ---: |
| Exact tensor GGN + search (gamma .001) | 0.355 ± 0.011 | 0.182 ± 0.002 | 350 |
| SOAP + search, tuned | 0.697 ± 0.064 | 0.210 ± 0.007 | 156 |
| Native SOAP, lr 1e-2 | 0.904 ± 0.098 | 0.161 ± 0.010 | 15 |
| Dense empirical Fisher + search | 0.705 ± 0.170 | 0.306 ± 0.093 | reached in 2/3 seeds |

**Next step:** the GGN advantage is mainly early and per step. No tested cheap surrogate holds it, so learning a predictor is not justified yet. The only local candidate near the bar is matrix-free full-GGN PCG with ~5 matvecs, which should be tested on training trajectories against tuned native SOAP.

## 2. Baseline suite, protocol hash and cost ledger

### Baseline suite

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`, run from
`residual-curvature-lab/` at 2026-10-07T12:02:27Z, before any experiment:
**106 passed**, 1 warning (`torch.jit.script` deprecation), 4.50 s reported by pytest
(5.96 s wall). Log: [logs/baseline_suite.log](logs/baseline_suite.log). `HANDOFF.md` mentions 103
tests; the three additional tests belong to the OT/GGN capacity runner added on 7 October
(`tests/test_combined_capacity.py`). The default collection contains no study tests
(106 collected, 0 under `studies/`).

### Protocol hash

SHA-256 of `AGENT_INSTRUCTIONS.md`:
`469b6eed9857f15c6d6933e132086b9a73dd6e688ebc3a3fd389eb529e3c5838`.

### Cost ledger

Dry run `20261007T121137-28c507f9` ([mirror](runs/20261007T121137-28c507f9/dryrun.json)):
three consumed steps of every training arm (seed 0, batch 512; the first consumed step of
every SOAP-based arm initializes statistics, so per-step costs average steps 2–3), three
searched-SOAP steps plus one exact-block construction for each of the nine check-4
architectures, and one fixed-state candidate evaluation (SOAP current-gradient map at saved
SOAP state 10: search plus four panels). That candidate gave independent decrease
0.0331906, matching the capacity screen's 0.033191. Measurements were sequential, one
thread, on an Apple M3 (4 performance + 4 efficiency cores).

| Arm (main ViT) | Seconds per step |
| --- | ---: |
| native_soap | 0.080 |
| adam_native | 0.069 |
| searched_soap | 0.906 |
| searched_soap_tuned (betas .9/.99, frequency 1) | 0.899 |
| ggn_exact_trace (gamma .01) | 3.933 |
| dense_ef (gamma .01) | 1.210 |
| soap_gradient_map | 0.924 |

Fixed-state candidate evaluation: 0.97 s; GGN action 0.064 s; OT block construction 2.99 s;
validation pass 0.3–0.5 s. Projections ([logs/cost_projection.json](logs/cost_projection.json),
[scripts/project_cost.py](scripts/project_cost.py)) count every protocol run, including all
possible tuning edge extensions:

| Check | Projected CPU-h | Budget | Fallback used | Actual CPU-h |
| --- | ---: | ---: | --- | ---: |
| 1 (with 2a/2b), 500 evaluation steps | 5.14 | 12 | no (300 steps would be 3.80) | 8.50 |
| 2c | 0.18 | 1 | no | 0.16 |
| 3 | 0.22 | 1 | no | 0.17 |
| 4 (4a+4b 1.14, 4c capped at 3 + 0.5 evaluation) | 4.64 | 6 | no | 1.19 (4a+4b; 4c not run) |

Actual CPU-hours ([logs/cost_actual.json](logs/cost_actual.json), [scripts/ledger.py](scripts/ledger.py)) sum
single-threaded process wall time with the sleep window excluded, and include the interrupted
tuning attempts. Check 1 exceeded its projection because steps ran about 1.6x slower with four
concurrent workers than in the sequential dry run (GGN 6.47 versus 3.93 s per step); it stayed
within its 12 CPU-h budget. The dry run itself took 0.02 CPU-h.

## 3. Checks

### Check 1: is the gap real?

**Step 1, reproduction gate (passed).** Run `20261007T121557-dfef403c`
([gate.json](runs/20261007T121557-dfef403c/gate.json)): seed 0, 50 consumed steps, default
settings, batch 512, the repository proposal functions (`run_mini_vit_allocation.propose`,
`run_transport_parsec_noise.propose`). Every arm reproduces the stored validation loss
exactly, at every stored step (0, 10, 25, 50):

| Arm | Stored step-50 loss | Reproduced | Absolute difference |
| --- | ---: | ---: | ---: |
| native_soap (lr .003) | 1.3543558422225512 | 1.3543558422225512 | 0 |
| searched_soap | 0.7316230218634457 | 0.7316230218634457 | 0 |
| adam_native (lr .003) | 1.2880076863671444 | 1.2880076863671444 | 0 |
| ggn_exact_trace (gamma .01) | 0.3276759798859086 | 0.3276759798859086 | 0 |

The gate run's archived `check1_train.py` differs from the version used for later check-1
runs only in `create()`, which later also archives `scripts/run_units.sh`. The evaluation run's
archived version additionally records per-step process CPU time and queues the most expensive
arms first; proposal and gating code is identical across all check-1 runs.

**Step 2, tuning (seed 0 only; validation data used for selection).** Run
`20261007T122018-998da6a3` ([selection](runs/20261007T122018-998da6a3/selection.json),
[all results](runs/20261007T122018-998da6a3/tuning_results.json)). Batch 512, all tensors,
search as in step 1; GGN with exact-trace damping and the shadow-SOAP metric. Selection by
validation loss at the end of the tuning horizon (500 steps; GGN 200 steps). No configuration
diverged. These are tuning results, not evidence for rankings.

| Arm | Setting | Step-50 loss | Final loss (horizon) | Final accuracy |
| --- | --- | ---: | ---: | ---: |
| native_soap | lr 3e-4 / 1e-3 / 3e-3 / **1e-2** / 3e-2 | 2.248 / 1.850 / 1.354 / 1.033 / 1.125 | .9555 / .2550 / .1825 / **.1713** / .2304 (500) | 80.4 / 92.7 / 94.6 / **94.6** / 92.6% |
| adam_native | lr 3e-4 / 1e-3 / 3e-3 / **1e-2** / 3e-2 | 2.053 / 1.671 / 1.288 / 1.227 / 1.993 | .7480 / .3840 / .2516 / **.1826** / .3297 (500) | 80.1 / 88.2 / 92.7 / **94.7** / 90.3% |
| searched_soap_tuned | betas .95/.95, freq 1 / 10 | .6025 / .7316 | .2545 / .2212 (500) | 92.1 / 93.3% |
| searched_soap_tuned | betas .9/.99, freq 1 / **10** | .5908 / .6674 | .2262 / **.2020** (500) | 93.5 / **93.8%** |
| searched_soap_tuned (extension) | betas .9/.99, freq 100 | — | .2069 (500) | — |
| ggn_exact_trace | gamma **.001** / .01 / .1 | .3609 / .3277 / .4366 | **.2071** / .2103 / .2206 (200) | **93.9** / 93.9 / 93.5% |
| ggn_exact_trace (extension) | gamma .0001 | — | .3044 (200) | — |

Selected: native SOAP lr 1e-2, Adam lr 1e-2, searched SOAP betas (.9,.99) with
precondition frequency 10, GGN gamma .001 (also used by `dense_ef`). Edge extensions (applied
once, DEVIATIONS #3): GGN gamma .001 was on the lower edge (gamma 1e-4 added) and frequency 10 on
the upper edge (frequency 100 added); neither extension improved the selection. The lr optima are
interior. The interrupted first attempts of the three GGN tuning units (DEVIATIONS #8) matched
the reruns exactly at every recorded step.

*Precondition frequency.* On seed 0, frequency 1 (fresh bases every step) was clearly better
early (step 50: .6025 versus .7316 at betas .95/.95; .5908 versus .6674 at .9/.99) but worse at
step 500 in both beta settings (.2545 versus .2212; .2262 versus .2020). Basis refresh
frequency therefore changes searched SOAP materially, in opposite directions early and late.
This is one seed and one horizon; it is not a separately replicated finding.

**Step 3, evaluation (fresh seeds 1, 2, 3; 500 consumed steps; no fallback needed).** Run
`20261007T143302-baba2664` ([tables incl. per-seed appendix](runs/20261007T143302-baba2664/check1_tables.md),
[summary](runs/20261007T143302-baba2664/check1_summary.json), figures
[check1_validation.png](figures/check1_validation.png), [check1_ratio.png](figures/check1_ratio.png)).
Workload `configs/mini_vit_allocation.yaml`: ViT width 16, expansion 32, 2 blocks, 2 heads, 28x28
MNIST, batch 512, CPU float64. Model init seed s, data order seed 102 + 1000 s, split seed 101.
Validation on the reserved 5,000 examples at steps 0, 10, 25, 50, …, 500. Scope: all 40 tensors.
Curvature arms: exact tensor blocks, gamma .001 with exact current trace (GGN trace for
`ggn_exact_trace`, empirical-Fisher trace for `dense_ef`), normalized inverse shadow-SOAP metric,
current-gradient RHS. Searched arms: one global ray search per step on the training minibatch
(zero plus native-radius multipliers 2^[-16..16], 18 golden refinements); every consumed step
advances SOAP statistics once. All 21 units completed (no divergence). Audits: solve residuals
≤ 8.1e-14, per-tensor action checks (steps 2/50/200/500) ≤ 6.1e-14, `dense_ef` per-example
gradient mean versus the GGN gradient ≤ 4.5e-13. All 7,485 searches were bracketed; 3 selected
zero (searched_soap). Checkpoints at steps 50/200/500 are stored externally.

Validation loss and accuracy, mean ± sd over seeds 1–3 (n = 3):

| Arm | Setting | Loss 50 | Loss 100 | Loss 200 | Loss 500 | Acc. 500 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ggn_exact_trace | gamma .001 | **0.3550 ± 0.0114** | **0.2654 ± 0.0183** | **0.2355 ± 0.0077** | 0.1824 ± 0.0021 | 94.45 ± 0.27% |
| searched_soap_tuned | betas .9/.99, freq 10 | 0.6973 ± 0.0644 | 0.3676 ± 0.0453 | 0.2617 ± 0.0185 | 0.2098 ± 0.0074 | 93.76 ± 0.24% |
| searched_soap | default | 0.7616 ± 0.0821 | 0.3983 ± 0.0433 | 0.2657 ± 0.0335 | 0.2377 ± 0.0197 | 92.82 ± 0.73% |
| native_soap | lr 1e-2 | 0.9041 ± 0.0980 | 0.4308 ± 0.0104 | 0.2361 ± 0.0294 | **0.1607 ± 0.0097** | **95.21 ± 0.47%** |
| adam_native | lr 1e-2 | 1.1126 ± 0.2119 | 0.6504 ± 0.0922 | 0.3119 ± 0.0396 | 0.2019 ± 0.0110 | 93.90 ± 0.21% |
| dense_ef | gamma .001 | 0.7051 ± 0.1695 | 0.4993 ± 0.0345 | 0.4112 ± 0.0452 | 0.3056 ± 0.0927 | 92.15 ± 2.00% |
| soap_gradient_map | default | 1.4742 ± 0.0913 | 1.1874 ± 0.0792 | 0.8940 ± 0.0979 | 0.4948 ± 0.1072 | 87.87 ± 2.34% |

Accuracies at steps 50/100/200 and every per-seed value are in the linked tables. Measured
seconds per step under four concurrent workers: GGN 6.47, dense_ef 1.86, searched arms 1.32–1.37,
native SOAP 0.108, Adam 0.088 (sequential dry run: 3.93, 1.21, 0.90–0.92, 0.080, 0.069).

**Verdict: C, mixed.** The ratio `searched_soap_tuned / ggn_exact_trace` is 2.05, 2.04 and 1.79
at step 50 (≥ 1.5 in all seeds) and 1.16, 1.20 and 1.09 at step 500. That is not ≥ 1.25 in all
seeds (A fails) and ≤ 1.10 in only one seed (B fails). Description: the gap peaks around step 25
(ratio ≈ 2.2), falls to about 1.1 by step 125, then stays between roughly 1.05 and 1.2 through
step 500, favouring GGN at step 500 in all three seeds. The large 50-step gap of the earlier pilot
replicates on fresh seeds with tuned baselines; most of it is early-phase. A smaller advantage
over tuned searched SOAP persists at 500 steps. It does not persist against every tuned
baseline: native SOAP with lr 1e-2, tuned at the 500-step horizon, has lower step-500 loss
than GGN in all three seeds (.1717/.1571/.1533 versus .1826/.1802/.1844). GGN was tuned at a
200-step horizon (protocol), which may disadvantage it at 500 steps.

**Targets.** T50 = 0.355037 and T100 = 0.265401 (seed-mean GGN losses at steps 50 and 100).

| Arm | T50 reached | Steps | Measured s | Sequential-estimate s | T100 reached | Steps | Measured s | Sequential-estimate s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ggn_exact_trace | 3/3 | 58.3 | 349.9 | 225.6 | 3/3 | 141.7 | 918.5 | 553.3 |
| searched_soap_tuned | 3/3 | 108.3 | 156.1 | 94.4 | 3/3 | 200.0 | 285.1 | 174.9 |
| searched_soap | 3/3 | 116.7 | 146.4 | 104.8 | 3/3 | 291.7 | 386.4 | 263.3 |
| native_soap | 3/3 | 141.7 | **15.0** | **11.3** | 3/3 | 191.7 | **20.3** | **15.3** |
| adam_native | 3/3 | 175.0 | 16.2 | 12.0 | 3/3 | 275.0 | 24.9 | 18.9 |
| dense_ef | 2/3 | — | — | — | 1/3 | — | — | — |
| soap_gradient_map | 0/3 | — | — | — | 0/3 | — | — | — |

Steps are seed means of the first validation step at or below the target (resolution 25 steps).
"Measured s" is cumulative training time excluding validation, I/O and audits, measured with
four concurrent workers (process CPU seconds agree within 2%). "Sequential estimate" is
steps-to-target times the sequential dry-run cost per step (DEVIATIONS #6).

**Wall-clock view: no.** GGN needs the fewest steps to both targets, but it does not reach T50 or
T100 in fewer seconds than the best SOAP or Adam arm, measured or sequential: native SOAP reaches
T50 in 15 s and T100 in 20 s, against GGN's 350 s and 919 s.

**Precondition frequency.** Only frequency 10 was evaluated on fresh seeds (selected by tuning).
The seed-0 tuning comparison above is the only evidence that frequency 1 changes searched SOAP:
better early, worse by step 500 in both beta settings. EMA/basis staleness therefore matters, but
its sign depends on the horizon.

#### 2a and 2b (inside the evaluation)

**2a, `dense_ef`: *no*.** Closure (L_tuned − L_dense_ef)/(L_tuned − L_ggn) with seed means:
−2.3% at step 50 (0.7051 versus 0.6973 tuned searched SOAP and 0.3550 GGN) and −571% at step 200
(0.4112 versus 0.2617 and 0.2355; the step-200 gap is small, which magnifies the ratio). Dense
per-example empirical-Fisher blocks, with the same damping rule, metric, solver, current gradient
and search as GGN, train *worse* than tuned searched SOAP and far worse than exact GGN. They also
vary strongly across seeds (step-500 loss 0.41/0.28/0.23). The earlier fixed-state finding that
dense F recovers most of the early local GGN gain at SOAP states does not carry over to training
trajectories. Gamma was not tuned for `dense_ef` (protocol); F's trace and spectrum differ from C's,
so the same gamma means different relative damping.

**2b, `soap_gradient_map`.** The current gradient through SOAP's frozen preconditioner, without
momentum and with the same search, trains far worse than searched SOAP: 1.4742 versus 0.7616 at
step 50 and 0.4948 versus 0.2377 at step 500. Over trajectories, SOAP's momentum is essential to
its performance. The geometry alone does not explain the gap between searched SOAP and GGN, and
neither does momentum alone: GGN uses the current gradient without momentum and leads early.

**Limitations.** Three seeds, one workload and one batch size; tuning on seed 0 with validation
data, different tuning horizons per arm (500 versus 200 for GGN), and a coarse lr grid. Measured
time is confounded by concurrent load (DEVIATIONS #5, #6, #9). The searched arms select step
sizes on the training minibatch, which is not a held-out search.

### Check 2: where is the missing information?

2a (`dense_ef`) and 2b (`soap_gradient_map`) run inside check 1's evaluation; see check 1.

#### 2c. Offline capacity of structured families (complete)

Run `20261007T123608-7f39721a` ([tables](runs/20261007T123608-7f39721a/check2c_tables.md),
[summary](runs/20261007T123608-7f39721a/summary.json), figure
[check2c_recovery.png](figures/check2c_recovery.png)). No training.

**Settings as used.** Six saved seed-0 states of
`runs/20261005T143029-9b582ed5` (searched-SOAP and exact-GGN trajectories at steps 10/25/50),
artifact checksums verified. Batch: the 512-example construction batch of each state. Scope: all
40 tensors (5,738 parameters), tensor partition. Damping: gamma .01 with exact GGN trace,
mu = max(.01 sum_b trace(C_bb)/D, 1e-8), Omega = normalized inverse frozen SOAP metric, identical
for every family at a state. Search: `search_candidate` (zero plus native-radius multipliers
2^[-16..16], 18 golden refinements) on construction loss; score = mean independent decrease on the
four analysis panels used by the capacity screen (development data, reused). Every family is a
per-tensor PSD matrix S_b solved densely as (S_b + mu Omega_b)^-1 rhs; PCG families use x0 = 0, the
SOAP map as preconditioner and exactly k operator actions. Every family is scored on the current
gradient and on the effective SOAP momentum (normalized to the gradient norm), so each RHS has
same-RHS references. 1-D tensors keep their exact block in every Kronecker family; for `ks_2_ef`
that block is the empirical-Fisher block F_bb (the family is computed from per-example gradients
only), still damped with the GGN mu.

**Sanity reproduction (passed).** SOAP current-gradient map and exact tensor GGN reproduce the
capacity screen's damping-table `identity`/`exact` rows, and `soap_basis_diag` reproduces "GGN
diagonal in SOAP basis + gradient", at all six states; maximum absolute difference 5.2e-12 (≤ 1e-9).
Maximum cached-GGN action error 1.1e-14, empirical-Fisher action error 5.2e-15 (independent
JVP/VJP route), all damped solve residuals ≤ 1e-8, all 204 searches bracketed or rejected.

| Family | Recovery, gradient | Mean decrease, gradient | Ray error g / m / factor | Storage (model vectors; x SOAP factors) | Matvecs per RHS | Tier |
| --- | ---: | ---: | --- | --- | ---: | --- |
| SOAP map (identity) | 0.0% | 0.019314 | .901 / .970 / .924 | 0 | 0 | reference |
| exact tensor GGN | 100.0% | 0.045883 | 0 / 0 / 0 | 399.9; 176 | 0 | reference |
| soap_basis_diag | 21.0% | 0.024883 | .768 / .907 / .708 | 1.00; 0.44 | 0 | reference |
| **kron_basis_diag** | **46.0%** | 0.031536 | .740 / .569 / .219 | 4.51; 1.99 | 0 | **insufficient** |
| ks_1 | 15.5% | 0.023421 | .939 / .916 / .475 | 3.58; 1.58 | 0 | insufficient |
| ks_2 | 15.9% | 0.023528 | .942 / .944 / .650 | 5.81; 2.56 | 0 | insufficient |
| ks_2_ef | 25.6% | 0.026123 | .941 / .938 / .696 | 5.81; 2.56 | 0 | insufficient |
| ks_3 | 18.5% | 0.024232 | .941 / .942 / .646 | 8.03; 3.54 | 0 | reference |
| pcg_block_2 / 3 / 5 / 10 | 9.9 / 16.1 / 29.7 / 50.4% | 0.021944 / 0.023595 / 0.027201 / 0.032717 | (10: .355 / .676 / .506) | 0 (4 work vectors) | 2 / 3 / 5 / 10 | reference |
| pcg_full_2 / 3 / 5 / 10 | 5.4 / −14.0 / 67.1 / 132.7% | 0.020746 / 0.015598 / 0.037141 / 0.054582 | (5: .817 / .925 / .909) | 0 (4 work vectors) | 2 / 3 / 5 / 10 | reference |
| exact_1d_only | 2.8% | 0.020059 | .924 / .972 / .919 | 1.31; 0.58 | 0 | reference |

Recovery is (D_family − D_SOAP-map)/(D_exact − D_SOAP-map) of six-state mean independent
decreases, no state dropped (aggregate exact headroom .026569). Ray errors are pooled block-teacher
A-energy errors after one global nonnegative scalar per RHS (`optim.capacity.energy_errors`) on the
gradient, the momentum and 16 factor-coordinate Gaussian RHS F0^-T z. Storage counts the oracle
parameters a family would keep (Kronecker factors/eigenbases plus diagonal, exact 1-D blocks); SOAP
factor storage is counted as its two eigenbases per 2-D tensor (sum out^2 + in^2 = 13,031 numbers).
PCG matvecs: block = tensor-block GGN action; full = one JVP plus one VJP. Matvec counts exclude
fitting: every non-PCG family here needs the exact blocks as an oracle.

Per-state current-gradient independent decreases (positive is better):

| Family | SOAP 10 | SOAP 25 | SOAP 50 | GGN 10 | GGN 25 | GGN 50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SOAP map | 0.033191 | 0.025898 | 0.002988 | 0.015570 | 0.017390 | 0.020846 |
| exact | 0.135327 | 0.085051 | −0.012308 | 0.039168 | 0.010830 | 0.017227 |
| kron_basis_diag | 0.069311 | 0.059277 | −0.006971 | 0.034918 | 0.014938 | 0.017741 |
| ks_2 | 0.046455 | 0.044541 | −0.004651 | 0.021154 | 0.019106 | 0.014561 |
| ks_2_ef | 0.051885 | 0.044166 | 0.001382 | 0.024192 | 0.019004 | 0.016111 |
| pcg_block_10 | 0.103624 | 0.049679 | −0.015622 | 0.030902 | 0.012345 | 0.015371 |
| pcg_full_5 | 0.165217 | 0.056960 | −0.024180 | 0.036299 | −0.001222 | −0.010230 |
| pcg_full_10 | 0.150223 | 0.108300 | −0.019280 | 0.091359 | 0.008827 | −0.011937 |

Local headroom is positive only at SOAP 10, SOAP 25 and GGN 10. At SOAP 50, GGN 25 and GGN 50,
exact tensor GGN does worse than the SOAP map, so per-state recovery ratios there change sign or
exceed 100% and are not interpretable; the aggregate keeps those states as required. At the three
positive-headroom states, `kron_basis_diag` recovers 35%, 56% and 82%.

**Observations.** (1) No cheap family reaches the 60% *promising* tier; the best,
`kron_basis_diag`, recovers 46.0% and is *insufficient*. It nonetheless beats every Kronecker sum,
including `ks_3`: the exact diagonal in a Kronecker eigenbasis keeps more useful geometry than
fitting the Kronecker sum itself in Frobenius norm. (2) The Kronecker sums barely improve on rank 1
(15.5%, 15.9%, 18.5%). With symmetrized factors, the rank-2 and rank-3 sums were indefinite in
most matrix-tensor fits (89–92 of 96) and needed clipping of up to 14% of their eigenvalue mass;
median relative Frobenius block errors were .42 / .32 / .27 for ranks 1/2/3. (3) The
empirical-Fisher fit `ks_2_ef` recovers more than the GGN fit `ks_2` (25.6% versus 15.9%). This
does not show that F is a better teacher; both are far from the bar. (4) Matrix-free full-GGN PCG
passes 60% only at 5 iterations (67.1%) and is not monotone in k (pcg_full_3: −14.0%). At 10
iterations it exceeds the block-diagonal exact teacher (132.7%), because it includes
cross-tensor curvature. Its benefit is concentrated at SOAP 10/25 and GGN 10; at the late
states it overshoots, like exact GGN. (5) 1-D tensors explain little: `exact_1d_only` recovers
2.8%.

**Momentum RHS.** The aggregate exact headroom on momentum is only .003260 (exact GGN .015916
versus SOAP map .012656), so momentum recovery ratios (e.g. 364% for `kron_basis_diag`) are
dominated by a small denominator and are not meaningful as recovery. In absolute terms several
structured families beat exact GGN on momentum (mean decreases: kron_basis_diag .024531,
pcg_block_10 .052385, pcg_full_5 .042814), which repeats the earlier observation that the RHS
interacts strongly with the operator.

**Limitations.** Six related states from one seed; panels reused development data; oracle
parameters from exact blocks; the Kronecker basis is the nearest-Kronecker-product eigenbasis, not
an optimized basis. PCG costs are matvec counts, not timed optimizer steps.

### Check 3: OT versus GGN compressibility

**Verdict (pre-declared reading): *OT is easier to compress*, by a thin margin and with the
caveats below.**

The main lab had not run `docs/ot_ggn_capacity_protocol_20261007.md` (no run under `runs/` names
it in `protocol.json`). The study ran `scripts/check3_ot_ggn_capacity.py`, which replicates
`scripts/run_ot_ggn_capacity.py::main()` with the three section-7 exceptions (output under
`$GAP_AUDIT_DATA/runs`; attachment copied, it existed, SHA-256 `994544d0…`; original runner archived).
Run `20261007T124647-1d8487a2` ([tables](runs/20261007T124647-1d8487a2/check3_tables.md),
[summary](runs/20261007T124647-1d8487a2/check3_summary.json), figure
[check3_recovery.png](figures/check3_recovery.png)). Counts as expected: 420 candidate rows,
204 operator states, 12 state/targets. Families, gates, tolerances and panels unchanged.

**Settings as used.** Six saved seed-0 states (as in 2c), batch 512 construction, all 40 tensors,
damping calibration beta = 1/(mu SOAPMetric.scale), gamma .01 with each target's own exact trace
(so mu differs between OT and GGN), the same normalized inverse frozen SOAP Omega, the same
52-evaluation global construction-loss search and four reused analysis panels. Recovery is
own-teacher: (D_family − D_SOAP-map)/(D_exact − D_SOAP-map) with the same target and same RHS,
six-state means.

**Audits (all passed at 1e-8).** Maximum factor error 6.3e-16, teacher action error 1.1e-14 (OT blocks
against independent probability JVP/VJP actions), cached-teacher energy error 2.1e-11, teacher
gradient residual 1.5e-14, backward error 9.7e-17, dense application error 4.4e-15; all 420
searches bracketed or rejected.

| Family | GGN recovery, g | OT recovery, g | Δ (pp) | GGN mean decrease, g | OT mean decrease, g | Both conditions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| diagonal | 21.0% | 23.4% | +2.4 | 0.024883 | 0.024801 | no |
| state_4 | 37.3% | 47.7% | +10.4 | 0.029232 | 0.030520 | yes |
| combined_state_3 | 46.4% | 57.0% | +10.5 | 0.031652 | 0.032685 | yes |

Two of the three named families meet both conditions (≥ 10 pp higher own-teacher current-gradient
recovery under OT, and an OT absolute mean independent decrease ≥ GGN's), so the pre-declared
reading is met. The GGN rows reproduce the earlier capacity screen (e.g. state_4 37.3%, teacher_16
56.1%, diagonal 21.0%). No family reaches the 80% development shortlist under either teacher
(best deployable: combined_state_3 57.0% under OT, 4.06 model vectors; teacher_16 57.2% under OT is
a teacher-selected reference).

Exact teacher independent decreases (current gradient g, momentum m):

| State | Exact GGN, g | Exact OT, g | Exact GGN, m | Exact OT, m |
| --- | ---: | ---: | ---: | ---: |
| SOAP 10 | 0.135327 | 0.141515 | 0.035343 | 0.035679 |
| SOAP 25 | 0.085051 | 0.065014 | 0.033845 | 0.030824 |
| SOAP 50 | −0.012308 | −0.016958 | 0.045150 | 0.032143 |
| GGN 10 | 0.039168 | 0.042338 | 0.037787 | 0.039784 |
| GGN 25 | 0.010830 | 0.008463 | −0.020162 | −0.019447 |
| GGN 50 | 0.017227 | 0.016327 | −0.036464 | −0.044636 |
| Mean | 0.045883 | 0.042783 | 0.015916 | 0.012390 |

**Caveats.** (1) Both qualifying margins (10.4 and 10.5 pp) barely clear the 10 pp threshold. (2)
Part of OT's higher recovery comes from a smaller denominator: the exact OT teacher has less
aggregate headroom over the SOAP map (.023469 versus .026569 for GGN). The absolute gains of
the compressed families under OT are about .001 in mean independent decrease. (3) By inverse
accuracy, OT is not easier: pooled gradient ray errors are slightly *higher* under OT for all three
named families (state_4 .616 versus .586; combined_state_3 .601 versus .568; diagonal .791 versus
.767). The advantage therefore appears in the nonlinear independent score, not in how closely the
compressed operator matches its own teacher. (4) On the momentum RHS the OT exact teacher does not
beat the SOAP map in aggregate (reference advantage −.000265), so OT momentum recovery is undefined
by the protocol's rule. (5) Six related states from one seed and reused panels; no OT-trajectory states;
no training.

### Check 4: architecture family and neuron-graph predictor

#### 4a. Family dataset (complete)

Run `20261007T125950-63edd820` ([units](runs/20261007T125950-63edd820/units_summary.json)). Nine
ViTs: blocks {1,2,3} x width {8,16,24}, heads 2, expansion 2 x width, patch 7, MNIST 28x28, batch 512;
1,250 to 16,546 parameters. 22 (architecture, seed) trajectories of 50 consumed searched-SOAP steps
(default settings, section-3 seed rule), snapshots after steps 10/25/50: 66 states, 1.9 GB
external. Each state stores parameters, the construction-batch gradient (first 512 analysis IDs),
the effective momentum RHS (ray check, maximum 4.1e-10), SOAP contexts and optimizer state, the raw SOAP
proposal, exact streamed tensor GGN blocks (microbatch 8, 768 MiB budget; per-tensor action checks
against `GGN.action`, maximum 2.2e-14) and the four analysis panel ID sets. **Reproduction gate passed:** the main
ViT (width 16, 2 blocks, seed 0) matches `runs/20261005T143029-9b582ed5/searched_soap-*` parameters
exactly (maximum absolute difference 0.0 at steps 10/25/50).

#### 4b. Headroom and family capacity across architectures (**stopped**: 21 of 22 units)

Run `20261007T143410-b579ef62` ([partial summary](runs/20261007T143410-b579ef62/summary_partial.json),
[tables](runs/20261007T143410-b579ef62/check4_tables.md), figure
[check4_headroom.png](figures/check4_headroom.png)). The run stopped when unit `b1_w8__s0` failed the
study's own 1e-12 agreement assertion between two Cholesky solves of the same damped exact-GGN system
(DEVIATIONS #11). Its three states are missing; every other unit passed all checks, and all 756
searches of the completed units were bracketed (none rejected). The results below are **partial**.
Architecture `b1_w8` is represented by seeds 1 and 2 only, and the train split by 21 of its 24 states.

Settings as in 2c: per-state mu = max(.01 trace(C)/D, 1e-8), normalized inverse frozen SOAP metric,
`search_candidate` with the native radius, four analysis panels, current gradient and effective
momentum. All states are from searched-SOAP trajectories. Recovery is the ratio of mean decreases
over the group's states, (D_family − D_SOAP-map)/(D_exact − D_SOAP-map).

| Architecture | Split | Params | Headroom, g | soap_basis_diag | kron_basis_diag | pcg_full_2 | pcg_full_5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| b1_w8 (seeds 1, 2) | train/validation | 1,250 | 0.058776 | 9.7% | 61.3% | 16.3% | 80.5% |
| b1_w16 | train/validation | 3,514 | 0.067484 | 8.5% | 61.9% | 20.9% | 89.1% |
| b2_w8 | train/validation | 1,850 | 0.035353 | 10.3% | 51.1% | 31.5% | 113.9% |
| b2_w16 (main ViT) | train/validation | 5,738 | 0.081347 | 10.7% | 35.6% | 35.7% | 74.7% |
| b1_w24 | held-out width | 6,802 | 0.055257 | 12.2% | 75.2% | 5.3% | 99.5% |
| b2_w24 | held-out width | 11,674 | 0.078524 | 13.0% | 55.4% | 20.2% | 101.1% |
| b3_w8 | held-out depth | 2,450 | 0.055623 | 11.0% | 47.2% | 27.9% | 102.6% |
| b3_w16 | held-out depth | 7,962 | 0.061216 | 12.8% | 53.2% | 4.7% | 68.2% |
| b3_w24 | held-out both | 16,546 | 0.066385 | 12.2% | 39.8% | 17.8% | 89.9% |

| Split | States | Headroom, g | soap_basis_diag | kron_basis_diag | pcg_full_2 | pcg_full_5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train (partial) | 21 | 0.065540 | 9.1% | 48.7% | 21.2% | 78.2% |
| validation | 12 | 0.052831 | 11.3% | 54.4% | 40.0% | 103.8% |
| held-out width | 12 | 0.066890 | 12.6% | 63.6% | 14.0% | 100.4% |
| held-out depth | 12 | 0.058420 | 11.9% | 50.3% | 15.7% | 84.6% |
| held-out both | 6 | 0.066385 | 12.2% | 39.8% | 17.8% | 89.9% |

**Observations (partial).** (1) Exact tensor GGN has positive current-gradient headroom over the
SOAP map in every architecture (aggregate 0.035–0.081 in mean independent decrease). It is largest
for the main ViT (0.081 over three seeds). The headroom is not specific to the main ViT. (2)
`kron_basis_diag` recovers 36–75% depending on architecture. It does best for one-block models and
worst for the main ViT (35.6%), so its capacity does *not* look like the main ViT's across the family:
it varies with depth and width without a monotone trend. The train-split value, 48.7% on 21 of 24
states, is below the 50% 4c gate bar as well. (3) Five-step full-GGN PCG recovers 68–114%
(train 78.2%, held-out 85–100%), while two steps recover 5–40%. (4) The SOAP-basis diagonal stays
at 9–13% everywhere, consistent with the main-ViT finding that this compression discards most of
the useful geometry. (5) Momentum headroom is near zero or negative for most splits (e.g. held-out
width −.005), so momentum recovery ratios are not reported as recovery; absolute values are in the
summary JSON.

**Limitations.** Partial run; 50-step searched-SOAP states only (no GGN-trajectory states); 2–3
seeds per architecture; the reused panels are development data; oracle parameters come from exact
blocks.

#### 4c. Neuron-graph predictor: skipped (gate failed)

The gate requires the best cheap family from 2c to recover ≥ 60% on the six main states **and**
≥ 50% on the train-split states of 4b. The best cheap family, `kron_basis_diag`, recovers 46.0% on
the six main states (2c), so the first condition fails independently of 4b. (The partial 4b train-split value, 48.7% on 21 of
24 states, is also below its 50% bar.) 4c was not run: even
with oracle parameters computed from exact GGN blocks, no cheap structured family captures enough
of the local exact-GGN headroom to be worth learning. A predictor can at best match its oracle
family, so the transfer question is moot until a family passes the oracle bar. No predictor was
trained and no predictor training trajectories were run.

## 4. Issues found in shared code

No defect was found in shared code, and no shared function was replaced by a study-local
workaround. (`study_ggn_propose` is the pre-approved parameterization of a hard-coded gamma, not a
workaround.)

One environment observation is relevant to shared numerical checks. On this machine (PyTorch 2.14.1
CPU float64, one thread, deterministic algorithms), `torch.cholesky_solve` with the same Cholesky
factor and identical RHS values in a newly allocated buffer can return different last bits. Two
separately computed, equally valid solves of an ill-conditioned block (condition number 6.8e7)
therefore differed by 1.2e-11 relative in the Euclidean norm while both had residuals of 1.1e-15
(DEVIATIONS #11). Minimal reproduction (from `residual-curvature-lab/`, any saved state with a badly
conditioned block, e.g. 4a state `b1_w8__s0/state-0010`):

```python
chol = torch.linalg.cholesky(system)            # one damped tensor block
a = torch.cholesky_solve(-g[s.slice, None], chol)
b = torch.cholesky_solve(-g[s.slice, None].clone(), chol)
torch.equal(a, b)                               # False for some tensors
```

Euclidean agreement checks between separately computed solves should therefore be sized to the
system's conditioning or replaced by residual/energy checks, as the capacity screen already did.
Bitwise trajectory reproduction (check 1 gate, 4a gate) was unaffected, since those recompute the
same operation sequence.

## 5. Deviations

Full list with times and reasons: [DEVIATIONS.md](DEVIATIONS.md). In summary (same numbering):

1. `study_ggn_propose(gamma)` copies the GGN branch with gamma as a parameter (pre-approved;
   bitwise identical at gamma .01); `dense_ef` reuses that body.
2. Complete per-step rows (all search trials) and batch IDs are kept as compressed files in the
   external runs; mirrored `training.json` files hold compact per-step summaries.
3. Edge-extension reading for categorical axes (frequency 10 → 100; beta pairs not extended).
4. Definition of a divergent configuration (none occurred).
5. Up to four concurrent single-threaded workers on a 4P+4E-core machine.
6. Wall-clock reported both as measured and as a sequential estimate.
7. The 2a closure category uses the smaller of the step-50 and step-200 closures.
8. An operational interruption (agent tool time limit) killed three GGN tuning units; they were
   rerun from scratch and matched the interrupted attempts at every recorded step.
9. Short analysis scripts sometimes ran alongside four experiment workers.
10. The machine slept from 13:00:05Z to 13:56:23Z; units were suspended, not killed.
11. Check 4b stopped at 21 of 22 units on the study's own 1e-12 solver-agreement assertion; the
    assertion was not loosened and the unit was not retried. A rerun requires the owner's approval.

Items 1–7 were recorded before the results they could affect; items 8–11 were recorded as they
happened. Check 3's mirror omits per-state
`candidates.json` files whose rows `all_candidates.json` repeats (`MIRROR_NOTE.json` lists their
checksums). Check 3 resumability is at the run level, because its runner replicates the main-lab
`main()` line for line. No rerun was needed.

## 6. Isolation audit and final suite

- **Isolation manifest: identical.** `isolation_manifest_before.txt` (2026-10-07T12:02:01Z,
  before any other write) and `isolation_manifest_after.txt` (16:45Z) list the same 130 files with
  the same SHA-256 digests; both manifests hash to `960eca98…436372a`. The manifest covers
  `residual-curvature-lab/` minus `runs/`, `datasets/`, `.venv/`, `build/`, `*.egg-info/`,
  `.pytest_cache/`, `__pycache__/` and this folder; it includes the inner `.git/` skeleton
  (rule 10 excludes only `../.git/`). The command is `scripts/isolation_manifest.sh`.
- No `__pycache__` or `.pyc` was created outside `.venv`. No directory was added to the
  repository's `runs/` (19 entries before and after). `.pytest_cache/` is unchanged, and no git write
  operation was used. `new_run` itself calls the read-only `git rev-parse` and `git status`.
- **Final suite:** `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`:
  **106 passed** (baseline count), 1 warning, 4.33 s ([log](logs/final_suite.log)).
- **Study tests:** `... -m pytest -q -p no:cacheprovider studies/gap_audit_20261007/tests`: **12 passed**
  ([log](logs/final_study_tests.log)), including the three 2a tests, the bitwise GGN-copy test and the
  Kronecker convention test. The default collection does not include them.
- Light mirrors total 27.87 MB (≤ 30 MB); external path `~/skipsoap-study-data/gap_audit_20261007/`
  (2.6 GB including checkpoints and saved states). The protocol file is unchanged (SHA-256 `469b6eed…`).

## 7. Reproduction commands

Run from `residual-curvature-lab/` with `export PYTHONDONTWRITEBYTECODE=1` (and optionally
`GAP_AUDIT_DATA`). Unit workers can run in parallel (`scripts/queue_runner.sh`,
`scripts/run_units.sh`, `scripts/run_pending.sh`); rerunning a `work` command skips verified
complete units.

```sh
P=studies/gap_audit_20261007/scripts
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m pytest -q -p no:cacheprovider studies/gap_audit_20261007/tests
sh $P/isolation_manifest.sh > /tmp/manifest.txt
.venv/bin/python $P/dryrun_cost.py && .venv/bin/python $P/project_cost.py <dryrun-run>
# Check 1 (gate, tuning with the one-time edge extension, evaluation)
.venv/bin/python $P/check1_train.py create --phase gate        # then: work --run R --unit U (each unit), finalize --run R
.venv/bin/python $P/check1_train.py create --phase tune        # work all units; extend --run R; work new units; finalize --run R
.venv/bin/python $P/check1_train.py create --phase eval --tuning <tune-run>   # work all units; finalize --run R
.venv/bin/python $P/summarize_check1.py --gate <gate> --tune <tune> --eval <eval> --dryrun <dryrun>
# Check 2c, check 3
.venv/bin/python $P/check2c_capacity.py && .venv/bin/python $P/summarize_check2c.py --run <2c-run>
.venv/bin/python $P/check3_ot_ggn_capacity.py && .venv/bin/python $P/summarize_check3.py --run <3-run>
# Check 4a/4b
.venv/bin/python $P/check4a_family.py create        # work each unit; finalize --run R
.venv/bin/python $P/check4b_headroom.py create --family <4a-run> --check2c <2c-run>   # work each unit
.venv/bin/python $P/summarize_check4.py --family <4a-run> --headroom <4b-run>
.venv/bin/python $P/ledger.py <dryrun> <gate> <tune> <eval> <2c> <3> <4a> <4b>
```

Run IDs used here are listed in [README.md](README.md).
