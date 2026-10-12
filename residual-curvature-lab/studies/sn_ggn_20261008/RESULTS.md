# SN x exact per-tensor GGN training — results

Study `sn_ggn_20261008`, protocol [PROTOCOL.md](PROTOCOL.md) (SHA-256 `1b796bbe…`, unchanged). Large
artifacts: `~/skipsoap-study-data/sn_ggn_20261008/runs/`. Light mirror:
[runs/20261008T182801-03c6c5c2/](runs/20261008T182801-03c6c5c2/). Figure: [sn_ggn.png](figures/sn_ggn.png).

**Status (2026-10-08, 19:20Z):** complete. All 12 units completed, all gates passed, no divergence, no
deviations.

## 1. Summary

- **SN at c = 0.95 does not change validation loss consistently for either optimizer.** All nine
  pre-declared readings are *mixed*. Seed means differ by at most 0.031 at steps 50, 100 and 250. At
  step 250: per-tensor GGN 0.212 without SN versus 0.213 with SN; native SOAP 0.213 versus 0.221.
- **GGN's advantage over native SOAP is unchanged by SN.** SOAP's loss divided by GGN's is 2.3–2.7
  without SN and 2.2–3.1 with SN at step 50, and 0.96–1.03 versus 0.89–1.13 at step 250.
- **SN reduces GGN's early fitting of its own minibatch, not its real progress.** In steps 1–50,
  32% of each GGN update's own-batch decrease appears on the next batch with SN, against 19% without,
  in every seed. The own-batch decrease shrinks (0.14 versus 0.21 per update); the next-batch decrease
  is unchanged within error (0.044 versus 0.040, ± 0.005). After step 50, GGN's next-batch decrease is
  zero within ± 0.001 per update in both conditions.
- Hyperparameters were tuned without SN and not retuned. This is a matched-settings test, not a
  comparison of tuned methods.

| Seeds 1–3, mean ± sd | Loss 50 | Loss 100 | Loss 250 | Transfer, steps 1–50 | Measured s/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-tensor GGN, no SN | 0.355 ± 0.011 | 0.265 ± 0.018 | 0.212 ± 0.011 | 19% | 4.94 |
| Per-tensor GGN, SN c = 0.95 | 0.353 ± 0.006 | 0.274 ± 0.008 | 0.213 ± 0.004 | 32% | 6.08 |
| Native SOAP, no SN | 0.904 ± 0.098 | 0.431 ± 0.010 | 0.213 ± 0.008 | 75% | 0.121 |
| Native SOAP, SN c = 0.95 | 0.935 ± 0.153 | 0.457 ± 0.068 | 0.221 ± 0.023 | 75% | 0.128 |

**Next step:** at this scale SN neither helps nor hurts either optimizer at matched settings. Whether SN
changes optimizer quality after retuning (γ or lr under SN), over longer horizons, or for PARSEC-L
remains open. The early transfer gain suggests SN damps GGN's same-batch overfitting, which is the
failure mode PARSEC-L is meant to avoid without search (ideas/parsec.md, N1).

## 2. Settings as used

- Workload: main ViT (`configs/mini_vit_allocation.yaml`; width 16, 2 blocks, 5,738 parameters),
  MNIST 28x28, batch 512, CPU float64, all 40 tensors. Seeds 1–3 (model init s, data order 102 + 1000 s,
  split 101). Validation on the reserved 5,000 examples at steps 0, 10, 25, 50, …, 250.
- Per-tensor GGN: the gap audit's `ggn_exact_trace`. Exact GGN blocks per tensor on the training
  minibatch; damping μ = max(.001 · tr C / D, 1e-8) with the exact current trace; normalized inverse
  shadow-SOAP metric; current-gradient right-hand side; one global ray search per step on the same
  minibatch (zero plus native-radius multipliers 2^[-16..16], 18 golden refinements). Native SOAP:
  lr 1e-2, no search. Both settings were tuned on seed 0 without SN (gap audit).
- SN: W_eff = W · min(1, 0.95/σ(W)) with exact σ, via torch parametrization on the 12 residual-branch
  linear weights (attention q/k/v/o, MLP w1/w2). The optimizers act on the raw weights W; the GGN is
  taken with respect to the raw parameters. SN is active in 10 of 12 tensors at initialization and in
  all 12 from step 10 (GGN) or step 50 (SOAP) on.
- Next-batch measurement: before each update, the loss of the batch that the next step will consume;
  after the update, the same batch again. Both lie outside the timed region.

## 3. Gates and checks (all passed)

- **Reproduction gate:** all six no-SN units reproduce the gap-audit evaluation units of the same arm
  and seed (`20261007T143302-baba2664`) at every validation step through 250, difference 0.0.
- GGN action checks (steps 2, 50, 200, 250) ≤ 2.1e-14; dense-solve residuals ≤ 8.1e-14; 1,494 searches,
  all bracketed, none rejected, none selected zero.
- Study tests (4) before and after: next-batch peek does not advance the data order; an SN GGN step
  passes the action audit and active SN tensors have gradients orthogonal to their raw weights; with
  SN inactive (c = 1e9) two GGN steps match the plain model to a relative 1e-10.

## 4. Results

**Pre-declared readings** (lower/higher in all three seeds, otherwise mixed):

| Reading | Step 50 | Step 100 | Step 250 |
| --- | --- | --- | --- |
| 1. GGN with SN versus without | mixed | mixed | mixed |
| 2. Native SOAP with SN versus without | mixed | mixed | mixed |
| 3. GGN's advantage (SOAP/GGN loss ratio) under SN | mixed | mixed | mixed |

Per-seed validation losses (seeds 1 / 2 / 3):

| Arm | Step 50 | Step 100 | Step 250 |
| --- | --- | --- | --- |
| GGN, no SN | .3483 / .3682 / .3486 | .2715 / .2798 / .2448 | .2132 / .2229 / .2007 |
| GGN, SN | .3459 / .3540 / .3584 | .2831 / .2684 / .2704 | .2088 / .2132 / .2172 |
| SOAP, no SN | .8848 / 1.0103 / .8172 | .4417 / .4212 / .4294 | .2206 / .2134 / .2042 |
| SOAP, SN | .9428 / 1.0833 / .7777 | .4598 / .5237 / .3878 | .2363 / .2331 / .1943 |
| Ratio SOAP/GGN, no SN | 2.54 / 2.74 / 2.34 | 1.63 / 1.50 / 1.75 | 1.03 / 0.96 / 1.02 |
| Ratio SOAP/GGN, SN | 2.73 / 3.06 / 2.17 | 1.62 / 1.95 / 1.43 | 1.13 / 1.09 / 0.89 |

Seed means at steps 10 / 25: GGN 1.129 / 0.548 without SN, 1.111 / 0.666 with SN; SOAP 2.065 / 1.478
versus 2.114 / 1.523.

**4. Transfer to the next batch** (paired measurement; pooled over seeds as mean next-batch decrease /
mean own-batch decrease, per-seed range in brackets; next-batch decrease per update ± standard error):

| Arm | Steps 1–50 | Steps 51–150 | Steps 151–250 |
| --- | --- | --- | --- |
| GGN, no SN | 19.0% [18.4, 19.6]; .040 ± .005 of .210 | −0.5% [−1.8, 1.1]; −.001 ± .001 of .167 | −0.3% [−1.0, 0.2]; −.001 ± .001 of .143 |
| GGN, SN | 31.7% [30.6, 34.3]; .044 ± .005 of .138 | 0.6% [0.3, 0.9]; .001 ± .001 of .153 | 1.4% [0.1, 2.3]; .002 ± .001 of .135 |
| SOAP, no SN | 75.2% [73.4, 78.3]; .030 ± .002 of .040 | 28.1% [24.4, 33.1]; .006 ± .001 of .021 | 3.4% [1.9, 4.9]; .0005 ± .0006 of .015 |
| SOAP, SN | 74.7% [74.0, 75.9]; .029 ± .002 of .039 | 31.2% [27.8, 33.0]; .006 ± .001 of .019 | 7.5% [4.7, 9.5]; .0010 ± .0006 of .013 |

The log-based estimator of the earlier analysis (own-batch loss before step t minus the next batch's
loss before step t + 1; last window 151–249) gives GGN 19.9 / 0.6 / 0.5% without SN and 30.4 / 0.7 / 0.3%
with SN; SOAP 75.4 / 28.0 / 11.7% and 76.8 / 33.2 / 10.0%. Only the GGN early difference is consistent
across seeds and estimators; the late SOAP differences lie within one or two standard errors.

**5. Targets** (gap audit T50 = 0.355037, T100 = 0.265401; first validation step at or below the target,
seed means; measured training seconds with up to four concurrent workers):

| Arm | T50 steps | T50 s | T100 steps | T100 s |
| --- | ---: | ---: | ---: | ---: |
| GGN, no SN | 58.3 | 319 | 141.7 | 722 |
| GGN, SN | 58.3 | 337 | 150.0 | 891 |
| SOAP, no SN | 141.7 | 16.9 | 191.7 | 23.3 |
| SOAP, SN | 141.7 | 17.9 | 191.7 | 24.3 |

**Exploratory.**
- *Raw spectral norms* (mean and maximum σ of the 12 raw branch weights, from 1.06 / 1.39 at step 0):

  | Arm | Step 10 | Step 250 |
  | --- | --- | --- |
  | GGN, no SN | 2.95 / 5.20 | 4.09 / 7.50 |
  | GGN, SN | 6.31 / 16.8 | 8.06 / 21.0 |
  | SOAP, no SN | 1.13 / 1.45 | 1.70 / 2.19 |
  | SOAP, SN | 1.11 / 1.48 | 1.54 / 1.98 |

  Without SN, GGN steps nearly triple the branch weights' spectral norm within ten steps; SOAP barely
  changes it. Under SN, raw norms grow further but the effective weights stay at σ = 0.95.
- *Curvature scale (GGN):* SN roughly halves the raw-parameter GGN trace. Mean trace in windows
  1–50 / 51–150 / 151–250 is 16,853 / 29,199 / 37,636 without SN and 7,086 / 13,877 / 15,231 with SN.
  μ follows the trace (exact-trace damping), and the raw step norm roughly doubles (2.21 / 1.83 / 1.75
  versus 4.69 / 3.78 / 3.66).
- *Cost:* measured seconds per step 4.94 versus 6.08 (GGN) and 0.121 versus 0.128 (SOAP). Two no-SN GGN
  units ran partly with fewer concurrent workers, so the measured 23% GGN overhead overstates SN's cost.
  The sequential dry run gives 3.92 versus 4.41 s (12%). The exact-σ SVD inside per-example Jacobians
  causes most of it; SNGP's power iteration would be cheaper.

## 5. Observations, hypotheses and limitations

Observations:
1. At the gap audit's tuned settings, SN at c = 0.95 neither improves nor degrades validation loss
   consistently for per-tensor GGN or native SOAP within 250 steps, and it leaves GGN's step-count
   advantage over SOAP intact (T50 in 58 versus 142 steps under both conditions).
2. SN changes the GGN trajectory in raw parameters substantially: 0.4–0.5 times the GGN trace, about
   twice the raw step norm and twice the raw spectral norms. The validation trajectories nevertheless
   agree (effective-weight step sizes were not measured).
3. With SN, GGN's same-batch decrease in steps 1–50 is a third smaller while its next-batch decrease is
   unchanged, so its transfer rises from 19% to 32% in every seed. After step 50, GGN updates make no
   measurable next-batch progress with or without SN, while SOAP's still transfer 28–31% in steps 51–150.
4. GGN's seed spread is smaller with SN (sd 0.004 versus 0.011 at step 250), from three seeds only.

Hypotheses (not tested): (a) retuning γ or lr under SN could change reading 1, because SN rescales the
raw-parameter curvature and the search radius is tied to SOAP's raw native step; (b) SN's lower
early sharpness (SN curvature study) lets the same-batch search pick steps that overfit less; (c) the
late absence of next-batch progress is the search policy (ideas/parsec.md N1), which SN does not fix.

Limitations: three seeds, one workload, 250 steps; hyperparameters tuned without SN; one bound
(c = 0.95; bounds 2 and 6 are inactive this early); exact σ rather than power iteration; GGN uses
same-batch search; validation on in-distribution data only; no Laplace or OOD evaluation.

## 6. Cost, isolation audit and suites

- Cost: **2.4 CPU-h** (unit wall time summed; budget 4, projection 3.0). Wall time 18:28–19:14Z with
  four workers; caffeinate attached; no sleep.
- Baseline and final repository suite: **106 passed** each. Study tests: **4 passed** before and after.
- Isolation manifests (18:24Z and 19:16Z, 2,228 files, this study folder excluded): one difference,
  `ideas/parsec.md`, which the agent edited at the owner's request (insight bank) during the run. No
  `__pycache__` outside `.venv` is newer than the study start; the repository `runs/` is unchanged.
- Light mirror 2.9 MB.

## 7. Reproduction

From `residual-curvature-lab/`, with `export PYTHONDONTWRITEBYTECODE=1`:

```sh
P=studies/sn_ggn_20261008/scripts
.venv/bin/python -m pytest -q -p no:cacheprovider studies/sn_ggn_20261008/tests
.venv/bin/python $P/train.py dryrun
R=$(.venv/bin/python $P/train.py create)
# queue every unit of `train.py pending --run $R` (queue_runner.sh, 4 workers), then:
.venv/bin/python $P/train.py finalize --run $R
.venv/bin/python $P/summarize.py $R
```
