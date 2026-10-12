# Spectral normalization and curvature — results (A–C)

Study `sn_curvature_20261008`, protocol [PROTOCOL.md](PROTOCOL.md). Large artifacts:
`~/skipsoap-study-data/sn_curvature_20261008/runs/`. Light mirrors: [runs/](runs/).

**Status (2026-10-08, 14:30Z):** complete. Both reproduction gates passed; all 16 units and 48
diagnostic states completed; two deviations ([DEVIATIONS.md](DEVIATIONS.md)).

## 1. Summary

- **SNGP spectral normalization changes little of the curvature picture in the first 200 steps.**
  By the pre-declared six-pair rule, almost every metric is *mixed* for every bound in both models.
- **Bounds 2 and 6 are mostly inactive.** In the ViT, σ(W) stays below 6 throughout and below 2
  until step 200, so those runs equal no SN up to roundoff. Only c = 0.95 is active from the start.
- **Consistent effects at c = 0.95:** in the ViT, the damped-inverse (A-energy) errors of the
  diagonal and empirical-Fisher approximations are lower in all six matched states. In the MLP, the
  GGN's effective rank is lower in all six.
- **Exact structural effect:** with respect to the raw weights, every SN-active tensor has an exact
  GGN null direction along W (scale invariance; ≤ 4.5e-16 in all 175 active tensor-states).
- **Distance preservation:** SN at 0.95 lowers representation distortion at step 10. After that,
  distortion falls under training in every condition and the difference disappears.

| ViT, c = 0.95 vs no SN (seeds 0/1 x steps 10/50/200) | SN 0.95 | No SN |
| --- | --- | --- |
| Diagonal inverse A-energy error (gradient) | 0.82–0.89 | 0.87–0.95 |
| Empirical-Fisher inverse A-energy error | 0.36–0.84 | 0.42–0.92 |
| Nearest-Kronecker error, top eigenvalue, headroom | mixed | — |

**Next step:** these short-horizon, in-distribution diagnostics do not by themselves explain a large
Laplace gain. The candidates are longer training (where σ(W) reaches the larger bounds), distance
measured on shifted inputs, and the estimator effect at tight bounds. Part E (training) or a
longer-horizon A–C rerun would test them.

## 2. Settings as used

- Models: the ViT (`configs/mini_vit_allocation.yaml`, 5,738 parameters, 28x28) and the residual
  MLP (`runs/20261002T121635-0dac3337/resolved.yaml`, 6,634 parameters, 14x14); batch 512, CPU
  float64, one thread.
- SN: W_eff = W·min(1, c/σ(W)) with exact σ, on the residual-branch linear weights (ViT: attention
  q/k/v/o and w1/w2, 12 tensors; MLP: w1/w2, 6 tensors); c ∈ {0.95, 2, 6} versus no SN; identical
  initial raw weights.
- Trajectories: searched SOAP with default settings, 200 consumed steps, seeds 0 and 1; states after
  steps 10, 50 and 200.
- Diagnostics: construction batch = first 512 analysis IDs; four analysis panels; GGN with respect to
  the raw parameters (primary) and with respect to the effective weights (an effective-weight twin
  network); damping gamma .01 with exact trace and the normalized inverse frozen SOAP metric;
  `search_candidate` at the native radius.

**Reproduction gates (passed, bitwise).** No-SN ViT seed 0: parameters at steps 10 and 50 equal the
gap-audit 4a states (maximum difference 0.0). No-SN MLP seed 0: validation losses at steps 10/25/50
equal the stored searched-SOAP values (difference 0.0).

**Numerical checks (all ≤ 1e-8).** Block action errors ≤ 3.4e-9; full-matrix action errors ≤ 4.5e-14;
diagonal blocks of the full matrices equal the block construction to 2.2e-15; the effective-weight
twin reproduces the logits exactly; damped solves pass; 192 searches all bracketed. Tests: SN at
c = 1e9 is bitwise inactive; C_raw = Pᵀ C_eff P for every SN tensor (P the Jacobian of the SN map).

## 3. Results

Seed means (two seeds) at steps 10/50/200; readings compare each bound with no SN over six matched
(seed, step) pairs. Full tables (both parameterizations, every metric):
[sn_tables.md](runs/20261008T130821-44610eee/sn_tables.md); raw values:
[sn_summary.json](runs/20261008T130821-44610eee/sn_summary.json); figure
[sn_curvature.png](figures/sn_curvature.png).

**SN activity.** Active layers (of 12 in the ViT, 6 in the MLP) at steps 10/50/200: ViT c = 0.95
12/12/12, c = 2 0/0/11.5, c = 6 0/0/0; MLP c = 0.95 6/6/6, c = 2 0/6/6, c = 6 0/4/6. Where SN is
inactive, the run equals no SN up to roundoff (the parametrized model orders its parameter vector
differently): 3 of 6 ViT states at c = 2 and 4 of 6 at c = 6 match no SN exactly. The pre-declared
rule labels such identical pairs "mixed"; they mean *no effect because SN is inactive*.

**A. Distance preservation** (q95/q5 of ‖φ(x)−φ(x′)‖/‖x−x′‖):

| Model | Representation | No SN | c = 0.95 | c = 2 | c = 6 | Reading (0.95/2/6) |
| --- | --- | --- | --- | --- | --- | --- |
| ViT | global, residual stream | 7.36/2.89/2.28 | 7.24/2.77/2.32 | 7.36/2.89/2.32 | 7.36/2.89/2.28 | mixed/mixed/mixed |
| ViT | local, head input | 13.0/5.55/4.39 | 9.02/5.42/3.51 | 13.0/5.55/4.35 | 13.0/5.55/4.39 | mixed/mixed/mixed |
| MLP | global | 3.26/1.91/1.96 | 2.18/1.92/2.34 | 3.26/1.93/2.20 | 3.26/1.89/1.98 | mixed/mixed/mixed |
| MLP | local | 4.84/2.15/1.96 | 2.45/2.01/2.46 | 4.84/2.00/2.26 | 4.84/2.13/2.07 | mixed/mixed/mixed |

**B. Curvature structure** (raw parameters):

| Model | Metric | No SN | c = 0.95 | Reading (0.95/2/6) |
| --- | --- | --- | --- | --- |
| ViT | top GGN eigenvalue | 179/137/12.5 | 73.9/136/30.1 | mixed/mixed/mixed |
| ViT | effective rank | 5.1/22.3/44.8 | 8.2/17.2/28.9 | mixed/mixed/mixed |
| ViT | cross-tensor energy | .72/.69/.87 | .78/.82/.88 | mixed/mixed/mixed |
| ViT | nearest-Kronecker error | .31/.61/.60 | .30/.61/.55 | mixed/mixed/mixed |
| ViT | exact-GGN headroom | .198/.003/−.015 | .118/.014/−.004 | mixed/mixed/mixed |
| MLP | top GGN eigenvalue | 18.4/3.4/6.3 | 4.4/3.1/4.1 | mixed/mixed/mixed |
| MLP | effective rank | 58.8/57.0/42.1 | 57.8/44.6/33.0 | **consistently lower**/mixed/mixed |
| MLP | cross-tensor energy | .50/.49/.08 | .52/.20/.11 | mixed/mixed/mixed |
| MLP | nearest-Kronecker error | .35/.64/.67 | .43/.70/.64 | mixed/mixed/mixed |

SN null directions: for every SN-active tensor, ‖C_bb vec(W_b)‖/(‖C_bb‖₂‖W_b‖) ≤ 4.5e-16. The
effective-weight GGN has no such null directions. Its top eigenvalue under c = 0.95 rises late
(ViT 49 at step 200 versus 30 raw; MLP 58 versus 4.1), because the raw parameterization damps
curvature by the SN factor c/σ.

**C. Estimation quality** (versus the exact tensor GGN on the same batch):

| Model | Estimator | No SN | c = 0.95 | Reading (0.95/2/6) |
| --- | --- | --- | --- | --- |
| ViT | diagonal, inverse A-energy error | .936/.879/.877 | .885/.859/.829 | **consistently lower**/mixed/mixed |
| ViT | empirical Fisher, inverse A-energy error | .461/.703/.907 | .412/.574/.831 | **consistently lower**/mixed/mixed |
| ViT | Kronecker-basis diagonal, inverse A-energy error | .835/.751/.746 | .799/.774/.759 | mixed/mixed/mixed |
| ViT | empirical Fisher, Frobenius error | .19/.45/.48 | .31/.24/.43 | mixed/mixed/mixed |
| ViT | MC Fisher (1 label), Frobenius error | .066/.20/.42 | .074/.20/.38 | mixed/mixed/mixed |
| ViT | Hutchinson-8 trace, relative RMS | .32/.18/.11 | .31/.17/.14 | mixed/mixed/mixed |
| ViT | rank-16 truncation error | .015/.056/.13 | .020/.042/.077 | mixed/mixed/mixed |
| MLP | empirical Fisher, inverse A-energy error | .784/.961/.975 | .771/.984/.961 | mixed/mixed/mixed |
| MLP | empirical Fisher, Frobenius error | .42/1.33/3.31 | .49/3.88/3.65 | mixed/mixed/mixed |

Per-seed values of the two consistent ViT findings (seed 0 steps 10/50/200, then seed 1): diagonal
.893/.872/.838, .877/.845/.820 versus .924/.876/.873, .947/.881/.882; empirical Fisher
.462/.498/.825, .362/.651/.837 versus .508/.656/.892, .415/.750/.923.

**Context (not rankings).** Validation loss at step 200, seed means: ViT no SN .265, c = 0.95 .274,
c = 2 .245, c = 6 .265. MLP no SN .951, c = 0.95 .898, c = 2 1.066, c = 6 1.345. Under searched SOAP
the MLP's validation loss rises after step 50 in every condition (no SN .464 at step 50, .951 at 200),
so its later states are unstable training states.

## 4. Observations, hypotheses and limitations

Observations: (1) Within 200 steps, only the tight bound c = 0.95 changes the ViT at all. Larger
bounds act only once σ(W) has grown past them. (2) At c = 0.95, cheap estimators approximate the
damped GGN *inverse* more accurately in the ViT, though their Frobenius block errors do not improve
consistently. (3) In raw parameters, SN removes one direction per normalized tensor from the
curvature (scale invariance) and lowers the MLP's effective rank. (4) SN's distance effect is
early-training only here; training itself reduces distortion strongly in every condition.

Hypotheses (not tested): a Laplace benefit from SN could come from (a) longer training, where σ(W)
would otherwise grow and larger bounds activate; (b) distance preservation on shifted or
out-of-distribution inputs, which these in-distribution pairs do not probe; (c) the better
damped-inverse accuracy of cheap estimators seen at c = 0.95; (d) the scale-invariance null
directions, which change how a Laplace prior or damping enters.

Limitations: two seeds; 200 searched-SOAP steps; states from SN-trained trajectories differ, so
effects mix the direct effect of SN on curvature with where training went; LayerNorm, GELU and
attention keep SN from guaranteeing bi-Lipschitz branches; nearest-Kronecker is an oracle bound,
not K-FAC; no Laplace predictive (D) or training comparison (E), by the owner's scope decision.

## 5. Cost, deviations, isolation audit and final suite

- Cost: dry run `20261008T130452-a6beee47` projected 2.7 CPU-h (budget 4). Under four workers,
  diagnostics ran 2.9x slower than alone. The remaining units were moved to two workers
  (DEVIATIONS #1). Actual main-run cost: **3.1 CPU-h** ([logs/cost_actual.json](logs/cost_actual.json)).
- Baseline suite before the study and final suite: **106 passed** each; study tests: **5 passed**.
- Isolation: no code of this study wrote outside its folder or data directory. The after-manifest
  differs from the before-manifest (1,563 files) in two files: the geometry report, edited by the agent
  eight seconds after the before-manifest (DEVIATIONS #2), and `ideas/PARSEC_ideas.md`, created at
  13:27:28Z outside this session. No new `__pycache__`; nothing added to the repository `runs/` by this study.
- Light mirrors 1.8 MB.

## 6. Reproduction

From `residual-curvature-lab/`, with `export PYTHONDONTWRITEBYTECODE=1`:

```sh
P=studies/sn_curvature_20261008/scripts
.venv/bin/python -m pytest -q -p no:cacheprovider studies/sn_curvature_20261008/tests
.venv/bin/python $P/dryrun_sn.py
.venv/bin/python $P/run.py create            # then: run.py work --run R --unit U for each unit (queue_runner.sh)
.venv/bin/python $P/run.py finalize --run R
.venv/bin/python $P/summarize_sn.py --run R
```
