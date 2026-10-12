# Transfer law and computable step sizes — results

Protocol [PROTOCOL.md](PROTOCOL.md) (SHA-256 in `logs/protocol_sha256.txt`, written before any run). Three
deviations are in [DEVIATIONS.md](DEVIATIONS.md), each hashed before the rerun it concerns. Outputs:
- `results/corpus_<state>.json` (Part A);
- `results/steps.json` (Part B, after the fix in DEVIATIONS #3; the pre-fix outputs are kept as `*_prefix_bug.json`);
- `results/summary.json` and `results/summary_stdout.txt` (readings and tables).

Offline only, on six banked states; no training.

**Status (2026-10-11, 20:52Z).**
- **Part A:** six states complete; gradient gates ≤ 2.2e-15; no boundary exclusions, no rejected searches.
- **Part B:** complete. All 72 reference gates (same, second and oracle × four operators × six states) reproduce
  the optimism follow-up with difference 0.0.
- **Wiener gate failed** (identity difference up to 2.1e-8 > 1e-8; DEVIATIONS #1), so reading B2 is *not
  evaluated*.
- **Stops and reruns** before this final run:
  - a reported-only search hit the grid boundary (DEVIATIONS #2);
  - one launch died without a traceback or crash report and was rerun unchanged as a harness-managed job;
  - a scale bug in Ô for `soap_map` was found after a complete run, fixed and rerun (DEVIATIONS #3).

## 1. Summary

| Reading | Result | Verdict |
| --- | --- | --- |
| A1 Corpus share vs (1 − r)/(1 + r), r = (1 − f)·B_noise/B (126 points) | Spearman 0.963, MAE 0.182 (bar ≤ 0.15) | **partial** |
| A2 1/B law (log-slope of measured r on predicted r) | median slope 0.78 (bar [0.8, 1.25]); median r ratio 0.66 (bar [0.5, 2]) | **partial** |
| A3 New-data share vs conditional prediction | Spearman 0.963, MAE 0.211 | **partial** |
| B1 Loss-free step α = γ̂·q/κ̂₁ vs second-batch search (24 points) | 0.0210 vs 0.0403 held-out: 52% (bar 90%); before the fix 48% | **fails** |
| B2 Wiener full step vs second-batch search | gate failed (forward identity 2.1e-8 at cond(A) 1.6e10; both residuals ≤ 1.6e-14) | **not evaluated** |

**What holds.**
- **Ranking.** The transfer law ranks transfer almost perfectly across batch sizes, metrics and states
  (Spearman 0.96).
- **Fixed metrics.** It is accurate for SGD and the SOAP map: MAE 0.07 and 0.11, and 0.04 for both at B ≥ 1024.
- **The noise algebra is exact.** The realized alignment γ = p̄/q̄ matches 1/(1 + r) to 0.023 on average. This
  is an identity of sampling without replacement, so it confirms the bookkeeping, not the model.
- **New data.** The new-data gap that the corpus formula cannot see is real and roughly predicted. At B = 4096
  (f = 0.82), exact GGN's corpus share is 0.92–0.99, but its validation share is 0.57–0.95, against a predicted
  0.55–0.91 from r_pop = B_noise/B.

**What fails, and why (post-hoc decomposition, not pre-declared).**
- **Overstated harm.** For exact GGN at B = 64 the law predicts strongly negative shares (−0.50 to −0.90),
  while the measured shares stay near zero (−0.11 to +0.24); MAE 0.63 at B ≤ 256.
- **The cause is the equal-curvature assumption,** not the noise term. With the batch-optimal step,
  share = 2γ − κ_target/κ_batch, so the implied curvature ratio is 2γ̄ − share.
  - Along a step fitted to a small batch, that batch's own curvature exceeds the corpus curvature:
    κ_corpus/κ_batch = 0.25 / 0.35 / 0.50 / 0.66 / 0.81 / 0.92 / 0.99 for exact GGN at B = 64 … 4096.
  - For the SOAP map it is 0.71 → 1.00, and for SGD 0.88 → 1.00.
  - The same-batch search therefore stays short, which limits the damage. The noise costs far less than the
    equal-curvature law predicts, and the measured r falls more slowly than 1/B (median slopes: SGD 0.84, SOAP
    map 0.84, exact GGN 0.57).
- **Opposite sign to the proposal.** The proposal expected the batch to exploit *low* curvature (curvature
  optimism). Here the batch curvature along its own step is *higher*. Building exact GGN from another batch's
  curvature (panel 1) changes its share by at most 0.06 (`exact_ggn_xc`), so that remedy has nothing to fix
  here.

**Step sizes from statistics (Part B).**
- **Pre-declared B1 fails:** the loss-free γ̂/λ̂ step reaches 52% of second-batch selection. The other loss-free
  rules are no better: single-batch `gamma_b` 55%, `slope_x` 60%. All of them stay below plain same-batch search
  (72%).
- **Post-hoc split by training step:**
  - **Steps 25 and 50 (16 points):** `gamma_x` reaches 96% of second-batch selection (0.0290 vs 0.0301). It beats
    same-batch search in 14 of 16 points and nearly doubles its held-out decrease (0.0290 vs 0.0154). There the
    batch is noise-dominated and the quadratic model holds along the step.
  - **Step 10 (8 points):** 8%. The GGN quadratic along D underestimates the real loss curvature over the step,
    so the computed steps are 1.06–2.49 times the second-batch step (geometric mean 1.82) and overshoot (exact
    GGN at SOAP-10: −0.060 vs +0.132). At steps 25 and 50 the ratio is 0.46–1.35 (geometric mean 0.97).
- **Wiener diagnostics** (B2 not evaluated):
  - A full Wiener step without search is catastrophic: −4.0 to −13.4 held-out loss.
  - So is the unshrunk full Newton step, −3.5 to −20.7. The failure is the missing step-length control, not the
    shrinkage.
  - With same-batch search, the Wiener-shrunk direction (0.0408) does not beat the plain exact-GGN direction
    (0.0460). Two of its six searches hit the grid boundary and took no step.

**Consequence for PARSEC.**
- Step-length choices should keep loss evaluations on a second batch: second-batch search at 100% here, or the
  optimism-corrected batch search at 94% (optimism follow-up).
- The noise statistics are good enough to *correct* a searched step late in training, but GGN curvature along
  the step is not good enough to *replace* the search early on.
- The share law is useful as a diagnostic of how much of a batch fit transfers, if extended by the curvature
  ratio κ_target/κ_batch.

## 2. Part A tables

Noise scale B_noise = tr(A⁻¹Σ)/S (examples) and the positive-transfer threshold f* = B_noise/(N + B_noise),
N = 5,000:

| State | SGD | SOAP map | Exact GGN (corpus) |
| --- | --- | --- | --- |
| SOAP-10 | 36 / 0.007 | 68 / 0.013 | 195 / 0.038 |
| SOAP-25 | 93 / 0.018 | 174 / 0.034 | 420 / 0.078 |
| SOAP-50 | 120 / 0.023 | 264 / 0.050 | 908 / 0.154 |
| GGN-10 | 55 / 0.011 | 128 / 0.025 | 338 / 0.063 |
| GGN-25 | 210 / 0.040 | 249 / 0.047 | 814 / 0.140 |
| GGN-50 | 130 / 0.025 | 307 / 0.058 | 1204 / 0.194 |

Means over the six states (measured / predicted corpus share; implied curvature ratio 2γ̄ − share):

| B (f) | SGD share | κ ratio | SOAP map share | κ ratio | Exact GGN share | κ ratio |
| --- | --- | ---: | --- | ---: | --- | ---: |
| 64 (0.013) | −0.08 / −0.16 | 0.88 | −0.20 / −0.44 | 0.71 | −0.01 / −0.75 | 0.25 |
| 128 (0.026) | +0.30 / +0.16 | 0.93 | +0.08 / −0.14 | 0.79 | +0.07 / −0.57 | 0.35 |
| 256 (0.051) | +0.48 / +0.46 | 0.88 | +0.31 / +0.19 | 0.85 | +0.20 / −0.31 | 0.50 |
| 512 (0.102) | +0.77 / +0.69 | 0.97 | +0.61 / +0.50 | 0.94 | +0.38 / +0.02 | 0.66 |
| 1024 (0.205) | +0.88 / +0.85 | 0.97 | +0.78 / +0.74 | 0.97 | +0.57 / +0.38 | 0.81 |
| 2048 (0.410) | +0.93 / +0.94 | 0.99 | +0.90 / +0.89 | 0.99 | +0.78 / +0.70 | 0.92 |
| 4096 (0.819) | +0.99 / +0.99 | 1.00 | +0.98 / +0.98 | 1.00 | +0.96 / +0.94 | 0.99 |

Per-state values for every size, including validation shares and both new-data predictions, are in
`results/summary_stdout.txt`. Reported calibrations:
- validation against the population form r_pop = B_noise/B: Spearman 0.935, MAE 0.203;
- the realized mechanism 2γ − 1 against the corpus share: Spearman 0.968, MAE 0.179.

## 3. Part B tables

Mean held-out decrease on panels 2–4. Six states per operator; 24 points for "All".

| Operator | same | second | oracle | `gamma_x` | `gamma_b` | `slope_x` | `gamma_x` / second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGD | 0.0155 | 0.0151 | 0.0180 | 0.0132 | 0.0140 | 0.0130 | 0.88 |
| SOAP map | 0.0192 | 0.0232 | 0.0246 | 0.0189 | 0.0198 | 0.0187 | 0.81 |
| Exact GGN | 0.0460 | 0.0654 | 0.0658 | 0.0169 | 0.0169 | 0.0267 | 0.26 |
| PARSEC-L | 0.0352 | 0.0574 | 0.0577 | 0.0350 | 0.0380 | 0.0375 | 0.61 |
| **All** | 0.0290 | 0.0403 | 0.0415 | **0.0210** | 0.0222 | 0.0240 | **0.52** |

`gamma_x` beats second-batch selection at 9 of 24 points; `gamma_b` and `slope_x` at 5 each.

Exact GGN per state (held-out decrease, panels 2–4):

| State | same | second | `gamma_x` | `slope_x` | Wiener + search | Full Newton | Full Wiener | Share, panel-1 vs batch curvature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SOAP-10 | 0.1324 | 0.1319 | −0.0601 | −0.0222 | 0.1312 | −20.73 | −13.36 | +0.65 vs +0.71 |
| SOAP-25 | 0.0803 | 0.0944 | 0.0730 | 0.0773 | 0.0659 | −7.20 | −4.92 | +0.32 vs +0.32 |
| SOAP-50 | −0.0081 | 0.0288 | 0.0285 | 0.0291 | 0 (boundary) | −7.09 | −5.48 | +0.01 vs −0.04 |
| GGN-10 | 0.0368 | 0.0582 | −0.0167 | −0.0012 | 0.0355 | −3.45 | −3.95 | +0.21 vs +0.22 |
| GGN-25 | 0.0127 | 0.0399 | 0.0401 | 0.0393 | 0.0122 | −3.94 | −4.85 | +0.09 vs +0.07 |
| GGN-50 | 0.0216 | 0.0391 | 0.0366 | 0.0377 | 0 (boundary) | −4.63 | −4.35 | +0.16 vs +0.13 |

The Wiener columns are diagnostics: their gate failed (DEVIATIONS #1). The Wiener filter keeps 50–76% of the
Newton step's energy.

## 4. Limitations

- One workload (ViT width 16, depth 2, MNIST), six states, 8 subsets per size in Part A and one construction batch
  per state in Part B.
- The curvature decomposition and the early/late split are post hoc.
- Part A uses a fixed A built on the whole corpus. In training, A is built from batches.
- N = 5,000: f* and the new-data gap refer to this corpus. B_noise up to about 1,200 implies f* of about 2.4% of
  the 50,000-example training split, under the same equal-curvature law that overstates harm here.
- The paper this proposal cites (arXiv 2605.11181) was not read; its relation to this theory is unverified.

## 5. Cost, isolation, suites

- **Wall time:** Part A 20:08:52–20:26:31Z (four, then two workers, about 9.5 minutes per state); final Part B
  run 20:48–20:51Z.
- **Isolation:** the manifests before and after are identical (3,472 files), and no new `__pycache__` was
  created.
- **Suites:** 128 passed before and after (repository 106, PARSEC-L 22).
- **Not in this study:** the repository's `.gitignore` was rewritten at the owner's request before this study
  started (outside the study scope; see the session report).
