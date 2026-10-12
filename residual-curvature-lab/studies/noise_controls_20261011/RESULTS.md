# Why the brake works — noise controls (results)

Protocol [PROTOCOL.md](PROTOCOL.md) (SHA-256 in `logs/protocol_sha256.txt`, written before the runs). External
runs `~/skipsoap-study-data/noise_controls_20261011/runs/`:
- tuning `20261011T130114-e83fd7ef`;
- constant-lr evaluation `20261011T130116-c3c4abb4`;
- tuned evaluation `20261011T131652-a952c81b`.

Light mirrors are in [runs/](runs/). Figure: [noise_controls.png](figures/noise_controls.png). Summary:
`runs/20261011T130116-c3c4abb4/noise_controls_summary.json`.

**Status (2026-10-11, 13:25Z):** all 38 units completed, no divergence, all gates passed.

## 1. Summary

- **The brake's advantage is noise removal, which standard practice already provides.** On seeds 1–3, at
  step 500:

  | Run | Validation loss |
  | --- | ---: |
  | SOAP, constant lr | 0.161 |
  | Brake, constant lr | 0.149 |
  | SOAP evaluated on an EMA of its weights (d = 0.98) | **0.127** |
  | SOAP with a 100-step final decay (WSD) | **0.129** |

  Both are better than the brake in all three seeds. By the pre-declared rule, explanation N (selective
  annealing / noise) is supported.
- **On top of decay or weight averaging, the brake adds nothing consistent.**
  - Brake + WSD 0.130 vs SOAP + WSD 0.129: *mixed* (the brake is better in two seeds).
  - Brake + EMA 0.123 vs SOAP + EMA 0.127: *mixed*.
  - Brake + cosine 0.143 vs SOAP + cosine 0.147: *mixed*.
- **The alternative explanations are not supported.**
  - *Statistics effect (S):* not detected. At matched training loss, the brake's median effective step is
    0.84–0.95 times SOAP's, smaller rather than larger.
  - *Edge of stability (E):* absent. SOAP's preconditioned sharpness is 10–27 throughout, against a limit of
    78, except one early spike of 61 (seed 1, step 10).
  - *Transient ringing (T):* removing momentum in the top directions is *mixed* at steps 50 and 100.
- **What remains in the brake's favour:**
  - *Anytime performance:* without a schedule it beats cosine-decayed SOAP at steps 200 and 300 in all seeds.
  - *Seed spread:* much smaller. sd 0.0008–0.0017 with WSD or EMA, against 0.011–0.012 for SOAP. This is
    exploratory, from three seeds.
- **Unplanned observation:** the momentum-free top-space variant (c = 16) beats the constant brake at step 500
  in all seeds (0.140 vs 0.149). With weight averaging it reaches 0.119, the lowest step-500 loss in this study.
  That combination was not a pre-declared comparison.

**Consequence for earlier claims:** stage D's "the brake beats every optimizer" held only against baselines
trained at constant learning rate without weight averaging. Against SOAP with a final decay or weight
averaging, the brake does not win.

**Next step:** compare future optimizers with decay and weight averaging switched on for every arm. The only
PARSEC-L candidate left from this study is the momentum-free top space; it needs a pre-declared test against
SOAP + WSD and SOAP + EMA.

## 2. Gates

- **Stage-D reproduction (validation, every common step):** constant SOAP and constant brake at lr 1e-2,
  seeds 0–3, reproduce the stage-D runs with difference **0.0**. For SOAP seed 1 this holds through step 275,
  where the stage-D unit stopped.
- **WSD vs constant:** WSD arms at peak 1e-2 equal the constant arm of the same seed through step 400
  (difference 0.0, seed 0).
- **Tests:** PARSEC-L (22) and harness (2) tests passed; repository suite 106 passed before and after.

## 3. Pre-declared readings

Seeds 1–3, settings from seed 0. Seed-0 selections:
- SOAP + cosine at peak 3e-2;
- SOAP + WSD, brake + cosine and brake + WSD at peak 1e-2;
- top-free at c = 16;
- EMA decay 0.98.

| Code | Reading | Result |
| --- | --- | --- |
| R1 | SOAP + cosine vs constant brake, step 500 | mixed (0.147 vs 0.149) |
| R1 | SOAP + WSD vs constant brake, step 500 | **better** (0.129 vs 0.149) |
| R2 | Brake + cosine vs SOAP + cosine, step 500 | mixed (0.143 vs 0.147) |
| R2 | Brake + WSD vs SOAP + WSD, step 500 | mixed (0.130 vs 0.129) |
| R3 | Constant brake vs SOAP + cosine, steps 200 / 300 | **better / better** |
| R3 | Constant brake vs SOAP + WSD, steps 200 / 300 | mixed / better |
| R4a | SOAP + EMA vs raw brake, step 500 | **better** (0.127 vs 0.149) |
| R4b | Brake + EMA vs SOAP + EMA, step 500 | mixed (0.123 vs 0.127) |
| R5 | Top-free (c = 16) vs constant brake, steps 50 / 100 / 500 | mixed / mixed / **better** |
| R6 | Statistics effect (matched-loss step ratio > 1.05) | not detected (ratio 0.84–0.95) |
| R7 | Edge-of-stability signature | absent |

**Interpretation, by the rules fixed in advance:**
- N is supported: R4a is better, and R1 is better for WSD.
- Nothing beyond iterate noise is established: R4b is mixed. R5 (early), R6 and R7 do not support T, S or E.

## 4. Tables

Validation loss, mean ± sd over seeds 1–3. Per-seed values at step 500: raw / EMA.

| Arm | 100 | 300 | 500 raw | 500 EMA (d 0.98) | Per seed, raw 500 | Per seed, EMA 500 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| SOAP, constant | 0.431 ± 0.010 | 0.205 ± 0.013 | 0.161 ± 0.010 | 0.127 ± 0.011 | .172 / .157 / .153 | .134 / .132 / .114 |
| Brake, constant | 0.385 ± 0.040 | 0.182 ± 0.004 | 0.149 ± 0.006 | 0.123 ± 0.002 | .145 / .156 / .146 | .125 / .122 / .123 |
| SOAP + cosine (3e-2) | 0.510 ± 0.016 | 0.196 ± 0.008 | 0.147 ± 0.010 | 0.148 ± 0.010 | .158 / .143 / .140 | .160 / .144 / .140 |
| SOAP + WSD (1e-2) | 0.431 ± 0.010 | 0.205 ± 0.013 | 0.129 ± 0.011 | 0.127 ± 0.012 | .132 / .137 / .117 | .135 / .133 / .113 |
| Brake + cosine (1e-2) | 0.392 ± 0.039 | 0.170 ± 0.009 | 0.143 ± 0.005 | 0.144 ± 0.005 | .138 / .147 / .145 | .138 / .148 / .145 |
| Brake + WSD (1e-2) | 0.385 ± 0.040 | 0.182 ± 0.004 | 0.130 ± 0.001 | 0.125 ± 0.002 | .130 / .129 / .131 | .127 / .123 / .125 |
| Top-free brake (c = 16) | 0.374 ± 0.016 | 0.172 ± 0.011 | 0.140 ± 0.005 | 0.119 ± 0.006 | .140 / .145 / .134 | .124 / .122 / .113 |

Seed-0 grids (step 500):
- SOAP cosine 0.170 / 0.176 / 0.158 and WSD 0.142 / 0.151 / 0.157, at peaks 1e-2 / 2e-2 / 3e-2;
- brake cosine 0.150 / 0.172 / 0.157 and WSD 0.127 / 0.152 / 0.143, at the same peaks;
- top-free c = 1 / 4 / 16: 0.168 / 0.165 / 0.157;
- SOAP-constant EMA d = 0.9 / 0.98 / 0.995: 0.144 / 0.136 / 0.170.

Full-network preconditioned sharpness lr·λ_max(P̂H), steps 10 / 25 / 50 / 100 / 200 / 300 / 400 / 500:

| Seed | SOAP | Brake |
| ---: | --- | --- |
| 1 | 61 / 13 / 22 / 12 / 15 / 17 / 18 / 14 | 32 / 23 / 35 / 26 / 23 / 17 / 18 / 14 |
| 2 | 13 / 12 / 27 / 17 / 11 / 13 / 11 / 10 | 19 / 18 / 28 / 29 / 24 / 19 / 15 / 17 |
| 3 | 23 / 21 / 16 / 19 / 13 / 16 / 20 / 11 | 22 / 20 / 14 / 20 / 19 / 18 / 17 / 14 |

The brake's trajectory runs at equal or higher sharpness than SOAP's, never near the limit 78. The step-50
values for SOAP match the offline checkpoint measurement exactly (21.94, 27.19, 16.09).

## 5. Observations and limitations

Observations:
1. A 100-step linear decay at the end, or an EMA of the weights, removes as much excess loss from constant-lr
   SOAP as the brake does, or more, at no extra cost.
2. The brake and these remedies largely overlap. The brake's residual value is anytime performance before
   decay, and lower seed spread.
3. Momentum-free top-space braking (c = 16) beats the momentum brake at step 500, and with weight averaging it
   has the lowest loss here. This was not pre-declared and is a candidate for a proper test.

Limitations:
- One workload, 500 steps, three seeds; settings chosen on seed 0.
- The decay length (100 steps) and the EMA decays come from small grids.
- The cosine arm for SOAP chose the largest peak tried (3e-2), so it may be under-tuned.

## 6. Reproduction

From `residual-curvature-lab/`, with `export PYTHONDONTWRITEBYTECODE=1`:

```sh
S=studies/noise_controls_20261011/scripts
.venv/bin/python -m pytest -q -p no:cacheprovider studies/noise_controls_20261011/tests studies/parsec_l_20261009/tests
T=$(.venv/bin/python $S/train.py create --phase tune); F=$(.venv/bin/python $S/train.py create --phase eval_fixed)
# queue `train.py pending` units of both runs (queue_runner.sh, 4 workers); orchestrate.sh $T QUEUE creates and queues eval_tuned
.venv/bin/python $S/summarize.py $T $F <eval_tuned run>
```
