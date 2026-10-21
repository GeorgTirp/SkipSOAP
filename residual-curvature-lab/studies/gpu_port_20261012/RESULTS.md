# Does PARSEC-L give the same results on the GPU, and what does float32 change? — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, approved and fixed before any run.
Deviations: [DEVIATIONS.md](DEVIATIONS.md) (D1: a replay print bug, fixed before any replay output; D2: two cost runs).

External runs `~/skipsoap-study-data/gpu_port_20261012/runs/`:
- MNIST units and replays: `20261012T084322-39ae45d3`;
- cost C1: `20261012T085621-870cc249`, with the repetition `20261012T085237-8f184eba`.

Light mirrors are in `runs/`. Summary: `results/summary.json`.

**Status (2026-10-12):** all 24 units completed without divergence. All gates passed, R1 passed, and float32 is
adequate by the pre-declared rule.

## 1. Summary

- **The port is exact where it can be.**
  - On the CPU the device-ready copy reproduces the original bitwise (G1).
  - GPU runs repeat bitwise in float64 and float32 (G2).
  - A saved state replays bitwise on the CPU (G3).
  - One GPU float64 step from an identical state agrees with the CPU to at most 3.3e-10 relative (R1, gate 1e-8),
    on plain and probe steps.
- **Trajectories are chaotic at the 1e-3 level.** GPU and CPU float64 runs from identical starts agree to about 1e-8
  at step 10, then separate. They differ by about 1e-4 at steps 25–50 and by 0.0025 (SOAP) in step-500 EMA loss.
  - A seed is SOAP's float32 basis refresh: across devices the bases differ by 3e-6 to 1.2e-5 at every refresh, as
    predicted in the protocol.
  - Float32 runs differ from float64 runs by 0.002–0.008, and this machine's float64 runs differ from the laptop's
    by up to 0.0047 (EMA, step 500).
  - Single-run differences below about 0.008 at step 500 are therefore not interpretable. That is the size of the
    smallest per-seed margin in the top-free confirmation.
- **The top-free result replicates on this machine and in float32 (R3(b)).** Top-free c = 64 + EMA is below SOAP +
  EMA at step 500 in every fresh seed:

  | EMA d = 0.98, step 500 | Seed 4 | Seed 5 | Seed 6 | Mean |
  | --- | ---: | ---: | ---: | ---: |
  | SOAP, CPU float64 (this machine) | 0.1373 | 0.1302 | 0.1311 | 0.1329 |
  | Top-free, CPU float64 (this machine) | **0.1307** | **0.1176** | **0.1216** | **0.1233** |
  | SOAP, GPU float32 | 0.1401 | 0.1340 | 0.1352 | 0.1364 |
  | Top-free, GPU float32 | **0.1339** | **0.1203** | **0.1138** | **0.1227** |
  | Laptop (top-free confirmation): SOAP / top-free | 0.136 / 0.129 | 0.134 / 0.114 | 0.135 / 0.119 | 0.135 / 0.121 |

- **Float32 verdict: adequate.** R3(b) holds. R3(a) fails as a 1e-3 criterion: the eight float32-vs-float64
  differences are 0.0023–0.0078. It is excused by the pre-declared rule, because the float64 device floor from R2
  (0.0025) exceeds 1e-3. In plain terms, float32 changes single runs about as much as changing the device does. It
  does not change the sign of the established effect in any seed.
- **Two findings about the code (pre-run tests):**
  - The attention key bias has an identically zero gradient, because softmax ignores a common shift of each row's
    logits. Its gradient and probes are rounding noise. In float32 that noise is about SOAP's ε, so these
    coordinates can drift by random steps without changing the loss.
  - Singular-vector signs differ between LAPACK and cuSOLVER. The held PARSEC-L directions therefore differ in
    sign, while the operator agrees to 1e-14.

**Next step:** the CIFAR-10 study (owner decisions of 12 October): exact per-tensor GGN with and without SN, tuned
SOAP and top-free PARSEC-L. It uses the depth-6 M1 model, batch 2048, float32 on the GPU, and needs a memory-lean
probe path (C1).

## 2. Gates and R1

| Gate | Result |
| --- | --- |
| G0 suites | repository 106, PARSEC-L 22, study 28: passed before and after |
| G1 copy = original, CPU float64, seed 0 | bitwise for `soap` and `topfree` (all validation values, final parameters) |
| G2 GPU repeatability | bitwise: `soap` and `topfree`, float64 and float32 |
| G3 CPU replay | 72 replayed steps, update error 0.0, state differences 0.0 |
| R1 one GPU float64 step | max relative update error 1.4e-10 (`soap`, plain), 3.3e-10 (`topfree`, plain), 3.1e-10 (`topfree`, probe); gate 1e-8 |

After non-refresh steps the moments agree to at most 6.7e-16 and the held operator W̃ᵀW̃ to 3.8e-12. After the 10
steps with a SOAP basis refresh, the bases differ by 3.1e-6 to 1.2e-5: float32 rounding, as predicted.

## 3. Trajectories (R0, R2)

**R2, GPU float64 minus CPU float64, seed 0** (validation loss, raw / EMA 0.98):

| Step | 10 | 25 | 50 | 100 | 200 | 500 |
| --- | --- | --- | --- | --- | --- | --- |
| `soap` | 1.5e-8 / 4.0e-9 | −5.2e-6 / −7.4e-5 | 2.1e-4 / −1.3e-4 | −5.6e-4 / 1.9e-4 | 4.2e-3 / 1.1e-3 | 8.3e-3 / 2.5e-3 |
| `topfree` | 1.4e-5 / −7.9e-8 | −8.1e-4 / −2.6e-5 | −2.3e-3 / −1.7e-3 | 6.8e-4 / −1.8e-3 | −2.2e-4 / −9.9e-4 | −3.5e-3 / −3.7e-4 |

Device floor (largest step-500 EMA difference): 0.0025.

**R0, this machine's CPU float64 against the laptop** (step 500, raw / EMA; laptop values from 4-decimal logs):

| | Seed 0 | Seed 4 | Seed 5 | Seed 6 |
| --- | --- | --- | --- | --- |
| `soap` here / laptop | 0.1646 / 0.1344 vs 0.1713 / 0.1359 | 0.1647 / 0.1373 vs 0.1645 / 0.1360 | 0.1606 / 0.1302 vs 0.1699 / 0.1345 | 0.1571 / 0.1311 vs 0.1658 / 0.1346 |
| `topfree` here / laptop | 0.1409 / 0.1153 vs 0.1385 / 0.1124 | 0.1526 / 0.1307 vs 0.1501 / 0.1292 | 0.1370 / 0.1176 vs 0.1356 / 0.1141 | 0.1416 / 0.1216 vs 0.1397 / 0.1195 |

The largest difference over all validation steps is 0.023 raw and 0.0047 on the EMA.

## 4. R3 (float32)

(a) |GPU float32 − CPU float64|, EMA 0.98, step 500:
- `soap`: 0.0078 / 0.0028 / 0.0038 / 0.0041 (seeds 0/4/5/6);
- `topfree`: 0.0023 / 0.0032 / 0.0027 / 0.0078.

None is at most 1e-3. (a) is excused because the device floor 0.0025 > 1e-3. (b) holds in all three seeds (table in
section 1). **Verdict: adequate.**

## 5. Cost C1 (GPU float32, batch 512, 60 timed steps)

| Model | Optimizer | Parameters | s/step, plain steps | s/step, probe steps | Mean s/step | Peak GiB | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| M1 (w64, d4) | AdamW | 208,074 | 0.033 | — | 0.034 | 1.28 | completed |
| M1 | SOAP (upstream) | 208,074 | 0.036 | — | 0.039 | 1.29 | completed |
| M1 | SOAP (PARSEC-L, rank 0) | 208,074 | 0.041 | — | 0.046 | 1.29 | completed |
| M1 | top-free, s = 1 | 208,074 | 0.087 | 0.357 | 0.116 | 1.88 | completed |
| M1 | top-free, s = 4 | 208,074 | 0.119 | — | — | 5.69 | out of memory at step 12 |
| M2 (w128, d6) | AdamW | 1,205,898 | 0.072 | — | 0.072 | 2.46 | completed |
| M2 | SOAP (upstream) | 1,205,898 | 0.078 | — | 0.086 | 2.49 | completed |
| M2 | SOAP (PARSEC-L, rank 0) | 1,205,898 | 0.088 | — | 0.099 | 2.51 | completed |
| M2 | top-free, s = 1 / s = 4 | 1,205,898 | — | — | — | 5.95 / 6.29 | out of memory at step 2 |

- The PARSEC-L wrapper costs 14–16% over upstream SOAP even at rank 0. Top-free costs about 3× SOAP on M1:
  - non-probe steps are about 2× SOAP's (the per-tensor shrinkage loop);
  - probe steps take 0.36 s.
- The probe path needs several times the raw probe storage n × D. The probe rows are collected in chunks and then
  concatenated, and batched extraction stacks tensors of equal shape. Top-free therefore runs out of memory at sizes
  where the raw probes would fit:
  - M1 at s = 4: 1.7 GB of probes;
  - M2 at s = 1: 2.5 GB of probes.
- The repetition `20261012T085237-8f184eba` agrees within 2.5% per cell, with identical peak memory (D2).

## 6. Limitations

- One workload (the 5.7k-parameter MNIST ViT) for R0–R3. Four seeds for R3; seed 0 only for R2.
- R3(a) as a fixed 1e-3 criterion does not hold. The adequacy verdict rests on the sign of the effect in three seeds
  and on the device floor.
- C1 measures 60 steps at batch 512 with lr 1e-3; the step times do not include validation or diagnostics.

## 7. Isolation and suites

- **Isolation:** the manifest outside this folder is identical before and after (863 files). No `__pycache__`
  outside `.venv`. The venv, MNIST and CIFAR-10 were set up with the owner's permission before the study
  (`.venv/`, `datasets/`; both excluded from manifests and git).
- **Insight bank:** `ideas/parsec.md` was updated after the after-manifest (rule 5: section 1 entry, rule 11, next
  test, changelog). It is the only file outside this folder changed by the study.
- **Suites:** repository 106 passed and PARSEC-L 22 passed, before and after. Study tests 28 passed before and
  after.
