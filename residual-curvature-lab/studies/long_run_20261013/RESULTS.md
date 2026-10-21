# Top-free PARSEC-L vs SOAP over 2,000 steps on CIFAR-10 — results

Protocol: [PROTOCOL.md](PROTOCOL.md), amended before any run to add `topfree_n9`; both SHA-256 values are in
`logs/protocol_sha256.txt`. Deviations: [DEVIATIONS.md](DEVIATIONS.md):
- D1: the evaluation was interrupted by the end of a session and resumed bitwise-equivalently.
- D2: the owner's stop, which came after all units had completed.

External runs `~/skipsoap-study-data/long_run_20261013/runs/`:
- tuning (seed 0): `20261013T120559-5d587678`;
- extension (lr 9e-2): `20261013T134808-5bd891bd`;
- evaluation (seeds 1–5): `20261013T142233-20ef534e`.

Summary: `results/summary.json`.

**Status (2026-10-13):** all 27 units completed without divergence. G2 held at every `topfree_n9` rebuild.

## 1. Summary

- **Over 2,000 steps, top-free PARSEC-L no longer beats SOAP (R1: *mixed*, leaning to SOAP).**
  - EMA validation loss at step 2000, each arm with its selected decay: 0.462 vs 0.457 (seed means).
  - Top-free is lower in only 1 of 5 seeds, by 0.027 in seed 4. It is higher by 0.008–0.020 in the other four.
- **Raw weights at step 2000 (R2: *mixed*):** 0.488 vs 0.489. Lower in 2 of 5; mean difference −0.001.
- **Accuracy is the one consistent difference:** top-free is higher in all 5 seeds, 0.847 vs 0.844.
- **N9 under top-free (R5): mixed everywhere.** EMA 0.460 vs top-free's 0.462 (lower in 2 of 5) and vs SOAP's 0.457
  (lower in 2 of 5).
- **Along the run (raw seed means):** top-free and top-free + N9 lead SOAP mid-run, and the lead is gone by step
  1500.

  | Step | 200 | 500 | 1000 | 1500 | 2000 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | SOAP | 1.126 | 0.844 | 0.738 | 0.717 | 0.489 |
  | top-free | 1.090 | 0.761 | 0.691 | 0.726 | 0.488 |
  | top-free + N9 | 1.055 | 0.756 | 0.792 | 0.751 | 0.490 |

  In the constant-lr phase the raw losses are noisy at lr 9e-2. The EMA differences at intermediate steps mix
  decays (0.98 vs 0.995) and are not comparable; they are in `summary.json`.
- **The learning-rate optimum moved and was not bracketed.** At 2,000 steps every arm selected lr 9e-2, the top of
  the grid after the one pre-declared extension. At 500 steps the selections were 1e-2 (SOAP) and 3e-2 (top-free).
  All three arms are therefore compared at a common, possibly suboptimal lr.
- **Conclusion:** the 500-step advantage of top-free PARSEC-L on CIFAR does not survive a run four times as long
  with a retuned lr. Whatever top-free does, it does not change where SOAP ends up. It changes how fast the raw
  iterate gets there in the middle of the run.

## 2. Tuning (seed 0, best EMA over decays 0.98, 0.995 and 0.998, at step 2000)

| lr | SOAP | top-free | top-free + N9 |
| ---: | ---: | ---: | ---: |
| 3e-3 | 0.819 | 0.871 | 0.824 |
| 1e-2 | 0.657 | 0.663 | 0.658 |
| 3e-2 | 0.471 | 0.551 | 0.596 |
| **9e-2** (extension) | **0.454** (d 0.98) | **0.465** (d 0.995) | **0.461** (d 0.98) |

SOAP is lowest at every lr on seed 0. Every arm's selection lies at the grid's top edge.

## 3. Evaluation (seeds 1–5, lr 9e-2)

| Step 2000 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SOAP, EMA 0.98 | **0.451** | **0.453** | **0.461** | 0.469 | **0.451** | **0.457** |
| top-free, EMA 0.995 | 0.461 | 0.473 | 0.470 | **0.442** | 0.462 | 0.462 |
| top-free + N9, EMA 0.98 | 0.460 | 0.444 | 0.475 | 0.449 | 0.468 | 0.460 |
| SOAP, raw | 0.481 | 0.485 | 0.488 | 0.503 | 0.488 | 0.489 |
| top-free, raw | 0.492 | 0.484 | 0.494 | 0.480 | 0.489 | 0.488 |
| SOAP accuracy | 0.844 | 0.848 | 0.844 | 0.843 | 0.839 | 0.844 |
| top-free accuracy | 0.846 | 0.851 | 0.848 | 0.848 | 0.843 | 0.847 |

**Readings:**
- **R1:** *mixed*. Top-free lower in 1 of 5; mean difference +0.0045.
- **R2:** *mixed*. Lower in 2 of 5; mean difference −0.0012.
- **R5:**
  - top-free + N9 vs top-free: EMA mixed (2 of 5, −0.0023), raw mixed (2 of 5);
  - top-free + N9 vs SOAP: EMA mixed (2 of 5, +0.0022), raw mixed (2 of 5).
- **Cost:** seconds per step are 0.22 for SOAP and 0.29 for both top-free arms.

## 4. Limitations

- **lr not bracketed:** the lr optimum lies at or above 9e-2 for every arm. An arm whose optimum lies further out could
  be disadvantaged at the common lr.
- **Recipe:** one dataset, one model and one schedule shape. The decay and warm-up proportions were copied from the
  500-step runs.
- **Not tested:** the OT-probe variant of top-free (`probe_variants`, N23) was not run at 2,000 steps. Its fairness
  control (`ggn_c_control_20261013`) was cancelled before running.
