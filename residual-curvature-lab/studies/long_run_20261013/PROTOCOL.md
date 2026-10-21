# Top-free PARSEC-L vs SOAP over 2,000 steps on CIFAR-10 — protocol

Study `long_run_20261013`. Written on 13 October 2026, before any run.

**Amended before any run** (13 October, owner decision after `n9_training_20261013`): a third arm, `topfree_n9`,
was added. The first version's SHA is kept in `logs/protocol_sha256.txt`.

**Request.** The owner asked on 13 October: "then do a longer top-free run with PARSEC-L and SOAP and compare them."
Top-free PARSEC-L beat SOAP + EMA in every seed of three tests, all at 500 steps: MNIST, its GPU replication, and
CIFAR (0.541 vs 0.554). This study asks whether the gain holds, grows or vanishes over a run four times as long.
It uses more seeds, because the CIFAR margins (0.007–0.019) are near the MNIST run-to-run floor.

## 1. Questions

1. Over 2,000 steps, does top-free PARSEC-L + EMA still beat SOAP + EMA?
2. How does the gap develop along the run (steps 500, 1000, 1500, 2000)?
3. Does top-free PARSEC-L with N9 bases beat plain top-free over the longer run? At 500 steps its EMA loss was lower
   in 2 of 3 seeds, and its raw loss at steps 25 and 200 was lower in every seed.

## 2. Workload

`cifar_ggn_20261012`'s CIFAR workload with a longer schedule:
- **Data:** CIFAR-10, 45k/5k split of the training set (the test split is never accessed); crop + flip.
- **Model:** M1d6 ViT; batch 2048; GPU float32, deterministic.
- **Length:** 2,000 steps, about 91 epochs (the 500-step runs were about 23).
- **Schedule:** linear warm-up over steps 1–100, constant, linear decay over steps 1601–2000 to 1% (WSD; the
  same proportions as the 500-step runs: 5% warm-up, 20% decay).
- **Regularization:** decoupled weight decay 0.05; weight EMAs d ∈ {0.98, 0.995, 0.998}.
- **Runner:** `n9_training_20261013`'s, with its `soap` and `topfree` paths, only the schedule constants changed.

## 3. Arms and tuning

| Arm | Settings |
| --- | --- |
| `soap` | SOAP (betas 0.95/0.95, refresh every 10 steps) |
| `topfree` | top-free PARSEC-L (rank 16, c = 64, `top_rhs='gradient'`, lr calibration, probes every 10 steps from step 2) |
| `topfree_n9` | `topfree` with SOAP's bases rebuilt after the step at every probe step from the probe factors, with momentum and second moments carried, and upstream's QR refresh off (`n9_training_20261013`'s arm and `n9.py`, with gate G2) |

- **Tuning:** all three arms on seed 0 over lr ∈ {3e-3, 1e-2, 3e-2}. A longer run may prefer a smaller lr, so no
  arm keeps its 500-step lr.
  - Selection: the lowest step-2000 EMA validation loss over the decays.
  - If the selected lr is at a grid edge, one extension unit at ×3 or ÷3 is run, and the selection is repeated.
- **Evaluation:** seeds 1–5, all three arms at their selected lrs (15 units).
- **Validation:** at steps 0, 10, 25, 50, 100, 200, …, 2000, raw and EMA.

## 4. Gates

- **G0:** suites before and after:
  - the repository, PARSEC-L, `cifar_ggn`, `cifar_sn_soap`, `basis_refresh_snr`, `n9_training`,
    `probe_variants`, `ladder_mechanism`;
  - this study's tests: unit plan and schedule; `soap` and `topfree` equal `cifar_ggn`'s over CPU steps when given
    the same schedule factor; `topfree_n9` rebuilds at step 2.
- **G2:** at every rebuild of `topfree_n9`, momentum and the second-moment sum are preserved to 1e-4 relative, and
  v′ ≥ 0. A failure stops the unit.
- **Divergence** is reported, not a gate failure, and counts as worse.

## 5. Pre-declared readings (seeds 1–5)

| Code | Reading |
| --- | --- |
| R1 | **Main:** `topfree` + EMA vs `soap` + EMA at step 2000, each with its selected decay. *Better* if lower in all 5 seeds, *worse* if higher in all 5, otherwise *mixed*. The number of seeds in which it is lower and the mean difference are reported. |
| R2 | **Raw weights at step 2000:** same comparison. |
| R3 | **The gap along the run (descriptive):** per seed, the EMA (selected decay) and raw differences at steps 500, 1000, 1500 and 2000. |
| R4 | Descriptive: tuning table, seed-mean curves, accuracy, seconds per step, and the comparison with the 500-step result (0.541 vs 0.554). |
| R5 | **N9 under top-free:** `topfree_n9` + EMA vs `topfree` + EMA at step 2000 (better or worse only if all 5 seeds agree, otherwise mixed; seeds lower and mean difference reported); the same for raw weights at step 2000 and for `topfree_n9` vs `soap`; the differences along the run as in R3. |

## 6. Cost, isolation and limitations

- **Cost:** `soap` about 8 min per unit, `topfree` and `topfree_n9` about 14 min. Tuning about 1.8 h, evaluation
  about 3 h. **About 5 GPU-hours.**
- **Writes:** only to `studies/long_run_20261013/` and `~/skipsoap-study-data/long_run_20261013/`. Hygiene as before.
- **Limitations:**
  - One model and one dataset; 2,000 steps are still short of a converged ViT.
  - Tuning uses a three-point grid on one seed.
  - The schedule's proportions are copied from the 500-step runs, not tuned.
