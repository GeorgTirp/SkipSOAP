# OT-probe top-free PARSEC-L over 1,000 steps, one seed (exploratory) — protocol

Study `ot_1000_20261013`. Written on 13 October 2026, before any run.

**Request.** The owner asked on 13 October, after stopping the PARSEC line: "Okay do 1000 steps now quickly here". This
followed the question of how long a single-seed OT-probe run takes.

**Purpose.** A quick look at whether OT-probe top-free PARSEC-L, the best 500-step variant (`probe_variants`: EMA 0.521
vs top-free 0.541 vs SOAP 0.554), keeps an edge at twice the length. At 2,000 steps plain top-free's edge was gone
(`long_run`). **It is exploratory: one seed, no tuning, no verdict.**

## Runs

All on CIFAR-10 with the M1d6 ViT and batch 2048, GPU float32 (as `cifar_ggn_20261012`), **seed 1**:
- **Length:** 1,000 steps.
- **Schedule:** warm-up over steps 1–50, constant, linear decay over steps 801–1000 to 1%. These are the same
  proportions as the 500- and 2,000-step runs.
- **Regularization:** weight decay 0.05; EMAs d ∈ {0.98, 0.995, 0.998}.

| Arm | Settings |
| --- | --- |
| `soap` | SOAP, lr 3e-2 (at 500 steps on seed 0, 3e-2 tied SOAP's best, 0.5432 vs 0.5430; at 2,000 steps SOAP preferred ≥ 9e-2) |
| `topfree` | top-free PARSEC-L with GGN probes, c = 64, lr 3e-2 |
| `topfree_ot` | top-free PARSEC-L with OT-metric probes, c = 4096 (the `probe_variants` selection), lr 3e-2 |

The runner is `long_run_20261013`'s with the schedule constants changed. `topfree_ot` uses `probe_variants`' OT probes
(`probes_ot.py`, copied unchanged) and the arm's own c.

## Reported (descriptive only)

- The validation loss at step 1000: raw, and the EMA at every decay. Each arm's best decay is reported, but nothing
  is selected.
- The raw curve at steps 100, 200, 500, 800 and 1000.
- Accuracy and seconds per step.

No better/worse reading is declared: a single seed cannot separate effects of the size seen so far (0.005–0.02)
from seed noise.

## Gates and hygiene

- **G0:** suites before and after, including this study's tests:
  - the schedule;
  - the `soap` and `topfree` arms equal `cifar_ggn`'s over CPU steps under the same schedule factor;
  - `topfree_ot` equals `probe_variants`' arm over CPU steps.
- **Writes:** only to `studies/ot_1000_20261013/` and `~/skipsoap-study-data/ot_1000_20261013/`.
- **Running:** detached from the session (`setsid nohup`). Cost about 16 GPU-minutes.
