# Fairness control for the OT probes: top-free PARSEC-L with GGN probes and a CIFAR-tuned c — protocol

Study `ggn_c_control_20261013`. Written on 13 October 2026, before any run.

**Request.** The owner asked on 13 October: "Okay yes queue that". This was in reply to the proposal: "GGN top-free with c
tuned on CIFAR the same way, so the OT result can be judged fairly."

**Context.** In `probe_variants_20261013`, top-free PARSEC-L with OT-metric probes, c = 4096 selected on seed 0 from
{64, …, 16384}, beat top-free with GGN probes at c = 64 in every seed (EMA at step 500: 0.521 vs 0.541; P1). But
GGN top-free's c = 64 was chosen on MNIST and never tuned on CIFAR. This study gives the GGN probes the same
seed-0 c search.

## 1. Questions

1. With c tuned on CIFAR for both, is top-free PARSEC-L with OT probes still better than with GGN probes?
2. Does tuning c on CIFAR improve GGN top-free over c = 64?

## 2. Workload and arms

The runner, workload and schedule are `probe_variants_20261013`'s: CIFAR-10, M1d6, batch 2048, 500 steps, lr 3e-2,
rank 16, `top_rhs='gradient'`, probes at steps 2, 12, 22, …

| Arm | Probes | c |
| --- | --- | --- |
| `topfree_c` | sampled-label GGN probes of the step's batch (`micro.lean_probes`), as `topfree` | tuned on seed 0 over {64, 256, 1024, 4096}; one extension beyond a selected edge (16 or 16384) |
| `topfree_ot` | OT-metric probes (`probe_variants`' `probes_ot.py`) | 4096, `probe_variants`' selection (not retuned) |

- **Tuning selection:** the lowest step-500 EMA validation loss over decays {0.9, 0.98, 0.995}, as in
  `probe_variants`.
- **Evaluation:** seeds 1–3, both arms (6 units).
- **Reference, not rerun:** `topfree` (c = 64), seeds 1–3, from `probe_variants` and `cifar_ggn` (identical
  bitwise).

## 3. Gates

- **G0:** suites before and after (all studies' tests, including this study's: the unit plan; `topfree_c` at
  c = 64 equal to `cifar_ggn`'s `topfree` over CPU steps; `topfree_ot` equal to `probe_variants`' arm over CPU
  steps).
- **G1:**
  - the `topfree_ot` evaluation units reproduce `probe_variants`' (`20261013T111924-eae7ad48`) bitwise in
    `validation.json` and `training.json`, timings excluded;
  - the `topfree_c` c = 64 tuning unit reproduces `cifar_ggn`'s tuning unit `topfree__lr0.03__s0`
    (`20261012T103803-7d7d6274`) bitwise in `validation.json`. Its `training.json` carries an extra logged field.

## 4. Pre-declared readings (seeds 1–3)

*Better* or *worse* means lower or higher validation loss in all three seeds; otherwise *mixed*.

| Code | Reading |
| --- | --- |
| C1 | **OT vs GGN at matched tuning:** `topfree_ot` vs `topfree_c`, raw at steps 50 and 200, EMA at 500. *OT is the better curvature source* if better at EMA 500. Otherwise P1's advantage is attributed to tuning c, and the OT line is closed for PARSEC-L. |
| C2 | **CIFAR-tuned c for GGN:** `topfree_c` vs `topfree` (c = 64), same points. |
| C3 | Descriptive: the tuning table over c; seed-mean curves; accuracy; κ statistics at probe steps. |

## 5. Cost, isolation and limitations

- **Cost:** about 4 min per unit, 10–11 units, so about 45 GPU-minutes. It runs after `long_run_20261013`.
- **Writes:** only to `studies/ggn_c_control_20261013/` and `~/skipsoap-study-data/ggn_c_control_20261013/`. Hygiene
  as before.
- **Limitations:**
  - One lr (3e-2, top-free's) for both.
  - c is tuned on one seed.
  - Three seeds, 500 steps.
