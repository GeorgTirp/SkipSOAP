# Top-free PARSEC-L with OT-metric probes and with probes from another batch — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, fixed before any run. No deviations.

External runs `~/skipsoap-study-data/probe_variants_20261013/runs/`:
- tuning over c (seed 0): `20261013T110125-b67b7a86`;
- extension (c = 16384): `20261013T111545-ce0f8e0e`;
- evaluation (seeds 1–3): `20261013T111924-eae7ad48`.

Summary: `results/summary.json`. G1: `results/g1.json`.

**Status (2026-10-13):** all 14 units completed without divergence. G1 passed: the `topfree` units reproduce
`cifar_ggn`'s bitwise.

## 1. Summary

- **P1: OT-metric probes beat GGN probes in every seed at every point.** EMA at step 500: 0.521 vs 0.541 (seed
  means; per seed −0.008, −0.008 and −0.044). Raw at step 50: 1.426 vs 1.547. Raw at step 200: 0.853 vs 0.891.
  Accuracy is higher in all three seeds (0.817 vs 0.809).
  - By the pre-declared rule the reading is *OT is the better curvature source here*. But the comparison is **not
    matched in tuning**: OT's c was tuned on CIFAR (selected c = 4096 from 64 to 16384), while GGN top-free's c = 64
    comes from MNIST and was never tuned on CIFAR.
  - The fairness control `ggn_c_control_20261013` (queued) gives the GGN probes the same c search. Until it reports,
    P1 does not distinguish "the OT metric is better" from "a CIFAR-tuned brake is better".
- **OT needs a far stronger c, as predicted.** At c = 64 OT top-free is poor (seed-0 EMA 0.813). The curve is flat
  from c = 1024 to 16384 (0.515, 0.512, 0.521).
  - The measured scale: the median η·λ_max at probe steps is about 1.1 with OT probes against about 9.7 with GGN
    probes, so OT's curvature is about 9× smaller.
  - At c = 4096 OT's brake is therefore about 64/9 ≈ 7× *stronger* than GGN top-free's at c = 64 (median κ_max
    about 4,500 vs 620). The control will show whether a stronger GGN brake does as well.
- **P2: probes from another batch, *not shown*.** The cross-batch arm is mixed at every point. EMA at step 500 is
  0.537 vs 0.541: lower in seeds 2 and 3, higher in seed 1. The curvature it brakes has the same scale as
  same-batch probes (η·λ_max about 9.7).
- **Cost:** no difference. Median seconds per step: 0.28 (OT), 0.28 (cross-batch), 0.31 (top-free).

## 2. Tuning of `topfree_ot` (seed 0, lr 3e-2)

| c | Raw 500 | EMA 0.9 | EMA 0.98 | EMA 0.995 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.7973 | 0.7970 | 0.8133 | 0.9458 |
| 256 | 0.6029 | 0.5986 | 0.5818 | 0.7026 |
| 1024 | 0.5432 | 0.5356 | 0.5154 | 0.6557 |
| **4096** | 0.5477 | 0.5414 | **0.5117** | 0.6611 |
| 16384 (extension) | 0.5432 | 0.5368 | 0.5213 | 0.6588 |

- **Selection:** c = 4096 with EMA 0.98. The first grid's edge, 4096, became interior after the pre-declared
  extension.
- **Reference on seed 0:** GGN top-free at c = 64, EMA 0.5427 (`cifar_ggn`).

## 3. Evaluation (seeds 1 / 2 / 3, seed mean)

| | raw 50 | raw 200 | EMA 500 | accuracy 500 |
| --- | --- | --- | --- | --- |
| `topfree` (GGN, c = 64) | 1.631 / 1.507 / 1.505 (1.547) | 0.917 / 0.863 / 0.893 (0.891) | 0.540 / 0.519 / 0.563 (0.541) | 0.807 / 0.816 / 0.804 |
| `topfree_ot` (c = 4096) | **1.433 / 1.443 / 1.402 (1.426)** | **0.894 / 0.825 / 0.841 (0.853)** | **0.532 / 0.511 / 0.519 (0.521)** | 0.812 / 0.821 / 0.819 |
| `topfree_xbatch` (GGN, c = 64) | 1.593 / 1.569 / 1.484 (1.549) | 0.923 / 0.851 / 0.889 (0.888) | 0.556 / 0.517 / 0.540 (0.537) | 0.801 / 0.817 / 0.814 |

**Seed-mean raw curve:**

| Step | 10 | 25 | 50 | 100 | 200 | 300 | 400 | 500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `topfree` | 2.007 | 1.814 | 1.547 | 1.169 | 0.891 | 0.743 | 0.648 | 0.567 |
| `topfree_ot` | 1.988 | 1.775 | **1.426** | 1.133 | 0.853 | 0.717 | 0.649 | 0.545 |
| `topfree_xbatch` | 2.033 | 1.791 | 1.549 | 1.173 | 0.888 | 0.735 | 0.657 | 0.563 |

**Readings:**
- **P1 (OT vs GGN):** better at raw 50, raw 200 and EMA 500. As pre-declared, *OT is the better curvature source
  here*, qualified by the tuning confound above.
- **P2 (other batch vs same batch):** mixed, mixed, mixed; *not shown*.

## 4. Interpretation and limitations

- **OT top-free is the best PARSEC-L variant measured so far:** EMA 0.521 against SOAP's 0.554 and top-free's 0.541
  on the same seeds. Its advantage is largest early (raw at step 50: 1.426 vs 1.547 and SOAP's 1.458).
- **Not yet attributable to the metric.** It may come from the metric, from a much stronger brake than GGN top-free
  ever used on CIFAR, or from both. `ggn_c_control_20261013` decides this.
- **The cross-batch idea shows no effect at this scale.** Same-batch vs independent-batch curvature makes no
  consistent difference to top-free's training. That fits the finding (`basis_refresh_snr`, Part C) that top-free's
  brake does not act as a noise filter on the batch.
- **Limitations:**
  - One lr (3e-2), three seeds, 500 steps.
  - c is tuned on seed 0 for OT only.
  - The cross-batch arm uses 2,048 extra examples every 10 steps.
