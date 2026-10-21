# N9 in training: SOAP and top-free PARSEC-L with bases from the probe GGN — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, fixed before any run. No deviations.

External runs `~/skipsoap-study-data/n9_training_20261013/runs/`:
- tuning (seed 0): `20261013T093921-70a35c6e`;
- extension: `20261013T100613-2b55bb1b`;
- evaluation (seeds 1–3): `20261013T101535-b93c206f`.

Summary: `results/summary.json`. G1: `results/g1.json`.

**Status (2026-10-13):** all 27 units completed without divergence.
- **G1 passed:** the `soap` and `topfree` units reproduce `cifar_ggn`'s bitwise in every non-timing field of
  `validation.json` and `training.json`.
- **G2 passed at all 450 basis rebuilds:**
  - momentum preserved to at most 3.2e-7 relative;
  - the second-moment sum preserved to at most 1.1e-6;
  - every second moment ≥ 0.
- **Cost:** a rebuild costs 0.10 s (`soap_eigh`) or 0.17–0.18 s (probe arms) every 10 steps. The median step time
  is unchanged (0.24 s for the SOAP arms, 0.28 s for the top-free arms).

## 1. Summary

- **N9 does not improve SOAP (N1: *not shown*).**
  - With bases rebuilt from the probe factors, SOAP selects lr 3e-2, three times its own, and is *worse* raw at
    steps 25, 50 and 200 in every seed.
  - At step 500 its EMA loss is mixed: 0.563 vs 0.554 (seed means). It is lower only in seed 2.
  - At SOAP's own lr 1e-2 on seed 0 it was also worse: 0.567 vs 0.543.
- **Rebuilding from SOAP's own factors behaves the same (N2: mixed at every point; N4: worse early, mixed at 500).**
  The control `soap_eigh` uses exact eigenvectors of SOAP's own factors on the same schedule. The probe source makes
  no consistent difference for SOAP.
- **Under top-free PARSEC-L, N9 helps modestly (N3).**
  - `topfree_n9` is better raw at steps 25 and 200 in all three seeds, and mixed at step 50.
  - At EMA step 500 it is mixed: lower in seeds 1 and 3, higher in seed 2 by 0.003. Seed means 0.531 vs 0.541.
  - Descriptive: raw loss at step 500 is 0.545 vs 0.567; accuracy at step 500 is higher in every seed (0.817,
    0.822 and 0.818 vs 0.807, 0.816 and 0.804).
  - Against SOAP + EMA, `topfree_n9` + EMA is lower in every seed (0.538, 0.522 and 0.532 vs 0.547, 0.538 and
    0.575). This is descriptive; it was not a pre-declared reading.
  - On tuning seed 0: 0.526 vs top-free's 0.543.
- **Offline vs training.** The offline advantage of the probe basis did not transfer to SOAP training. Offline, it
  gave a better diagonal *curvature* model (`basis_refresh_snr`, A1 and A2, 9 of 9). But SOAP's diagonal is
  Adam's second moment of batch gradients, not a curvature diagonal. Rebuilding the basis also forces those
  second moments to be carried across large basis changes, under an assumption (Σ diagonal in the old basis) that a
  large move breaks. The lr shift to 3e-2 in both rebuild arms points the same way: the rebuild changes SOAP's
  effective step scale.

## 2. Tuning (seed 0)

EMA 0.98 validation loss at step 500 (the selection's decay in every case):

| lr | `soap_eigh` | `soap_n9` | `topfree_n9` |
| ---: | ---: | ---: | ---: |
| 3e-3 | 0.6673 | 0.6898 | 0.6632 |
| 1e-2 | 0.5480 | 0.5674 | 0.5509 |
| **3e-2** | **0.5455** | **0.5487** | **0.5255** |
| 9e-2 (extension) | 0.8702 | 0.8688 | 0.6763 |

- **Reference on seed 0** (`cifar_ggn`): SOAP 0.5430 at lr 1e-2; top-free 0.5427 at lr 3e-2.
- **Edge:** all three arms selected the grid's top edge. The pre-declared extension to 9e-2 was much worse each
  time, so 3e-2 stands.

## 3. Evaluation (seeds 1 / 2 / 3, seed mean)

| Validation loss | raw 25 | raw 50 | raw 200 | EMA 500 |
| --- | --- | --- | --- | --- |
| `soap` (1e-2) | 1.709 / 1.748 / 1.733 (1.730) | 1.434 / 1.450 / 1.490 (1.458) | 0.818 / 0.792 / 0.868 (0.826) | 0.547 / 0.538 / 0.575 (0.554) |
| `soap_eigh` (3e-2) | 1.815 / 1.871 / 1.776 (1.821) | 1.631 / 1.587 / 1.536 (1.585) | 0.993 / 0.893 / 0.989 (0.959) | 0.581 / 0.526 / 0.575 (0.560) |
| `soap_n9` (3e-2) | 1.820 / 1.839 / 1.812 (1.824) | 1.533 / 1.560 / 1.562 (1.551) | 0.965 / 0.900 / 0.932 (0.932) | 0.575 / 0.528 / 0.586 (0.563) |
| `topfree` (3e-2) | 1.849 / 1.815 / 1.779 (1.814) | 1.631 / 1.507 / 1.505 (1.547) | 0.917 / 0.863 / 0.893 (0.891) | 0.540 / 0.519 / 0.563 (0.541) |
| `topfree_n9` (3e-2) | **1.758 / 1.807 / 1.750 (1.771)** | 1.501 / 1.574 / 1.505 (1.527) | **0.850 / 0.837 / 0.837 (0.841)** | 0.538 / 0.522 / 0.532 (0.531) |

**Readings:**

| Code | raw 25 | raw 50 | raw 200 | EMA 500 | Verdict |
| --- | --- | --- | --- | --- | --- |
| N1 `soap_n9` vs `soap` | worse | worse | worse | mixed | *not shown* |
| N2 `soap_n9` vs `soap_eigh` | mixed | mixed | mixed | mixed | *not shown* |
| N3 `topfree_n9` vs `topfree` | **better** | mixed | **better** | mixed (2 of 3) | — |
| N4 `soap_eigh` vs `soap` | worse | worse | worse | mixed | — |

**Descriptive:**
- **Steps to SOAP's seed-mean raw loss at step 50 (1.458):** 50–75 for SOAP, 75 for every other arm.
- **Steps to SOAP's seed-mean raw loss at step 200 (0.826):** SOAP 200–225; `topfree_n9` 225; `topfree` 225–250;
  `soap_n9` 250–275; `soap_eigh` 275–325.
- **Seed-mean raw loss at step 500:** SOAP 0.588, `soap_eigh` 0.579, `soap_n9` 0.583, `topfree` 0.567, `topfree_n9`
  0.545.

## 4. Interpretation and limitations

- **For SOAP, N9 is closed.** Neither the probe source nor exact rebuilding of SOAP's own factors helps it at 500
  steps. Both make it slower early, partly because the rebuild arms run at 3× SOAP's lr.
- **For top-free PARSEC-L, N9 is a modest candidate.**
  - **Evidence:** consistently better mid-run raw loss, better final raw loss and accuracy, and an EMA loss lower
    in 2 of 3 seeds. Both arms run at the same lr, so this comparison has no lr confound.
  - **Hypothesis, not tested:** top-free's brake works in SOAP's whitened coordinates. A basis aligned with the
    probe curvature may let the 16 held directions capture more of the curvature that matters.
  - **What it would take:** more seeds or a longer run to settle the EMA reading.
- **Limitations:**
  - **Scale:** three seeds; 500 steps; a 3 + 1 point lr grid on one seed.
  - **Approximate carry:** the second-moment carry-over is an approximation that this study does not test
    separately. A variant that resets or re-estimates v after a rebuild was not run.
  - **Rebuild schedule:** both rebuild arms rebuild on the top-free probe schedule from step 2, rather than at
    SOAP's steps 11, 21, …
