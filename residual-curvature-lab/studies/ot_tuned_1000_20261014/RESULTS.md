# Tuned OT-probe top-free PARSEC-L vs tuned SOAP over 1,000 steps on CIFAR-10 — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, fixed before any run. No deviations.

External runs `~/skipsoap-study-data/ot_tuned_1000_20261014/runs/`:
- tuning (seed 0): `20261014T154042-ade2ef51`;
- extensions: `20261014T161902-9f733a55`;
- evaluation (seeds 1–3): `20261014T162934-085e376e`.

Summary: `results/summary.json`.

**Status (2026-10-14):** all 15 units completed. Suites passed before and after (12 suites). The isolation manifest
is unchanged.

## 1. Summary

- **T1: tuned OT-probe top-free beats tuned SOAP in every seed.**
  - EMA validation loss at step 1000, each arm with its selected lr and decay: 0.460 vs 0.483 (seed means). Per
    seed the differences are −0.033, −0.007 and −0.027.
  - Accuracy is higher in all three seeds (0.841 vs 0.828).
- **T2: raw weights at step 1000 are also better in every seed:** 0.484 vs 0.608.
- **The two arms selected different lrs.**
  - SOAP selected lr 3e-2 (with EMA 0.995); OT selected lr 9e-2 and c = 4096 (EMA 0.98). Both selections are
    interior after one extension each (SOAP 1e-2; OT 0.27, which was much worse).
  - On tuning seed 0, at **matched lr**:
    - **at 3e-2:** SOAP is better (0.475 vs 0.483);
    - **at 9e-2:** OT is far better (0.468 vs 0.544).
  - So OT top-free's advantage is that it *uses* a 3× larger lr productively over 1,000 steps, while SOAP at that lr
    gets a worse EMA.
- **Along the run (raw seed means), OT's raw iterate is behind until the decay phase.** That is consistent with its
  larger lr:
  - step 500: 0.726 vs 0.642;
  - step 800: 0.687 vs 0.620;
  - step 1000: 0.484 vs 0.608.

  During the decay (steps 801–1000), OT's raw loss drops by 0.20 and SOAP's by 0.01.

## 2. Tuning (seed 0; best EMA over decays 0.98, 0.995, 0.998, at step 1000)

| lr | SOAP | OT, c = 1024 | OT, c = 4096 |
| ---: | ---: | ---: | ---: |
| 1e-2 (extension) | 0.556 | — | — |
| 3e-2 | **0.475** (d 0.995) | 0.500 | 0.483 |
| 9e-2 | 0.544 | 0.469 | **0.468** (d 0.98) |
| 0.27 | 1.776 | — | 1.943 (extension) |

## 3. Evaluation (seeds 1 / 2 / 3, seed mean)

| | EMA 1000 (selected decay) | raw 1000 | raw 500 | raw 200 | accuracy 1000 |
| --- | --- | --- | --- | --- | --- |
| SOAP (lr 3e-2, d 0.995) | 0.494 / 0.463 / 0.490 (0.483) | 0.619 / 0.587 / 0.618 (0.608) | 0.652 / 0.624 / 0.650 (0.642) | 0.902 / 0.860 / 0.909 (0.890) | 0.826 / 0.833 / 0.826 |
| OT top-free (lr 9e-2, c 4096, d 0.98) | **0.461 / 0.456 / 0.463 (0.460)** | **0.486 / 0.480 / 0.487 (0.484)** | 0.731 / 0.701 / 0.746 (0.726) | 1.053 / 1.012 / 1.083 (1.049) | **0.839 / 0.842 / 0.842** |

**Readings:**
- **T1:** *better* in all seeds; *OT top-free beats SOAP*.
- **T2:** *better* in all seeds.
- **Cost:** 0.28 s per step for OT top-free against 0.22 s for SOAP (1.3×).

## 4. Interpretation and limitations

- **The strongest PARSEC-L result so far:** a tuned-vs-tuned win in every seed, with large margins on the raw
  weights.
- **What the margins rest on:**
  - **The lr selection.** The gain comes with OT top-free running at 9e-2, where SOAP does worse. At SOAP's best
    lr the two are close on seed 0. Mechanistically, the brake on the sharpest directions (c = 4096 in the OT
    metric) appears to let a larger lr stay productive.
  - **The length.** 1,000 steps is not 2,000. In `long_run`, SOAP itself used lr 9e-2 well at 2,000 steps (warm-up
    100), and GGN top-free's edge vanished.
- **What is not shown:**
  - Whether OT top-free keeps this edge at 2,000 steps, where SOAP's own lr optimum moves up.
  - Whether the OT metric matters, compared with GGN top-free with a tuned c and lr (N23).
- **Limitations:**
  - Three seeds; one tuning seed.
  - A two-point c grid.
  - SOAP's raw loss barely improves in the decay phase at its selected lr. Its EMA (d 0.995) carries its result, so
    the raw comparison partly reflects the different selected lrs.
