# GGN vs OT probes at matched tuning: top-free PARSEC-L over 1,000 steps — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, fixed before any run. No deviations.

External runs `~/skipsoap-study-data/ggn_ot_fairness_20261014/runs/`:
- tuning (seed 0): `20261014T175554-afe4ca25`;
- extension (lr 0.27 at c = 256): `20261014T183530-957bc167`;
- evaluation (seeds 1–3) and the G1 rerun: `20261014T184201-ec924e59`.

References: `ot_tuned_1000_20261014`'s evaluation run `20261014T162934-085e376e`. Summary: `results/summary.json`;
G1: `results/g1.json`.

**Status (2026-10-14):** all 11 units completed. Suites passed before and after (13 suites).
- **G1 passed:** the rerun `topfree_ot__lr0.09__c4096__s1` reproduces `ot_tuned_1000`'s unit bitwise in
  `validation.json` and `training.json`.
- **Isolation manifest:** the only changes are 16 new files in `studies/bulk_krylov_20261014/`, which was being
  prepared during this run. Nothing outside the two study folders changed.

## 1. Summary

- **F1: OT-probe top-free beats tuned GGN-probe top-free in every seed.** By the pre-declared reading, *the OT metric
  matters*. The margin is small:
  - EMA validation loss at step 1000: 0.460 vs 0.466 (seed means). Per seed the differences are −0.0074, −0.0077
    and −0.0024.
  - Accuracy is higher in every seed (0.841 vs 0.835).
  - Raw weights at step 1000 are mixed (−0.011, −0.005, +0.003).
- **F2: tuned GGN-probe top-free vs tuned SOAP is mixed.** EMA 0.466 vs 0.483 (seed means). It is better by 0.026
  and 0.025 on seeds 1 and 3, and level on seed 2 (+0.0006).
- **Most of N23's gain comes from tuning the brake and lr.**
  - Tuned GGN top-free reaches 74% of OT top-free's EMA gain over SOAP (0.017 of 0.023).
  - The OT metric adds the remaining 0.006.
- **GGN tuning selected lr 9e-2, c = 256, EMA 0.98, close to OT's choice.**
  - OT selected lr 9e-2 and c = 4096. Since GGN curvature is about 9× OT's, OT's c = 4096 corresponds to about
    GGN c = 450, and the GGN selection lies between 256 and 1024.
  - Both lr and c are interior: the lr extension at 0.27 diverged to EMA 1.86.
  - With c = 64 at lr 9e-2, as used before this study, the seed-0 EMA is 0.483 instead of 0.469.

## 2. Tuning (seed 0; best EMA over decays 0.98, 0.995, 0.998, at step 1000)

| lr | c = 64 | c = 256 | c = 1024 |
| ---: | ---: | ---: | ---: |
| 3e-2 | 0.496 (d 0.995) | 0.503 (d 0.995) | 0.491 (d 0.995) |
| 9e-2 | 0.483 (d 0.98) | **0.469** (d 0.98) | 0.478 (d 0.98) |
| 0.27 (extension) | — | 1.863 (d 0.98) | — |

The references on the same seed: OT top-free 0.468 (lr 9e-2, c = 4096, d 0.98) and SOAP 0.475 (lr 3e-2, d 0.995).

## 3. Evaluation (seeds 1 / 2 / 3, seed mean)

| | EMA 1000 (selected decay) | raw 1000 | raw 500 | raw 200 | accuracy 1000 |
| --- | --- | --- | --- | --- | --- |
| SOAP (lr 3e-2, d 0.995), reference | 0.494 / 0.463 / 0.490 (0.483) | 0.619 / 0.587 / 0.618 (0.608) | 0.652 / 0.624 / 0.650 (0.642) | 0.902 / 0.860 / 0.909 (0.890) | 0.826 / 0.833 / 0.826 |
| GGN top-free (lr 9e-2, c 256, d 0.98) | 0.468 / 0.464 / 0.465 (0.466) | 0.497 / 0.485 / 0.484 (0.489) | 0.722 / 0.729 / 0.720 (0.724) | 1.057 / 1.055 / 1.041 (1.051) | 0.832 / 0.835 / 0.839 |
| OT top-free (lr 9e-2, c 4096, d 0.98), reference | **0.461 / 0.456 / 0.463 (0.460)** | 0.486 / 0.480 / 0.487 (0.484) | 0.731 / 0.701 / 0.746 (0.726) | 1.053 / 1.012 / 1.083 (1.049) | **0.839 / 0.842 / 0.842** |

**Readings:**
- **F1:** *better* in all seeds; *the OT metric matters*.
- **F2:** *mixed*.
- **F3:** the two top-free arms track each other along the run (raw 200 and 500 within 0.003 in the seed means). Both
  are behind SOAP until the decay phase, as in `ot_tuned_1000`.
- **Cost** (mean over all 1,000 steps, including the probe steps every 10 steps): GGN top-free 0.371 s, OT top-free
  0.362 s, SOAP 0.221 s per step. Both top-free arms cost about 1.65× SOAP's wall time.
  - `ot_tuned_1000`'s RESULTS.md gives 1.3×. That is the median step, which leaves out the probe steps.

## 4. Interpretation and limitations

- **What changed in the picture:**
  - N23's win over SOAP is mostly a tuning effect of the top-free brake. Tuned GGN top-free gets three quarters of
    it, and it beats SOAP clearly in two seeds out of three.
  - The OT metric's own contribution is consistent in sign but small: 0.006 in EMA loss and 0.006 in accuracy. That
    is about the size of the seed-to-seed spread of either arm (0.004 GGN, 0.007 OT).
- **The reading is a sign test on three seeds.** Under no difference, all three seeds in one direction has
  probability 1/4 (either direction), or 1/8 for the declared direction. The verdict follows the protocol, but it
  is weak evidence for a metric effect.
- **Tuning effort:**
  - GGN was tuned on a 2 × 3 grid plus one extension. OT's reference was tuned on 2 × 2 plus one, with c ∈ {1024,
    4096}.
  - The finer GGN grid would, if anything, favour GGN.
  - OT's c was not tried above 4096. The OT selection was interior in lr but at the upper edge of its c grid.
- **Not shown:**
  - Whether either top-free arm keeps its edge over SOAP at 2,000 steps. GGN top-free's 500-step edge vanished there
    in `long_run`.
  - Whether SOAP with a longer warm-up at lr 9e-2 closes the gap. The brake may act mainly as an automatic
    warm-up for the sharp directions.
  - Any comparison at equal wall time: SOAP could run about 1,650 steps in the same time.
- **Limitations:** three seeds; one tuning seed; the references come from another study (licensed by G1).
