# Does in-loop top-space damping beat weight averaging? — confirmation protocol

Study `topfree_confirm_20261011`, approved by the owner on 11 October 2026 ("Yes lets run this here").

It tests a hint found after the fact in `noise_controls_20261011`. There, the momentum-free top-space brake
(`top_rhs='gradient'`, c = 16) evaluated on an EMA of its weights reached 0.119, against 0.127 for SOAP + EMA, on
seeds 1–3. Those seeds generated the hypothesis, so this test uses **fresh seeds 4–6**. Written before any run.

## 1. Question and decision rule

Does damping the sharp top directions *inside* training (no momentum there, shrink 1/(1 + c·lr·λ)) give a lower
loss than averaging the weights afterwards, also against SOAP with a final decay?

- **Yes** (both readings in section 5 are *better*): in-loop damping adds something beyond averaging. Build the
  per-direction Kalman version next, informed by the checkpoint diagnostics.
- **No** (either reading *mixed* or *worse*): stop the in-loop line. For this setup the answer is SOAP + decay or
  weight averaging.

## 2. Workload and rules

- **Workload:** as in stage D and the noise controls. Main ViT, depth 2, branch scaling off, MNIST 28x28, batch
  512, CPU float64, 500 steps; validation on the reserved 5,000 at 0, 10, 25, …, 500.
- **PARSEC-L settings:** rank 16, s = 4 sampled-label probes every 10 steps, learning-rate calibration, norm blend
  0, per-tensor extraction.
- **Seeds:** settings chosen on seed 0; evaluation on seeds 4, 5, 6, never used before (model init s, data order
  102 + 1000 s).
- **Isolation:** write only to `studies/topfree_confirm_20261011/` and
  `~/skipsoap-study-data/topfree_confirm_20261011/`; `PYTHONDONTWRITEBYTECODE=1`; at most four workers;
  manifests and suites before and after.

## 3. Arms

| Arm | Settings |
| --- | --- |
| `soap` constant | lr 1e-2 |
| `soap` WSD | peak 1e-2; linear decay over steps 401–500 to 1% (peak selected in the noise controls, fixed here) |
| `brake` constant | c = 4, lr 1e-2 (reference; not in the readings) |
| `topfree` constant | lr 1e-2, c ∈ {16, 64} on seed 0; the selected c on seeds 4–6 |
| `topfree` WSD | selected c, peak 1e-2 (exploratory; not in the readings) |

**Every arm** tracks bias-corrected weight EMAs with d ∈ {0.9, 0.98, 0.995} (evaluation only).

**Seed-0 selection (5 units):**
- for each arm, the EMA decay with the lowest step-500 EMA validation loss;
- for `topfree`, the c whose best EMA loss at step 500 is lowest.

**Evaluation:** seeds 4–6, 15 units.

## 4. Gates and diagnostics

- **Gates:** seed-0 units that also exist in the noise-controls tuning run (`soap` constant, `soap` WSD, `brake`
  constant, `topfree` c = 16) must reproduce it bitwise (validation difference ≤ 1e-12 at every step). PARSEC-L
  and harness tests must pass.
- **Diagnostics:**
  - full-network preconditioned sharpness at steps 10, 25, 50, 100, 200, 300, 400, 500 for `soap` constant,
    `brake` constant and `topfree` constant on seeds 4–6;
  - for seed 4, full checkpoints (parameters, optimizer state including the held PARSEC-L directions, EMA
    weights) every 10 steps for `soap` constant, `brake` constant and `topfree` constant. These are for an
    offline check of the noise (Ornstein–Uhlenbeck) model: spread of the top-direction coordinates around the
    EMA against η'σ²/(2λ), and correlation time.

## 5. Pre-declared readings

Seeds 4–6, step 500, each arm on its seed-0-selected EMA decay. *Better* means lower validation loss in all
three seeds.

| Code | Comparison |
| --- | --- |
| P1 | `topfree` constant + EMA vs `soap` constant + EMA |
| P2 | `topfree` constant + EMA vs `soap` WSD + EMA |

Verdict: *in-loop damping adds beyond averaging* if P1 and P2 are both better; otherwise *not confirmed*.

Reported but not part of the verdict:
- raw (unaveraged) values;
- `topfree` vs `brake`, both with EMA;
- `topfree` WSD;
- sharpness;
- seed spread.

## 6. Cost

20 units at about 0.05–0.1 s per training step, plus evaluation of four weight copies, sharpness on 9 units
and checkpoints on 3 units: about 0.5 CPU-h, roughly 10–15 minutes with four workers. Budget 1 CPU-h.
