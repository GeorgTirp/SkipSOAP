# Why the brake works — noise controls (protocol)

Study `noise_controls_20261011`. Approved by the owner on 11 October 2026: "The noise theories seem quite
realistic so lets run 1 and 2 to check". It adds the discriminating controls proposed for the competing
explanations (EMA of the weights, learning-rate decay, effective bulk step, sharpness time course). Written
before any run.

## 1. Explanations under test

The brake is PARSEC-L with learning-rate calibration, c = 4, lr 1e-2, norm blend 0. In stage D it beat
constant-lr SOAP in all seeds.

| Code | Explanation | Discriminating control |
| --- | --- | --- |
| N | Selective annealing: sharp whitened directions are underdamped under EMA momentum, so lr there buys no speed, only noise, and 57–70% of the noise sits there | LR decay removes the same excess; EMA of weights removes iterate noise |
| S | Statistics: less jitter in sharp directions means smaller second moments and cleaner bases, so a larger effective bulk step | Effective step size at matched training loss |
| T | Transient ringing: underdamped sharp directions ring after large early gradients, and damping them helps early | Momentum-free top space |
| E | Edge of stability | Full-network preconditioned sharpness over training |

## 2. Workload, data, rules

- **Workload:** as in stage D. Main ViT, depth 2, branch scaling off, MNIST 28x28, batch 512, CPU float64,
  500 steps; validation on the reserved 5,000 at 0, 10, 25, …, 500.
- **Seeds:** settings chosen on seed 0; evaluation on seeds 1–3; gradient batches in the gap-audit order.
- **Isolation:** write only to `studies/noise_controls_20261011/` and
  `~/skipsoap-study-data/noise_controls_20261011/`; `PYTHONDONTWRITEBYTECODE=1`; at most four workers;
  tolerance 1e-8 (bitwise for reproduction); manifests and suites before and after.

## 3. Arms

All use `PARSECL`: rank 16, s = 4 sampled-label probes on the gradient batch every 10 steps, extract
refresh, norm blend 0, per-tensor extraction.

| Arm | Description |
| --- | --- |
| `soap` | Rank 0 (upstream SOAP bit for bit) |
| `brake` | Learning-rate calibration, c = 4 |
| `topfree` | Brake, but inside the held top subspace the step uses the bias-corrected current gradient instead of SOAP's momentum (`top_rhs='gradient'`); c ∈ {1, 4, 16} |

**Schedules** (set as `group['lr']` before every step; the brake's κ = c·lr_t·λ follows lr_t):
- constant;
- cosine, lr_t = peak · ½(1 + cos(π (t−1)/500));
- WSD, peak for t ≤ 400, then linear to peak/100 at t = 500: lr_t = peak · (501 − t)/100.

**Seed-0 tuning (17 units):**
- `soap` and `brake`, constant at lr 1e-2 (references and gates);
- `soap` and `brake` × {cosine, WSD} × peak ∈ {1e-2, 2e-2, 3e-2};
- `topfree`, constant at lr 1e-2, × c ∈ {1, 4, 16}.

**Selection:** per (arm, schedule), the peak with the lowest step-500 validation loss; for `topfree`, the c with
the lowest step-500 loss.

**Evaluation (seeds 1–3, 21 units):** `soap` and `brake` constant; the four selected schedule arms; selected
`topfree`.

## 4. Measurements

- **Per step:** training-batch loss before and after; next-batch loss before and after (outside the timed
  region); lr_t.
- **Every 10 steps:** r̂ diagnostic (true-label per-example pass), as in stage D.
- **At every validation step:**
  - validation loss and accuracy of the raw weights and of bias-corrected EMAs of the weights with decay
    d ∈ {0.9, 0.98, 0.995}. The EMAs are tracked from step 1, are evaluation only, and do not affect
    training.
  - effective step size e = lr_t/(√v̂ + ε) over all rotated coordinates of all tensors (v̂ = SOAP's
    bias-corrected second moment): median and geometric mean. Also the mean training-batch loss over the
    preceding 25 steps.
- **Constant `soap` and `brake` arms, seeds 0–3:** full-network preconditioned sharpness
  S = lr_t·√(1−β₂ᵗ)·λ_max(F0ᵀ H F0) at steps 10, 25, 50, 100, 200, 300, 400, 500. This uses the full Hessian
  on the first 512 analysis examples, Lanczos with 30 iterations, and F0 from the current SOAP state.

## 5. Gates

- `soap` and `brake` constant at lr 1e-2 reproduce the stage-D runs bitwise (validation difference ≤ 1e-12 at
  every common step):
  - seeds 1–3 against `20261010T180135-101694d8` (brake) and `20261010T171814-46f283ad` (SOAP; seed 1
    through step 275, where that unit stopped);
  - seed 0 against `20261010T171813-415c766d`.
- WSD arms with peak 1e-2 equal the constant arm of the same seed through step 400.
- PARSEC-L tests (22) and harness tests pass. A failed gate stops that arm; divergence is a result.

## 6. Pre-declared readings

Seeds 1–3, settings from seed 0. Better or worse means lower or higher validation loss in all three seeds;
otherwise mixed.

| Code | Reading |
| --- | --- |
| R1 | Decay vs brake: SOAP + S vs constant brake at step 500, for S ∈ {cosine, WSD} |
| R2 | Additive with decay: brake + S vs SOAP + S at step 500 |
| R3 | Anytime: constant brake vs SOAP + S at steps 200 and 300 |
| R4 | Iterate averaging, with the decay d chosen on seed 0 by `soap`-constant EMA loss at step 500: (a) SOAP + EMA vs raw brake at step 500; (b) brake + EMA vs SOAP + EMA at step 500 |
| R5 | Transient (T): `topfree` (selected c) vs constant brake at steps 50 and 100; step 500 reported |
| R6 | Statistics (S): ratio of the brake's median effective step to SOAP's at matched training loss. *Present* if > 1.05 over the common loss range at steps ≥ 100 in all seeds, else *not detected* |
| R7 | Edge of stability (E): *signature present* if SOAP's S rises to ≥ 58 (0.75 × 78) and stays within ±25% of 78 for two consecutive logged steps in all seeds; else *absent* |

**Interpretation rules, fixed in advance:**
- **N supported** if R4a is better or mixed, i.e. SOAP + EMA is not worse than the brake, *or* R1 is better or
  mixed for some schedule, i.e. decayed SOAP is not worse than the brake at step 500.
- **Something beyond iterate noise** if R4b is better (the brake still helps on top of weight averaging). Its
  nature is then indicated by R5, R6 and R7.

## 7. Cost

38 units at about 0.05–0.1 s per training step, plus evaluation of the raw weights and three EMAs, r̂, and
sharpness for 8 units: about 1 CPU-h, roughly 20 minutes with four workers. Budget 2 CPU-h.
