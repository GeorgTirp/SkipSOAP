# CIFAR-10 on a depth-6 ViT: exact per-tensor GGN with and without SN, tuned SOAP and top-free PARSEC-L — protocol

Study `cifar_ggn_20261012`. Written on 12 October 2026, before any run. The owner approved it on 12 October with one
change: no GGN tuning, γ = 1e-2. That change is incorporated below.

The owner's request (12 October): run one exact layerwise GGN with big batches on the new model, a tuned SOAP and
PARSEC. For the GGN, also measure distance preservation and run with and without the SN regularizer, because the
interaction of distance preservation and the quadratic model is still open. Owner decisions:
- model M1 at depth 6, because depth is where distance preservation becomes interesting;
- batch 2048;
- SN bound c = 1;
- quadratic models along the step for the full GGN, the per-tensor GGN and the Hessian.

The GPU port (`studies/gpu_port_20261012`) found float32 on the GPU adequate.

## 1. Questions

1. Does exact per-tensor GGN keep its per-step advantage over tuned SOAP at this scale and batch, early (steps 50
   and 200) and at the end (step 500)?
2. Does top-free PARSEC-L + EMA beat tuned SOAP + EMA on CIFAR-10, as it did on MNIST?
3. Does SN (c = 1) change how the exact GGN trains?
4. Does SN raise distance preservation through depth? Does it make the quadratic model along the GGN step more
   accurate? Does distance preservation track quadratic-model fidelity over training?

## 2. Workload

- **Data:** CIFAR-10 with the GPU port's split: 45,000 training and 5,000 validation examples from the official
  training set (seed 101), normalized with the training split's channel statistics. **The official test split is
  reserved and never accessed.**
- **Augmentation**, for training and selection batches: random crop with 4-pixel zero padding of the normalized
  image (padding equals the training mean) and horizontal flip, drawn per image from a seeded CPU generator.
  Validation, distance and next-batch inputs are not augmented.
- **Model M1d6:** ViT with patch 4 (65 tokens), width 64, depth 6, 4 heads, MLP 256, pre-LayerNorm; 308,042
  parameters in 104 tensors. It is the lab's `TransformerBlock` and parameter naming (`studies/gpu_port_20261012`,
  `scripts/cifar.py`); initialization in float64 under the model seed, then rounded to float32.
- **Run length:** batch 2048 for every arm; 500 steps (about 23 epochs for SOAP and PARSEC-L).
- **Numerics:** GPU float32 with TF32 off and deterministic algorithms. Gradients and GGN actions are summed over
  microbatches of 512 with exact mean-loss weights; the Hessian quadratic uses microbatches of 256. Loss evaluations
  of the GGN step search are single no-grad passes of the 2,048-example selection batch.

## 3. Arms

| Arm | Settings |
| --- | --- |
| `soap` | SOAP as PARSEC-L with rank 0 (the lab's `soap` arm); betas (0.95, 0.95), preconditioner refresh every 10 steps, decoupled weight decay 0.05; peak lr tuned |
| `topfree` | PARSEC-L: rank 16, s = 1 sampled-label probe per example (2,048 probe rows) every 10 steps, per-tensor extraction (randomized SVD, 2 power steps), learning-rate calibration c = 64, `top_rhs='gradient'`, norm blend 0; SOAP settings as `soap`; peak lr tuned |
| `ggn` | Stage D's exact per-tensor GGN step: dense blocks from the training batch; damping μ = max(γ tr C / D, 1e-8) in the normalized inverse metric of a shadow SOAP (lr 3e-3, no weight decay); current-gradient RHS; step length chosen on a second 2,048-example *selection* batch by the lab's broad direction search (`search_candidate`); **γ = 1e-2, not tuned** (owner decision) |
| `ggn_sn` | `ggn` with SN, W_eff = W·min(1, c/σ(W)), c = 1, exact σ (an SVD per weight), on the 36 residual-branch weights (attention query/key/value/output, MLP w1/w2); the optimizer acts on the raw weights; parameters keep the plain model's names (`cifar.CifarViTSN`); γ = 1e-2 |

- **Schedules:**
  - `soap` and `topfree`: linear warm-up over steps 1–25, then constant, then linear decay over steps 401–500 to
    1% (WSD).
  - `ggn` arms: the selected step is multiplied by the same decay factor (no warm-up). They apply decoupled weight
    decay as SOAP would at the reference lr 3e-3, with the same schedule.
- **Weight EMA:** every arm tracks bias-corrected weight EMAs with d ∈ {0.9, 0.98, 0.995}, used only for evaluation.
- **Data per step:** the GGN arms draw a selection batch from a separate order, so they see 4,096 examples per
  step against 2,048 for the others.
- **Implementation of the exact GGN** (`scripts/ggn.py`): the step of the lab's `streamed_blocks`/`solve_blocks`,
  solved in SOAP-whitened coordinates.
  - With SOAP's factor F0 = Q·diag(d^−½) (P = F0F0ᵀ), the exact identity C + μ·scale·P⁻¹ = F0⁻ᵀ(F0ᵀCF0 + μ′I)F0⁻¹
    gives δ_b = −F0 (K_b + μ′I)⁻¹ F0ᵀ g_b. Here K_b = F0ᵀ C_bb F0 is accumulated from whitened Jacobian rows, and
    μ′ = μ·scale.
  - **Why (pre-run dry run):** at step 2 the original-coordinate systems C_bb + μΩ_b had condition numbers of
    1.7e7–1.8e8, because SOAP's denominators span seven decades after one statistics step.
    - Float32 Cholesky failed on the first attention key weight.
    - The float32 rounding of C_bb (smallest eigenvalue about −3e-9) was as large as μΩ's smallest eigenvalue
      (2e-9).
    - The whitened system's condition number is about 1 + κ_max.
  - Per-example Jacobians come from vmap over examples. Tensors are processed in groups of at most 2.5 GiB of blocks,
    recomputing the Jacobian per group. tr C, hence μ, comes from unwhitened rows in the first pass.
  - **Study tests, in float64 on the MNIST ViT:**
    - the unwhitened blocks and trace agree with the lab's to 1e-12;
    - the whitened blocks equal F0ᵀ C_bb F0 to 1e-10;
    - the step solves the lab's own damped systems to a backward error of 1e-10;
    - the step agrees with the lab's solution to 1e-6 (the early systems are ill-conditioned).

## 4. Tuning and evaluation

- **Tuning on seed 0 (8 units, no diagnostics):** `soap` and `topfree`, peak lr ∈ {1e-3, 3e-3, 1e-2, 3e-2}. The GGN
  arms are not tuned (owner decision). They use γ = 1e-2 and the EMA decay d = 0.98, the decay every arm selected in
  the top-free confirmation; all three decays are reported.
- **Selection:** for `soap` and `topfree`, the lr and EMA decay with the lowest step-500 EMA validation loss.
  - If the selected lr lies at an edge of its grid, one more point beyond that edge is run on seed 0 before
    evaluation (lr ÷ 3 or × 3; an *extension* run). The selection is then repeated over both runs.
  - A diverged unit counts as worse than any finite one.
- **Evaluation:** seeds 1, 2, 3, with diagnostics (12 units).
  - Arms: `soap`, `topfree`, `ggn`, `ggn_sn`.
  - Seed s sets model initialization s, data order 102 + 1000 s, augmentation 5000 + s, selection order
    7102 + 1000 s, selection augmentation 6000 + s and probes 9000 + s.

## 5. Diagnostics (evaluation units)

**Every 10 steps**, along the step actually taken (d = θ_{t+1} − θ_t, including schedule and weight decay):

- **Quadratic-model fidelity** (`scripts/measure.py`), on two batches:
  - *own*: the step's augmented training batch;
  - *next*: the next 2,048 training examples, not augmented and not used for this step.

  On each batch:
  - measured: actual = L(θ+d) − L(θ), the gradient term gᵀd, the full GGN dᵀCd (forward-mode product), the
    per-tensor GGN Σ_b d_bᵀC_bb d_b (per-example Jacobians, one pass) and the Hessian dᵀHd (double backward);
    - **pre-run checks against finite differences:** in torch 2.14.1 LayerNorm's forward-over-forward second
      derivative is wrong (12% off), and forward-over-reverse through SN's SVD fails with a dtype mismatch in
      float32 on CUDA. Double backward agrees with finite differences and with float64 forward-over-reverse;
  - predictions gᵀd + q/2 and ratios actual / prediction, for q in {full, per-tensor, Hessian};
  - coupling = dᵀCd / Σ_b d_bᵀC_bb d_b;
  - Gauss–Newton gap = dᵀHd / dᵀCd.
- **Distance preservation** at θ_{t+1} (the SN curvature study's measure, extended over depth):
  - representations: the residual stream after the embedding and after each block (whole stream; CLS row from
    block 1 on), and the head input;
  - global ratios over 1,000 fixed pairs of validation images;
  - local ratios for 512 validation images with x′ = x + 10⁻³‖x‖ξ;
  - reported: quantiles 1/5/50/95/99 and distortion q95/q5.
- **Spectral norms:** σ of the 36 residual-branch weights, raw and effective, and SN activity.

**Every 50 steps:** block-Jacobian deviation, the RMS singular value of R_k − I for each block, from 4 probes at the
local inputs.

**Validation:** raw and EMA losses and accuracy on the 5,000 examples at steps 0, 10, 25, 50, …, 500.

**Checkpoints** (external): parameters, EMAs and optimizer or shadow-SOAP state at steps 50, 100, 250 and 500.

## 6. Gates

- **G0:** the repository suite (106) and the PARSEC-L tests (22) pass before and after; so do the study tests:
  - GGN equivalence with the lab reference in float64;
  - lean probes bitwise equal to `ggn_probes`;
  - microbatched gradient, GGN action and HVP equal to full-batch or independent references;
  - quadratic-check terms equal to dense references;
  - SN: an inactive bound gives the plain model, and active SN gives gradients orthogonal to the raw weight;
  - augmentation correctness;
  - the unit plan.
- **GGN numerics**, in float32, with the lab's `tolerance32` = 1e-4:
  - the per-tensor whitened residual ‖(K_b + μ′I)z + F0ᵀg_b‖/‖F0ᵀg_b‖ ≤ 1e-4 at every step; the
    original-coordinate residual is reported;
  - the block-against-action audit ≤ 1e-4 for every tensor at steps 2, 100, 250 and 500. It compares the whitened
    block with F0ᵀ C F0 u from the microbatched GGN action, relative to the damped action ‖F0ᵀCF0u + μ′u‖ (an error
    below that cannot move the step by more than the tolerance).
    - Pre-run dry run: relative to F0ᵀCF0u alone, the dead attention key biases fail (1.7), because both sides
      are rounding noise.
    - For live tensors the two definitions agree, since the curvature dominates μ′.
  - the step search must bracket (the lab's `search_candidate` stops at the radius boundary).
  - κ_max per tensor is logged at the audit steps (power iteration).

  A failure stops the unit; record it in `DEVIATIONS.md` before any rerun.
- **Divergence** (a non-finite loss or parameters) ends a unit as diverged. It is reported, not a gate failure.

## 7. Pre-declared readings (evaluation seeds 1–3)

*Better* or *worse* means lower or higher validation loss in all three seeds; otherwise *mixed*. EMA readings use
each arm's seed-0-selected decay.

| Code | Reading |
| --- | --- |
| R1 | `ggn` vs `soap`: raw weights at steps 50 and 200; EMA at step 500 |
| R2 | `topfree` + EMA vs `soap` + EMA at step 500 |
| R3 | `ggn_sn` vs `ggn`: raw at steps 50 and 200; EMA at step 500 |
| R4 | SN and distance preservation: local distortion q95/q5 of the head input and of the stream after block 6, at steps 100 and 500. *SN raises preservation* if `ggn_sn` is lower than `ggn` in all seeds, for both representations at both steps. |
| R5 | SN and the quadratic model: mean \|1 − ratio\| of the full-GGN model on the *next* batch, over measurement steps 10–100 and 110–500. *SN improves the quadratic model* if `ggn_sn` is lower than `ggn` in all seeds in both windows. Reported alongside: the same for the Hessian and per-tensor models, on both batches, plus coupling and Gauss–Newton gap. |
| R6 | Interaction (descriptive, no verdict): per unit, Spearman correlation over the 50 measurement steps between local distortion of the head input and \|1 − ratio_full\| on the next batch; also pooled over the six GGN units, and over all twelve units. |

Also reported: accuracy; steps and seconds to the targets (`soap`'s seed-mean raw loss at steps 50 and 200); seconds
per step; and the step-length and damping traces of the GGN arms. A ratio is reported as missing when its predicted
change is 0; medians accompany means.

Evidence from the GPU port (rule 11 in `ideas/parsec.md`): single runs differ by 0.002–0.008 at step 500 for
numerical reasons alone. Readings rest on all three seeds, and differences of that size are not interpreted.

## 8. Cost (dry runs, `logs/dryrun*.log`; 12 steps per arm, seed 0)

| Arm | Seconds per step | Peak GPU | Per 500-step unit |
| --- | ---: | ---: | ---: |
| `soap` | 0.22 | 4.7 GiB | about 2 min (+6 min of diagnostics in evaluation) |
| `topfree` | 0.29 (probe steps 1.55) | 4.8 GiB | about 3 min (+6 min) |
| `ggn` | 23.3, of which 19.1 building and solving the blocks | 5.2 GiB | about 3.3 h (+7 min) |
| `ggn_sn` | 39.3, of which 32.2 building and solving the blocks | 5.2 GiB | about 5.5 h (+15 min) |

- **Overheads:** an audit step costs about 50 s (`ggn`) or 100 s (`ggn_sn`) more. One diagnostic row takes 7–8 s
  without SN and 18 s with SN, every 10 steps.
- **Dry-run numerics:** whitened residuals 0.5–2.9e-5 and action audits about 4e-6, against 1e-4. κ_max at step 2
  was 1.3e6 (`ggn`, γ = 1e-2). At step 10, `ggn` and `ggn_sn` reached validation loss 1.51 and 1.64 (one seed,
  not a reading); `soap` at lr 3e-3 reached 2.05.
- **Budget (after the owner's decision):**
  - tuning: about 0.5 h, plus about 10 min if an extension is needed;
  - evaluation: about 1 h for `soap`/`topfree`, 10 h for `ggn` and 17 h for `ggn_sn`.

  About 29 GPU-hours in total. The queue runs the cheap arms first; `scripts/orchestrate.sh` chains the phases.

## 9. Isolation and resources

- **Writes:** only to `studies/cifar_ggn_20261012/` and `~/skipsoap-study-data/cifar_ggn_20261012/`. The study
  imports the lab and `studies/gpu_port_20261012/gpu_parsec`, and archives them with each run.
- **Hygiene:** `PYTHONDONTWRITEBYTECODE=1`; pytest with `-p no:cacheprovider`; isolation manifests and suites
  before and after.
- **Processes:** one GPU process at a time.

## 10. Limitations known in advance

- One model, three evaluation seeds and 500 steps (about 23 epochs), far from a converged ViT.
- The GGN arms see twice the data per step (the selection batch). Per-step comparisons favour them in data;
  comparisons per example or per second favour SOAP.
- The GGN arms' decay and weight decay are imposed on a searched step, by declared choice.
- γ = 1e-2 is not tuned for this workload, by owner decision. On MNIST the gap audit's per-tensor GGN was tuned to
  γ = 1e-3, which float32 cannot solve reliably here. `ggn_sn` uses the same γ.
- The GGN blocks and solves are float32 (tolerance 1e-4).
- The attention key biases have zero gradient. In float32 their coordinates may drift without effect on the loss.
- At c = 1, SN clips most residual-branch weights from step 0. The loss is then invariant to the raw scale of each
  clipped weight, so the raw-parameter GGN is singular along it and only damping acts there.
