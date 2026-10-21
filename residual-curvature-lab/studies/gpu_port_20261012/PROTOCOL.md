# Does PARSEC-L give the same results on the GPU, and what does float32 change? — protocol

Study `gpu_port_20261012`. Draft of 12 October 2026, written before any run; awaiting the owner's approval.

All PARSEC-L evidence so far comes from one workload on the laptop CPU in float64. The next study (CIFAR-10 with
larger ViTs, owner decision of 12 October) needs the GPU, and on this GPU it needs float32. This study checks the
port on the established MNIST workload before any new science, and measures the cost of the candidate CIFAR models.

## 1. Questions and decision rule

1. Does the device-ready copy of PARSEC-L compute exactly what the original computes? (G1)
2. Does the GPU in float64 agree with the CPU in float64, one step at identical states (R1) and along whole
   trajectories (R2)?
3. Does float32 on the GPU keep the established result: top-free PARSEC-L (c = 64) + EMA below SOAP + EMA in every
   fresh seed? (R3)
4. What do the candidate CIFAR-10 ViTs cost per step and in GPU memory under each optimizer? (C1)

Decision:
- **All gates pass and R3 is adequate:** the CIFAR study is drafted for float32 on the GPU, with its model size and
  probe count chosen from C1.
- **R3 not adequate:** float32 is not used as is. Report, and let the owner decide (for example float64 for the
  curvature extraction only).
- **Any gate fails:** stop and report. Record deviations in `DEVIATIONS.md` before any rerun.

## 2. Machine and environment

- RTX 3070, 8 GB, compute capability 8.6; its float64 rate is 1/64 of float32. Ryzen 7 5800X, 16 threads, 31 GB
  RAM, Linux 6.8, Python 3.12.3.
- `.venv` with the `tested-requirements.txt` pins; torch 2.14.1+cu130 (driver CUDA 13.2). The lab is on the path
  through a `.pth` file in the venv, not an editable install.
- Determinism: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off for matmul
  and cuDNN, float32 matmul precision `highest`. CPU units run single-threaded, as on the laptop.
- The laptop's run data (`~/skipsoap-study-data`) is not on this machine. CPU float64 references are regenerated
  here (owner decision, 12 October). The laptop ran macOS, so bitwise agreement with its runs is not expected.
- Known before the study: the repository suite (106) and the PARSEC-L tests (22) pass here. The harness tests of
  `topfree_confirm_20261011` and `damping_sweep_20261011` fail because they read the laptop's run data; this is
  reported, not changed.
- Seen in the pre-run study tests: the attention key bias has an identically zero gradient. A key bias adds q_i·b_k
  to every logit of row i, and softmax ignores such shifts. Its gradient and probe rows are therefore pure rounding
  noise: about 1e-17 in float64, where they differ between devices by O(1) in relative terms. PARSEC-L's κ for it
  is about 1e-26. In float32 the noise is about 1e-8, the same order as SOAP's ε = 1e-8, so these coordinates can
  drift by random steps of size about lr without changing the loss. Reported, not changed; all registered
  parameters stay in the optimizer.

## 3. Code

- `gpu_parsec/`: a copy of `studies/parsec_l_20261009/parsec_l/`. Its only change is that every random number
  (the randomized-SVD sketch and the sampled probe labels) is drawn from a CPU generator and then moved to the
  device. CPU and GPU units therefore use the same random numbers, and on the CPU the copy is the original.
- `scripts/cifar.py`: CIFAR-10 split and ViT for part C only. The split is 45,000 training / 5,000 validation examples,
  drawn from the official training set with a seeded permutation; the manifest is stored. **The official test
  split is reserved and never accessed.** The ViT takes 3 × 32 × 32 images, patch 4 (65 tokens). It reuses the
  lab's `TransformerBlock` and keeps the lab's parameter names, so `Registry` works unchanged.
- `scripts/train.py`: the top-free confirmation's runner with a device and dtype switch. It also checkpoints the
  data order and both generators, so R1 can replay single steps.
- Float32 units start from the float64 initialization rounded to float32, so precision is the only difference at
  step 0.

## 4. Workload and arms (G1, G2, R0–R3)

The top-free confirmation's workload, unchanged:
- main ViT (width 16, depth 2, patch 7, branch scaling off); MNIST 28×28; batch 512; 500 steps;
- validation on the reserved 5,000 at steps 0, 10, 25, 50, …, 500;
- weight EMAs d ∈ {0.9, 0.98, 0.995}; readings use d = 0.98, the decay every arm selected on seed 0;
- seed s gives model initialization s and data order 102 + 1000 s.

| Arm | Settings |
| --- | --- |
| `soap` | constant lr 1e-2 (PARSEC-L with rank 0, norm blend 0) |
| `topfree` | constant lr 1e-2; rank 16; s = 4 sampled-label probes every 10 steps; learning-rate calibration c = 64; norm blend 0; `top_rhs='gradient'` |

| Configuration | Seeds | Purpose |
| --- | --- | --- |
| CPU float64, original `parsec_l` | 0 | G1 |
| CPU float64, copy | 0, 4, 5, 6 | G1, R0, reference for R1–R3 |
| GPU float64, copy, run twice | 0 | G2, R2 |
| GPU float32, copy | 0, 4, 5, 6; seed 0 twice | G2, R3 |

There are 2 arms × 12 configuration-seed runs, so 24 units in total. The seed-0 CPU float64 copy units also save
full states at every step in the windows 50–62, 250–262 and 488–500.

## 5. Gates

- **G0 suites:** repository suite and PARSEC-L tests pass, against the original and against the copy; study tests
  pass. Run before and after, with isolation manifests.
- **G1 code equivalence:** on the CPU in float64, the copy reproduces the original bitwise (validation difference
  0.0 at every validation step, final parameters equal) for both arms, seed 0.
- **G2 GPU repeatability:** the two runs of each repeated GPU unit are bitwise equal, in float64 and in float32.
- **G3 replay:** on the CPU, loading a saved state and taking one step reproduces the saved next state bitwise.
  R1 depends on this.

## 6. Pre-declared readings

**R0 (cross-machine, reported only):** CPU float64 here versus the laptop. Seed 0 is compared with the laptop's
4-decimal validation logs at every validation step, and seeds 4–6 with the top-free confirmation's step-500 table.

**R1 (one step, CPU float64 versus GPU float64):** start from each saved state in the windows and take one step
on the GPU, with the same batch and random numbers.
- **Parameter update: gate, at most 1e-8** (the lab default) for every step, by the relative error
  ‖Δθ_GPU − Δθ_CPU‖ / ‖Δθ_CPU‖. Steps are reported in two classes, plain and probe (where PARSEC-L re-extracts its
  directions).
- **State after the step: reported, no gate.** A SOAP basis refresh happens at the end of a step, so it does not
  enter that step's Δθ, but it does change the state the next step starts from. Upstream SOAP computes its
  eigenbases in float32 even for float64 parameters (`third_party/soap/soap.py`, `get_orthogonal_matrix` and
  `get_orthogonal_matrix_QR`), and it sorts tied eigenvalues with an unstable `argsort`. Bases in rank-deficient
  factors (for example the CLS token's 16 × 16 factor, built from rank-one gradients) are therefore set by rounding,
  and they can differ between devices by more than rounding. For refresh steps, report the largest change of the
  SOAP bases and of the rotated moments.
- **Held PARSEC-L factor: reported, no gate.** It is compared through its operator W̃ᵀW̃, because the signs of
  singular vectors follow the library's convention and differ between LAPACK and cuSOLVER (seen in a pre-run test:
  raw differences of exactly 2·max|W̃|, operators equal to about 1e-14).

**R2 (trajectory, GPU float64 versus CPU float64, seed 0):** report the validation-loss difference at every
validation step, raw and on the EMA. The difference at step 500 on the d = 0.98 EMA is the **device floor** used
in R3.

**R3 (float32, GPU float32 versus CPU float64):**
- **(a)** For each arm and each seed 0, 4, 5, 6: the step-500 EMA (d = 0.98) validation loss differs by at most
  1e-3. That is one seventh of the smallest per-seed margin (0.007) in the top-free confirmation. If the device
  floor from R2 itself exceeds 1e-3, (a) is reported but does not count towards the verdict.
- **(b)** The confirmation's reading P1 holds in float32: `topfree` + EMA is below `soap` + EMA at step 500 in
  each of seeds 4, 5 and 6.
- **Verdict:** float32 is adequate if (b) holds and (a) holds for all eight arm-seed pairs (or (a) is excused by
  the device floor as above). Otherwise it is not adequate.

**C1 (cost, reported; sets the CIFAR study's model and probe count):**
- **Models:** CIFAR-10 ViTs M1 (width 64, depth 4, 4 heads, MLP 256, about 0.21 M parameters) and M2 (width
  128, depth 6, 4 heads, MLP 512, about 1.2 M parameters).
- **Optimizers:** AdamW; SOAP, timed twice, as upstream `SOAP` and as PARSEC-L with rank 0 (the lab's `soap`
  arm), so the wrapper's own overhead is visible; and top-free PARSEC-L (c = 64) with s ∈ {1, 4}. Script:
  `scripts/cost.py`, one process per cell.
- **Setup:** GPU float32, batch 512, training-split images, weight decay 0.05, 10 warm-up steps then 60 timed
  steps.
- **Measured:** median seconds per step, separately for probe and non-probe steps; peak allocated GPU memory; and
  whether the unit runs out of memory. M2 at s = 4 is predicted to need about 10 GB of probes and to run out of
  memory.

## 7. Isolation and resources

- **Writes:** only to `studies/gpu_port_20261012/` and `~/skipsoap-study-data/gpu_port_20261012/`.
- **Hygiene:** `PYTHONDONTWRITEBYTECODE=1`; pytest with `-p no:cacheprovider`; isolation manifests and suites
  before and after.
- **Processes:** at most four CPU workers and one GPU process at a time, unless the owner sets other limits.

## 8. Cost

- **CPU:** 10 float64 units at about 1 minute each, single-threaded.
- **GPU:** 14 units of the small model (launch-bound; estimated 1–3 minutes each), plus C1 at about 16
  model-optimizer cells × 70 steps.
- **Total:** about 1 CPU-hour and 1 GPU-hour, roughly 30–45 minutes of wall time. Budget 2 hours.
