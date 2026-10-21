# Power-law spectra and cheap bulk preconditioning at CIFAR top-free states — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, fixed before any run. One deviation:
[DEVIATIONS.md](DEVIATIONS.md) D1. The first attempt crashed in a fit helper on `head.bias` (10 entries) before
any output; the helper was fixed and a test added, then everything was rerun from the start.

Outputs: `results/bulk/topfree_s<seed>_t<step>.json` (copies of `~/skipsoap-study-data/bulk_krylov_20261014/bulk/`)
and `results/summary.json` / `summary.md`, from `scripts/summarize.py`.

**Status (2026-10-14):** all 12 states completed in about 61 s each. Suites passed before and after (14 suites,
including this study's 10 tests). The isolation manifest is unchanged.
- **Gate L0** passed in every state.
- **Consistency with the ladder:** R* and top-free's per-tensor reductions equal `ladder_mechanism`'s bitwise in all
  12 states, with the same μ.
- **Spectrum accuracy:** the largest Ritz residual of the 200th eigenpair is 1.7e-3 (relative).

The early states are steps 10, 25 and 50 of seeds 1–3; step 100 is descriptive. "Share" is the pooled share of the
gap G between top-free's step and the exact per-tensor GGN step on the damped per-tensor quadratic.

## 1. Summary

- **P1, single power law: not supported.** R² ≥ 0.95 holds in at least 80% of matrix tensors only from step 50 on.
  - At steps 10 and 25 the share is 0.67–0.80, although the median R² is 0.96–0.98.
  - At steps 10 and 25 the spectrum is a **broken power law:** the top 16 fall as about k^−1 (α_top 0.70–1.02).
    Ranks 17–200 fall about twice as steeply (α_bulk 1.79–2.10).
  - The break flattens with training. By step 100, α_top is 0.63–0.68 and α_bulk 0.99–1.01.
  - Early on, the bulk exponent grows with depth: at step 10 from 1.2 (input) to 2.8 (block 5) and 3.3 (head).
- **P2, bulk exponent from the probes: not supported.** α_trace is within 0.15 of α_bulk in only 3–15% of matrix
  tensors early (46–62% at step 100).
  - The probes themselves are accurate:
    - λ₁: median true/probe ratio 0.91–1.02;
    - trace: 0.97–1.02;
    - top-16 slope: median α_slope within 0.025 of median α_top in every state.
  - What fails is the single-law extrapolation. It underestimates λ₁₇ by a median factor of 1.6–2.4.
- **K1, spectrum only is not enough: supported.**
  - The owner's power-law diagonal closes 1.7–18% of G (median 5.9%).
  - It matches the exact (Jacobi) diagonal, diag K + δ (median 6.4%). So the power law assigns the diagonal about
    as well as the exact diagonal does, and that level is low.
  - **Correction (2026-10-14, after the run):** the protocol calls `diag_exact` "the ceiling of any diagonal in
    SOAP's basis". It is not a mathematical ceiling. A diagonal fitted to the current gradient can reproduce any
    step, the exact one included. It is the exact-diagonal (Jacobi) baseline for gradient-independent diagonals.
- **K2, bulk reachable with few products: m = 8.**
  - Deflated Krylov closes **67–76%** of G with 8 products in every early state.
  - With 4 products: 47–63% (median 51%), below half in 3 of 9.
  - With 2 products: 31–52%.
  - The top-16 deflation adds 22–25 points at every m. Plain Krylov reaches medians of 13%, 26% and 47%.
- **K3, Chebyshev with the probe interval: not supported.**
  - Shares range from −21% to +20%.
  - In every state the global step is not a descent direction (α₀ = 0), so its held-out decrease is 0.
  - Because λ₁₇ is underestimated, bulk eigenvalues above the interval are amplified.
- **K4, cheap online full-GGN CG: not supported in all 9 states.**
  - 5 CG steps with the full GGN of 256 examples beat top-free on held-out loss in 7 of 9 early states. It lost at
    step 50 for seeds 1 and 2.
  - Median held-out decrease: 0.068 vs top-free's 0.024, the same as the exact per-tensor GGN step (0.068).
  - Its per-tensor share is low (median 5.6%) because it solves the coupled full-GGN system, not the per-tensor
    blocks.

## 2. Part K: shares and held-out decreases (9 early states)

| Rung | Products per tensor | Share of G, median (range) | Held-out decrease, median | Beats top-free held-out |
| --- | ---: | --- | ---: | ---: |
| top-free | — | 0 | 0.024 | — |
| exact per-tensor GGN | dense | 1 | 0.068 | 9/9 |
| `diag_exact` | — | 0.064 (0.019–0.189) | 0.038 | 4/9 |
| `diag_powerlaw` | — | 0.059 (0.017–0.184) | 0.028 | 4/9 |
| `krylov_2` / `_4` / `_8` | 2 / 4 / 8 | 0.13 / 0.26 / 0.47 | 0.032 / 0.047 / 0.066 | 5/9 / 8/9 / 8/9 |
| `deflated_krylov_2` / `_4` / `_8` | 2 / 4 / 8 | **0.35 / 0.51 / 0.70** (min 0.31 / 0.47 / 0.67) | 0.058 / **0.075 / 0.078** | 8/9 each |
| `deflated_chebyshev_4` / `_8` | 4 / 8 | 0.02 / 0.01 (−0.21 to 0.20) | 0 | 0/9 |
| `full_cg_sub` (256 examples) | 5 (full GGN) | 0.056 (−0.15 to 0.13) | 0.068 | 7/9 |
| `full_cg_full` (2,048 examples) | 5 (full GGN) | 0.065 (−0.17 to 0.13) | 0.076 | 7/9 |

- **Step 100 (descriptive):** deflated Krylov with 8 products closes 76–79%, and with 4 products 54–56%.
- **The one loss of the deflated Krylov rungs:** state s1_t50, where most rungs' held-out decreases are about
  0.18–0.20, top-free's included.
- **Truncated solves generalize as well as the exact one:** the deflated Krylov steps with 4 or 8 products have
  larger held-out decreases than the exact per-tensor step (medians 0.075 and 0.078 vs 0.068).

**Cost (K5):** the median time per full-GGN product, including CG's vector operations and the whitening of every
tensor, is **0.070 s at 256 examples** and **0.46 s at 2,048**. A SOAP step takes 0.22 s. Five sub-batch products
(0.35 s) therefore cost about 1.6 SOAP steps. Peak GPU memory was 4.7 GiB.

## 3. Part P: spectra (medians over matrix tensors)

| State | P1 fraction | P2 fraction | median R² | α_top | α_bulk | α_trace | α_slope | true/predicted λ₁₇ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1_t10 / s2_t10 / s3_t10 | 0.79 / 0.69 / 0.67 | 0.03 / 0.03 / 0.03 | 0.975–0.98 | 0.93–1.02 | 2.01–2.10 | 1.07–1.14 | 0.93–1.02 | 1.57–1.68 |
| s1_t25 / s2_t25 / s3_t25 | 0.77 / 0.77 / 0.67 | 0.05 / 0.08 / 0.05 | 0.96–0.98 | 0.70–0.96 | 1.79–1.93 | 1.00–1.09 | 0.72–0.96 | 1.64–2.44 |
| s1_t50 / s2_t50 / s3_t50 | 0.95 / 0.97 / 0.95 | 0.13 / 0.15 / 0.13 | 0.97 | 0.71–0.78 | 1.42–1.53 | 0.95–0.98 | 0.71–0.79 | 1.84–1.98 |
| step 100 (descriptive) | 0.95–0.97 | 0.46–0.62 | 0.99 | 0.63–0.68 | 0.99–1.01 | 0.83–0.87 | 0.63–0.69 | 1.63–1.86 |

**P3** (`results/summary.json`, `readings.P3`):
- **Depth:** at step 10, α_bulk rises with depth, from 1.23 at the input through 1.67 (block 0) and 2.81 (block 5)
  to 3.33 at the head. By step 100 the range is 0.78–1.17, with the head at 1.59.
- **Tensor kind:** query and key have the steepest top (α_top about 1.25 at step 10). Among the block tensors, the
  MLP, value and output projections have the steepest bulk (2.25–2.5 at step 10); the head is steeper still (3.3). Position embedding and patch projection are close to a single
  power law throughout (α_top and α_bulk within 0.4).

## 4. Interpretation and limitations

- **For the owner's power-law idea:**
  - **The spectrum is roughly a power law, with an early break.** The probes measure its top accurately, but a
    single law fitted from the top and the trace does not predict the bulk.
  - **A spectrum without directions gains little.** Even assigned well, it reaches only the exact-diagonal level of
    about 6%.
- **For cheap bulk preconditioning:**
  - **The directions are cheap to find with products.** The top-16 probe subspace plus 4–8 deflated Krylov
    products reaches half to three quarters of the per-tensor GGN advantage. On held-out loss these steps match or
    beat the exact step.
  - **The cheapest realistic version** is 5 CG steps on the full GGN of 256 examples, matrix-free and preconditioned
    by SOAP's whitening. It matches the exact per-tensor step on held-out loss at median, but not in every state.
  - **A Chebyshev polynomial needs a measured interval, not a power-law guess.** For example, a few Lanczos steps on
    the deflated operator could supply it. That is not tested here.
- **What is not shown:**
  - **Training benefit.** All results are one step ahead at fixed states. The exact GGN step wins early in training
    only (rule 9), and its CIFAR training arms stopped at the float32 audit gate.
  - **Online per-tensor products.** Offline the blocks are dense. Online, per-tensor products would need probe
    rows or per-tensor Jacobian products, which are not costed here.
- **Limitations:**
  - **States:** 12, at steps 10–100 of one top-free run per seed.
  - **Spectrum:** the top 200 eigenvalues only, for tensors of up to 16,384 entries.
  - **Model:** the damped per-tensor quadratic as the score.
  - **Held-out check:** a single selection batch and a single evaluation batch per state.
