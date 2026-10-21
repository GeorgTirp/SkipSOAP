# Top-free PARSEC-L with OT-metric probes and with probes from another batch — protocol

Study `probe_variants_20261013`. Written on 13 October 2026, before any run.

**Request.** The owner asked for two small runs on 13 October:
- "one small PARSEC run with the OT metric instead of GGN so we can close the chapter";
- "something a bit crazy: what if for generalization over batches the probe measures were made on other batches
  than SOAP?"

Both change only where top-free PARSEC-L's curvature directions come from. Everything else is the CIFAR top-free
arm of `cifar_ggn_20261012`.

## 1. Questions

1. **OT:** does top-free PARSEC-L train better when its probes sketch the OT metric (the lab's dynamic graph
   Wasserstein pullback) instead of the GGN?
2. **Other batch:** does it train better when its GGN probes come from an independent batch rather than the batch
   of the step? The idea: same-batch curvature is fitted to that batch (rule 8, N20), and curvature from another
   batch reflects the population.

## 2. Workload

`cifar_ggn_20261012`'s:
- **Data:** CIFAR-10, 45k/5k split of the training set; the test split is never accessed.
- **Model and training:** M1d6 ViT; batch 2048; 500 steps with warm-up and WSD decay; weight decay 0.05; EMAs.
- **Numerics:** GPU float32, deterministic.

The runner is `n9_training_20261013`'s, with its `topfree` path unchanged. It reproduced `cifar_ggn`'s units
bitwise in CPU tests, and in G1 there.

## 3. Arms

All arms are top-free PARSEC-L: rank 16, `top_rhs='gradient'`, lr calibration, lr 3e-2. Probes are drawn at steps
2, 12, 22, …

| Arm | Probe rows (s = 1 per example) | c |
| --- | --- | --- |
| `topfree` | sampled-label GGN probes of the step's batch (`micro.lean_probes`), as in `cifar_ggn` | 64 |
| `topfree_ot` | **OT-metric probes** of the step's batch (below) | tuned on seed 0 ∈ {64, 256, 1024, 4096} |
| `topfree_xbatch` | sampled-label GGN probes of an **independent batch**: 2,048 examples from a separate data order (seed 7102 + 1000·seed), augmented with their own generator (6000 + seed), drawn only at probe steps, at the step's parameters | 64 |

**OT-metric probes.** The lab's metric (`curvature/wasserstein.py`, used in the gap audit's OT arms) is
M = (1/N) Σ_i J_p,iᵀ (L(p_i) + 11ᵀ/K)⁻¹ J_p,i. Here:
- J_p = (diag p − ppᵀ) J_z is the Jacobian of the probabilities;
- L(p) is the Laplacian of the complete class graph with edge mobility (p_j + p_k)/2.

With (L + 11ᵀ/K) = RRᵀ (Cholesky), the probe row of example i is the gradient of z_iᵀv_i, with
v_i = (diag p_i − p_ip_iᵀ) R_i⁻ᵀ ξ_i / √N and ξ_i ~ N(0, I_K) drawn on the CPU from the probe generator. Then
Σ_i E[u_iu_iᵀ] = M. The cost is the same per-example backward pass as the GGN probes.

**Why OT gets a grid over c.** Near uniform predictions the OT curvature is about 1/K of the GGN's, and it shrinks
faster as predictions sharpen. With c fixed, the brake κ = c·lr·λ would be much weaker for OT. The grid spans 1×
to 64× of top-free's c, so that OT is compared at its best brake strength, not at GGN's. The cross-batch arm keeps
c = 64: its probes estimate the same curvature, from other examples.

## 4. Tuning and evaluation

- **Tuning:** `topfree_ot` on seed 0 over c ∈ {64, 256, 1024, 4096} at lr 3e-2.
  - Selection: the lowest step-500 EMA validation loss over the three decays.
  - If a grid edge is selected (64 or 4096), one extension unit beyond it (c = 16 or 16384) is run, and the
    selection is repeated.
- **Evaluation:** seeds 1–3 for `topfree`, `topfree_ot` (selected c) and `topfree_xbatch`. That is 9 units.
- **Records:** validation at steps 0, 10, 25, 50, …, 500; training records every step, including PARSEC-L's probe
  information.

## 5. Gates

- **G0:** suites before and after:
  - the repository (106), PARSEC-L (22), `cifar_ggn` (11), `cifar_sn_soap` (12), `basis_refresh_snr` (9) and
    `n9_training` (9);
  - this study's tests: OT probes against the lab's dense Wasserstein blocks; GGN probes unchanged; the
    cross-batch stream; `topfree` equal to `cifar_ggn`'s bitwise over CPU steps.
- **G1:** the `topfree` evaluation units reproduce `cifar_ggn`'s (`20261012T110529-d747f9fc`) bitwise in
  `validation.json` and `training.json` (timings excluded).
- **Divergence** is reported, not a gate failure, and counts as worse.

## 6. Pre-declared readings (seeds 1–3)

*Better* or *worse* means lower or higher validation loss in all three seeds; otherwise *mixed*.

| Code | Reading |
| --- | --- |
| P1 | **OT vs GGN:** `topfree_ot` vs `topfree`, raw at steps 50 and 200, EMA at 500. *OT is the better curvature source here* only if better at EMA 500. Otherwise the OT line is closed for PARSEC-L. |
| P2 | **Other batch vs same batch:** `topfree_xbatch` vs `topfree`, same points. *Curvature from another batch is better* if better at EMA 500. |
| P3 | Descriptive: the tuning table over c; seed-mean curves; κ_max and the correction energy at probe steps; seconds per step. |

## 7. Cost, isolation and limitations

- **Cost:** about 4 min per unit, 13 units, so about 1 GPU-hour.
- **Writes:** only to `studies/probe_variants_20261013/` and `~/skipsoap-study-data/probe_variants_20261013/`.
  Earlier studies are imported, never modified. Hygiene as before.
- **Limitations:**
  - **One lr:** lr is not retuned for the variants (top-free's 3e-2), and only OT's c is tuned.
  - **Batch budget:** the cross-batch arm sees 2,048 extra examples every 10 steps.
  - **Scale:** one model, three seeds, 500 steps.
