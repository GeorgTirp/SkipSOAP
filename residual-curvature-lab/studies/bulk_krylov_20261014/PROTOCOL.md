# Power-law spectra and cheap bulk preconditioning at CIFAR top-free states — protocol

Study `bulk_krylov_20261014`. Written on 14 October 2026, before any run. Offline.

**Request.** The owner asked on 14 October: "Do the GGN OT fairness test and the power-law experiment now". This
followed their question whether a power law of the GGN spectrum, fitted from the top-k probes together with SOAP's
statistics, could give the correct bulk preconditioning cheaply.
- The owner cited Meterez et al. 2026 for power-law eigenvalue distributions within a model; that paper was not
  available to verify here.
- The proposed answer was that a spectrum alone cannot supply the bulk directions, but a few curvature-vector
  products (Krylov) can, with the top-k probes as deflation and the power law for damping and polynomial design.

This study tests both halves.

## 1. Questions

1. **Power law:** is the whitened per-tensor GGN spectrum a power law? Is its exponent stable across tensors, depth
   and training time? Can it be predicted from the top-16 probe eigenvalues and the probe trace?
2. **Spectrum only:** can a diagonal built from the fitted power law recover the bulk gain? The power-law values are
   assigned to whitened coordinates by the rank of the probe-estimated diagonal.
3. **Krylov:** how many curvature-vector products, with and without top-16 deflation, recover the gap between
   top-free's step and the exact per-tensor GGN step? Does a power-law-informed Chebyshev polynomial do as well
   without inner products? Does a full-GGN SOAP-preconditioned CG on a 256-example sub-batch, which is cheap
   online, carry the gain to held-out loss?

## 2. States and quantities

- **States:** `ladder_mechanism_20261013`'s replayed top-free states (verified bitwise against `cifar_ggn` there):
  seeds 1–3 at steps 10, 25, 50 and 100, read-only from `~/skipsoap-study-data/ladder_mechanism_20261013/states/`.
  The batch of the next step is used, with that study's selection and evaluation batches.
- **Per tensor b, in SOAP-whitened coordinates (as in the ladder):**
  - K_b, the exact whitened GGN block;
  - A_b = K_b + δI, with δ = μ·scale and μ = 1e-2·tr C / D;
  - z_g, the whitened gradient;
  - y* = −A_b⁻¹z_g with reduction R*;
  - the reference gap G = ΣR* − ΣR(topfree), where top-free's own next step is whitened.
- **Probes:** s = 1 per example (`micro.lean_probes`), whitened. Per tensor they give:
  - U_fresh, the top-16 directions (PARSEC-L's randomized SVD);
  - κ̂_1…κ̂_16, their eigenvalues (squared singular values);
  - the probe trace T̂ = ‖Z‖²_F;
  - the probe diagonal d̂_i = Σ_j z_ji².

## 3. Part P: the power law

**Exact spectrum.** For each matrix tensor: the top 200 eigenvalues of K_b (exact eigh up to 512 columns, otherwise
subspace iteration with 16 iterations and Rayleigh–Ritz), and tr K_b.
- **Fits:** α_top is the log-log slope of λ_k vs k over k = 1…16; α_bulk the slope over k = 17…200. Each comes with
  the R² of its fit.

**Probe prediction:**
- α_slope: the log-log slope of κ̂_k over k = 1…16;
- α_trace: solves Σ_{k=1}^{D_b} κ̂_1 k^{−α} = T̂ (bisection);
- λ̂_17 = κ̂_1·17^{−α_trace}, the predicted top of the bulk.

**Readings** (9 early states: steps 10, 25 and 50, seeds 1–3; step 100 is descriptive):

| Code | Reading |
| --- | --- |
| P1 | **Power law holds:** in every early state, R² ≥ 0.95 for the fit over k = 1…200 in ≥ 80% of matrix tensors. |
| P2 | **Predictable from probes:** in every early state, \|α_trace − α_bulk\| ≤ 0.15 in ≥ 80% of matrix tensors. |
| P3 | Descriptive: α_top and α_bulk by tensor type and depth, and their change over steps. |

## 4. Part K: bulk preconditioning rungs

All rungs are scored by the pooled share of G, (ΣR_X − ΣR(topfree))/G, with Galerkin reductions for subspace rungs and
ray reductions otherwise (as in the ladder).

| Rung | Construction (per tensor unless stated) |
| --- | --- |
| `diag_exact` | y = −z_g / (diag K_b + δ): the ceiling of any diagonal in SOAP's basis |
| `diag_powerlaw` | y = −z_g / (λ̂_rank(i) + δ), with λ̂_k = κ̂_1 k^{−α_trace} assigned by the rank of d̂_i: the owner's spectrum-only idea |
| `krylov_m`, m ∈ {2, 4, 8} | Galerkin in span{z_g, A z_g, …, A^{m−1} z_g} (the m-step CG iterate) |
| `deflated_krylov_m`, m ∈ {2, 4, 8} | Galerkin in U_fresh ∪ span{P z_g, (PAP)P z_g, …}, with P = I − U_freshU_freshᵀ (deflated CG) |
| `deflated_chebyshev_m`, m ∈ {4, 8} | exact solve in U_fresh, plus the degree-m Chebyshev approximation of (λ + δ)⁻¹ on [δ, λ̂_17 + δ] applied to P z_g through products with PAP. No inner products; the interval comes from the probe power law. |
| `full_cg_sub` | across all tensors at once: 5 steps of CG on F0ᵀ C_sub F0 + δI (whitened, so SOAP-preconditioned), with C_sub the full GGN of the first 256 examples of the batch (matrix-free forward and backward products) |
| `full_cg_full` | the same with the full 2,048-example batch (reference for the sub-batch) |

**Held-out check:** for each rung's global step, α₀·2^k with k ∈ {−14, …, 2} is chosen on the selection batch, and the
decrease is measured on the evaluation batch. Whether the choice is interior is reported.

**Readings** (9 early states):

| Code | Reading |
| --- | --- |
| K1 | **Spectrum only is not enough:** `diag_powerlaw` closes < 20% of G in all 9 states. |
| K2 | **Bulk reachable cheaply:** the smallest m ∈ {2, 4, 8} for which `deflated_krylov_m` closes ≥ 50% of G in all 9 states, or none. |
| K3 | **Polynomial vs Krylov:** whether `deflated_chebyshev_8` closes ≥ 50% of G in all 9 states. |
| K4 | **Cheap online candidate:** whether `full_cg_sub`'s held-out decrease exceeds top-free's in all 9 states; its gap share is reported. |
| K5 | Descriptive: every rung's share and held-out decrease; time per full-GGN product at batch 2,048 and 256. |

**Gate L0** (as in the ladder): no rung's per-tensor reduction exceeds R* by more than 1e-3 relative. A violation stops
the state.

## 5. Gates, cost, isolation and limitations

- **G0:** suites before and after, including this study's tests: Krylov and Galerkin against dense solves; deflated
  CG; Chebyshev against the exact inverse on its interval; the power-law fit and the trace bisection; whitened CG
  against a dense solve; state loading.
- **Cost:** about 1.5 min per state, 12 states. **About 25 GPU-minutes.** It runs after `ggn_ot_fairness_20261014`.
- **Writes:** only to `studies/bulk_krylov_20261014/` and `~/skipsoap-study-data/bulk_krylov_20261014/`;
  `ladder_mechanism`'s snapshots are read only.
- **Limitations:**
  - **Model:** one step ahead, on the damped per-tensor quadratic.
  - **Online cost:** per-tensor Krylov products are dense offline. Online they would need per-tensor products, for
    example from stored probe rows, which K5 does not cost.
  - **Spectrum:** the top 200 eigenvalues are a partial spectrum for the large tensors.
