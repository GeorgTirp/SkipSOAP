# PARSEC-L: separate PyTorch implementation

A standalone implementation of PARSEC-L (ideas/parsec.md, section 1; ideas/PARSEC_ideas.md, section 2),
separate from the lab's `pns_eigenadam` port (`src/residual_curvature/optim/parsec.py`, unchanged). It is
a `torch.optim.Optimizer` built on the pinned upstream SOAP (`third_party/soap`) and works on any
module whose output is cross-entropy logits, including transformers with token-shared weights.
Nothing outside this folder was modified.

## The method

Per tensor, SOAP's step is −η_t Q(m̃/d), with bases Q, rotated momentum m̃ and d = √ṽ + ε. PARSEC-L
splits it into two half-steps around the whitened momentum z = m̃/√d and shrinks z:

    z' = z − V (I + VᵀV)⁻¹ Vᵀ z,        step = −η_t Q (z'/√d)

V Vᵀ is a rank-r approximation of the whitened GGN K_C = β F0ᵀ C F0, with F0 = Q diag(d^-½),
β = 1/(μ·scale), μ = max(γ tr C / D, floor) and scale = D / Σ_b tr(P_b⁻¹). These are the lab's
damping calibration and SOAP-metric normalization. For V = W diag(√κ) with orthonormal W the shrinkage
is z − W diag(κ/(1+κ)) Wᵀ z.

- **r = 0, or no probes ever given:** upstream SOAP, bit for bit.
- **r ≥ rank(K_C):** −η_t μ·scale (C + μ·scale P⁻¹)⁻¹ u, the damped GGN step in SOAP's metric applied to
  SOAP's momentum u.
- **One learning rate and one momentum,** both SOAP's. The pure shrinkage step (`norm_blend=0`) changes
  only the direction mix. The standard setting `norm_blend=1` then rescales the whole step to SOAP's
  global norm. The shrinkage acts in whitened coordinates, so the physical norm can end up on either
  side of SOAP's, and the rescale factor on either side of 1.

**Curvature source.** Probe rows u_j with Σ_j E[u_j u_jᵀ] = C. `ggn_probes` returns per-example
gradients with labels drawn from the model's own predictive distribution, scaled by 1/√(N s) for N
output rows and s draws. With `exact=True` it uses all K classes weighted by √p, which reproduces the
GGN exactly; this needs one output row per example. Probe rows are full per-example gradients, so
token-shared weights are handled correctly.

**At a probe step**, from the whitened probes Z = √β F0ᵀ Uᵀ:

| `refresh` | What it does |
| --- | --- |
| `'extract'` (default) | Randomized SVD, oversampling 8, 2 power steps: the scheme that passed offline (97.8%). |
| `'track'` | One warm-started power step from the held subspace plus Rayleigh–Ritz (the pasted design, I3/N6). Falls back to extraction when nothing is held, or when the invariance residual exceeds `residual_threshold`. |
| `'exact'` | Full SVD (tests, small tensors). |

**Between probe steps**, the low-rank curvature is held in physical coordinates and re-whitened with
the current F0 and β. It is transported exactly when SOAP changes a basis. μ is held from the last
probe step.

## Switches (all off by default; signals always logged in `optimizer.last_info`)

| Option | Effect | Default |
| --- | --- | --- |
| `rank`, `gamma`, `probe_every`, `refresh`, `oversample`, `power_steps`, `seed` | Core settings | 16, 0.01, 10, `'extract'`, 8, 2, 0 |
| `probe_until=N` | Hand-over (I7): from step N + 1 on, no probes and no correction, so plain SOAP. | None |
| `norm_blend=a`, a in [0, 1] | Rescales the whole step, globally and never per tensor, by (‖SOAP step‖ / ‖PARSEC-L step‖)^a. a = 1 restores SOAP's step norm so its tuned lr carries over (N12); a = 0 gives the pure shrinkage step. **Standard: 1** (owner decision, 9 Oct, on one-seed smoke evidence). | **1** |
| `batched` | Extracts equal-shape tensors with batched QR/SVD. `'auto'` batches off the CPU only: on one CPU thread the per-tensor loop measured 8–14% faster. | `'auto'` |
| `signals` | Computes ρ at probe steps (about 25% of a refresh); forced on by the ρ trigger. | True |
| `calibration='lr'`, `strength=c` | Learning-rate calibration (N17): β = c·lr·√(1 − β₂ᵗ). SOAP's bias-corrected step is then the proximal step of its own metric, and κ_k = c·lr·λ_k is the edge-of-stability ratio. With damping calibration, ηλ_max is still logged per tensor. | `'damping'`, 1 |
| `rhs='gradient'` | Shrinks the SOAP map of the current gradient instead of SOAP's momentum update. SOAP's statistics advance as usual. | `'momentum'` |
| `step(selection=f)` with `select_grid`, `select_refine` | Chooses the step length on a selection batch (`f()` returns its loss at the current parameters): α ∈ {0, 2^k}, then golden refinement in log α. | off; (−4, 8), 12 |
| `select_compare_soap`, `select_handover=k` | Also evaluates SOAP's own step on the selection batch and takes the better one; after k consecutive SOAP wins, switches permanently to plain SOAP (model-free hand-over). | off, None |
| `step(true_gradients=...)` | Logs the optimism ratio r for SOAP's metric and for PARSEC-L's operator. Uses per-example true-label gradients (`per_example_gradients`); the signal is cross-fitted from two halves. | off |
| `soap_refresh='triggered'` or `'both'` with `rho_threshold=τ` | Per-side SOAP basis refresh, using upstream's QR step for that mode only, when ρ (off-diagonal energy of the probe partial trace in SOAP's basis) has risen by τ since that side's last refresh. `'triggered'` disables SOAP's periodic refresh. | `'periodic'`, None |
| `residual_threshold=e` | Track mode: re-extract when the tracked subspace's invariance residual exceeds e. | None |
| `stop_energy=c`, `stop_patience=n` | Stop probing and continue as plain SOAP once the mean correction energy 1 − ‖z'‖²/‖z‖² between probe steps stays below c for n probe steps. | None, 3 |

**Branch scaling** (option 2 of the SN follow-up) is a model option, not an optimizer one:
`with_branch_scaling(config, exponent, alpha)` sets the lab models' fixed `branch_scale` to
α/depth^exponent. **Standard: exponent 1 (α/depth)** (owner decision, 9 Oct; one-seed smoke: it helped at depth
4, not at depth 2). **exponent = 0 (with α = 1) is the plain network**, bit for bit; 0.5 gives α/√depth.

**Block Jacobian deviation** (for the depth test, N14): `block_jacobian_deviation(model, x, probes,
power_steps)` measures how far each residual block's Jacobian R_k is from the identity at the actual
block inputs. It returns:

- the RMS singular value of R_k − I, from Gaussian JVP probes (forward mode only);
- the largest singular value, by power iteration with JVP and VJP.

Not implemented: SOAP bases built from the probe GGN (N9), `merge_dims`, `normalize_grads`.

## Usage

```python
from parsec_l import PARSECL, ggn_probes, with_branch_scaling
from residual_curvature.models import make_model

config = with_branch_scaling(config)      # standard: branch scale 1/depth; exponent=0. for the plain model
model = make_model(config)
optimizer = PARSECL(model.parameters(), lr=1e-2, betas=(.95, .95), weight_decay=0., rank=16, gamma=.01, probe_every=10)  # norm_blend=1 standard
for x, y in batches:
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(model(x), y).backward()
    probes = ggn_probes(model, x, samples=4)[0] if optimizer.wants_probes() else None
    optimizer.step(probes=probes)          # optimizer.last_info: correction energy, mu, kappa, residuals, rho
```

Run from `residual-curvature-lab/` with this folder on `sys.path`. The upstream SOAP import is
`third_party.soap.soap`.

## Verification

`tests/test_parsec_l.py`, 21 tests, all pass:

- **No curvature** (rank 0, or rank 16 without probes), 12 steps: parameters bitwise equal to
  persistent upstream SOAP. They agree with the lab's `soap_propose` chain to 2e-12 relative; that
  chain applies θ + (θ_new − θ) and reloads state every step.
- **Full rank with exact probes** (ViT, 16-example probe batch, after 4 SOAP steps): the step solves
  the lab's damped per-tensor system (C + μΩ) x = −u, built from `streamed_blocks`, `SOAPMetric` and
  `solve_blocks`, to a relative residual of 1.1e-12 (gate 1e-10).
  - The forward difference to the lab's own solution is 3e-9, consistent with condition numbers up to
    1.4e9 (input projection).
  - The step differs from native SOAP by more than 10%, so the check is not vacuous.
- **Truncated rank** (r = 4): equals the exact top-r shrinkage built from the exact blocks with the
  lab's `capacity.factor`, to 1e-8 relative.
- **Probes:** exact probes reproduce every exact GGN block to 1e-12; sampled rows equal autograd
  per-example gradients with the returned labels to 1e-12.
- **Track mode:** the first probe step extracts, later ones track, and `residual_threshold = 0` forces
  re-extraction. Tracking an exact invariant subspace keeps it (residual ≤ 1e-10, same κ).
- **Signals and triggers:**
  - ρ = 0 for probes diagonal in the basis;
  - a per-side refresh equals upstream's QR for that mode bitwise and carries the momentum over
    (1e-5, float32 bases);
  - the ρ trigger refreshes a side;
  - the stop rule stops probing and zeroes the correction.
- **Persistence:** a state-dict round trip continues bitwise.
- **Branch scaling:** exponent 0 leaves config and model output bitwise unchanged.
- **Hand-over:** after `probe_until`, steps equal upstream SOAP from the same state, bit for bit.
- **Norm blend:** at a = 1 the step norm equals SOAP's to 1e-10; at a = 0.5 it is the geometric mean of
  SOAP's and PARSEC-L's norms; the direction is unchanged (cosine 1 to 1e-12).
- **Batched extraction** equals per-tensor extraction to 1e-10, in a setting where the randomized range
  finder is exact. Turning ρ off leaves the step bitwise unchanged.
- **Learning-rate calibration** (c = 1 and 4): at full rank the step equals −η F0 (I + β F0ᵀCF0)⁻¹ F0ᵀu,
  built densely from the lab's blocks and factors, to 1e-9; the logged ηλ_max equals κ_max/c.
- **Gradient RHS** without curvature equals the SOAP map of the gradient to 1e-10.
- **Selection:** the step is α times the unselected step to 1e-12, and no grid point beats the chosen α.
  When SOAP's own step wins, the hand-over yields SOAP's parameters bit for bit and stops probing and
  selection.
- **Optimism diagnostic:** the optimism and signal terms for SOAP's metric and PARSEC-L's operator
  match a dense computation to 1e-10 and 1e-8.
- **SVD fallback:** when LAPACK's SVD does not converge, an eigendecomposition of the Gram matrix is
  used (this happened with exact probes under learning-rate calibration).
- **Block Jacobian:** zero for branch scale 0. The RMS estimate is within 10% of the exact Jacobian's,
  and the power-iteration spectral norm within 1e-6 of the exact one.

## Smoke run (one seed, untuned; not an evaluation)

`scripts/smoke.py` writes `logs/smoke.json`. Setup: main ViT, batch 512, seed 1, lr 1e-2 for both
arms (SOAP's tuned value), γ = 0.01, r = 16, s = 4 probes on the training batch every 10 steps.

| Arm | Val. loss 50 | Val. loss 100 | s/step (sequential) |
| --- | ---: | ---: | ---: |
| Native SOAP (this optimizer, rank 0) | 0.885 | 0.442 | 0.041 |
| PARSEC-L, extract | 0.719 | 0.343 | 0.085 |
| PARSEC-L, track | 0.720 | 0.338 | 0.080 |

- **SOAP check:** the SOAP row reproduces the gap audit's native SOAP for seed 1 (0.8848 and 0.4417).
- **Cost:** probes take 0.22 s per probe step; the correction adds about 0.02 s per step. For
  comparison, the exact per-tensor GGN takes about 4–6 s per step on this workload.
- **Correction energy:** rises from 0.24 to about 0.5–0.6.
- **Held-subspace staleness:** ten steps after a refresh, the held subspace's invariance residual is
  0.5–0.86, so a shorter refresh interval may matter.
- **Not yet a comparison:** this is one seed without tuning. The stage-D test is still to be done.

## Smoke ablations (one seed, untuned, 100 steps; not an evaluation)

`scripts/ablations.py` writes `logs/ablations.json`. Setup: main ViT, batch 512, seed 1, lr 1e-2, γ 0.01,
r 16, s = 4 probes every 10 steps. The `parsec_l` row used batched extraction (the default before `'auto'`).

| Arm | Val. 25 | Val. 50 | Val. 100 | Optimizer ms per probe step | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Native SOAP | 1.499 | 0.885 | 0.442 | — | |
| PARSEC-L | 1.261 | 0.717 | 0.343 | 291 | |
| Per-tensor extraction | 1.247 | 0.719 | 0.343 | 259 | Differs from batched only by random draws |
| ρ signal off | 1.261 | 0.717 | 0.343 | 214 | Same step as with ρ, about 75 ms cheaper |
| Hand-over at 30 | 1.261 | 0.701 | 0.375 | 293 | |
| Hand-over at 50 | 1.261 | 0.717 | 0.399 | 287 | |
| Norm blend 0.5 | 1.270 | 0.707 | 0.353 | 283 | Rescale factor mean 1.07, max 1.15 |
| Norm blend 1 | 1.245 | 0.672 | 0.356 | 285 | Rescale factor mean 1.14, max 1.28 |

Probes cost about 213 ms per probe step in every arm. Readings (n = 1, so only indications):

- **Hand-over:** handing over to SOAP at step 30 or 50 makes step 100 worse (0.375 and 0.399 versus
  0.343). Within 100 steps, the correction still helps after step 50.
- **Norm blend:** PARSEC-L's step is on average only 12% shorter than SOAP's. Restoring SOAP's norm helps
  early (0.672 at step 50) and slightly hurts at step 100.
- **CPU cost:** batched extraction is slower than the per-tensor loop on one CPU thread. Hence `'auto'`.

**Branch scaling and block Jacobians** (PARSEC-L, same settings). The RMS deviation ‖R_k − I‖ per block is
measured on 64 validation examples, with spectral norms from 20 power steps:

| Depth, scale | Val. 100 | RMS per block, step 0 | RMS per block, step 100 | Spectral per block, step 100 |
| --- | ---: | --- | --- | --- |
| 2, s = 1 (plain) | 0.343 | 1.72, 0.49 | 2.96, 1.47 | 23.9, 10.5 |
| 2, s = 1/2 | 0.384 | 0.93, 0.48 | 2.34, 1.17 | 24.7, 7.9 |
| 4, s = 1 (plain) | 0.396 | 1.72, 0.49, 0.39, 0.31 | 2.48, 1.11, 0.76, 0.57 | 19.3, 8.8, 5.6, 5.8 |
| 4, s = 1/4 | 0.319 | 0.54, 0.41, 0.36, 0.31 | 0.58, 0.64, 0.56, 0.56 | 4.4, 4.3, 3.7, 3.9 |

- **Blocks are far from identity:** in the plain networks they already are at initialization (the first
  block's RMS deviation is 1.7) and move further during training.
- **Likely cause** (untested): the CLS token enters the first block with a tiny norm (CLS and
  position embeddings are initialized with std 0.02), and pre-LayerNorm divides by that token's scale.
- **Depth 4 with 1/depth scaling** keeps every block near 0.6 RMS and below 4.4 spectral, against 2.5 and
  19 when plain. It also had the lowest validation loss at step 100.
- **At depth 2** the same scaling did not help.
- These are single runs; the depth test (N14) needs seeds and tuning.

## Files

- `parsec_l/optimizer.py`: `PARSECL`.
- `parsec_l/probes.py`: `ggn_probes`.
- `parsec_l/branch_scaling.py`: `depth_branch_scale`, `with_branch_scaling`.
- `parsec_l/diagnostics.py`: `block_jacobian_deviation`.
- `tests/`: run with
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider studies/parsec_l_20261009/tests`.
- `scripts/smoke.py`, `logs/smoke.json`; `scripts/ablations.py`, `logs/ablations.json`.
