# Spectral normalization, distance preservation and curvature (A–C)

Study ID `sn_curvature_20261008`. Selected by the project owner on 8 October 2026. The owner
found that SNGP's spectral normalization (SN) substantially improves Laplace approximations in an
uncertainty paper, and asked how distance preservation and SN interact with curvature and
curvature estimation in this lab. Owner decisions: SNGP-style SN first; a small bound grid; both
the ViT (primary, since later work targets transformers) and the residual MLP; parts A–C only
(no Laplace predictive, no training comparisons; part E is decided afterwards). This file is the
prospective protocol, drafted before any run of this study (moved into this folder at 13:05Z
after the geometry follow-up finished); it is not edited after the first
run, and departures go to `DEVIATIONS.md`.

## Questions

- **A, distance preservation.** How strongly does SN make the learned representation
  distance-preserving (bi-Lipschitz) in this ViT and MLP during training?
- **B, curvature structure.** How does SN change the exact GGN: spectrum, effective rank,
  cross-tensor coupling, Kronecker approximability, the exact-GGN local headroom over the SOAP map,
  and the null directions that SN's scale invariance creates?
- **C, curvature estimation.** How does SN change the error of standard curvature estimates
  relative to the exact GGN: diagonal, Kronecker, empirical Fisher, Monte Carlo Fisher, Hutchinson
  trace and low-rank truncation?

## Isolation

As in the earlier side studies: write only to this folder and `$SN_CURVATURE_DATA` (default
`~/skipsoap-study-data/sn_curvature_20261008/`); read/import only from the main lab; never modify
the main lab's protected paths or `studies/gap_audit_20261007/` or
`studies/geometry_followup_20261008/`; `PYTHONDONTWRITEBYTECODE=1`; pytest with
`-p no:cacheprovider`; no git writes, packages or downloads; no official MNIST test split;
shared-code bugs reported, not fixed. Isolation manifest of everything outside this folder written
first (2026-10-08T13:03:56Z, 1,563 files, including the finished gap-audit and geometry folders) and
compared at the end. Copied code: `families.py` from the gap audit (SHA-256 b1dee84a...), `common.py`
from the geometry follow-up (88f0f9f1..., paths changed). At most four concurrent single-threaded processes in total.
Numerical tolerance 1e-8; study checks use residual, action or energy criteria only. `HANDOFF.md`
is not edited; a `HANDOFF_PROPOSED_ENTRY.md` is written here.

## Models and spectral normalization

- **ViT:** `configs/mini_vit_allocation.yaml` (5,738 parameters, 40 tensors, MNIST 28x28).
- **MLP:** `runs/20261002T121635-0dac3337/resolved.yaml` (residual MLP, MNIST 14x14, width 16,
  expansion 32, 3 blocks; 6,634 parameters, 22 tensors).
- Batch 512, CPU float64, one thread, deterministic algorithms, split seed 101.

**SN map (SNGP).** For a hidden weight W, W_eff = W · min(1, c / σ(W)), with σ the largest singular
value. Applied to the linear weights inside the residual branches: ViT
`blocks.*.attention.{query,key,value,output}.weight` and `blocks.*.{w1,w2}.weight`; MLP
`blocks.*.{w1,w2}.weight`. Not applied to the input projection, embeddings, LayerNorm, biases or
the head. No GP output layer. σ is computed exactly (`torch.linalg.matrix_norm(W, 2)`, an SVD)
instead of SNGP's one-step power iteration, so every forward pass is deterministic and stateless.
The gradient flows through σ, which matches the converged power-iteration gradient u vᵀ. The map is
a `torch.nn.utils.parametrize` parametrization; the trainable parameters are the raw W.
Conditions: no SN, and c ∈ {0.95, 2, 6}. Initial raw weights are identical across conditions
(same seed).

The bi-Lipschitz argument needs each residual branch to be a contraction. With LayerNorm and GELU
inside the branches, SN on the linear layers bounds the branch's Lipschitz constant only up to
those factors, and dot-product attention is not globally Lipschitz. A therefore measures distance
preservation empirically instead of assuming it.

## Trajectories (context only, no rankings)

Seeds s ∈ {0, 1} (model seed s, data order seed 102 + 1000 s). Searched SOAP with default settings
(lr .003 native radius, betas .95/.95, preconditioning frequency 10, global ray search
`search_candidate`, zero plus 2^[-16..16] native-radius multipliers and 18 golden refinements),
200 consumed steps; SOAP sees the raw parameters. Snapshots after steps 10, 50 and 200. Validation
on the 5,000 reserved examples at steps 0, 10, 25, 50, 100, 150, 200, recorded as context. Units:
2 models x 4 conditions x 2 seeds = 16 trajectories, 48 states.

**Reproduction gates (≤ 1e-9; > 1e-6 stops the study).** No-SN ViT seed 0: parameters at steps 10
and 50 equal `~/skipsoap-study-data/gap_audit_20261007/runs/20261007T125950-63edd820/b2_w16__s0`
states (maximum absolute difference). No-SN MLP seed 0: validation losses at steps 10/25/50 equal
`runs/20261002T121635-0dac3337` `searched_soap` (0.9028134971683025, 0.4536378557680572,
0.4911033240812833). Study test: SN with c = 1e9 (inactive) gives bitwise-identical logits.

## Diagnostics at every saved state

All on the 512-example construction batch (first 512 analysis IDs) unless stated. The exact GGN,
empirical-Fisher and sampled-Fisher blocks come from one `information_blocks` pass with a declared
tensor-workspace budget of 3,072 MiB (the MLP's 3,136-parameter input tensor needs it; the repository
default of 768 MiB applies to training-time curvature only). The GGN is taken
with respect to the **raw trainable parameters** (what SOAP and GGN steps act on); the GGN with
respect to the **effective weights** (the same network with W_eff as plain weights) is reported
alongside for B and C.

**A. Distance preservation.**
- Representations: φ_res, the residual stream after the last block (ViT: CLS token before the final
  LayerNorm; MLP: the block output), and φ_head, the input to the head (ViT: after the final
  LayerNorm; MLP: equal to φ_res).
- Global ratios r = ‖φ(x)−φ(x′)‖ / ‖x−x′‖ for 2,000 fixed random pairs from analysis IDs 512–2559
  (disjoint from construction and panels): quantiles 1/5/50/95/99% and distortion q95/q5.
- Local ratios for 512 construction inputs with x′ = x + ε ξ, ‖ξ‖ = 1 (fixed Gaussian
  directions), ε = 1e-3 ‖x‖: the same quantiles.
- Per-layer σ(W) and σ(W_eff), and the share of SN-active layers (σ(W) > c).

**B. Curvature structure.**
- Exact full GGN (all cross-tensor blocks; dense assembly from streamed Jacobian factors, checked
  against `GGN.action` and the exact tensor blocks at 1e-8): eigenvalues by `eigvalsh`; top
  eigenvalue, trace, share of trace in the top 16 eigenvalues, entropy effective rank, numerical
  rank (eigenvalues > 1e-10 x top).
- Cross-tensor energy: off-diagonal tensor-block share of ‖C‖_F².
- SN null directions: for each SN-active tensor, ‖C_bb vec(W_b)‖ / (‖C_bb‖₂ ‖W_b‖), raw
  parameterization (expected ≈ 0 by scale invariance).
- Kronecker approximability: GGN-energy-weighted relative Frobenius error of the nearest Kronecker
  product (Van Loan–Pitsianis, rank 1) over 2-D tensors, and of the Kronecker-eigenbasis diagonal
  (`kron_basis_diag`).
- Local headroom: independent decreases (four analysis panels, `measurement`) of the SOAP
  current-gradient map, exact tensor GGN, `soap_basis_diag` and `kron_basis_diag`, each with
  `search_candidate` at the native radius; damping gamma .01 with exact trace and the normalized
  inverse frozen SOAP metric (as in the gap audit's 2c/4b). Reported: exact-GGN headroom over the
  SOAP map and each family's recovery.

**C. Curvature estimation (each versus the exact tensor GGN on the same batch).**
- Exact diagonal; nearest Kronecker (above); empirical Fisher (per-example gradients); Monte Carlo
  Fisher with 1, 4 and 16 model-sampled labels (`information_blocks`, two repeats); Hutchinson trace
  with 8 Rademacher probes (16 independent repeats, relative RMS error); rank-16 truncation of the
  full GGN (eigen, relative Frobenius error).
- Error measures: relative Frobenius error pooled over tensor blocks; for the diagonal,
  Kronecker-basis diagonal, empirical Fisher and MC-1 Fisher also the damped block-teacher A-energy
  ray errors on the gradient and on 16 factor-coordinate Gaussian RHS
  (`optim.capacity.energy_errors`), damping as in B.

## Pre-declared readings (descriptive)

For each model, condition c and metric, compare with no SN at the matched (seed, snapshot): six
pairs per model and c. *Consistently lower* if lower in all six, *consistently higher* if higher in
all six, otherwise *mixed*. Pre-declared metrics: distortion q95/q5 (global and local, φ_res and
φ_head); top GGN eigenvalue; effective rank; cross-tensor energy share; nearest-Kronecker error;
diagonal error; empirical-Fisher error; MC-1 Fisher error; Hutchinson-8 error; rank-16 truncation
error; A-energy ray errors (gradient) of diagonal, Kronecker-basis diagonal and empirical Fisher;
exact-GGN headroom; `kron_basis_diag` recovery. Raw and effective parameterizations are reported
separately. These are descriptive comparisons on two seeds, not significance tests; trajectories
under SN differ, so differences combine the direct effect of SN on curvature with its effect on
where training goes.

## Budget and stop conditions

Dry run first (three trajectory steps per model/condition and one full diagnostic state per
model), then projected CPU-hours. Budget 4 CPU-h; fallback: seed 0 only. Stop on a reproduction gate
off by more than 1e-6, a numerical check failing at 1e-8, a failing baseline suite, a budget
exceeded after the fallback, or a required change outside this folder or its data directory.

## Outputs

`README.md`, `RESULTS.md` (summary ≤ 12 lines, settings, tables, figures, readings, limitations,
deviations, isolation audit, final suite, reproduction commands), `DEVIATIONS.md`,
`HANDOFF_PROPOSED_ENTRY.md`, light mirrors (json/md/png/yaml, ≤ 30 MB).
