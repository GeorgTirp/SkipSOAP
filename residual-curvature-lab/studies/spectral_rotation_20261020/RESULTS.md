# Stable spectrum or rotating eigenvectors? Per-tensor GGN spectra along CIFAR training — results

Protocol: [PROTOCOL.md](PROTOCOL.md), SHA-256 in `logs/protocol_sha256.txt`, fixed before any run. No deviations.

Outputs: `results/spectra/{soap,topfree}_s{1,2,3}.json` (copies of `~/skipsoap-study-data/spectral_rotation_20261020/spectra/`)
and `results/summary.json` / `summary.md`, from `scripts/summarize.py`.

**Status (2026-10-20):** all six trajectories completed, in 10–12 min each.
- **G1:** each replay equals its run's checkpoint parameters bitwise at steps 50, 100, 250 and 500 (24 of 24).
- **Suites:** passed before and after (15 suites, including this study's 6 tests).
- **Isolation manifest:** the only changes are 16 new files in `studies/queue_curvature_20261020/`, the next study,
  prepared during this run.
- **Spectrum accuracy:** the largest Ritz residual of the 200th eigenpair is 2.5e-3, at step 10; from step 50 on it
  is below 2e-4. No eigenvalue was excluded from the shape distance.

All numbers are medians over the 39 matrix tensors of one trajectory. Ranges are across seeds 1–3.

## 1. Summary

- **Verdict (V), both optimizers: stable spectrum, rotating eigenvectors.** M1 and M2 hold for SOAP and top-free;
  M3 does not.
- **Late in training (steps 250–500), the spectrum's shape barely moves:**

  | Interval | Shape distance (log units) |
  | --- | --- |
  | 10 steps | 0.03–0.05 |
  | 50 steps | 0.04–0.08 |

  The eigenspaces move a lot. A 50-step-old top-16 subspace captures only 0.70–0.81 of the current top-16
  curvature; after 10 steps it is already 0.84–0.88.
- **Early training is different:** the shape changes strongly as well. Over steps 10→60 the shape distance is
  1.1–1.6, and the old top-16 subspace captures only 0.16–0.32.
- **The power law straightens to α ≈ 1:**
  - The per-tensor raw GGN spectrum (ranks 1–200) goes from α ≈ 1.8–2.1 at step 10 to α ≈ 0.9–1.1 at steps 250–500,
    with R² ≈ 0.99.
  - The early kink (top about 1.0, bulk about 2.1–2.3) fades: late α_top is 0.65–0.86 and α_bulk 0.96–1.17.
  - The late exponent is close to the α ≈ 0.96 reported for Meterez et al.'s language model; that paper was not
    verified here.
- **Rotation, not batch noise, limits a cached top subspace.** At the same weights, the top-16 subspace of one
  2,048-image batch captures 0.93–0.999 of another batch's top-16 curvature (the batch floor, reading B).
  - The noise-to-drift ratio R = (1 − floor capture₁₆)/(1 − capture₁₆ after 10 steps):

    | Step | SOAP | top-free |
    | ---: | --- | --- |
    | 50 | 0.03–0.04 | 0.02 |
    | 100 | 0.08–0.13 | 0.04–0.06 |
    | 250 | 0.25–0.28 | 0.14–0.16 |
    | 400 | 0.39–0.60 | 0.28–0.30 |

    Drift dominates throughout. Batch noise becomes relatively larger late in training.
- **Top-free rotates faster than SOAP,** in line with its 3× larger lr (3e-2 vs 1e-2): late 50-step capture₁₆ is
  0.70–0.77 against 0.76–0.81.
- **Curvature grows during the lr decay:** the total trace rises by e^0.27–e^0.47, i.e. 1.3–1.6×, from step 250 to
  step 500.

## 2. Readings

| Code | SOAP | top-free |
| --- | --- | --- |
| M1 (shape distance ≤ 0.10 at gap 10, ≤ 0.20 at gap 50, late, every seed) | **holds** (0.029–0.037; 0.036–0.079) | **holds** (0.035–0.050; 0.054–0.059) |
| M2 (capture₁₆ ≤ 0.90 at gap 50, late, every seed) | **holds** (0.757–0.809) | **holds** (0.699–0.769) |
| M3 (capture₁₆ ≥ 0.95 at gap 50, late, every seed) | no | no |
| V | **stable spectrum, rotating eigenvectors** | **stable spectrum, rotating eigenvectors** |

## 3. Rotation against gap (T) and early against late (E)

capture₁₆ for each anchor→state pair:

| Pair | Gap | SOAP | top-free | Shape distance (SOAP; top-free) |
| --- | ---: | --- | --- | --- |
| 10→20 | 10 | 0.66–0.70 | 0.55–0.63 | 0.15–0.33; 0.19–0.33 |
| 50→60 | 10 | 0.82–0.85 | 0.80–0.82 | 0.12–0.18; 0.22–0.26 |
| 100→110 | 10 | 0.86–0.90 | 0.81–0.85 | 0.04–0.06; 0.05–0.08 |
| 250→260 | 10 | 0.86–0.88 | 0.84–0.85 | 0.03–0.04; 0.04–0.05 |
| 400→410 | 10 | 0.86–0.88 | 0.86–0.87 | 0.03; 0.04–0.05 |
| 10→60 | 50 | 0.25–0.32 | 0.16–0.20 | 1.06–1.36; 1.25–1.56 |
| 50→100 | 50 | 0.55–0.59 | 0.46–0.49 | 0.33–0.42; 0.43–0.66 |
| 100→150 | 50 | 0.70–0.72 | 0.60–0.62 | 0.14–0.19; 0.15–0.24 |
| 250→300 | 50 | 0.76–0.79 | 0.70–0.71 | 0.05; 0.05–0.06 |
| 400→450 | 50 | 0.80–0.81 | 0.75–0.77 | 0.04–0.08; 0.05–0.06 |
| 250→500 | 250 | 0.64–0.66 | 0.54–0.56 | 0.16–0.22; 0.16–0.22 |
| 10→500 | 490 | 0.09 | 0.04–0.05 | 1.81–1.93; 1.94–2.20 |

- **Capture by subspace size:** capture after 10 steps rises with the subspace size (late, both optimizers: k = 1
  0.78–0.84, k = 200 0.87–0.91). This is consistent with eigenvectors mixing within nearly tied groups
  rather than leaving the top space entirely.
- **Overlap:** the plain eigenspace overlap is lower than capture (late gap 10: 0.73–0.76), because capture weights
  directions by curvature.
- **Decay phase:** states 410 and 450 lie in the decay phase. Rotation there is about as fast as at step 250 (gap 50:
  0.75–0.81), although the lr has fallen to about half by step 450.

## 4. Batch floor (B) and spectrum (S)

| Step | Floor capture₁₆ (SOAP; top-free) | Floor capture₂₀₀ | α (ranks 1–200), SOAP; top-free | Top-16 trace share |
| ---: | --- | --- | --- | --- |
| 10 | 0.998–0.999; 0.997–0.999 | 0.999 | 1.84–1.94; 1.98–2.05 | 0.80–0.86 |
| 50 | 0.993–0.995; 0.996 | 0.99 | 1.39–1.46; 1.49–1.61 | 0.63–0.72 |
| 100 | 0.988–0.989; 0.991–0.992 | 0.98 | 1.21–1.22; 1.26–1.29 | 0.53–0.59 |
| 250 | 0.960–0.969; 0.976–0.978 | 0.92–0.95 | 0.99–1.01; 1.05–1.10 | 0.40–0.50 |
| 400 | 0.928–0.944; 0.957–0.963 | 0.86–0.92 | 0.92–0.96; 1.01–1.05 | 0.35–0.45 |
| 500 | — | — | 0.89–0.91; 0.99–1.02 | 0.33–0.42 |

The batch floor deepens with training. Late in training the curvature of one batch is less representative of
another's, mostly in the bulk (k = 200: 0.86–0.92 at step 400).

## 5. Interpretation and limitations

- **For the owner's question:** in the second half of these runs the per-tensor GGN spectrum behaves as Meterez et
  al. describe. It is a nearly fixed power law near α = 1, while the eigenvectors keep rotating: about 12–16% of the
  top-16 curvature leaves a cached subspace within 10 steps, and 19–30% within 50. Early training (the window where
  curvature methods have paid off so far) has neither property.
- **For PARSEC and the queue theory:**
  - **Spectral quantities can be calibrated rarely; directions must be refreshed often.** Spectral quantities
    include the brake scale, damping, a power-law prior and polynomial intervals. Ten-step-old directions already
    miss 12–16% of the top curvature.
  - **Pooling batches buys little for the top subspace.** Rotation exceeds batch noise at every measured state.
    R stays at or below the 0.3–0.8 break-even estimated in `queue_curvature_20261020`'s protocol, except for SOAP
    at step 400 (and one top-free case at 0.303). That study tests this directly (Part F).
- **Limitations:**
  - **Model and runs:** one small ViT and 500-step runs, so the "second half" is steps 250–500.
  - **GGN object:** raw (unwhitened) per-tensor blocks, on one fixed batch.
  - **Spectrum depth:** the top 200 eigenpairs only.
  - **Ties:** capture at a fixed k includes mixing across nearly tied eigenvalues at the boundary.
  - **Thresholds:** M1–M3 use thresholds fixed in the protocol, not derived from a noise model.
