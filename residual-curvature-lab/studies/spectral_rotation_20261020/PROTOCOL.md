# Stable spectrum or rotating eigenvectors? Per-tensor GGN spectra along CIFAR training — protocol

Study `spectral_rotation_20261020`. Written on 20 October 2026, before any run. Offline: no new training, only
bitwise replays of existing runs.

**Request.** The owner asked on 20 October: "Test 2. on the safe snapshots". Item 2 of the previous discussion
asked whether, as Meterez et al. (2026) report for a language model, the GGN spectrum is a stable power law in the
second half of training. If so, is the distribution fixed while the eigenvectors rotate?
- Meterez et al. was not available for verification here.
- In `bulk_krylov_20261014`, the whitened spectrum changed strongly over steps 10–100: a broken power law
  straightening, with the top-16 trace share falling from about 0.75 to 0.33.

## 1. Questions

1. **Late training:** in the second half of training (steps 250–500), is the shape of the per-tensor GGN spectrum
   stable over 10 and 50 steps?
2. **Rotation:** over the same intervals, do the top eigenspaces rotate? That is, does a subspace computed 50 steps
   earlier miss a substantial part of the current top curvature?
3. **Early vs late:** how do both compare with early training (steps 10–150)?
4. **Batch vs parameters:** at fixed weights, how much does the top eigenspace change between two disjoint batches,
   compared with 10 steps of training on one fixed batch? This is descriptive and feeds the owner's queue question.

## 2. Trajectories and states

- **Runs:** `cifar_ggn_20261012`'s evaluation units `soap__lr0.01__s{1,2,3}` and `topfree__lr0.03__s{1,2,3}`
  (`runs/20261012T110529-d747f9fc`).
- **Setup:** CIFAR-10 M1d6 ViT, batch 2048, 500 steps. Warm-up over 25 steps, constant to step 400, then linear
  decay to 1% over steps 401–500.
- **Replay:** each run is replayed from step 0 with `ladder_mechanism_20261013`'s `Replay`. That code was verified
  bitwise at steps 50 and 100 there.
- **States:** parameters after steps 10, 20, 50, 60, 100, 110, 150, 250, 260, 300, 400, 410, 450 and 500.
- **Anchors:** steps 10, 50, 100, 250 and 400.

## 3. Quantities

**Object.** The raw per-tensor GGN block C_b = (1/N) Σ_i J_ibᵀ (diag p_i − p_i p_iᵀ) J_ib, computed on a **fixed
reference batch A**.
- It is computed in parameter coordinates, not SOAP-whitened ones, so it does not depend on the optimizer's
  preconditioner.
- Batch A has 2,048 unaugmented training images: the first draw of a dedicated `DataOrder` with seed 20261020.
- Fixing the batch removes batch-to-batch variation from comparisons across steps.
- **Batch B** is the next 2,048 images of the same order. It is used only at the anchors (question 4).
- Only the 39 matrix tensors are used (two-dimensional, both sides above 1, as in `bulk_krylov`).

**Per state and tensor:**
- the top 200 eigenpairs, computed as in `bulk_krylov`: exact eigh up to 512 columns, otherwise 16 subspace
  iterations with 400 columns and Rayleigh–Ritz;
- tr C_b;
- the Ritz residual of the 200th pair;
- power-law fits over ranks 1–200, 1–16 and 17–200;
- the top-16 share of the trace.

**Pair metrics** (anchor a, later state t; U_k is the top-k eigenspace, λ_i are eigenvalues):
- **Capture:** capture_k(a→t) = tr(U_k(a)ᵀ C(t) U_k(a)) / Σ_{i≤k} λ_i(t), for k ∈ {1, 4, 16, 64, 200}.
  - This is the fraction of the current top-k curvature that the anchor's subspace still captures, i.e. what a
    cached subspace would deliver.
  - It is at most 1 (Ky Fan), with equality iff the subspace is the current top-k eigenspace (up to ties).
- **Overlap:** overlap_k(a→t) = ‖U_k(a)ᵀ U_k(t)‖²_F / k.
- **Shape distance:** d_shape(a→t) = RMS over i = 1…200 of |log(λ_i(t)/tr C(t)) − log(λ_i(a)/tr C(a))|.
  - It is invariant to the overall curvature scale, which is reported separately as log(tr C(t)/tr C(a)).
  - Non-positive eigenvalues are excluded and counted.
- **Batch floor at each anchor:** capture_k(A→B) and overlap_k(A→B), with A's eigenspace evaluated on batch B's
  GGN at the same weights.

Each pair statistic is the median over the 39 matrix tensors.

## 4. Readings (per optimizer, seeds 1–3)

**Late pairs:**
- gap 10: 250→260 and 400→410;
- gap 50: 250→300 and 400→450.

The thresholds are conventions fixed here. A shape distance of 0.1 corresponds to a typical 10% change in
trace-normalized eigenvalues. A capture of 0.9 means a cached top-16 subspace misses 10% of the current top-16
curvature.

| Code | Reading |
| --- | --- |
| M1 | **Stable shape, late:** d_shape ≤ 0.10 for both gap-10 late pairs and ≤ 0.20 for both gap-50 late pairs, in every seed. |
| M2 | **Rotation, late:** capture_16 ≤ 0.90 for both gap-50 late pairs, in every seed. |
| M3 | **No rotation, late:** capture_16 ≥ 0.95 for both gap-50 late pairs, in every seed. |
| V | **Verdict:** *stable spectrum, rotating eigenvectors* if M1 and M2; *stable spectrum and eigenvectors* if M1 and M3; *spectrum not stable* if M1 fails; otherwise *mixed*. |
| E | Descriptive: the same metrics for the early pairs (10→20, 50→60, 100→110; 10→60, 50→100, 100→150) against the late ones. |
| T | Descriptive: capture_k and overlap_k for k ∈ {1, 4, 16, 64, 200}, against the gap, for every anchor and later state (gaps 10–490). |
| B | Descriptive: batch floor at each anchor against the gap-10 parameter change from that anchor. |
| S | Descriptive: α over ranks 1–200, 1–16 and 17–200, the fit R², and the top-16 trace share per state (does the late spectrum become a single power law?). |

States 410 and 450 lie in the decay phase, where steps shrink. This is reported, not corrected.

## 5. Gates

- **G0:** suites before and after, including this study's tests:
  - capture and overlap identities and bounds;
  - the scale invariance of d_shape;
  - spec grouping;
  - the raw dense block against matrix-free GGN products on a tiny CPU batch;
  - the fit helpers (copied).
- **G1:** each replay equals the run's saved checkpoint parameters bitwise at steps 50, 100, 250 and 500. Any
  failure stops the study.

## 6. Cost, isolation and limitations

- **Cost:**
  - 14 states plus 5 batch-B passes per trajectory, 6 trajectories;
  - about 30 s per dense pass, plus replay at 0.2–0.4 s per step;
  - **about 1–1.5 GPU-hours**, run detached.
- **Writes:** only to `studies/spectral_rotation_20261020/` and `~/skipsoap-study-data/spectral_rotation_20261020/`.
  `cifar_ggn` checkpoints and `ladder_mechanism` code are read only.
- **Limitations:**
  - **Model and run length:** one small ViT and 500-step runs, so the "second half" is steps 250–500, not a long
    language-model run.
  - **GGN object:** per-tensor blocks only, and on one fixed batch.
  - **Spectrum depth:** the top 200 eigenpairs only.
  - **Ties:** the capture metric's boundary at k mixes nearly tied eigenvalues by design.
