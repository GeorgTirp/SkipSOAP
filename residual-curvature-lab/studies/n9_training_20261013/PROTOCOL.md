# N9 in training: SOAP and top-free PARSEC-L with bases from the probe GGN — protocol

Study `n9_training_20261013`. Written on 13 October 2026, before any run.

**Request.** The owner asked for this on 13 October: "a training test of N9 on CIFAR. That means SOAP with its bases taken
from the probe factors (the same per-step cost as top-free's probe pass), with Adam's second moments carried into
the new basis. It would also make sense to try the same bases under top-free PARSEC-L."

**Basis.** Offline, at CIFAR SOAP states (`basis_refresh_snr_20261013`), the eigenbasis of the probe factors gave a
markedly better diagonal curvature model than SOAP's basis (A1, A2: 9 of 9). Post hoc, it also gave a larger
held-out decrease at steps 10–50. This study asks whether that becomes a training gain.

## 1. Questions

1. Does SOAP train better early, and at the end, when its eigenbases come from the probe factors (N9)?
2. Is any gain due to the *source* of the basis (the probe GGN), or only to refreshing exactly and on N9's
   schedule? The control `soap_eigh` refreshes at the same steps by an exact eigendecomposition of SOAP's own
   factors.
3. Does top-free PARSEC-L gain from the same bases?

## 2. Workload

The CIFAR workload of `cifar_ggn_20261012`:
- **Data:** CIFAR-10, 45k/5k split of the training set; the test split is never accessed. Augmentation crop + flip.
- **Model:** M1d6 ViT, 308,042 parameters.
- **Training:** batch 2048 in microbatches of 512, 500 steps. Warm-up over 25 steps, then WSD decay over steps
  401–500. Decoupled weight decay 0.05; weight EMAs d ∈ {0.9, 0.98, 0.995}.
- **Numerics:** GPU float32, deterministic.

`scripts/train.py` is `cifar_ggn`'s runner with the GGN arms and the diagnostics removed and the N9 arms added.
Its `soap` and `topfree` paths are unchanged.

## 3. Arms

| Arm | Bases |
| --- | --- |
| `soap` | upstream SOAP: eigh of GG at step 1, then one power-iteration QR step on GG at the end of steps 11, 21, …; lr 1e-2, as selected in `cifar_ggn` (not retuned) |
| `soap_eigh` | **control:** upstream's QR refresh switched off. At the end of steps 2, 12, 22, …, the bases are replaced by the exact eigenvectors of SOAP's own current factors GG. |
| `soap_n9` | **N9:** upstream's QR refresh switched off. At steps 2, 12, 22, …, s = 1 sampled-label probes of the step's batch are drawn at the step's parameters (`micro.lean_probes`, as top-free). After the step, the bases are replaced by the exact eigenvectors of the probe factors S_L = Σ_j G_jG_jᵀ, S_R = Σ_j G_jᵀG_j. |
| `topfree` | top-free PARSEC-L (rank 16, c = 64, `top_rhs='gradient'`, probes at steps 2, 12, 22, …); lr 3e-2, as selected in `cifar_ggn` |
| `topfree_n9` | top-free PARSEC-L whose SOAP bases are rebuilt as in `soap_n9`, from the probes it already draws (no extra probe pass) |

- **Which modes change:** only the modes SOAP preconditions are rebuilt (every mode of every matrix). 1-D tensors
  stay unrotated.
- **Initial basis:** at step 1, every arm uses upstream's eigh of the first gradient's factors.
- **Rebuilding a basis, Q → Q′** (`scripts/n9.py`):
  - **Momentum:** carried exactly. It goes back to original coordinates with Q and into Q′, as at an upstream
    refresh.
  - **Adam's second moments:** carried into the new basis by assuming the second-moment matrix is diagonal in the
    old basis, Σ = Q diag(v) Qᵀ. Then v′ = diag(Q′ᵀ Σ Q′), per mode v′ = (M∘M) v with M = Q′ᵀQ. This preserves Σv
    and the trace.
  - **Eigendecompositions:** float64 on the CPU (deterministic), cast to the parameters' float32.
- **PARSEC-L's held factors:** they are stored with their extraction bases and are carried to new bases by its own
  `transport` (unchanged).

## 4. Tuning and evaluation

- **Baselines:** `soap` and `topfree` keep `cifar_ggn`'s seed-0 selections (lr 1e-2 and 3e-2, EMA decay 0.98).
- **Tuning:** `soap_eigh`, `soap_n9` and `topfree_n9` on seed 0, with lr ∈ {3e-3, 1e-2, 3e-2}.
  - Selection: the lowest step-500 EMA validation loss over the three decays.
  - If the selected lr lies at a grid edge, one extension unit at ×3 or ÷3 is run for that arm, and the selection
    is repeated.
  - The grid omits 1e-3, which was far behind for both baselines in `cifar_ggn` (0.83 and 0.82 vs 0.54).
- **Evaluation:** seeds 1–3, all five arms at their lrs (15 units).
- **Seeds:** seed s sets initialization, data order, augmentation and the probe generator (as in `cifar_ggn`).
- **Records:** validation at steps 0, 10, 25, 50, …, 500, raw and EMA. Training records every step, including each
  basis rebuild's numerical checks.

## 5. Gates

- **G0:** suites before and after:
  - the repository (106), PARSEC-L (22), `cifar_ggn` (11), `cifar_sn_soap` (12) and `basis_refresh_snr` (9);
  - this study's tests: the rebuild against dense references; the momentum carried; v′ equal to diag(Q′ᵀΣQ′);
    `soap` and `topfree` equal to `cifar_ggn`'s arms bitwise over CPU steps; the N9 arms run.
- **G1:** the `soap` and `topfree` evaluation units reproduce `cifar_ggn`'s evaluation units
  (`20261012T110529-d747f9fc`) bitwise in every field of `validation.json` and `training.json` except wall-clock
  timings. That run had diagnostics on; they do not touch the training state.
- **G2, rebuild numerics,** at every rebuild:
  - the momentum in original coordinates changes by at most 1e-4 relative;
  - Σv′ = Σv to 1e-4 relative;
  - v′ ≥ 0.
  A failure stops the unit.
- **Divergence:** a non-finite loss is reported and counts as worse; it is not a gate failure.

## 6. Pre-declared readings (evaluation seeds 1–3)

*Better* or *worse* means lower or higher validation loss in all three seeds; otherwise *mixed*. Each arm's EMA uses
its selected decay.

| Code | Reading |
| --- | --- |
| N1 | **N9 in SOAP:** `soap_n9` vs `soap`, raw at steps 25, 50 and 200, EMA at 500. *N9 speeds up SOAP early* if better at both 25 and 50. |
| N2 | **Source vs schedule:** `soap_n9` vs `soap_eigh`, same four points. *The probe source matters* if better at both 25 and 50. |
| N3 | **N9 in top-free:** `topfree_n9` vs `topfree`, same four points. |
| N4 | Descriptive: `soap_eigh` vs `soap` at the same points; seed-mean curves; steps to `soap`'s seed-mean raw loss at steps 50 and 200; seconds per step and per rebuild; the rebuild checks; the tuning table. |

## 7. Cost (estimate; the dry runs follow)

- **Per step:** `soap` 0.22 s; `soap_eigh` about 0.25 s; `soap_n9` about 0.38 s (a probe pass every 10 steps,
  about 1.3 s); `topfree` 0.29 s; `topfree_n9` about 0.32 s.
- **Total:** tuning (9 units, up to 3 extensions) about 0.5 h; evaluation (15 units) about 0.75 h. **About
  1.3 GPU-hours.**

## 8. Isolation and limitations

- **Writes:** only to `studies/n9_training_20261013/` and `~/skipsoap-study-data/n9_training_20261013/`. Earlier
  studies are imported or read, never modified.
- **Process:** `PYTHONDONTWRITEBYTECODE=1`; pytest `-p no:cacheprovider`; isolation manifests before and after; one
  GPU process at a time.
- **Limitations:**
  - One model, three seeds, 500 steps.
  - The baselines' lrs come from `cifar_ggn`. The new arms are tuned on a three-point grid.
  - **v′ is an approximation:** carrying v assumes a diagonal second moment in the old basis. Upstream SOAP instead
    only permutes v at its QR refresh, because the basis moves little there; N9's basis can move a lot.
  - N9 refreshes on top-free's probe schedule, so the first rebuild comes at step 2 rather than step 11. The control
    `soap_eigh` shares that schedule.
