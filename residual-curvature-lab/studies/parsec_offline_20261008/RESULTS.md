# PARSEC-L offline tests — results

Study `parsec_offline_20261008`, protocol [PROTOCOL.md](PROTOCOL.md), from `ideas/PARSEC_ideas.md`
stages B and C. Large artifacts: `~/skipsoap-study-data/parsec_offline_20261008/runs/`. Light mirrors:
[runs/](runs/). Figure: [parsec_offline.png](figures/parsec_offline.png).

**Status (2026-10-08, 16:58Z):** complete. All gates passed; no unit failed.

## 1. Summary

- **I2, probe extraction: passes.** PARSEC-L built from s = 4 model-sampled labels per example keeps
  **97.8%** of the exact operator's local gain (r = 16, momentum RHS, 72 states; kill bar 70%). Every
  (s, r, RHS) setting keeps 93–104%, even with s = 1. Subspace error is small (mean sin²Θ 0.05–0.07 at s = 4).
- **In absolute terms (exploratory),** exact and probe PARSEC-L (r = 16) recover about **63%** of exact
  tensor-GGN local headroom on the 69 states shared with gap-audit 2c/4b. That is above the 60% tier the
  Kronecker families missed (best 46%), and it holds on the momentum RHS as well as the gradient.
- **I3, one-step tracking: killed.** For r = 4, 8 and 16, one power step per update tracks the subspace
  worse than a fresh probe extraction (median sin²Θ .088/.081/.077 versus .044/.055/.073). The subspaces
  move ~0.05–0.06 per step, faster than one power step follows.
- **SN-1 (S2), partial test: absent by the pre-declared rule** in both models. Stream-writer Kronecker
  errors are not consistently lower under SN at c = 0.95, though local distance distortion is.
  Exploratory: the ViT stream writers are lower in 5 of 6 pairs (first-in-branch 3 of 6); the MLP's
  degenerate step-200 states are discussed below.
- **Unplanned observation:** without SN (and at c = 6) the lab's residual MLP loses every residual
  branch by step 200 under searched SOAP (0% active GELU units, zero branch curvature). At c ≤ 2 the
  branches stay alive.

**Next step:** probes are a sufficient curvature source for PARSEC-L locally. Re-extraction per refresh,
not one-step tracking, is the right scheme. The remaining question is training (idea-bank stage D:
PARSEC-L with s = 4 probes and refresh-based extraction versus tuned native SOAP).

## 2. Settings as used

All states: 512-example construction batch (first 512 analysis IDs); four analysis panels; CPU float64,
one thread. PARSEC-L operator per tensor: P_r = β F0 [I − Σ κ/(1+κ) w wᵀ] F0ᵀ with exact-trace damping
μ = max(.01 tr C/D, 1e-8), β = 1/(μ · SOAPMetric.scale) (damping calibration), all 40+ tensors.
Candidates −P_r v for v = current gradient or effective SOAP momentum, each with `search_candidate`
(native radius, 52 evaluations) and `measurement`. Probes: model-sampled labels from the exact
per-example Jacobian; whitened probe matrix √β F0ᵀUᵀ; randomized SVD with oversampling 8 and 2 power
steps; μ and β exact (the probe trace is a diagnostic).

**Gates (all passed, bitwise).** B1: SOAP map and exact teacher_r (r = 4/8/16) on both RHS at the six
main states reproduce the capacity screen (difference 0.0). B2: trajectory parameters at steps 25 and 50
equal the 4a states (0.0). C: recomputed blocks of the no-SN ViT seed 0 at steps 10/50 equal the stored
4a blocks. All per-tensor action checks ≤ 1e-8; all searches bracketed.

**Tests (5 passed).** Probe rows equal plain per-example backprop gradients with sampled labels (the
layer-parallel route computes the same objects); the probe expectation equals the GGN blocks; the
randomized range finder and cores match the exact teacher core; transport is consistent; role
classification.

## 3. B1 (I2): probe-estimated versus exact PARSEC-L

Run `20261008T162657-67bfad3a` (6 main + 66 family states). R = (D̄_probe − D̄_SOAP-map)/(D̄_exact-PARSEC-L −
D̄_SOAP-map), mean independent decreases:

| Setting | All 72, gradient | All 72, momentum | Family 66, gradient | Family 66, momentum |
| --- | ---: | ---: | ---: | ---: |
| s = 1, r = 4 / 8 / 16 | 94.3 / 102.4 / 98.7% | 96.8 / 100.7 / 92.9% | 93.2 / 101.0 / 98.4% | 96.7 / 100.7 / 93.8% |
| s = 4, r = 4 / 8 / 16 | 100.3 / 103.5 / 102.7% | 100.9 / 101.9 / **97.8%** | 99.1 / 103.6 / 103.8% | 100.7 / 102.6 / 99.4% |

**Pre-declared reading:** R(s = 4, r = 16, momentum) = 97.8% ≥ 70% — **I2 is not killed.**

On the six main states the ratios are unstable because exact PARSEC-L gains little there. At r = 4 on the
gradient its mean (0.0156) is below the SOAP map's (0.0193), so the ratio is meaningless. At r = 16:
gradient 70.6%, momentum 76.1% (s = 4).

Absolute means (all 72 states): SOAP map 0.0296 (gradient) / 0.0258 (momentum); exact PARSEC-L r = 16
0.0675 / 0.0678; probe s = 4, r = 16 0.0686 / 0.0669. Main states only: SOAP map 0.0193 / 0.0127, exact
PARSEC-L r = 16 0.0342 / 0.0462 (the capacity screen's teacher_16), probe s = 4 0.0298 / 0.0382.

Diagnostics (all states): energy-weighted mean sin²Θ to the exact subspaces 0.048 / 0.056 / 0.069
(s = 4, r = 4/8/16) and 0.118 / 0.124 / 0.135 (s = 1); relative error of the leading κ 8.8% (s = 4) and
13.4% (s = 1); probe trace error 2.8% (s = 4) and 5.2% (s = 1).

**Exploratory: share of exact tensor-GGN headroom** (69 states scored in both this study and gap-audit
2c/4b; the SOAP-map rows agree to 7.8e-14). Exact tensor GGN on the gradient 0.0892 versus SOAP map
0.0302 (headroom 0.0590):

| Operator | Gradient | Momentum |
| --- | ---: | ---: |
| exact PARSEC-L, r = 16 | 63.0% | 63.6% |
| probe PARSEC-L, s = 4, r = 16 | 64.4% | 62.7% |
| probe PARSEC-L, s = 1, r = 16 | 61.1% | 58.0% |
| exact PARSEC-L, r = 8 | 56.5% | 60.2% |
| probe PARSEC-L, s = 4, r = 8 | 58.3% | 61.0% |

(Shares are (D − D_SOAP-map, gradient)/(exact-GGN headroom, gradient); exact GGN on momentum scores only
0.0318.) For comparison on the main six states (2c): `kron_basis_diag` 46.0%, `pcg_full_5` 67.1%.

## 4. B2 (I3): drift and one-step tracking

Run `20261008T162659-e0b80fae`; main ViT, searched SOAP seed 0, states 20–50. Median over steps 21–50
of the energy-weighted mean sin²Θ to the exact subspace:

| r | Drift per step | One-step tracking, exact K | One-step tracking, probes (s = 4) | Fresh probe extraction | Reading |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 0.059 | 0.081 | 0.088 | **0.044** | kill |
| 8 | 0.054 | 0.072 | 0.081 | **0.055** | kill |
| 16 | 0.051 | 0.062 | 0.077 | **0.073** | kill (marginal) |

**Pre-declared reading: I3 is killed** (probe tracking worse than fresh extraction for every r). Even
tracking with the exact matrix is worse than re-extracting from probes. A single power step cannot follow
subspaces that rotate this much per step; persistent tracking would need several steps per update, i.e.
a refresh. Drift is largest early in the window (maximum 0.24 at step 21; also high at steps 22–26 and 48)
and does not concentrate at SOAP's basis refreshes (steps 30, 40, 50).

## 5. C (SN-1, partial): stream writers versus first-in-branch layers

Run `20261008T162701-af2b0339` (the 48 SN-study states; that study's recipe: exact σ, residual-branch
weights only, c ∈ {0.95, 2, 6}). Nearest-Kronecker (rank-1) relative error of the effective-weight GGN
per group, seeds 0/1 at steps 10/50/200:

| Model | Group | No SN | c = 0.95 | Reading (c = 0.95, 6 pairs) |
| --- | --- | --- | --- | --- |
| ViT | stream writers | .263/.582/.665, .119/.662/.657 | .194/.531/.651, .266/.626/.619 | mixed (5 of 6 lower) |
| ViT | first-in-branch | .306/.567/.534, .308/.642/.604 | .198/.576/.723, .317/.485/.522 | mixed (3 of 6 lower) |
| MLP | stream writers | .598/.409/0*, .482/.579/0* | .478/.306/.420, .441/.398/.363 | mixed |
| MLP | first-in-branch | .560/.265/0*, .540/.594/0* | .478/.391/.449, .487/.330/.554 | mixed |

\* Degenerate: the residual branches are dead (zero curvature), so the error is trivially 0.

**Pre-declared reading: S2 absent** in both models. Local distance distortion is lower at step 10
under c = 0.95 (SN study), but the stream-writer error is not consistently lower. At c = 2 (ViT, two
active pairs) stream writers are lower and first-in-branch higher, which is S2's pattern, but too few
pairs to read.

**Exploratory (degenerate states excluded).** MLP at steps 10/50: stream-writer error lower under
c = 0.95 in 4 of 4 pairs, first-in-branch in 3 of 4. Together with the ViT's 5 of 6, the direction
matches S2, but the declared all-pairs criterion is not met.

**Exploratory: residual-branch collapse in the MLP.** Fraction of active GELU units per block at step
200 (seeds 0/1): no SN 0/0/0 and 0/0/0; c = 6 .06/0/0 and .41/.03/0; c = 2 .62/.42/.34 and .58/.60/.48;
c = 0.95 .58/.47/.50 and .52/.50/.62. Without SN the branch curvature at step 200 is 1e-23 to 1e-43,
and the network has reduced to input projection plus head. That is consistent with its validation loss
rising after step 50 (0.46 → 0.95). SN at c ≤ 2 keeps the branches alive but does not give a good
validation loss either (0.85–1.15 at step 200). This concerns searched SOAP on this MLP at batch 512 and
is not a general claim about SN.

## 6. Implications for PARSEC-L (hypotheses, not tested here)

1. Shared sampled-label probes (s = 4, about 4 backward passes per refresh) are enough to build
   PARSEC-L locally; s = 1 loses a few points.
2. Refresh by re-extraction (randomized SVD, 2 power steps) rather than one-step tracking; the refresh
   interval should be short, because the subspaces rotate by ~0.05 sin²Θ per step.
3. Locally, PARSEC-L reaches about two thirds of exact tensor-GGN headroom on both RHS. Whether this
   survives training, where exact GGN's per-step gain mostly fit the minibatch, is the open question for
   stage D. PARSEC-L's native single-lr step, without same-batch line search, may avoid that failure mode.

## 7. Limitations

Local, construction-batch scores on reused development panels; saved states come mostly from 50-step
searched-SOAP trajectories; the main-state subset is small and its ratios are unstable; probes come from
the exact Jacobian here (the backprop route is equal by a test, its speed is not measured); SN-1 uses
the SN study's recipe, not the UQ paper's; no training.

## 8. Cost, isolation audit and final suite

- Cost: 1.0 CPU-h (budget 3), three workers.
- Baseline and final suite: **106 passed**; study tests: **5 passed**.
- Isolation: no code of this study wrote outside its folder or data directory. The after-manifest
  differs from the before-manifest (16:22:12Z, 1,818 files) only in `ideas/PARSEC_ideas.md`, which was
  edited at 16:29:02Z outside this session (it gained an idea I15); this study only read it (protocol
  cites SHA-256 `cd3958f9…`). No new `__pycache__`; the repository `runs/` is unchanged by this study.
- Light mirrors 1.1 MB.

## 9. Reproduction

From `residual-curvature-lab/`, with `export PYTHONDONTWRITEBYTECODE=1`:

```sh
P=studies/parsec_offline_20261008/scripts
.venv/bin/python -m pytest -q -p no:cacheprovider studies/parsec_offline_20261008/tests
B1=$(.venv/bin/python $P/b1_extraction.py create); B2=$(.venv/bin/python $P/b2_tracking.py create); C=$(.venv/bin/python $P/c_sn_split.py create)
# work every unit (b1_extraction.py / b2_tracking.py / c_sn_split.py work --run R --unit U; queue_runner.sh), then finalize each run
.venv/bin/python $P/summarize_offline.py $B1 $B2 $C
```
