# Verification record

## 2026-10-08: matched OT/GGN study execution and independent review

The main suite after adding combined cores/graph identity passed 106 tests in
4.66 seconds, with one upstream warning. The study's separate tests were run on
resume with `.venv/bin/python -B -m pytest -q -p no:cacheprovider studies/gap_audit_20261007/tests`:
12 passed in 4.07 seconds, one warning.

`scripts/review_ot_ggn_capacity.py` verifies the isolated study's execution of our
protocol, external run `20261007T124647-1d8487a2`, without modifying it. All 48
archived sources pass hashes; current imported implementation matches. Six parent
arrays, twelve operator artifacts, light mirrors, paired physical/factor/g/m RHS,
420 search records, 204 operators and non-mutation flags are checked. Maximum
recorded teacher action error 1.13e-14, teacher equation residual 1.52e-14,
backward error 9.75e-17 and dense application discrepancy 4.43e-15.

Independently replayed 120 stored operator/scalar proposals (five families, both
RHS, both targets, six states) and all their construction/panel losses. Maximum
loss discrepancy is 4.44e-16. No search, fitting or training was repeated. Review
JSON/source hashes are in `runs/20261008_ot_ggn_capacity_review/`; see the
[report](ot_ggn_capacity_results_20261008.md). Broader study training results are
attributed to its report and were not independently checkpoint-replayed here.

## 2026-10-07: inverse-capacity screen

`.venv/bin/python -m pytest -q`: 103 passed, one upstream deprecation warning,
4.49 seconds after completion. Seven capacity tests cover factors/inverses,
nonorthogonal bases, structured applications, population fits, spectral bounds,
global ray errors and zero-curvature damping calibration.

The completed run is `runs/20261007T093903-8d7d86f1`. Its summarizer verifies
42 source files, six parent arrays, 12 operator artifacts, both nested resumed
source archives, 324 bracketed searches and model/context/RNG non-mutation.
Maximum teacher gradient equation residual is 1.59e-14; backward error 1.28e-16;
cached-teacher energy discrepancy 1.79e-11; completed dense representation
comparison 3.19e-9. Logged condition-amplified gradient forward discrepancy is
3.09e-8, distinguished from the operator Frobenius check. See the
[protocol amendments](learnable_capacity_protocol_20261007.md) and
[report](learnable_capacity_results_20261007.md) for three interrupted attempts,
the two numerical audit types and unchanged 1e-8 check tolerances.

## 2026-10-01: initial synthetic verification

Executed using the repository's `.venv/bin/python` on macOS-15.7.3-arm64-arm-64bit-Mach-O.
Python 3.13.3, CPU float64, one PyTorch thread. Dependencies are frozen in
`tested-requirements.txt`. No MNIST download/training was performed.

```bash
python -m pip install --no-build-isolation -e '.[dev]'
python -m pytest -q
python -m residual_curvature.cli smoke --config configs/synthetic_smoke.yaml
python -m residual_curvature.cli estimate --config configs/mnist_snapshots.yaml
python -m residual_curvature.cli snapshots --config configs/mnist_snapshots.yaml --dry-run
python -m residual_curvature.cli train --config configs/mnist_training_pilot.yaml --dry-run
python -m pip wheel --no-build-isolation --no-deps . -w /private/tmp/residual-curvature-wheels
python -m residual_curvature.cli --help  # also checked from /private/tmp
```

Final test result: **30 passed in 2.61 seconds**, one upstream PyTorch deprecation
warning. All tests run offline. The wheel includes the installed command package,
SOAP source and MIT license, and the editable command works outside the source tree.
Vendored SOAP source and license were compared byte-for-byte with the pinned checkout.

Latest final smoke: [`runs/20261001T075554-7ece8564`](../runs/20261001T075554-7ece8564/REPORT.md).
The snapshot reports 211 parameters / 16 tensors. All three gamma settings and
4/8/16/32 probes with two independent repeats are retained. Source checksums in this
run were verified against the delivered implementation. Since local git has no
initial commit, provenance uses the actual source-tree SHA256 revision, not an
invented commit. Earlier debug runs remain isolated in separate ignored directories.

| Numerical audit | Measured value |
| --- | --- |
| Dense-factor versus matrix-free action relative error | 7.44733e-16 |
| Maximum microbatch factor relative error | 8.74261e-17 |
| Worst full output-dual normalized solve residual | 3.00448e-12 |
| Worst SOAP-preconditioned CG normalized residual | 6.46541e-13 |
| Maximum allocation KKT residual | 1.66533e-16 |
| Dense/dual solution relative difference (last gamma) | 5.02976e-09 |
| Damped dense condition (last gamma) | 1.76103e+09 |
| Dense GGN minimum eigenvalue | -1.87647e-16 |
| PSD roundoff tolerance | 3.75438e-14 |
| Whole snapshot analysis seconds | 1.26355 |
| Process-lifetime peak RSS MiB | 317.375 |

All CG solves converged. The original independent audit exposed Woodbury cancellation
at an early SOAP state; residual refinement fixes the declared backward-error check
without relaxing its 1e-8 threshold. Forward-error differences are interpreted using
the measured condition, as documented in decisions.md. Algebraic tests still use
1e-9 relative and 1e-11 absolute tolerances for well-conditioned identities.

This smoke does not favor the proposal uniformly. At gamma=0.01, native SOAP's common-grid
construction decrease is 0.229166, versus 0.103861 for exact full GGN and 0.0633898
for exact diagonal GGN. The corresponding panel means are 0.0746378, 0.0529317 and
0.0284375. These choices hit the upper finite-grid boundary. Raw diagonal candidates
have negative full-model predicted decrease, while cone/span/full model values obey
the restricted-oracle ordering. Nonlinear radius limits and sketch error remain
separate measured issues; a positive quadratic opportunity does not establish better
training. No post-hoc grid redesign was performed.

MNIST dry-run output is retained in [mnist_snapshot_estimate.json](mnist_snapshot_estimate.json)
and [mnist_training_estimate.json](mnist_training_estimate.json): 6,634 parameters,
640 outputs, maximum GGN rank 576, estimated exact-output workspace 240,200,704 bytes
(229.07 MiB), below the 256 MiB guard. Dense parameter-space reference would exceed
both the default dimension cap and its workspace budget. Runtime is not extrapolated
from the toy. The next supported experiment is the deliberately selected seed-0
MNIST fixed-state pilot described in the README; decoder/statistic-reuse work is deferred.
