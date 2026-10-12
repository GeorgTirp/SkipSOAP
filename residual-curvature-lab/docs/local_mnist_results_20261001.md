# Local MNIST results — 2026-10-01

The user-authorized local experiments are complete. MNIST was downloaded once;
training and measurements then ran locally on CPU in float64 with one PyTorch thread.
The task uses deterministic 14x14 downsampling, not original-resolution MNIST.
No official test-set evaluation, hyperparameter sweep or replicated training was run.

## Runs and artifacts

- [SOAP trajectory and all fixed-state diagnostics](../runs/20261001T080647-fd8e50e5/REPORT.md):
  100 consumed iterations, snapshots at 10/50/100, all three damping settings,
  ranks 4/8/16/32, two independent sketch repeats, and four independent panel batches.
- [Three-arm comparison report](../runs/20261001T081542-3836b346/REPORT.md).
- [Comparison plot](../runs/20261001T081542-3836b346/comparison.png) and
  [per-state validation CSV](../runs/20261001T081542-3836b346/validation.csv).
- [Sketch-diagonal run](../runs/20261001T081543-323a92f8/training.json) and
  [sketch-coupled run](../runs/20261001T081556-39824518/training.json).

Each trajectory has resolved configuration, split fingerprint, source provenance,
training measurements and a final atomic checkpoint. Snapshot artifacts include
checksummed Jacobians, factors, reduced matrices, raw/selected candidates, SOAP state,
trial losses and independent-panel reductions. Exact code matching each run's recorded
SHA256s is archived under its `source/` directory; the comparison runner is also saved.
The later checkpoint safeguard for off-cadence snapshot failures changes persistence
only and does not alter these completed trajectories or their numerical settings.

## Training measurements

All three arms used identical initial parameters and the same 100 batches of 64
examples; the runner verified paired batch IDs. Rank 8 and gamma 0.01 were declared
before inspecting results. Native SOAP used its configured learning rate; sketch
arms used the existing true-gradient quadratic and Armijo backtracking. Validation
used all 5,000 examples in the reserved pool at iterations 0/10/50/100, without
selecting settings. Each arm accepted 99 updates after the initialization-only call.

| Method | Validation loss at 50 | Final validation loss | Final accuracy | Sum of training-step seconds |
| --- | --- | --- | --- | --- |
| SOAP | 0.908774 | 0.385377 | 88.98% | 0.672 |
| Sketch diagonal | 1.302483 | 4.463031 | 57.54% | 4.094 |
| Sketch coupled | 1.244505 | 4.005140 | 53.72% | 1.753 |

Time sums include candidate construction and any scalar search, but exclude loading,
checkpoints, validation, snapshots and plotting. The SOAP run's elapsed loop time
including its three snapshot measurements was about 52.7 seconds. Snapshot analysis
took 15.37/16.31/16.18 seconds respectively; process-lifetime peak RSS was about
910 MiB. The earlier 229 MiB estimate describes curvature allocations, not complete
process memory or all automatic-differentiation intermediates. These are small local
prototype measurements, not a general performance comparison.

## Numerical checks and interpretation

All snapshot GGN/microbatch gates passed; worst GGN-action error was 1.62e-15 and
worst exact output-dual solve residual was 2.67e-10, below the 1e-8 threshold.
All SOAP-preconditioned CG references converged. Tolerances were not relaxed.

The sketch training settings failed this feasibility comparison. Both arms worsened
on validation between iterations 50 and 100 even though every accepted step reduced
its own construction minibatch loss. Diagonal/coupled arms required 1,153/1,203
backtracking loss evaluations over the run, and their largest update norms were
25.11/25.82 versus 0.231 for native SOAP. No search was exhausted; there were no
rejected nonzero updates. Passing the same-batch Armijo condition did not ensure
useful transfer to other examples.

At the fixed SOAP states, rank-8 relative curvature Frobenius errors ranged from
1.16 to 1.85 across the prespecified snapshots/repeats. This establishes poor global
curvature accuracy at these ranks and states, not a causal explanation for every
training failure. Large update norms and later validation deterioration are consistent
with inadequate curvature/step-scale control in this setting. Metric error, rank,
damping and nonlinear step choice are not isolated by this pilot.

Common-grid comparisons frequently selected the largest allowed radius, including
the gamma=0.01 native/exact/sketch comparisons at all three snapshots. Their direction
rankings therefore concern the finite prespecified window, not unconstrained nonlinear
optima. At iteration 100, native and both rank-8 sketch candidates improved their
construction batch but worsened the independent-panel mean. The negative transfer
and boundary observations are retained; no grid or optimizer setting was retuned.

Coupled versus diagonal training changes later gradients and SOAP statistics as well
as immediate cross-tensor curvature. One seed with these starting settings does not
establish a universal optimizer ranking. It provides no reason to scale up now.
The next useful work is a prospectively specified, bounded investigation of sketch
rank, damping/metric conditioning and step magnitude, before replicated training.

## Commands and checks

Executed from the repository using `.venv/bin/python`:

```bash
python -m pytest -q
python -m residual_curvature.cli snapshots --config configs/mnist_snapshots.yaml --dry-run
python -m residual_curvature.cli data --dataset mnist --download
python -u -m residual_curvature.cli snapshots --config configs/mnist_snapshots.yaml
python -u scripts/run_local_training_pilot.py \
  --snapshot-run runs/20261001T080647-fd8e50e5 \
  --config configs/mnist_training_pilot.yaml
```

The network sandbox initially blocked downloading; the authorized retry completed.
Final regression result: **32 passed in 2.91 seconds**, with one upstream PyTorch
deprecation warning. Added checks verify validation weighting across uneven batches,
non-mutation during evaluation, and preservation of a failed snapshot's exact consumed
step/checkpoint. No processes from these runs remain active.
