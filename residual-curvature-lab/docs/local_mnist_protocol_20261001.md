# Selected local experiment — 2026-10-01

The user explicitly authorized downloading MNIST and starting local training with
measurements. This activates the previously prepared pilot. No cloud jobs or large
datasets are involved.

First run the unmodified scientific settings in `configs/mnist_snapshots.yaml`:
seed 0, 100 consumed SOAP iterations, batch 64, CPU float64, one PyTorch thread,
14x14 downsampled MNIST, snapshots at iterations 10/50/100. Retain all three damping
settings, ranks 4/8/16/32, two sketch repeats, all direction trials and four independent
panel batches. The estimated curvature workspace is 229.07 MiB; process RSS also
includes data, libraries and differentiation workspace. No runtime estimate is assumed.

If numerical gates pass, run a bounded three-arm training pilot (plain SOAP and
matched sketch diagonal/coupled), each with the existing seed-0 initialization and
paired 100-batch data order. Use the already configured rank 8 and gamma 0.01, without
tuning from snapshot results. Plain SOAP uses its native update; sketch arms use the
declared Armijo policy and advance statistics once per consumed gradient. Evaluate
the reserved validation pool for loss and accuracy at initialization and iterations
10/50/100, without changing any settings. The official test set remains reserved.
This is a feasibility pilot, not replicated evidence or a matched tuning-budget claim.

All failures are retained. A failed numerical identity or exact-reference solve stops
escalation until its cause is understood. Training progress/checkpoints are saved
before snapshot measurement so the failing state can be inspected. This persistence
change does not alter updates, batches, random streams or numerical tolerances.
