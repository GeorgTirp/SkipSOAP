# Experiment protocol

Stage 0 runs network-free CPU float64 numerical audits and a tiny synthetic snapshot.
The default network has input 6, width 4, expansion 8, two residual blocks, three classes
and batch size four. Separate deterministic seeds govern initialization, data, probes
and analysis. No dropout, augmentation, clipping, update normalization or decay is used.

Stage 1 is prepared for a seed-0, 100-iteration SOAP pilot on downsampled 14×14 MNIST,
width 16, expansion 32, three blocks and batch 64. The loader saves the split manifest,
raw-data fingerprint, resizing policy and example IDs. Analysis construction and four
panel batches are disjoint and fixed across snapshots/methods. Analysis states occur
after consumed iterations 10, 50 and 100. The first SOAP initialization consumes an
iteration without an accepted parameter update. This distinction is logged.

Each snapshot proposes from a clone and discards next state. Native SOAP uses momentum;
the quadratic candidates use the true gradient. Native-versus-proposal is an optimizer
comparison. Exact diagonal-versus-full and matched sketch diagonal-versus-coupled are
interaction comparisons. All stages include input, normalization, biases and head.
No trained skip connection is removed as a purported isolated coupling intervention.

Raw candidates are stored before search. The local direction grid uses native norm
as its common radius, includes zero and `2^j` for j=-8,...,3, and gives every nonzero
direction the same 13-evaluation budget. It logs trials, boundary choices and zero
selections. A zero native radius makes the comparison unavailable. Each prediction
and actual loss correspond to the same displacement. The independent panel evaluates
the construction-selected update without rebuilding or retuning it. Its values are
transfer diagnostics, not hyperparameter selectors or test-set generalization claims.

The optional training path uses raw SOAP or true-gradient Armijo backtracking for
Algorithm A, with common initial scalar, halving, acceptance constant and maximum
evaluations. An ascent direction is rejected. Every consumed gradient advances SOAP
statistics once, including rejected updates. Trials and probes never advance moments.
Checkpoint restoration is audited against uninterrupted parameters, optimizer state,
used contexts and data-order state; monitoring equality uses actual state equality.

Stage 2 develops approximation quality, converged versus one message sweep, and
Algorithm B calibration. Stage 3 selects paired-seed training only after fixed-state
diagnostics are understood. No sweep, replicated training, MNIST download or MNIST
experiment is automatically run. Stage 4 decoder and Stage 5 integration remain gated.

Interpret seven questions separately: transport correctness; exact interaction
opportunity; fixed-direction versus rotation opportunity; sketch accuracy; statistic
reuse quality; nonlinear/independent-batch/training behavior; and runtime/memory cost.
Save per-state results before pooling. Sketch repeats are not independent training
seeds. No unstable gain ratio is reported; a future ratio needs a declared denominator
and explicit near-zero/negative-denominator handling. Signed pair contributions use
unordered pairs multiplied by two and must reconstruct the quadratic form.

The allocation budget limits estimated curvature workspace, not process RSS including
PyTorch/libraries/data. Dense D cap and output-dual cap are both 2048. Over-cap exact
runs fail before allocating reference matrices, with matrix-free APIs available for
an explicitly smaller/different experiment. No silent reference substitution occurs.
RSS is process-lifetime high-water memory and is labelled as such. Probe refresh,
SOAP proposal, full solve and whole snapshot timing are recorded; these are prototype
measurements, not scalable-efficiency evidence.
