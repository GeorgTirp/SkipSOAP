# Selected mini ViT SOAP allocation pilot

The user requested a small ViT and KKT-based SOAP tensor allocation. This protocol
is recorded before MNIST runs. It changes the model to a 5,738-parameter ViT:
28x28 grayscale images, 7x7 patches (16 image tokens plus learned CLS), width 16,
two heads, two pre-LayerNorm attention/MLP blocks, expansion 32, final LayerNorm
and 10-class head. No dropout or augmentation. The prior MLP used 14x14 inputs,
so this is a new workload, not an isolated causal test of architecture alone.

Use seed 0, the existing data-seed/split, CPU float64 and batch 512. Run four paired
50-consumed-iteration trajectories: native SOAP, globally searched SOAP, SOAP with
diagonal reduced-curvature allocation, SOAP with full nonnegative reduced-curvature
allocation. The first SOAP call initializes statistics without a parameter update.
All arms use identical initial weights and training example order. SOAP moments
advance once per consumed gradient; allocation preserves its unit tensor directions.

For allocation, E contains the unit native tensor directions. Compute the exact
small K=E^T C E from directional JVPs, without full parameter GGN blocks. With
phi=-E^T g and H=K+mu E^T Omega E, the full amplitudes solve
min_{a>=0} a^T H a/2-phi^T a. Diagonal amplitudes are [phi]_+/diag(H). Recover shares
as a^2/sum(a^2). Negative-alignment directions remain eligible in the full problem.
These are restricted allocation oracles, not guaranteed nonlinear or held-out optima.

Omega is the normalized inverse SOAP metric. For training curvature arms use
mu=max(.01*estimated_trace(C)/D, 1e-8) with eight candidate-independent Rademacher
Hutchinson probes, seeded by iteration and analysis seed. This avoids a full exact
GGN solely to set damping. Log each trace estimate and its samples; do not label
it exact. The matrix K itself is exact up to numerical tolerance. This same
training damping rule applies to an optional exact tensor-GGN arm.

Every searched arm uses zero and common native-radius multipliers 2^j,
j=-16,...,16, then 18 golden-section evaluations within the best grid bracket.
Log rejection and boundary selections. A positive upper-bound selection fails
the search gate. Only construction loss selects the scalar. Native SOAP uses
lr=.003 without searching. Validation uses all 5,000 reserved examples at
iterations 0/10/25/50, without tuning or early stopping. No official test evaluation.

On the searched-SOAP trajectory at iterations 10/25/50, save fixed-state comparisons
of SOAP, frozen -Pg, both allocation variants and exact tensor GGN. Use the first
512 analysis examples for construction and the next four disjoint 512-example
batches as independent panels. Snapshot damping uses exact trace and gamma .01
for all candidates; scope, gradient and SOAP context are identical. Reuse of the
analysis pool makes these diagnostic comparisons, not a fresh confirmatory study.

Tensor-workspace cap: 768 MiB, with total process RSS reported separately. Check
exact tensor actions against independent JVP-Hz-VJP products and solve/KKT
residuals at 1e-8. Benchmark the first snapshot and extrapolate the 49-update exact
tensor trajectory before selecting it. Launch that optional paired arm only if
exact tensor GGN beats searched SOAP on mean independent loss decrease at >=2/3
states, the mean difference is positive, all numerical/search checks pass, and
estimated exact-arm time is <=300 seconds. Otherwise retain the snapshot result.

Save complete source/config provenance, checkpoints, allocation amplitudes/shares,
KKT residuals, projected curvature, scalar trials, validation loss and accuracy,
per-step time and independent-panel values. No architecture, rank or damping
sweep, decoder experiment, new data download or cloud run is part of this pilot.
