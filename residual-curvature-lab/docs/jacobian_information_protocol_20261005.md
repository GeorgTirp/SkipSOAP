# Output-Jacobian information versus SOAP representation

The user asks what separates SOAP from exact tensor GGN and clarified that
"residual mapping" means the full parameter-to-output-error Jacobian, rather than
skip/branch interactions. This selected diagnostic keeps the original ViT and
all residual derivatives unchanged. No new training, architecture or tuning sweep
is selected. Record this protocol before inspecting new candidate results.

Use saved searched-SOAP and exact tensor-GGN states at iterations 10/25/50 from
`runs/20261005T135920-d7ce29b2`. Construction is the first 512 analysis IDs, as
before. Evaluation uses four previously unused 512-example analysis panels at
positions 2560:4608. They do not select settings. All six states are reported;
do not retain only states where the GGN direction wins.

At each state clone one SOAP proposal/context from its saved optimizer and the
current construction gradient. Compute C_bb = mean J_ib^T H_i J_ib exactly,
H_i=diag(p_i)-p_i p_i^T. Separately compute dense empirical Fisher blocks
F_bb = mean g_ib g_ib^T, g_ib=J_ib^T(p_i-onehot(y_i)). This retains parameter
correlations but observes only the actual label-gradient direction at each
example. F is NOT SOAP's actual statistic or an unbiased estimate of C for fixed
observed labels. Their difference changes output weighting/rank as well as which
Jacobian directions are observed; do not claim a finer identification than this.

Use the same true gradient, tensor partition, normalized inverse-SOAP metric,
mu=max(.01*trace(C)/D,1e-8), and global scalar-search policy for curvature controls.
Do not recompute damping from F or select it on evaluation panels. Include both
raw F and F rescaled within each tensor to match trace(C_bb). Trace matching
uses oracle information and is a diagnostic control, not a deployable estimate.

Information/representation controls: dense C, dense trace-matched F, exact
curvature diagonal in the actual SOAP basis for C, and the corresponding diagonal
for F. Match physical tensor traces of the diagonal models to C as well. This
separates changing the curvature target from discarding within-basis correlations.
Retain native SOAP and frozen -P g. Add C solves and its SOAP-coordinate diagonal
using an effective momentum RHS reconstructed from -P^-1 delta_SOAP and normalized
to the current gradient norm. Verify the frozen SOAP map returns the native ray.
This controls RHS direction without assuming its saved float32 bases are orthogonal.

Also draw labels from each example's current categorical prediction, independently
across examples and draws. At ranks 1/4/16 and two fixed repeats, average outer
products of J^T(p-onehot(sampled_label)). Before optional scaling, this estimator
has expectation C, unlike the true-label empirical Fisher. Ranks are nested within
each repeat; two local generators/streams are fixed before evaluation. Use raw
unbiased sampled matrices, the same exact-C damping, and dense tensor solves.
This asks whether a few extra output-error directions can recover the oracle
without changing its information target. The diagnostic computes a full Jacobian
and reuses it; its runtime does not estimate a practical VJP-only implementation.

Finally swap complete tensor norms between native SOAP and exact GGN directions,
then apply the same scalar search. This tests within-tensor directions separately
from allocation. Zero directions with nonzero requested norms fail explicitly.
All 17 candidates receive the common zero + 2^[-16..16] native-radius grid and
18 golden refinements on construction loss only. Positive boundary choices fail.

CPU float64, one thread, microbatch 8, existing 768 MiB tensor-workspace ceiling.
Retaining eight curvature-block collections adds about 140 MiB to the prior
workspace estimate. The selected work is six fixed-state audits, approximately
a few minutes; no automatic training followup. Save source/checkpoint hashes,
exact IDs, sampled labels, matrix/direction discrepancies, all searches and panel
values. Preserve model/optimizer/RNG and validate gradients, exact blocks and solves.

Interpretation: these interventions can identify local direction-quality differences
at matched states. They cannot uniquely attribute the full 50-step training gap,
prove a universal SOAP limitation, or establish that skip connections are the cause.
A training claim requires a separately selected paired continuation/replication.
Paper context: [SOAP](https://arxiv.org/abs/2409.11321) and
[full/layerwise GGN study](https://arxiv.org/abs/2510.09378).
