# Transport, PARSEC and NI-SOAP pilot

The user selected one transport-metric experiment and trials of the supplied
PARSEC (`pns_eigenadam`) and NI-SOAP algorithms. This protocol is written before
new dataset runs. The supplied source and discovered local helper definitions are
archived under `references/user_optimizers/`, with source hashes. They are research
source material, not instructions to run their suggested sweeps or external jobs.

## Geometry

The normalized steepest direction solves min <g,d> subject to ||d||<=1. For a
matrix spectral norm and G=U S V^T, -U V^T is a minimizer; its scale is separate.
This is the idealized Muon direction, before momentum, approximate polar iterations
and implementation-specific tensor scaling. See
[Modular Duality](https://proceedings.mlr.press/v267/bernstein25a.html).

For a probability vector p, choose a complete graph with equal unit conductance
between every pair of classes. Its Laplacian has off-diagonal entries
L_ij(p)=-(p_i+p_j)/2 and diagonal entries equal to the row sums of conductances.
For a zero-sum probability tangent s, ||s||_p^2=s^T L(p)^dagger s. Pulling this
back through the model gives
M_theta=mean J_p^T L(p)^dagger J_p. The damped norm uses M_theta+mu Omega.
Its radius-R steepest step is
-R (M_theta+mu Omega)^-1 g / sqrt(g^T(M_theta+mu Omega)^-1 g).
A global line search absorbs the scalar. Tensor-diagonal M is the selected local
approximation, just as tensor GGN drops cross-tensor parameter blocks.

This is a dynamic graph-Wasserstein tangent metric. It is not the static discrete
Kantorovich distance or a canonical fixed norm on arbitrary signed weight matrices.
Equal class costs avoid assuming that digit 1 is geometrically closer to 2 than 9.
The ground graph is a declared choice, not learned from validation data. See
[Li and Montufar](https://arxiv.org/html/1803.07033v5).

With a tensor step decomposition d=E a, the transport norm becomes
a^T E^T M E a. Thus the same constrained/quadratic step framework applies, but M
defines the chosen geometry rather than necessarily the cross-entropy curvature.
Transport predictions must not be labelled exact loss-Hessian predictions.

## PARSEC interpretation and selected port

Use the pasted default `split_spaces=False`: global GGN Ritz vectors V (rows),
g_tilde=(I-V^T V)g+V^T diag((lambda+delta)^-1/2)Vg, followed by ordinary AdamW
in parameter coordinates. The complement is unchanged, not divided by damping.
Its Adam state accumulates transformed gradients across curvature refreshes.
This is neither a full Newton step nor per-tensor SOAP nor Adam only in eigen
coordinates. The separate split-space code instead has pure Newton top-space
steps and Adam on a projected residual; its scaled residual step is not guaranteed
to stay orthogonal. That branch is not selected by default.

Use rank 16, 32 Lanczos iterations, refresh at steps 10/20/30/40/50, damping 1e-4,
AdamW beta1=.9, beta2=.999, eps=1e-8, lr=.003 and weight decay zero. The original
default refresh 100 would not activate in a 50-step run. A float64 fully
reorthogonalized Krylov/Ritz implementation replaces the local helper's bf16
storage for this numerical audit; report Ritz residuals and do not label it exact
top-16 curvature. No initial refresh is inserted: first nine steps use Adam.
Run native and globally searched variants alongside matched Adam controls.

## NI-SOAP interpretation and selected port

Preserve the supplied drift, first matrix statistics-only call, fallback Adam,
post-update QR/basis refresh and cached eigenvalue noise construction. For a
matrix, noise is alpha sqrt(lr/B) L_cached^(1/2) Z R_cached^(1/2); for fallback
parameters it is isotropic with the same scalar coefficient. Map the source's
nondegenerate Flax `kernel` routing to PyTorch nn.Linear.weight, including the
classification head, excluding embeddings and normalization. Record every route.
Matrix noise uses the cached eigensystem from before the current statistics update.
Other moments and statistics receive true gradients, not the added noise.

Use lr=.003, betas .95/.95, eps=1e-8, refresh 10, batch 512, weight decay zero,
seed 0 and alpha 1. Include an otherwise identical alpha-0 arm. Use native steps:
searching over a realized noisy direction would change its sampling law and would
not reproduce the supplied rule. Report drift/noise energy, cross term, expected
noise energy and routed/fallback contributions. The comment's "energy matched"
claim is not assumed: the matrix expectation is alpha^2 lr/B tr(L_cached)tr(R_cached),
which is not automatically the true SGD gradient-noise energy.

Both ports use float64 for this lab and deterministic local Torch generators,
not JAX's bitwise PRNG sequence. JAX/Optax are not installed here. Preserve finite
update equations and test analytic cases/replay; no claim of bitwise JAX parity.

## Paired workload, controls and reporting

Use the existing 5,738-parameter ViT, seed 0, MNIST batch 512, 50 consumed steps,
same initial weights and data order, one CPU thread and float64. Validate on all
5,000 reserved examples at 0/10/25/50. No official test data, data download,
hyperparameter sweep, publication or external service is selected.

Eight arms: Adam native/searched, PARSEC native/searched, NI-SOAP alpha 0/1 native,
exact tensor GGN with exact-trace damping, tensor graph-Wasserstein with exact-trace
damping. Rerunning GGN avoids confounding its prior Hutchinson damping estimate
with exact transport trace. Curvature arms use the same shadow SOAP metric and
mu=max(.01*trace(metric)/D,1e-8), equivalently matched trace-normalized geometry.
Both have the initial SOAP statistics-only call. Other ports retain their native
initialization behavior; report accepted updates as well as consumed iterations.

Searched arms use zero + 2^[-16..16] native-proposal-radius multipliers and 18
golden refinements. Curvature-arm radius comes from shadow SOAP. Each stochastic
or stateful proposal is made once; search never advances optimizer/noise state.
Positive boundary selections fail, rejected steps still advance optimizer state.

Verify operator actions/solves, matrix PSD, probability mass/gauge invariance,
native step decomposition, moments, zero-noise limit, non-mutation and exact
checkpoint replay. Preserve configurations, source and data hashes, all scalar
trials, checkpoint states and per-step cost. Run arms sequentially, retain failures,
and compare both iteration and measured step-time results. Estimated cost is
several minutes; no automatic continuation beyond 50 steps or followup sweep.
