# Mathematical model and contracts

For each example, the model applies an affine input projection, pre-norm residual
blocks `h[i+1] = h[i] + a W2 GELU(W1 LayerNorm(h[i]))`, and an affine logit head.
The registry includes every bias and normalization parameter, with stable name,
shape, count, slice, role, stage and orientation. Linear weights are `(out,in)`.
The primary partition is individual stored tensors. The alternative `block` partition
groups each residual stage, input projection and head; allocation directions still
refer to individual native SOAP tensors and are explicitly named in the records.

With T examples and mean cross-entropy, `r=(p-onehot(y))/T` and
`Hz=blockdiag(diag(p)-p p^T)/T`. The true gradient is `g=J^T r` and the GGN is
`C=J^T Hz J`. Weighted/masked operators use `w/sum(w)` consistently in the loss,
residual and metric. A test covers analytic categorical covariance and zero-weight
positions. This is not an empirical gradient outer product or the nonlinear-network
Hessian. A linearized-output Hessian audit provides an independent check.

The scoring model is `g^T d + (d^T C d + mu d^T Omega d)/2`, with positive damping.
`Omega` is identity or the frozen SOAP inverse map normalized to trace/D=1.
For each fixed state, all interaction comparisons use identical gradient, scope,
metric and damping. The gamma grid is `[.001,.01,.1]` with
`mu=max(gamma trace(C)/D,1e-8)`. Snapshots use the exact output-factor trace;
optional Algorithm A training labels its estimated trace separately.

Reference routes are dense Jacobian/output metric, JVP–metric–VJP products and the
output-factor Woodbury solve. With `C=UU^T`, `Q=Omega^-1`,
`d=-Qg/mu + QU solve(mu I + U^T QU, U^T Qg)/mu`.
The softmax square root is blockwise symmetric-eigenfactored; eigenvalues below
`100 eps max(abs(eigenvalue))` are removed, the tolerance is logged, and substantial
negative values fail. The output null mode is never inverted. Classification rank
is at most `batch*(classes-1)`. Cholesky/linear solves are used, not dense inverses.
Matrix-free preconditioned CG logs actual residual, iterations and convergence.

`R_i=I+dF_i/dh_i`, `B_i=dF_i/dtheta_i`, `T_i=dz/dh_i` give
`M_i=T_i^T Hz T_i=R_i^T M_(i+1) R_i` and
`C_ii=B_i^T M_(i+1) B_i`. For i<j,
`C_ij=B_i^T (R_j ... R_(i+1))^T M_(j+1) B_j`.
The explicit transport tests include LayerNorm/GELU and all within-stage tensors.
The hidden-block message sweep freezes input/head proposals, or seeds the terminal
message with a supplied head output change. Its result is one reverse Gauss–Seidel
sweep, not a full solve. Arbitrary subsequent/nonzero-initialized sweeps are audited
through the global sketch accumulator, which retains all current proposals.

For allocation, E embeds each nonzero native tensor unit direction on disjoint support.
Zero directions are recorded and excluded; uphill directions are retained.
`phi=-E^T g`, `K=E^T C E`, `KH=K+mu E^T Omega E`. The cone minimizes
`-phi^T tau + tau^T KH tau/2` for tau>=0. Cholesky reduces this to SciPy NNLS,
followed by primal, dual, active-stationarity and complementarity checks in original
coordinates. Squared shares are `tau_i²/sum(tau²)`. The signed span allows negative
coefficients. Optimized native ray <= cone <= signed span <= full-space damped value;
a diagonally constructed candidate has no such guarantee under the full model.

Shared categorical output probes are `(onehot(draw)-p)/sqrt(T*k)` with p detached.
All parameters share identical global probes. `Y=J^T Xi` is constructed by VJPs;
an explicit stagewise implementation and microbatch sum agree for identical probes.
Coupled `YY^T+mu Omega` and diagonal `blockdiag(Y_b Y_b^T)+mu Omega` preserve
identical diagonal blocks. Tensorwise Woodbury implements diagonal and reverse sweeps.
Nested ranks reuse probe prefixes with corrected `sqrt(k)` scaling. PSD and conditional
unbiasedness of a curvature estimate do not imply accuracy or unbiased inverse estimates.
