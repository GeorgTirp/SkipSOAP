# NI-SOAP learning-rate search extension

The user explicitly requested line search for both supplied algorithms after the
initial pilot. PARSEC already has native/searched trajectories and matched Adam
controls in `runs/20261006T100557-f1efb932`. This extension selects only two new
trajectories, NI-SOAP alpha 0 and 1 with learning-rate search. It does not rerun or
replace the eight completed arms. This protocol precedes the extension run.

Use the same ViT, initialization, seed, MNIST split, batch 512, 50 consumed steps,
data order, CPU float64 and one thread. Retain all NI-SOAP hyperparameters and
routing from the original protocol. Validate at 0/10/25/50 on all 5,000 reserved
examples. Preserve native trajectories as the control for adding search.

At each step, compute one true gradient, one optimizer-state proposal and one
deterministic local noise draw. Decompose the native lr=.003 proposal into drift
d and noise n. Search nonnegative learning-rate multiplier a along

`Delta(a) = a d + sqrt(a) n`, with `eta = .003 a`.

This respects the supplied algorithm's linear learning-rate dependence of drift
and square-root dependence of noise. Multiplying the whole noisy step by a would
instead alter their relative scaling. No candidate advances moments or resamples
noise. The state advances once even if a=0 is selected. Each next step uses a new
prespecified local draw; selected learning rates do not change the random stream.

Use the existing broad policy: zero plus powers 2^[-16..16], followed by 18
golden-section evaluations if an interior positive grid point wins. Reject the
zero choice explicitly and fail positive upper-boundary choices. Search uses
construction loss only. For alpha zero this reduces exactly to ordinary global
direction search with native radius. For alpha one the path is curved, not a
fixed ray. Both receive the same evaluation budget, not guaranteed equal realized
counts when zero wins.

This is a loss-selected stochastic algorithm. Although the candidate formula
at fixed eta retains the source scaling, eta depends on the realized draw, so
conditional noise after selection need not be zero-mean or have the fixed-eta
covariance. Saved scaled expected energies are pre-selection formula diagnostics,
not unconditional expectations for the selected noisy step. Do not claim search
preserves the native stochastic law or removes every hyperparameter confound.

Before training verify that decomposition matches fresh native proposals at
several learning rates with the same draw/state, that alpha-zero search matches
the ordinary ray search, and that replay/non-mutation checks pass. At completion
verify all final checkpoints, paired batches against the parent run, searches,
energy decomposition and archived hashes. Report selected learning-rate ranges,
rejections, validation and measured step costs, with native NI-SOAP, searched
PARSEC/Adam and earlier SOAP as labelled controls. No tuning sweep or new test-set
evaluation is selected.
