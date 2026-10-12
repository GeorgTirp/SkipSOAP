# Selected local MNIST larger-batch oracle protocol

User authorized batch 256 followed by 512 on 2 October 2026. This protocol is
written before those measurements. The first phase isolates construction-batch
size at the existing SOAP states 10, 50 and 100 from
`runs/20261001T080647-fd8e50e5`; their training history and moments used batch 64.
It is not a new batch-256 or batch-512 training trajectory.

Use the unchanged 14x14 residual MLP, all 6,634 parameters in 22 separate tensor
blocks, seed 0, CPU float64 and the saved pre-proposal SOAP states. Compare native
SOAP, its direction after the common search, frozen -Pg, and exact tensor GGN.
The gradient and SOAP proposal are recomputed on each construction batch.

Accumulate full tensor GGN blocks in microbatches of eight with mean normalization
over the complete batch. Do not allocate a full parameter GGN, full-batch Jacobian,
or full output factor. Solve the damped tensor systems by Cholesky. Check every
block's action against an independent JVP-Hz-VJP calculation; action and solve
relative errors must pass 1e-8. Store exact blocks, directions, contexts and hashes.
The conservative tensor-workspace budget is 768 MiB; also log process peak RSS,
which includes data and libraries and is a different quantity.

Use the existing normalized inverse-SOAP damping metric and
mu=max(gamma*trace(C)/D, 1e-8). The primary gamma is 0.01. The prespecified
sensitivity settings 0.001 and 0.1 remain reported individually. No independent
panel selects damping or a step size.

Normalize every direction to the same native SOAP norm. Evaluate zero and the
common radius multipliers 2^j for j=-16,...,16. If the best positive grid point is
interior, refine its neighboring interval with 18 golden-section evaluations.
Retain the best observed construction loss, with a maximum budget of 52 loss
evaluations per direction. Report zero rejections and upper-bound selections;
an unbracketed positive optimum does not pass the search gate. This finite local
search does not claim global nonlinear optimality.

Use analysis examples [0:B] for construction and [512*i:512*i+B], i=1,...,4,
for the four independent batches. They are disjoint, and batch-256 examples are
nested in batch-512 regions. Preserve the original train/validation/analysis split
and never access the official test set. First finish all three 256 snapshots,
then all three 512 snapshots. No failed run is overwritten.

The gate for a short paired training followup at a given batch size is: at primary
gamma, exact tensor GGN beats searched SOAP on mean independent loss decrease at
at least two of the three states, the average difference across states is positive,
and all numerical/search gates pass. A followup must be labelled separately from
these fixed-state measurements and use a protocol recorded before training.

Commands:

```sh
.venv/bin/python scripts/run_large_batch_oracle.py --dry-run
.venv/bin/python scripts/run_large_batch_oracle.py
```
