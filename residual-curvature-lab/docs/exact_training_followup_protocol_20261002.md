# Short exact tensor GGN training followup

Recorded after the fixed-state 256/512 comparison and before training. Only batch
512 passed the previously declared primary-gamma opportunity gate: independent
loss-decrease differences at states 10/50/100 were -0.018766, +0.054106, +0.003531.
Batch 256 did not pass. The gate is a spending rule, not statistical significance.

Run three arms at batch 512 for 50 consumed iterations from identical seed-0
random initialization and identical training example order: native SOAP,
line-searched SOAP, and exact tensor GGN with the same line search. The first
iteration initializes SOAP statistics without an update in every arm. Evaluate
the reserved 5,000-example validation pool at iterations 0, 10, 25 and 50. Do not
select settings or stop early using validation. Do not evaluate the official test.

All arms use the existing model and SOAP settings. Search uses the fixed-state
protocol's common broad radius grid and refinement. The GGN arm uses the
prespecified primary gamma 0.01, exact current-batch trace, and normalized inverse
SOAP damping. Every arm advances its own SOAP statistics once per consumed
gradient; curvature and search never update statistics. GGN directions use the
true gradient, while SOAP proposals retain momentum. Save the solver residuals,
search trials, minibatch measurements and checkpoints. A positive search boundary
or failed numerical solve stops the run with the failure retained.

The primary trajectory comparison is exact GGN versus line-searched SOAP, with
native SOAP retained to expose the effect of searching SOAP itself. Compare both
equal consumed examples and measured training-step cost. One seed and 49 updates
cannot establish a broad optimizer ranking. This is a SOAP-metric-damped tensor
GGN reference, not a reproduction of the LLM paper's training setup.

```sh
.venv/bin/python scripts/run_exact_training_followup.py --oracle-run runs/20261002T121124-74f10aae
```
