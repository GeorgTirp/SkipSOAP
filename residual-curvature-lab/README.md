# Residual curvature laboratory

Start with [HANDOFF.md](HANDOFF.md) for experiment history, findings, run locations,
reproduction commands and the active task. [STATUS.md](STATUS.md) tracks implementation.

An independent, installable PyTorch research prototype for testing whether richer
within-tensor and cross-tensor GGN information improves residual-network updates. The first delivery
implements the tiny residual MLP references, official SOAP adapter, allocation oracle,
shared-probe prototype, snapshot runner and bounded training path. It makes no claim
that the proposed optimizer improves training. See [STATUS.md](STATUS.md).

The first authorized local MNIST pilots are complete. See
[the measured results](docs/local_mnist_results_20261001.md), including the retained
negative results for the rank-8 sketch training arms.

The requested batch-256 and batch-512 exact tensor GGN comparisons are also complete.
See [the larger-batch results](docs/local_large_batch_results_20261002.md) for the six
fixed-state measurements and the gated 50-iteration paired batch-512 training followup.
The exact reference achieved lower final validation loss than searched SOAP in that
short run, with lower accuracy and substantially greater measured compute cost.
The [snapshot protocol](docs/large_batch_oracle_protocol_20261002.md) and
[training protocol](docs/exact_training_followup_protocol_20261002.md) record the
selected settings. These bounded runners are separate from the generic training CLI.

The subsequent [downstream-curvature and allocation audit](docs/downstream_allocation_results_20261005.md)
is complete at the three batch-512 states. Shared residual transport matches direct
VJPs, but the first curvature-diagonal prototype does not meet its approximation
targets. Optimizing allocation for SOAP's own directions helps at two states and
hurts at the third; copying exact GGN shares gives little improvement. These are
fixed-state diagnostics, not a new successful optimizer or training result.

The [mini ViT allocation pilot](docs/mini_vit_results_20261005.md) is also complete.
At batch 512 after 50 consumed iterations, exact tensor GGN achieved validation
loss .325714 and accuracy 90.30%, versus searched SOAP .731623 and 77.40%, at 4.76x
its measured training-step time. Diagonal and full KKT allocation finished worse
than searched SOAP despite full allocation's favorable shared-state diagnostics.
These are one-seed results with untuned settings; the report retains all five arms.

The [output-information diagnostic](docs/jacobian_information_results_20261005.md)
then separates curvature source, diagonal compression, momentum and tensor norms
at six saved ViT states. Dense per-example gradient covariance recovers much of
the early local GGN advantage; preserving correlations matters more than having
the exact GGN target at those states. Later-state reversals and momentum effects
are retained. These findings do not yet establish a new training method.

The [transport/PARSEC/NI-SOAP pilot](docs/transport_parsec_noise_results_20261006.md)
and [NI-SOAP learning-rate search](docs/ni_soap_lr_search_results_20261006.md) are
complete. Searched PARSEC improves its matched Adam control (loss 1.007097 versus
1.239108). NI-SOAP noise worsens both native and searched results; searched losses
are .973104 with noise and .867103 without. Tensor graph-Wasserstein reaches
.381630 versus the exact-trace tensor-GGN control .327676. These are paired,
single-seed 50-step ViT trials; reports preserve settings, costs, approximation
errors, negative findings and the supplied optimizer semantics.

The [learnable inverse-capacity screen](docs/learnable_capacity_results_20261007.md)
is complete: cheap rank-4/16 state bases with teacher coefficients recover
37.3%/41.6% of aggregate local GGN headroom. Teacher-selected directions nearly
close the early SOAP-state gap, but performance depends on calibration, training
state and momentum. No eligible family passes the declared gate; predictor
learning and new paired training remain unrun within that sequence. The report
retains both calibrations, negative states and three interrupted numerical attempts.

The [matched OT/GGN follow-up](docs/ot_ggn_capacity_results_20261008.md) was executed
by the isolated [gap-audit study](studies/gap_audit_20261007/RESULTS.md) and independently
verified on resume. Diagonal plus three state-derived directions recover 57.0% of
OT's own local headroom versus 46.4% for GGN, at 4.063 extra model vectors. Neither
reaches 80%; the OT advantage is modest in absolute loss decrease. The study also
extends training to 500 steps on three fresh seeds: GGN's early per-step advantage
shrinks, and tuned native SOAP ends with lower loss and much lower time to target.
The main suite has 106 passing tests; the isolated study suite has 12. Large study
artifacts live under `~/skipsoap-study-data/gap_audit_20261007/`; preserve them
alongside the repository's lightweight mirrors.

Python 3.13.3 and PyTorch 2.14.1 were tested on Apple Silicon CPU. Exact installed
versions are in [tested-requirements.txt](tested-requirements.txt). CPU float64 is the
default. CUDA configuration is available but has not been audited on this machine.

```bash
cd residual-curvature-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r tested-requirements.txt
python -m pip install -e '.[dev]'
python -m pytest
python -m residual_curvature.cli smoke --config configs/synthetic_smoke.yaml
```

Each run creates a fresh directory under `runs/`, with resolved settings, provenance,
split manifest, atomic checkpoint, raw arrays and checksums, candidate metrics,
all scalar-search trials, independent-panel evaluations, CSV and plots. Run names
are printed by the CLI. Rebuild plots using:

```bash
python -m residual_curvature.cli summarize --run-dir runs/RUN_NAME
```

The prepared MNIST experiment is explicitly **downsampled 14×14 MNIST**, not the
original-resolution task. These commands are deliberate experiment selections;
implementation does not run them automatically:

```bash
python -m pip install -e '.[mnist]'
python -m residual_curvature.cli estimate --config configs/mnist_snapshots.yaml
python -m residual_curvature.cli snapshots --config configs/mnist_snapshots.yaml --dry-run
python -m residual_curvature.cli data --dataset mnist --download
python -m residual_curvature.cli snapshots --config configs/mnist_snapshots.yaml
python -m residual_curvature.cli train --config configs/mnist_training_pilot.yaml --dry-run
python -m residual_curvature.cli train --config configs/mnist_training_pilot.yaml
```

The MNIST pilot consumes 100 batches; its ordinary first SOAP call initializes bases
without moving parameters. Snapshots follow iterations 10, 50 and 100. Pools of
50,000/5,000/5,000 examples reserve training, validation and analysis before training;
the official test set is not evaluated. Snapshot construction uses the first analysis
batch, followed by four disjoint panel batches. No panel selects settings.

For an interrupted run, keep the exact configuration and add
`--resume RUN_DIRECTORY/checkpoint.pt` to `train` or `snapshots`. Restoration creates
a new output directory and preserves consumed iterations, accepted updates, data
permutation position, optimizer context and RNG. Load only trusted local checkpoints.

All YAML files resolve against `config.DEFAULT`; unknown keys fail. `resolved.yaml`
is standalone, including every SOAP default. The training pilot uses plain SOAP.
Algorithm A can be selected prospectively with `training.method: sketch_diagonal`
or `sketch_coupled` and `training.policy: training_backtracking`. Those trajectories
use a labelled sketch-trace damping estimate. They are not launched by the smoke run.

The small ViT is available through `configs/mini_vit_allocation.yaml` and the
bounded runner `scripts/run_mini_vit_allocation.py`; its
[prospective protocol](docs/mini_vit_protocol_20261005.md) covers the selected pilot.
The generic decoder/Stage 4 configuration remains deferred and never falls back
to an MLP. Algorithm B, replicated training and large-model integration are later
gated work. Consult [the protocol](docs/experiment_protocol.md) before running them.

Source maps: `curvature/` contains operators, exact solves, transport and sketches;
`optim/` contains SOAP instrumentation and allocation; `experiments/` owns candidate
evaluation, search and training. Framework-neutral operator contracts are kept in
`curvature/interfaces.py`. No existing research repository was imported or modified.
