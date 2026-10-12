# SN x exact per-tensor GGN training — protocol

Study `sn_ggn_20261008`, requested 8 October 2026: "Can we run exact layerwise GGN with and without SN
for 250 steps now?" It answers whether SNGP spectral normalization changes how the exact per-tensor
GGN optimizer trains, and whether it changes GGN's advantage over native SOAP. This is the first direct
optimizer comparison under SN (ideas/parsec.md, section 4).

## 1. Isolation rules (as in the earlier side studies)

- Write only to `studies/sn_ggn_20261008/` and `~/skipsoap-study-data/sn_ggn_20261008/`. Read and
  import only elsewhere. No git write operations, no new packages, no downloads, no official MNIST test
  split. `PYTHONDONTWRITEBYTECODE=1` for every Python call; pytest with `-p no:cacheprovider`.
- At most four concurrent single-threaded processes.
- Numerical tolerance 1e-8 for action checks and solve residuals. Never loosen a gate; stop and report.
- Isolation manifests before and after; repository suite before and after.

## 2. Design

- Workload: main ViT (`configs/mini_vit_allocation.yaml`; width 16, 2 blocks, 5,738 parameters), MNIST
  28x28, batch 512, CPU float64, all 40 tensors. Fresh seeds 1, 2, 3 (model init seed s, data order
  seed 102 + 1000 s, split seed 101), as in the gap-audit evaluation.
- Optimizers, hyperparameters as tuned *without* SN in the gap audit (not retuned):
  - `ggn`: the gap audit's `ggn_exact_trace`: exact per-tensor GGN blocks; damping
    μ = max(.001 · tr C / D, 1e-8) with the exact current trace; normalized inverse shadow-SOAP metric;
    current-gradient RHS; one global ray search per step on the training minibatch. Code: gap-audit
    `arms.py`, copied unchanged.
  - `soap` (reference arm, cheap): native SOAP, lr 1e-2.
- SN conditions: none; c = 0.95 (the only bound active from step 0 in the ViT; SN curvature study).
  SN as in that study: W_eff = W · min(1, c/σ(W)) with exact σ on the 12 residual-branch linear weights
  (attention q/k/v/o, MLP w1/w2), via torch parametrization; the optimizer acts on the raw weights.
- 12 units: {ggn, soap} x {no SN, c = 0.95} x seeds 1–3, 250 consumed steps each.
- Measurements:
  - validation loss and accuracy on the reserved 5,000 examples at steps 0, 10, 25, 50, …, 250;
  - σ(W) of each SN tensor and the number of active SN tensors at the same steps;
  - per step: own-batch loss before and after (from the step), the loss of the *next* batch before and
    after the update (outside the timed region), step norm, μ, trace, search outcome, seconds;
  - action audits (per-tensor GGN versus independent JVP/VJP actions) at steps 2, 50, 200, 250;
    checkpoints at steps 50 and 250.

## 3. Gates

- **Reproduction gate.** Every no-SN unit must reproduce the gap-audit evaluation unit of the same arm
  and seed (`20261007T143302-baba2664`) at every validation step up to 250. A difference above 1e-9
  stops that unit, and the study is reported as failed.
- Action checks and dense-solve residuals ≤ 1e-8, or the unit stops.
- Study tests:
  - `peek` returns the next batch without advancing the data order;
  - a GGN step on the SN model passes the action audit, and every active SN tensor's gradient is
    orthogonal to its raw weight;
  - with SN inactive (c = 1e9), two GGN steps agree with the plain model to a relative 1e-10.

## 4. Pre-declared readings

Better or worse means lower or higher validation loss in all three seeds; otherwise mixed.

1. **SN effect on GGN:** `ggn` with c = 0.95 versus without, at steps 50, 100 and 250.
2. **SN effect on native SOAP:** the same reading for `soap`.
3. **SN effect on GGN's advantage:** the ratio soap/ggn of validation losses per seed, with and without
   SN, at steps 50, 100 and 250. The advantage *grows under SN* if the ratio is higher with SN in all
   three seeds, *shrinks* if lower in all three, otherwise mixed.
4. **Minibatch fitting:** transfer = mean next-batch decrease / mean own-batch decrease per update, in
   windows 1–50, 51–150 and 151–250, from the paired next-batch measurement. The log-based estimator
   of the earlier analysis is reported alongside.
5. Steps and measured seconds to the gap-audit targets T50 = 0.355037 and T100 = 0.265401.

Exploratory, labelled as such: σ trajectories, μ and trace, step norms.

## 5. Limitations known in advance

- Hyperparameters were tuned without SN. A loss under SN may reflect untuned γ or lr rather than SN.
- Three seeds, one workload, 250 steps.
- At c = 0.95 the loss is scale-invariant in each active SN tensor's raw weight W. The gradient is then
  orthogonal to W, and the raw-parameter GGN is zero along W. The step's component along W comes only
  from the damping and SOAP-metric term. As ‖W‖ grows, the effective step of that tensor shrinks.
- SN at c = 2 or 6 is inactive this early and is not tested.

## 6. Budget

Sequential dry run (`logs/dryrun.json`): 3.92 s per GGN step without SN, 4.41 s with SN. With the
1.6x slowdown observed under four workers, about 2.9 CPU-h for the GGN units and under 0.1 CPU-h for
SOAP. Budget: 4 CPU-h. If the projection is exceeded by more than 50%, stop and report.
