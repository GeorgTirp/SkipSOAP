# Gap audit study: agent instructions (checks 1–4)

Study ID: `gap_audit_20261007`. Authorized by the project owner (Georg) on 7 October 2026
by providing this file. This file is the prospective protocol for the study. Do not edit
it after you start; record every departure, with reason and time, in `DEVIATIONS.md`.
Launch prompt: "Read studies/gap_audit_20261007/AGENT_INSTRUCTIONS.md and execute it
exactly, starting with section 4." Keep a task list for the checks.

## 0. Purpose

`HANDOFF.md` reports, for one seed and 50 steps on the 5,738-parameter ViT, that exact
tensor GGN with global search reaches validation loss about .33 versus about .73 for
searched SOAP. Fixed-state diagnostics attribute much of the local gain to within-tensor
geometry that the SOAP-coordinate diagonal discards. This study asks, in order:

1. Is the gap real? Does it survive three fresh seeds, a 10x longer horizon and tuned
   baselines?
2. Is the missing information contained in per-example gradients? Can a structured family
   (Kronecker sums, a better Kronecker basis, few-matvec Krylov) hold it when given
   oracle parameters?
3. Are graph-Wasserstein (OT) teacher blocks easier to compress than GGN blocks? This runs
   the already-written OT/GGN capacity protocol.
4. Does the SOAP-to-GGN headroom exist across a small ViT family? Can a small
   permutation-equivariant network on the neuron graph predict a structured operator
   that transfers to held-out widths and depths? (Gated.)

This is a self-contained side study. It must not change, overwrite or confuse anything in
the main lab. Run nothing beyond what is listed here: no extra sweeps.

## 1. Isolation rules (hard constraints)

1. Write only to `studies/gap_audit_20261007/` (this folder) and to the external data
   directory `$GAP_AUDIT_DATA` (default `~/skipsoap-study-data/gap_audit_20261007/`).
   The external directory sits outside the OneDrive-synced project. All run directories,
   checkpoints and arrays go there.
2. Read and import only. Never modify, rename, delete or reformat anything in `src/`,
   `scripts/`, `tests/`, `configs/`, `docs/`, `references/`, `third_party/`, `runs/`,
   `datasets/`, `.venv/`, `HANDOFF.md`, `STATUS.md`, `README.md`, `AGENTS.md`,
   `pyproject.toml` or `tested-requirements.txt`.
3. Never create a run directory under the repository's `runs/`. Every config passed to
   `residual_curvature.provenance.new_run` must set `output` to `$GAP_AUDIT_DATA/runs`.
4. Set `PYTHONDONTWRITEBYTECODE=1` for every Python call, so that importing `scripts/*.py`
   writes no `__pycache__` into shared folders. Run pytest with `-p no:cacheprovider`.
5. Use no git write operations (add, commit, stash, checkout, reset, clean, rm, branch).
6. Install no packages and add no dependencies; use torch, numpy, scipy, matplotlib,
   PyYAML and pytest from `.venv`. Download nothing. MNIST is local. Never access the
   official MNIST test split.
7. If you find a bug in shared code, do not fix it. Write a minimal reproduction under
   "Issues found in shared code" in `RESULTS.md`. Work around it in a clearly named
   study-local function and say so in the affected section.
8. Study tests live in `studies/gap_audit_20261007/tests/` and are run by explicit path.
   The default suite (`testpaths = ["tests"]`) must not collect them.
9. Do not edit `HANDOFF.md` or `STATUS.md`, even though `AGENTS.md` asks for that after
   experiments. Write `HANDOFF_PROPOSED_ENTRY.md` in this folder instead; a human merges it.
10. Isolation audit. Before writing anything else, save a SHA-256 manifest to
    `isolation_manifest_before.txt`. It covers every file in `residual-curvature-lab/`
    except `runs/`, `datasets/`, `.venv/`, `build/`, `*.egg-info/`, `.pytest_cache/`,
    `__pycache__/`, `../.git/` and `studies/gap_audit_20261007/`. At the end, recompute it
    as `isolation_manifest_after.txt`, confirm the two are identical, and report the
    result.

## 2. Folder layout

```
studies/gap_audit_20261007/
  AGENT_INSTRUCTIONS.md        this file (frozen)
  README.md                    for collaborators: what this is, status, where results are
  RESULTS.md                   the single results document for all checks
  DEVIATIONS.md                every departure from this file
  HANDOFF_PROPOSED_ENTRY.md    draft ledger rows and insights for the main HANDOFF
  isolation_manifest_before.txt, isolation_manifest_after.txt
  configs/   study YAML configs (derived from configs/mini_vit_allocation.yaml)
  scripts/   all study code (common.py, check1_*.py, check2_*.py, ...)
  tests/     study tests
  runs/      light mirrors of external runs (json/md/png/yaml only; <= 30 MB total)
  figures/   PNGs regenerated from stored JSON by summarize scripts
  logs/      stdout/stderr of long runs
```

`README.md` begins: "Isolated side study. Nothing here changes the main lab, its scripts
or its runs. Large artifacts live outside the synced folder at <path>. Start with
RESULTS.md."

## 3. Execution conventions

- Run from `residual-curvature-lab/` with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ...`.
- `scripts/common.py` puts the repository `scripts/` directory on `sys.path` and resolves
  `$GAP_AUDIT_DATA`. Import existing code rather than copying it, e.g.
  `from run_mini_vit_allocation import propose, search_candidate`,
  `from run_transport_parsec_noise import propose as transport_propose, optimizer_settings`,
  `from run_exact_training_followup import evaluate`,
  `from run_large_batch_oracle import measurement, state_equal`,
  `from residual_curvature.experiments.training import commit`.
  If you must change behaviour, copy the function as `study_<name>` and log it.
- Use CPU, float64, `torch.set_num_threads(1)` and `torch.use_deterministic_algorithms(True)`.
  You may run up to four independent single-threaded processes in parallel.
- Numerical tolerance stays 1e-8 (`c['solver']['tolerance64']`). Never loosen a
  tolerance or gate to make a run pass. Stop and report instead.
- Create every run through `new_run(c)` with output under `$GAP_AUDIT_DATA/runs`. Archive
  the study scripts used, this file and every imported repository script into
  `<run>/source/` with SHA-256 checksums, following `scripts/run_transport_parsec_noise.py`.
  Write `complete.json` on success and `failure.json` on failure. After each run, copy
  its json/md/png/yaml files into `runs/<run-id>/` here, together with `EXTERNAL_PATH.txt`.
- Resumability: each (check, arm, seed) unit writes its own completion marker. Rerunning
  a script verifies the checksums of completed units and skips them. If a session ends
  mid-check, write the status into `RESULTS.md`.
- Run long jobs in the background with logs in `logs/`. Poll the log; do not busy-wait.
- Seeds: the model init seed is `s`. The data order is
  `DataOrder(manifest['training'], 102 + 1000*s)`; `s = 0` reproduces the existing order
  (`data_seed + 1 = 102`). Keep `data_seed = 101` (the split) fixed for every seed.
  Probe and analysis seeds are unchanged.

## 4. Before any experiment

1. Read `HANDOFF.md`, `AGENTS.md`, `docs/writing_style.md` and
   `docs/mathematical_model.md`. Also read `docs/mini_vit_results_20261005.md`,
   `docs/jacobian_information_results_20261005.md`,
   `docs/transport_parsec_noise_results_20261006.md`,
   `docs/learnable_capacity_results_20261007.md` and
   `docs/ot_ggn_capacity_protocol_20261007.md`.
2. Write the isolation manifest (rule 10).
3. Run `.venv/bin/python -m pytest -q -p no:cacheprovider`. Record the pass count and
   duration in `RESULTS.md` under "Baseline suite". If anything fails, stop.
4. Record the SHA-256 of this file in `RESULTS.md`.
5. Cost dry run: time three steps of every training arm and one fixed-state candidate
   evaluation, and write the projected CPU-hours per check into `RESULTS.md`.
   Budgets: check 1 (with 2a/2b) <= 12 CPU-h, check 2c <= 1, check 3 <= 1, check 4 <= 6.
   If a projection exceeds its budget, use that check's named fallback and log it.

## 5. Check 1: is the gap real?

**Workload (fixed):** `configs/mini_vit_allocation.yaml` (ViT width 16, expansion 32,
2 blocks, 2 heads, 28x28 MNIST, batch 512, CPU float64). Validation is the reserved
5,000 examples.

**Arms.** All reuse existing proposal code. A consumed step always advances SOAP
statistics exactly once, as in the existing runners.

| Arm | Implementation |
| --- | --- |
| `native_soap` | `run_mini_vit_allocation.propose('native_soap', ...)`, `c['soap']['lr']` tuned |
| `adam_native` | `transport_propose('adam_native', ..., settings, routes=None)`, settings from `optimizer_settings('adam_native')` with `lr` tuned |
| `searched_soap` | `run_mini_vit_allocation.propose('searched_soap', ...)`, default SOAP settings |
| `searched_soap_tuned` | as above with tuned `betas` / `precondition_frequency` |
| `ggn_exact_trace` | exact tensor GGN, exact-trace damping, shadow-SOAP metric, current gradient, global search, as `transport_propose('ggn_exact_trace', ...)`. That function hard-codes gamma .01, so for tuning use a study copy `study_ggn_propose(gamma)` containing only that branch (pre-approved; still log it) |
| `dense_ef` | check 2a |
| `soap_gradient_map` | check 2b |

**Step 1: reproduction gate (seed 0, 50 steps, default settings).** These arms must
reproduce the stored step-50 validation loss to within 1e-9 absolute:

- `native_soap` and `searched_soap`, from `runs/20261005T135920-d7ce29b2/validation.json`;
- `adam_native` and `ggn_exact_trace` (gamma .01), from
  `runs/20261006T100557-f1efb932/validation.json`.

A difference above 1e-6 stops the study until it is explained. A difference in
(1e-9, 1e-6] is recorded, and you may continue.

**Step 2: tuning (seed 0 only).** Select by validation loss at the end of the tuning
horizon:

- `native_soap`: lr in {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}, 500 steps.
- `adam_native`: lr in the same grid, 500 steps.
- `searched_soap_tuned`: betas in {(.95,.95), (.9,.99)} x `precondition_frequency` in
  {1, 10}, 500 steps.
- `ggn_exact_trace`: gamma in {.001, .01, .1}, 200 steps.

If the best value lies on a grid edge, extend one grid point in that direction, once.
A divergent or NaN run is a failed configuration, not an error. Tuning results are
reported but are not evidence for rankings.

**Step 3: evaluation (fresh seeds 1, 2, 3).** Run all seven arms for 500 consumed steps
with the tuned settings. Validate at steps 0 and 10, then every 25 steps up to 500. Store
the per-step train-batch loss, step seconds (excluding validation and I/O, as in the
existing runners), all search trials, and checkpoints at steps 50, 200 and 500.
Fallback if over budget: 300 steps.

**Report:**

- A table of mean ± sd (n=3) validation loss and accuracy at steps 50/100/200/500 for
  every arm, plus a per-seed appendix.
- A figure of validation loss against step and against cumulative training seconds
  (log y): seed mean, with thin per-seed lines.
- A curve of the ratio `searched_soap_tuned / ggn_exact_trace` over steps.
- Steps-to-target and seconds-to-target for targets T50 and T100. These are the mean
  `ggn_exact_trace` validation losses at steps 50 and 100 over seeds 1–3. "Not reached"
  is allowed.
- A verdict in these pre-declared descriptive categories (not significance tests):
  - **A, gap persists:** at step 500, `searched_soap_tuned` loss >= 1.25 x
    `ggn_exact_trace` loss in all three seeds.
  - **B, early phase only:** at step 500 the ratio is <= 1.10 in at least two seeds,
    while at step 50 it is >= 1.5 in at least two seeds.
  - **C, mixed:** anything else; describe it.
- Report the wall-clock view separately: does GGN reach T50 and T100 in fewer seconds
  than the best SOAP or Adam arm?
- State whether `precondition_frequency = 1` (fresh bases every step) materially changes
  searched SOAP. This bears on EMA and basis staleness.

## 6. Check 2: where is the missing information, and can a structured family hold it?

### 2a. Dense empirical-Fisher training arm (`dense_ef`)

This is identical to `ggn_exact_trace` except for the tensor blocks:
F_bb = (1/B) sum_i g_ib g_ib^T, where g_i is the gradient of example i's loss (so
mean_i g_i = g). Compute per-example gradients with `torch.func.vmap(grad(...))`.

- Damping: mu = max(gamma * trace(F) / D, floor), using the gamma selected for
  `ggn_exact_trace` (no separate tuning).
- Same shadow-SOAP metric, `layerwise.solve_blocks`, current-gradient RHS, and
  `search_candidate` with the native SOAP radius.
- The first consumed step is SOAP initialization, as in the other curvature arms.

Study tests:

1. The mean of the per-example gradients equals `GGN(...).gradient` to 1e-12 relative.
2. The blocks equal `curvature.information.information_blocks(...)[0]['empirical_fisher']`
   on a 16-example batch to 1e-10 relative.
3. The solve residual is <= 1e-8.

### 2b. Current-gradient SOAP-map training arm (`soap_gradient_map`)

Each step, `soap_propose` advances SOAP statistics once. The candidate is
`-SOAPMetric(registry, contexts).map(g)`: the current gradient through SOAP's
preconditioner, without momentum. It then goes through `search_candidate` with the native
radius. Over a trajectory, this separates SOAP's geometry from SOAP's momentum.

2a and 2b run inside check 1's evaluation (seeds 1–3, 500 steps). Pre-declared reading:
`dense_ef` *suffices* if its seed-mean validation loss closes >= 75% of the
`searched_soap_tuned` -> `ggn_exact_trace` gap at both steps 50 and 200. It is *partial*
at 25–75% and *no* below 25%.

### 2c. Offline capacity of structured families (six saved states)

**Inputs.** Use
`runs/20261005T143029-9b582ed5/{searched_soap,exact_tensor_ggn}-{0010,0025,0050}/`.
`arrays.pt` holds the keys `parameters`, `gradient`, `momentum_rhs`, `contexts`,
`ggn_blocks`, `empirical_fisher_blocks` and `raw_soap`; `diagnostics.json` holds
`construction_ids`, `panel_ids`, `step` and `trajectory`. Verify the `artifacts.json`
checksums first. Rebuild the objective, panels, `GGN`, `SOAPMetric`,
mu = max(.01 * sum_b trace(C_bb) / D, floor) and Omega exactly as
`scripts/run_learnable_capacity.py::audit` does, and reuse `measurement` and
`search_candidate`.

**Sanity reproduction.**

- The SOAP current-gradient map and exact tensor GGN candidates must reproduce the
  capacity screen's per-state independent decreases to 1e-9. These are the rows
  `identity` and `exact` of the damping table, e.g. .033191 / .135327 at SOAP step 10.
  The raw JSON is in `runs/20261007T093903-8d7d86f1`.
- `soap_basis_diag` must reproduce "GGN diagonal in SOAP basis + gradient" from
  `docs/jacobian_information_results_20261005.md`. Use
  `optim.downstream_soap.exact_coordinate_diagonal`.

**Families.** All are per tensor, damped with the same mu * Omega_b, and solved densely
for this diagnostic:

- `soap_basis_diag`: the exact GGN diagonal in SOAP's saved bases (reproduction).
- `kron_basis_diag`: let A1 ⊗ B1 be the nearest Kronecker product to C_bb (Van
  Loan–Pitsianis rearrangement, leading singular pair, signs chosen so that A1 and B1
  are symmetric PSD). With orthonormal eigenvectors U = U_A ⊗ U_B, use
  U diag(diag(U^T C_bb U)) U^T. This tests whether any Kronecker eigenbasis diagonal
  recovers the gain, i.e. whether separable neuron-graph filters can work.
- `ks_1`, `ks_2`, `ks_3`: sum_{k<=r} A_k ⊗ B_k from the top-r singular pairs of the
  rearranged block, with symmetrized factors. If the sum is not PSD, clip the negative
  eigenvalues of the n x n approximation and record the minimum eigenvalue and the
  clipped trace fraction.
- `ks_2_ef`: `ks_2` fitted to the empirical-Fisher block F_bb, still damped with the GGN
  mu. This asks whether a teacher computable from per-example gradients can feed the
  structure.
- `pcg_block_k`, k in {2, 3, 5, 10}: k iterations of SOAP-preconditioned CG on
  blockdiag(C_bb) + mu * Omega, starting from x0 = 0 with the current gradient as RHS.
  The preconditioner is the SOAP map, i.e. Omega^{-1} up to scale.
- `pcg_full_k`, k in {2, 3, 5, 10}: the same on the full damped GGN,
  `op.action(v) + mu * metric.apply(v)`, including cross-tensor terms. This is the
  deployable matrix-free yardstick: each iteration is one JVP plus one VJP. k = 1 is
  omitted because, from zero, it is a rescaled SOAP map and identical after search.
- `exact_1d_only`: exact blocks for 1-D tensors (biases, LayerNorm), the SOAP
  current-gradient map elsewhere. This controls for how much of a family's score comes
  from the small 1-D tensors.

**Conventions.**

- The registry flattens an (out, in) tensor row-major, so (A ⊗ B) vec(W) = vec(A W B^T).
  Verify this with a study test before use.
- 1-D tensors use their exact block in every Kronecker family. Tensors with a size-1
  axis (e.g. `cls_token`, 1x16) are exact under Kronecker structure anyway.
- Write your own truncated PCG and count matvecs. (`curvature.solves.cg` spends an extra
  action per iteration on its residual check.)

**Scoring.** Use the same global construction-loss search and four panels as the
capacity screen. Score every family on the current gradient; score the structured-operator
families on the effective momentum RHS too. Report:

- per-state independent decreases;
- aggregate current-gradient local recovery,
  (D_family − D_SOAP-map) / (D_exact − D_SOAP-map), over all six states, none dropped;
- block-teacher A-energy errors on the gradient, the momentum and 16 factor-coordinate RHS,
  as in `optim.capacity.energy_errors`;
- storage, as numbers per tensor, in model vectors and relative to SOAP's own factor
  storage;
- matvec counts.

Pre-declared tiers for the cheap-to-apply families (`kron_basis_diag`, `ks_1`, `ks_2`,
`ks_2_ef`): >= 80% meets the previous screen's quality bar; 60–80% is *promising*; below
60% is *insufficient*. `ks_3` and `pcg_*` are references.

## 7. Check 3: OT versus GGN compressibility

The protocol is `docs/ot_ggn_capacity_protocol_20261007.md` and the runner is
`scripts/run_ot_ggn_capacity.py`. It had not been run when this file was written.

1. First check whether the main lab has run it since. Look for a directory under `runs/`
   whose `protocol.json` names that document and that has a `complete.json`. If one
   exists, do not rerun it; analyse that run read-only and say so.
2. Otherwise write `scripts/check3_ot_ggn_capacity.py`. It imports `audit`, `read`,
   `PARENT`, `TRAINING` and `FAMILIES_MATCHED` from the existing modules and replicates
   `main()` of `scripts/run_ot_ggn_capacity.py` line for line, with three exceptions:
   - `c['output']` points to `$GAP_AUDIT_DATA/runs`;
   - the hard-coded attachment is copied if it exists; otherwise
     `{"attachment_missing": true, "expected_path": ...}` is recorded in `protocol.json`;
   - the original runner file is archived as well.

   Expected counts: 420 candidate rows, 204 operator states and 12 states. Do not change
   families, gates, tolerances or panels.
3. Analysis (`scripts/summarize_check3.py`) follows the protocol: all families, both RHS
   and both targets; per-state and aggregate results; own-teacher recovery; absolute
   independent decreases; the exact-OT versus exact-GGN teacher decrease at every state.
   Pre-declared reading: *OT is easier to compress* if, for at least two of {`diagonal`,
   `state_4`, `combined_state_3`}, two things hold:
   - OT's own-teacher current-gradient recovery exceeds GGN's by >= 10 percentage points;
   - the family's absolute mean independent decrease under OT is >= its decrease under
     GGN.

   Otherwise report *no evidence that OT is easier to compress*.

## 8. Check 4: architecture family and neuron-graph predictor (gated)

### 4a. Family dataset (always runs)

- Architectures: blocks in {1, 2, 3} x width in {8, 16, 24}, heads 2,
  expansion = 2 x width, patch 7, 28x28 MNIST, batch 512 (nine architectures).
- Each (architecture, seed) runs 50 consumed steps of searched SOAP (default settings)
  with the seed rule from section 3. Snapshots follow steps 10, 25 and 50.
- Seeds: {0, 1, 2} for training architectures, {0, 1} for held-out ones.

Width 16 with 2 blocks is the main ViT. Its seed-0 parameters at steps 10/25/50 must
match `runs/20261005T143029-9b582ed5/searched_soap-*/arrays.pt['parameters']` to 1e-12
(reproduction gate).

Store each snapshot in `$GAP_AUDIT_DATA`. Expect a few GB.

- parameters;
- the gradient on the construction batch (the first 512 analysis IDs, as in the main lab);
- the effective momentum RHS, rebuilt as in `scripts/run_jacobian_information.py::audit`:
  `-metric.map(native, inverse=True)` normalized to the gradient norm, with its ray check;
- SOAP contexts and the SOAP optimizer state;
- the raw SOAP proposal;
- exact tensor GGN blocks from `layerwise.streamed_blocks` (microbatch 8, budget 768 MiB),
  with the per-tensor action check against `GGN.action` at 1e-8;
- the four existing panel ID sets.

Splits:

- *train*: width in {8, 16} x blocks in {1, 2}, seeds 0 and 1;
- *validation*: the same architectures, seed 2;
- *held-out-width*: width 24, blocks in {1, 2};
- *held-out-depth*: blocks 3, width in {8, 16};
- *held-out-both*: width 24, blocks 3.

Fallback if over budget: drop the step-25 snapshots.

### 4b. Headroom and family capacity across architectures (always runs)

Evaluate every state with:

- the SOAP current-gradient map;
- exact tensor GGN;
- `soap_basis_diag`;
- the best cheap family from 2c (highest aggregate recovery among `kron_basis_diag`,
  `ks_1`, `ks_2` and `ks_2_ef`; ties go to lower storage);
- `pcg_full_2` and `pcg_full_5`.

Use the same search, panel and mu rules as 2c. Report, per architecture and state, the
exact-GGN local headroom (D_exact − D_SOAP-map) and each family's recovery. The question
is whether the headroom and the family's capacity look like the main ViT's across width
and depth.

### 4c. Neuron-graph predictor (only if the gate passes)

**Gate.** The best cheap family from 2c recovers >= 60% on the six main states and
>= 50% on the train-split states of 4b. If not, skip 4c and explain why in `RESULTS.md`:
there is no operator worth learning yet.

**Target.** The family's oracle parameters for each 2-D tensor: the factor pairs
(A_k, B_k) for `ks_r`; for `kron_basis_diag`, (A1, B1) plus the diagonal in their
eigenbasis. Fix the scale gauge so that ||A_k||_F = 1. 1-D tensors use the SOAP map
(cheap and deployable) and are not predicted.

**Predictor.** Plain PyTorch, <= 50k parameters, float32 for training and float64 for
evaluation.

- Graph: one bipartite neuron graph per 2-D tensor. Nodes are output and input neurons;
  edges are the matrix entries.
- Edge features, standardized per tensor: W, g, effective momentum, the SOAP-rotated
  gradient Q_L^T G Q_R, and SOAP's Adam denominator in rotated coordinates.
- Node features: row and column norms of W, g and momentum, and the diagonals of SOAP's
  left/right statistics (from the stored optimizer state).
- Global features: tensor role from the registry, depth fraction and log width.
- Layers: 3–4 permutation-equivariant row/column exchangeable layers (Hartford et al.
  2018: a·X + b·rowmean + c·colmean + d·mean + node terms, then a nonlinearity),
  hidden size 32.
- Output: node embeddings H_out (m x d) and H_in (n x d) per Kronecker term, giving
  A_k = H_out H_out^T / d + softplus diagonal, and B_k likewise (SPD by construction).
  For `kron_basis_diag`, an extra edge head predicts the log-diagonal in the eigenbasis
  of the predicted (A1, B1).
- Loss: the block-teacher A-energy error of the induced damped inverse on the RHS
  {current gradient, momentum, 8 factor-coordinate Gaussians}, normalized per tensor
  (definitions as in `optim.capacity.energy_errors`), plus 0.1 x the gauge-fixed
  Frobenius error to the oracle factors.
- Training: Adam, with early stopping on the validation split, <= 3 CPU-h.

**Evaluation.** On all three held-out groups, and on the train split to show overfitting:

- local recovery with the standard search and panels, for the current gradient and the
  momentum;
- the *transfer ratio*: predicted recovery divided by oracle-family recovery on the same
  states;
- A-energy errors;
- wall-clock time of predictor forward plus solve, versus the SOAP map and `pcg_full_2`.

Pre-declared success: on held-out-width and on held-out-depth, the transfer ratio is
>= 0.5 and the predicted recovery is >= `pcg_full_2`'s. Report held-out-both separately.
Run no training trajectories with the predictor in this study.

## 9. Reporting

`RESULTS.md` sections:

1. **Summary** (<= 12 lines): the verdict for each check in its pre-declared category,
   the one table a collaborator must see, and what it implies for the next step.
2. Baseline suite, protocol hash, and a cost ledger (projected versus actual CPU-hours
   per check).
3. Checks 1–4: settings as used (batch, scope, partition, damping, search, seeds,
   steps), tables, figure links, verdict and limitations.
4. Issues found in shared code.
5. Deviations (summary; the full list lives in `DEVIATIONS.md`).
6. The isolation audit result and the final suite result.
7. Reproduction commands.

Follow `docs/writing_style.md`:

- separate identities, approximations, observations and hypotheses;
- name the batch, scope, damping and search for every comparison;
- report negative results plainly;
- no single-seed rankings;
- say that validation data were used for tuning on seed 0 only.

`HANDOFF_PROPOSED_ENTRY.md` holds one ledger row per check in the HANDOFF table format,
plus at most five candidate "insights to preserve".

## 10. Stop conditions

Stop if any of these happens:

- a reproduction gate fails by more than 1e-6;
- a numerical check fails at 1e-8;
- the baseline suite fails;
- a budget would be exceeded even after its fallback;
- continuing would require modifying anything outside this folder or `$GAP_AUDIT_DATA`.

When you stop, write the current state into `RESULTS.md`, mark the check "stopped" and
end the session. Never loosen tolerances or change gates to continue.

## 11. Final checklist

- [ ] Every check is completed, or explicitly skipped or stopped with reasons.
- [ ] `.venv/bin/python -m pytest -q -p no:cacheprovider` passes with the baseline count.
- [ ] `.venv/bin/python -m pytest -q -p no:cacheprovider studies/gap_audit_20261007/tests`
      passes.
- [ ] The isolation manifest is unchanged.
- [ ] Light mirrors total <= 30 MB, and the external path is recorded.
- [ ] `README.md`, `RESULTS.md`, `DEVIATIONS.md` and `HANDOFF_PROPOSED_ENTRY.md` are
      written.
