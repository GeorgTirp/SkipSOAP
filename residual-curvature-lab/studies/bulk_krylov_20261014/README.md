# bulk_krylov_20261014

Offline study: is the whitened per-tensor GGN spectrum a power law that the top-16 probes predict (Part P), and how
cheaply can the bulk gain of the exact per-tensor GGN step be recovered (Part K)? Read `PROTOCOL.md` first
(SHA-256 in `logs/protocol_sha256.txt`).

- [RESULTS.md](RESULTS.md): P1 and P2 not supported (broken power law early; the probes don't predict the bulk), K1
  supported, K2 m = 8, K3 and K4 not supported.
- [DEVIATIONS.md](DEVIATIONS.md): D1, a fit-helper crash before any output; fixed and rerun.
- [HANDOFF_PROPOSED_ENTRY.md](HANDOFF_PROPOSED_ENTRY.md): proposed ledger entry.

## Layout

- `scripts/run.py`: the driver over `ladder_mechanism_20261013`'s top-free states (seeds 1–3, steps 10, 25, 50, 100),
  read-only from `~/skipsoap-study-data/ladder_mechanism_20261013/states/`. One output per state in
  `$BK_DATA/bulk/` (default `~/skipsoap-study-data/bulk_krylov_20261014/`), mirrored to `results/bulk/`. Existing
  outputs are skipped, so an interrupted run resumes at the next state.
- `scripts/bulk.py`: Parts P and K at one state (rungs in `RUNGS`; gate L0).
- `scripts/spectral.py`: top eigenpairs, log-log fits, the trace bisection, Krylov bases, Chebyshev iteration, CG.
- `scripts/summarize.py`: readings P1–P3 and K1–K5 into `results/summary.json` and `results/summary.md`.
- `scripts/orchestrate.sh`: waits for `ggn_ot_fairness_20261014`, then manifests and suites before, the run, the
  summary, manifests and suites after. Logs in `logs/`.
- `scripts/common.py`, `states.py`, `algebra.py`: copied from `ladder_mechanism_20261013` (paths changed in
  `common.py`; the others carry a header note and are otherwise unchanged).
- `tests/`: run by explicit path, `pytest -p no:cacheprovider studies/bulk_krylov_20261014/tests`.

## Implementation choices the protocol leaves open (fixed before any run)

- **Probe draws:** the ladder's probe seed per state (`9300 + 1000·seed + step`) and its sketch seed (+1), so
  U_fresh is the ladder's `fresh` subspace.
- **Dense pass:** the ladder's calls exactly (first group with all parameters and chunk 64, then chunk 128), so the
  blocks, R* and top-free's reductions can be compared with the ladder's results (`summarize.py`, column "ladder
  bitwise"). The full-GGN CG rungs need μ before that pass; tr C is taken by one extra call without blocks, and its
  value is recorded beside the dense pass's.
- **Top-200 spectrum:** subspace iteration over 16 iterations as declared, with 200 oversampling columns (400 in
  all) so the 200th Ritz value converges. The Ritz residual of the 200th pair is recorded.
- **Chebyshev:** m steps of Saad's Algorithm 12.1, so the iterate lies in the same m-dimensional Krylov space as
  `deflated_krylov_m`'s bulk part. The interval is [δ, λ̂₁₇ + δ]; if α_trace is undefined, only the top solve is
  used.
- **Full CG timing:** wall time of the 5-step CG divided by 5. It includes CG's vector operations and the whitening
  of every tensor.
- **Matrix tensors (P1, P2):** two-dimensional shape with both sides above 1 (so `cls_token` is excluded).
