# SkipSOAP — notes for Claude Code sessions

This repository holds `residual-curvature-lab/`: research on curvature-corrected SOAP optimizers (PARSEC,
PARSEC-L). All results so far come from a small ViT on MNIST (width 16, depth 2), run on a laptop CPU in float64.
The work now moves to larger networks on a GPU machine. Run commands from `residual-curvature-lab/`; its scripts
use paths relative to it (`configs/…`, `runs/…`).

## Read first

- `residual-curvature-lab/ideas/parsec.md` is the living PARSEC insight bank. It holds design rules with evidence
  levels, idea status (I1–I15, N1–N20), reference numbers, next tests and a changelog. Start here.
- `residual-curvature-lab/AGENTS.md` holds the research contracts; `docs/writing_style.md` sets how results are
  written.
- `residual-curvature-lab/HANDOFF.md` and `STATUS.md` are the owner's ledger (see rule 3).
- `residual-curvature-lab/studies/<name>_<date>/` has one folder per study, with `PROTOCOL.md`, `RESULTS.md`,
  `DEVIATIONS.md`, `scripts/` and `results/`.
- `residual-curvature-lab/studies/parsec_l_20261009/` holds the PARSEC-L optimizer (`parsec_l.PARSECL`, a
  subclass of upstream SOAP) and its 22 tests.

## Where things stand (2026-10-12)

- **Best PARSEC-L so far:**
  - learning-rate calibration (κ = c·lr·λ);
  - `top_rhs='gradient'`: no momentum in the top-16 whitened GGN directions per tensor;
  - c = 64;
  - a weight EMA (d = 0.98).

  On fresh seeds it beat SOAP + EMA and SOAP + WSD decay + EMA in every seed (validation loss 0.121 vs 0.135).
  Stronger damping, up to freezing those directions, did not help.
- **Mechanism open:** the gain is not an edge-of-stability effect. A noise-only (Ornstein–Uhlenbeck) explanation
  failed its offline check, so a per-direction Kalman gain was not built.
- **Step lengths:** same-batch line search overfits the batch. Choose step lengths with losses on a second batch.
  Step sizes computed from statistics alone failed early in training.
- **Comparison rule:** every optimizer comparison gives all arms a learning-rate decay (e.g. WSD) and weight
  averaging.
- **Proposed next tests:**
  - the offline capture and valley-floor check on the saved checkpoints (`parsec.md`, section 6);
  - then top-free PARSEC-L with c = 64 on longer runs and larger models.

## Working rules

These are the owner's standing instructions from earlier sessions. Ask if a task seems to need an exception.

1. **Protocol first.** Write `PROTOCOL.md` with pre-declared readings before running anything, and record its
   SHA-256 in the study's `logs/`. Launch expensive training only after the owner has approved the protocol.
2. **Gates are fixed.**
   - Never loosen a tolerance or gate to make a run pass, never relabel a failed check, and never change a
     protocol after the fact to favour a result. Stop and report.
   - Record deviations in the study's `DEVIATIONS.md` before any rerun.
   - The default numerical tolerance is 1e-8.
3. **Isolation.**
   - Write only to the current study folder and its external data directory.
   - Read and import, but never modify `src/`, `scripts/`, `tests/`, `configs/`, `docs/`, `references/`,
     `third_party/`, `runs/`, `datasets/`, `HANDOFF.md`, `STATUS.md`, `README.md`, `AGENTS.md`, `pyproject.toml`
     or `tested-requirements.txt`.
   - `AGENTS.md` asks for ledger updates. Propose them in the study's `HANDOFF_PROPOSED_ENTRY.md` instead of
     editing `HANDOFF.md` or `STATUS.md`.
   - If you find a bug in shared code, report it; don't fix it.
4. **Owner's file.** `ideas/PARSEC_ideas.md` belongs to the owner: read only.
5. **Insight bank.** After every PARSEC-related study, update `ideas/parsec.md`: rules, idea table and changelog.
   Verify every number against the RESULTS files, cite the study, and never delete negative results.
6. **Hygiene.**
   - Set `PYTHONDONTWRITEBYTECODE=1` for every Python call, and run pytest with `-p no:cacheprovider`.
   - Take isolation manifests and run the test suites before and after each study.
   - Gate new code against earlier runs, bitwise where the computation is unchanged.
7. **Ask first.** No git write operations (add, commit, push, …) unless the owner asks. Install no packages and
   download nothing without asking. Never use the official MNIST test split.

## Moving to the GPU machine

- **Not in git** (see `residual-curvature-lab/.gitignore`): the virtual environment, `datasets/` (copy MNIST there
  by hand), run outputs and checkpoints. Five small files under `runs/` stay tracked because study scripts read
  them as gates; the split manifest is the main one.
- **External study data:** the laptop's `~/skipsoap-study-data` (3.2 GB) is not in the repository. The
  noise-model follow-up needs `topfree_confirm_20261011` from it (the seed-4 checkpoints, 160 MB).
- **Process limits:** the laptop runs used at most four single-threaded CPU processes. Ask the owner for the GPU
  machine's limits.
- **Suggested first step,** before new experiments: reproduce a CPU reference run on the GPU in float64, with the
  agreement tolerance declared in a protocol.
