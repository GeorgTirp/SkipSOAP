"""Driver (PROTOCOL.md, section 2): Parts P and K at ladder_mechanism_20261013's top-free states. One GPU process.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  run.py [--only SEED STEP]
Outputs: $BK_DATA/bulk/topfree_s<seed>_t<step>.json and copies in studies/bulk_krylov_20261014/results/bulk/.
A state whose output exists is skipped, so an interrupted run resumes at the next state.
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import common

import torch  # noqa: E402

import bulk  # noqa: E402
import states  # noqa: E402

SEEDS = [1, 2, 3]
STEPS = [10, 25, 50, 100]
PROBE_SEED = 9300  # ladder_mechanism's: the same probe draws and sketches as its ladder
LADDER_STATES = Path.home()/'skipsoap-study-data'/'ladder_mechanism_20261013'/'states'


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def output(seed, step):
    return common.DATA/'bulk'/f'topfree_s{seed}_t{step:04d}.json'


def save(seed, step, value):
    path = output(seed, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), allow_nan=False)+'\n')
    mirror = common.STUDY/'results'/'bulk'
    mirror.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, mirror/path.name)


def restored(seed, step, device, data):
    replay = states.Replay('topfree', seed, device, data)
    replay.restore(torch.load(LADDER_STATES/f'topfree_s{seed}_t{step:04d}.pt', weights_only=False))
    return replay


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs=2, type=int, metavar=('SEED', 'STEP'))
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    device = torch.device('cuda')
    data = states.GGN_TRAIN.Data(device)
    plan = [tuple(args.only)] if args.only else [(seed, step) for seed in SEEDS for step in STEPS]
    for seed, step in plan:
        if output(seed, step).exists():
            print(f'bulk s{seed} t{step}: exists, skipped', flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        out = bulk.run_state(restored(seed, step, device, data), PROBE_SEED+1000*seed+step)
        out['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        out['source_checksums'] = {name: common.checksum(name) for name in [str(common.PROTOCOL), *common.loaded_sources()]}
        save(seed, step, out)
        print(f'bulk s{seed} t{step}: {out["seconds"]:.0f} s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
