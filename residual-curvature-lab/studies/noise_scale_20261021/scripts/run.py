"""Driver (PROTOCOL.md, sections 2-5): replay each cifar_ggn unit bitwise (gate G1) and measure the noise-scale
quantities and fractional-Newton steps at the declared states (gate G2 inside noise.analyse). One GPU process.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  run.py [--only ARM SEED] [--states 10 ...] [--output DIR]
Outputs: $NS_DATA/noise/<arm>_s<seed>.json, mirrored to studies/noise_scale_20261021/results/noise/. --output
redirects a pre-run smoke elsewhere (then nothing is mirrored). A trajectory whose output exists is skipped.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import common

import torch  # noqa: E402

import noise  # noqa: E402
import states  # noqa: E402

ARMS, SEEDS = ['topfree', 'soap'], [1, 2, 3]
STATES = [10, 25, 50, 100, 200, 400]
G2_STATES = [10, 400]
GATE_STEPS = [50, 100, 250]


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def gate(replay):
    unit = states.EVAL/f"{replay.arm_name}__lr{states.LRS[replay.arm_name]:g}__s{replay.seed}"
    stored = torch.load(unit/'checkpoints'/f'step-{replay.t:04d}.pt', weights_only=False)['parameters']
    return bool(torch.equal(replay.flat().cpu(), stored))


def trajectory(arm, seed, device, data, chosen_states):
    replay = states.Replay(arm, seed, device, data)
    partition_positions = noise.partition(data)
    reference_x = noise.reference_inputs(data)
    out = {'arm': arm, 'seed': seed, 'lr': states.LRS[arm], 'gate': {}, 'states': {}, 'seconds': {},
           'partition': {'batch': noise.PARTITION_BATCH, 'count': noise.PARTITION_COUNT,
                         'sha256': common.ids_sha256([p for b in partition_positions for p in b])}}
    started = time.perf_counter()
    while replay.t < max(chosen_states):
        x, y = replay.next_batch()
        replay.step(x, y)
        t = replay.t
        if t in GATE_STEPS:
            out['gate'][str(t)] = gate(replay)
            if not out['gate'][str(t)]:
                raise RuntimeError(f'Gate G1 failed: {arm} seed {seed} step {t}')
        if t not in chosen_states:
            continue
        began = time.perf_counter()
        result, steps, flat = noise.analyse(replay, data, partition_positions, reference_x, t in G2_STATES)
        result['score'] = noise.score_steps(replay, data, flat, steps, result.pop('scale_terms'))
        result['lr_factor_next'] = states.GGN_TRAIN.lr_factor(t+1)
        out['states'][str(t)] = result
        common.synchronize(device)
        out['seconds'][str(t)] = time.perf_counter()-began
        print(f'{arm} s{seed} t{t}: {out["seconds"][str(t)]:.0f} s, G2 {result["g2"] or "-"}', flush=True)
    out['total_seconds'] = time.perf_counter()-started
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs=2, metavar=('ARM', 'SEED'))
    parser.add_argument('--states', nargs='+', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    device = torch.device('cuda')
    data = states.GGN_TRAIN.Data(device)
    chosen_states = args.states or STATES
    target = args.output or common.DATA/'noise'
    plan = [(args.only[0], int(args.only[1]))] if args.only else [(a, s) for a in ARMS for s in SEEDS]
    for arm, seed in plan:
        path = target/f'{arm}_s{seed}.json'
        if path.exists():
            print(f'{arm} s{seed}: exists, skipped', flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        out = trajectory(arm, seed, device, data, chosen_states)
        out['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        out['source_checksums'] = {name: common.checksum(name) for name in [str(common.PROTOCOL), *common.loaded_sources()]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(clean(out), allow_nan=False)+'\n')
        if args.output is None:
            mirror = common.STUDY/'results'/'noise'
            mirror.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, mirror/path.name)
        print(f'{arm} s{seed}: done in {out["total_seconds"]:.0f} s, peak {out["peak_cuda_bytes"]/2**30:.1f} GiB', flush=True)


if __name__ == '__main__':
    sys.exit(main())
