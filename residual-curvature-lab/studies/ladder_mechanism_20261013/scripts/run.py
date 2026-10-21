"""Driver (PROTOCOL.md): replays with gate S1, then Part L (ladder) and Part N (control). One GPU process per command.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  run.py replay | ladder | control
Outputs: $LM_DATA/<part>/*.json and copies in studies/ladder_mechanism_20261013/results/<part>/.
"""
import argparse
import json
import math
import shutil
import sys

import common

import torch  # noqa: E402

import control  # noqa: E402
import ladder  # noqa: E402
import states  # noqa: E402

SEEDS = [1, 2, 3]
SNAPSHOTS = {'soap': [9, 19, 20, 49, 59, 60, 99, 109, 110],
             'topfree': [9, 10, 19, 20, 25, 49, 50, 59, 60, 99, 100, 109, 110]}
GATE_STEPS = [50, 100]
LADDER_STEPS = [10, 25, 50, 100]
CONTROL_TIMES = [20, 60, 110]
PROBE_SEED = 9300


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save(part, name, value):
    target = common.DATA/part
    target.mkdir(parents=True, exist_ok=True)
    path = target/f'{name}.json'
    path.write_text(json.dumps(clean(value), allow_nan=False)+'\n')
    mirror = common.STUDY/'results'/part
    mirror.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, mirror/path.name)


def snapshot_path(arm, seed, step):
    return common.DATA/'states'/f'{arm}_s{seed}_t{step:04d}.pt'


def load(arm, seed, step):
    return torch.load(snapshot_path(arm, seed, step), weights_only=False)


def restored(arm, seed, step, device, data):
    replay = states.Replay(arm, seed, device, data)
    replay.restore(load(arm, seed, step))
    return replay


def replay_command(device, data):
    (common.DATA/'states').mkdir(parents=True, exist_ok=True)
    gate = {}
    for arm, steps in SNAPSHOTS.items():
        for seed in SEEDS:
            replay = states.Replay(arm, seed, device, data)
            while replay.t < max(steps):
                x, y = replay.next_batch()
                replay.step(x, y)
                if replay.t in steps:
                    torch.save(replay.snapshot(), snapshot_path(arm, seed, replay.t))
                if replay.t in GATE_STEPS:
                    unit = states.EVAL/f"{arm}__lr{states.LRS[arm]:g}__s{seed}"
                    stored = torch.load(unit/'checkpoints'/f'step-{replay.t:04d}.pt', weights_only=False)['parameters']
                    gate[f'{arm}_s{seed}_t{replay.t}'] = bool(torch.equal(replay.flat().cpu(), stored))
            print(f'{arm} seed {seed}: replayed to {replay.t}', flush=True)
    passed = all(gate.values())
    save('replay', 'gate_s1', {'passed': passed, 'checks': gate})
    if not passed:
        raise RuntimeError(f'Gate S1 failed: {gate}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['replay', 'ladder', 'control'])
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    device = torch.device('cuda')
    data = states.GGN_TRAIN.Data(device)
    if args.command == 'replay':
        replay_command(device, data)
    elif args.command == 'ladder':
        for seed in SEEDS:
            for step in LADDER_STEPS:
                out = ladder.run_state(restored('topfree', seed, step, device, data), PROBE_SEED+1000*seed+step)
                save('ladder', f'topfree_s{seed}_t{step:04d}', out)
                print(f'ladder s{seed} t{step}: {out["seconds"]:.0f} s', flush=True)
    else:
        for arm in SNAPSHOTS:
            for seed in SEEDS:
                for t in CONTROL_TIMES:
                    out = control.run_point(restored(arm, seed, t-1, device, data), load(arm, seed, t), load(arm, seed, t-11),
                                            seed_offset=100000*(arm == 'topfree')+1000*seed+t)
                    save('control', f'{arm}_s{seed}_t{t:04d}', out)
                    print(f'control {arm} s{seed} t{t}: {out["seconds"]:.0f} s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
