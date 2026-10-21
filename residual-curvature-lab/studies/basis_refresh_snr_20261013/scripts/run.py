"""Driver (PROTOCOL.md): replays with gate S1, then Parts A, B and C. One GPU process per command.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  run.py replay | ladder | refresh | snr  [--seeds 1 2 3] [--steps ...]
Outputs: $BRS_DATA/<part>/*.json (full) and studies/basis_refresh_snr_20261013/results/<part>/*.json (copies).
"""
import argparse
import json
import math
import shutil
import sys

import common

import torch  # noqa: E402

import ladder  # noqa: E402
import refresh  # noqa: E402
import snr  # noqa: E402
import states  # noqa: E402

SNAPSHOTS = [10, 11, 25, 50, 51, 100, 250, 251]
GATE_STEPS = [50, 100, 250]
LADDER_STEPS = [10, 25, 50, 100, 250]
WINDOW_STARTS = [11, 51, 251]
SNR_STEPS = [50, 100, 250]


def clean(value):
    """Non-finite floats to None (JSON without NaN or infinity)."""
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
    return path


def snapshot_path(seed, step):
    return common.DATA/'states'/f'soap_s{seed}_t{step:04d}.pt'


def restored(seed, step, device, data):
    replay = states.Replay(seed, device, data)
    replay.restore(torch.load(snapshot_path(seed, step), weights_only=False))
    return replay


def replay_command(seeds, device, data):
    """Replay each seed's SOAP unit to step 251, save the snapshots, and check gate S1."""
    (common.DATA/'states').mkdir(parents=True, exist_ok=True)
    gate = {}
    for seed in seeds:
        replay = states.Replay(seed, device, data)
        while replay.t < max(SNAPSHOTS):
            x, y = replay.next_batch()
            replay.step(x, y)
            if replay.t in SNAPSHOTS:
                torch.save(replay.snapshot(), snapshot_path(seed, replay.t))
            if replay.t in GATE_STEPS:
                stored = torch.load(states.GGN_EVAL/f'soap__lr0.01__s{seed}'/'checkpoints'/f'step-{replay.t:04d}.pt',
                                    weights_only=False)['parameters']
                gate[f's{seed}_t{replay.t}'] = bool(torch.equal(replay.flat().cpu(), stored))
        print(f'seed {seed}: replayed to {replay.t}', flush=True)
    passed = all(gate.values())
    save('replay', 'gate_s1', {'passed': passed, 'checks': gate})
    if not passed:
        raise RuntimeError(f'Gate S1 failed: {gate}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['replay', 'ladder', 'refresh', 'snr'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--steps', type=int, nargs='+')
    parser.add_argument('--posthoc', action='store_true', help='ladder with the extended transfer grid (DEVIATIONS.md D2)')
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    device = torch.device('cuda')
    data = states.SNS_TRAIN.Data(device)
    if args.command == 'replay':
        replay_command(args.seeds, device, data)
        return
    for seed in args.seeds:
        if args.command == 'ladder':
            for step in args.steps or LADDER_STEPS:
                grid = ladder.POSTHOC_MULTIPLIERS if args.posthoc else ladder.MULTIPLIERS
                out = ladder.run_state(restored(seed, step, device, data), ladder.PROBE_SEED+1000*seed+step, grid)
                save('ladder_posthoc' if args.posthoc else 'ladder', f's{seed}_t{step:04d}', out)
                print(f'ladder s{seed} t{step}: {out["seconds"]:.0f} s', flush=True)
        elif args.command == 'refresh':
            for step in args.steps or WINDOW_STARTS:
                rows = refresh.window(restored(seed, step, device, data))
                save('refresh', f's{seed}_t{step:04d}', {'seed': seed, 'start': step, 'rows': rows})
                print(f'refresh s{seed} t{step}: {sum(r["seconds"] for r in rows):.0f} s', flush=True)
        else:
            for step in args.steps or SNR_STEPS:
                out = snr.run_state(seed, step, device, data)
                save('snr', f's{seed}_t{step:04d}', out)
                print(f'snr s{seed} t{step}: {out["seconds"]:.0f} s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
