"""Driver (PROTOCOL.md, sections 2-5): replay each cifar_ggn unit bitwise (gate G1) and measure GGN spectra and
anchor-pair metrics at the declared states. One GPU process.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  run.py [--only ARM SEED]
Outputs: $SR_DATA/spectra/<arm>_s<seed>.json and copies in studies/spectral_rotation_20261020/results/spectra/.
A trajectory whose output exists is skipped.
"""
import argparse
import json
import math
import shutil
import sys
import time

import common

import torch  # noqa: E402

import rotation  # noqa: E402
import states  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402

ARMS, SEEDS = ['soap', 'topfree'], [1, 2, 3]
STATES = [10, 20, 50, 60, 100, 110, 150, 250, 260, 300, 400, 410, 450, 500]
ANCHORS = [10, 50, 100, 250, 400]
GATE_STEPS = [50, 100, 250, 500]
BATCH_SEED = 20261020
EIGEN_SEED = 4242


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def output(arm, seed):
    return common.DATA/'spectra'/f'{arm}_s{seed}.json'


def save(arm, seed, value):
    path = output(arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), allow_nan=False)+'\n')
    mirror = common.STUDY/'results'/'spectra'
    mirror.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, mirror/path.name)


def reference_batches(data):
    """Batches A and B: the first two draws of a dedicated order over the training positions, unaugmented."""
    order = DataOrder(list(range(len(data.manifest['training']))), BATCH_SEED)
    first, second = order.take(states.BATCH), order.take(states.BATCH)
    return data.batch(first), data.batch(second), first, second


def gate(replay):
    unit = states.EVAL/f"{replay.arm_name}__lr{states.LRS[replay.arm_name]:g}__s{replay.seed}"
    stored = torch.load(unit/'checkpoints'/f'step-{replay.t:04d}.pt', weights_only=False)['parameters']
    return bool(torch.equal(replay.flat().cpu(), stored))


def trajectory(arm, seed, device, data, batches):
    (xa, _), (xb, _) = batches[0], batches[1]
    replay = states.Replay(arm, seed, device, data)
    model, registry = replay.arm.model, replay.arm.registry
    out = {'arm': arm, 'seed': seed, 'lr': states.LRS[arm], 'states': {}, 'gate': {}, 'seconds': {}}
    anchors = []
    started = time.perf_counter()
    while replay.t < max(STATES):
        x, y = replay.next_batch()
        replay.step(x, y)
        t = replay.t
        if t in GATE_STEPS:
            out['gate'][str(t)] = gate(replay)
            if not out['gate'][str(t)]:
                raise RuntimeError(f'Gate G1 failed: {arm} seed {seed} step {t}')
        if t not in STATES:
            continue
        began = time.perf_counter()
        flat = replay.flat().detach()
        generator = torch.Generator().manual_seed(EIGEN_SEED+t)
        records, kept = rotation.state_pass(model, registry, flat, xa, generator, anchors=anchors, keep=t in ANCHORS)
        entry = {'step': t, 'lr_factor': states.GGN_TRAIN.lr_factor(t), 'A': records}
        if t in ANCHORS:
            floor, _ = rotation.state_pass(model, registry, flat, xb, generator, floor_of=kept)
            entry['B'] = floor
            anchors.append((t, kept))
        out['states'][str(t)] = entry
        common.synchronize(device)
        out['seconds'][str(t)] = time.perf_counter()-began
        print(f'{arm} s{seed} t{t}: {out["seconds"][str(t)]:.0f} s', flush=True)
    out['total_seconds'] = time.perf_counter()-started
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs=2, metavar=('ARM', 'SEED'))
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    device = torch.device('cuda')
    data = states.GGN_TRAIN.Data(device)
    batch_a, batch_b, first, second = reference_batches(data)
    plan = [(args.only[0], int(args.only[1]))] if args.only else [(a, s) for a in ARMS for s in SEEDS]
    for arm, seed in plan:
        if output(arm, seed).exists():
            print(f'{arm} s{seed}: exists, skipped', flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        out = trajectory(arm, seed, device, data, (batch_a, batch_b))
        out['batches'] = {'A_sha256': common.ids_sha256(first), 'B_sha256': common.ids_sha256(second)}
        out['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        out['source_checksums'] = {name: common.checksum(name) for name in [str(common.PROTOCOL), *common.loaded_sources()]}
        save(arm, seed, out)
        print(f'{arm} s{seed}: done in {out["total_seconds"]:.0f} s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
