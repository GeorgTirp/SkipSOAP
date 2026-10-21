"""Driver (PROTOCOL.md, sections 2-6): replay each top-free unit with queue bookkeeping (gate G1), then Parts V and S
at the declared states. One GPU process.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  run.py [--only SEED]
Outputs: $QC_DATA/queue/topfree_s<seed>.json (mirrored to studies/queue_curvature_20261020/results/queue/) and the
per-example arrays $QC_DATA/queue/topfree_s<seed>_examples.pt. A seed whose output exists is skipped.
"""
import argparse
import copy
import json
import math
import shutil
import sys
import time

import common

import torch  # noqa: E402

import queue_analysis  # noqa: E402
import states  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.optim.soap import SOAPMetric  # noqa: E402

SEEDS = [1, 2, 3]
STATES = [50, 100, 250, 400]
GATE_STEPS = [50, 100, 250]
POOL = queue_analysis.POOL
PLACEBO_SEED, RANDOM_SEED, VISIT_SEED = 777000, 880000, 990000


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def output(seed):
    return common.DATA/'queue'/f'topfree_s{seed}.json'


def gate(replay):
    unit = states.EVAL/f"topfree__lr{states.LRS['topfree']:g}__s{replay.seed}"
    stored = torch.load(unit/'checkpoints'/f'step-{replay.t:04d}.pt', weights_only=False)['parameters']
    return bool(torch.equal(replay.flat().cpu(), stored))


def needed_steps():
    """Weights kept: t-8 ... t+7 around every state (and t itself)."""
    return {s for t in STATES for s in range(t-POOL, t+POOL)}


def trajectory(seed, device, data):
    replay = states.Replay('topfree', seed, device, data)
    model, registry = replay.arm.model, replay.arm.registry
    total = len(data.manifest['training'])
    placebo = DataOrder(list(range(total)), PLACEBO_SEED+seed)
    last_visit = torch.full((total,), -1, dtype=torch.long)
    placebo_visit = torch.full((total,), -1, dtype=torch.long)
    keep = needed_steps()
    flats, positions, snaps = {}, {}, {}
    out = {'seed': seed, 'gate': {}, 'states': {}, 'seconds': {}}
    started = time.perf_counter()
    while replay.t < max(STATES)+POOL:
        taken = replay.order.take(states.BATCH)  # the calls of Replay.next_batch, with the positions kept
        x, y = data.batch(taken, replay.augment)
        replay.step(x, y)
        t = replay.t
        positions[t] = taken
        last_visit[torch.tensor(taken)] = t
        placebo_visit[torch.tensor(placebo.take(states.BATCH))] = t
        if t in keep:
            flats[t] = replay.flat().detach().clone()
        if t in GATE_STEPS:
            out['gate'][str(t)] = gate(replay)
            if not out['gate'][str(t)]:
                raise RuntimeError(f'Gate G1 failed: seed {seed} step {t}')
        if t in STATES:
            contexts = copy.deepcopy(states.contexts(replay.arm.optimizer, registry))
            order = DataOrder(list(range(total)), RANDOM_SEED+1000*seed+t)
            snaps[t] = {'t': t, 'flat': flats[t], 'contexts': contexts, 'scale': SOAPMetric(registry, contexts).scale,
                        'eta': states.LRS['topfree']*states.GGN_TRAIN.lr_factor(t+1),
                        'ages': torch.where(last_visit >= 0, t-last_visit, last_visit),
                        'placebo': torch.where(placebo_visit >= 0, t-placebo_visit, placebo_visit),
                        'rand_positions': [order.take(states.BATCH) for _ in range(POOL)]}
        if t-POOL in STATES:
            s = t-POOL
            snap = snaps.pop(s)
            snap['flats'], snap['positions'] = flats, positions
            began = time.perf_counter()
            visit, arrays = queue_analysis.visit_phase(model, registry, data, snap['flat'], snap['ages'], snap['placebo'],
                                                       positions[s+1], positions[s],
                                                       torch.Generator().manual_seed(VISIT_SEED+1000*seed+s))
            result = queue_analysis.analyse(model, registry, data, snap, seed)
            result['visit'] = visit
            result['lr_factor'] = states.GGN_TRAIN.lr_factor(s+1)
            out['states'][str(s)] = result
            out.setdefault('examples', {})[str(s)] = arrays
            common.synchronize(device)
            out['seconds'][str(s)] = time.perf_counter()-began
            print(f'seed {seed} t{s}: {out["seconds"][str(s)]:.0f} s', flush=True)
    out['total_seconds'] = time.perf_counter()-started
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', type=int)
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    device = torch.device('cuda')
    data = states.GGN_TRAIN.Data(device)
    for seed in ([args.only] if args.only else SEEDS):
        if output(seed).exists():
            print(f'seed {seed}: exists, skipped', flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        out = trajectory(seed, device, data)
        examples = out.pop('examples')
        out['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        out['source_checksums'] = {name: common.checksum(name) for name in [str(common.PROTOCOL), *common.loaded_sources()]}
        path = output(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(examples, path.with_name(f'topfree_s{seed}_examples.pt'))
        path.write_text(json.dumps(clean(out), allow_nan=False)+'\n')
        mirror = common.STUDY/'results'/'queue'
        mirror.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, mirror/path.name)
        print(f'seed {seed}: done in {out["total_seconds"]:.0f} s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
