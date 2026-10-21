"""Cost measurement C1 (PROTOCOL.md, section 6): seconds per step and peak GPU memory of the candidate CIFAR-10 ViTs
under AdamW, SOAP and top-free PARSEC-L, GPU float32, batch 512, 10 warm-up and 60 timed steps.

Each cell runs in its own process, so an out-of-memory cell cannot distort another cell's peak memory.
Gradients come from an ordinary backward pass, as a GPU training run would compute them.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  cost.py create | cell --run DIR --cell NAME | finalize --run DIR
"""
import argparse
import statistics
import sys
import time
import traceback
from pathlib import Path

import common
from common import read

import torch  # noqa: E402

import cifar  # noqa: E402
from gpu_parsec import PARSECL, ggn_probes  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402
from third_party.soap.soap import SOAP  # noqa: E402

BATCH, WARMUP, TIMED, LR, WEIGHT_DECAY, PROBE_SEED = 512, 10, 60, 1e-3, .05, 9000
OPTIMIZERS = {  # name: probe samples per example (None: no probes)
    'adamw': None, 'soap_upstream': None, 'soap': None, 'topfree_s1': 1, 'topfree_s4': 4}


def cells():
    return {f'{model}__{optimizer}': {'model': model, 'optimizer': optimizer, 'samples': samples}
            for model in cifar.MODELS for optimizer, samples in OPTIMIZERS.items()}


def make_optimizer(name, model):
    """AdamW; upstream SOAP; SOAP as PARSEC-L with rank 0 (the lab's `soap` arm); top-free PARSEC-L (c = 64)."""
    soap = dict(lr=LR, betas=(.95, .95), weight_decay=WEIGHT_DECAY, precondition_frequency=10)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if name == 'soap_upstream':
        return SOAP(model.parameters(), **soap)
    if name == 'soap':
        return PARSECL(model.parameters(), **soap, rank=0, norm_blend=0., signals=False)
    return PARSECL(model.parameters(), **soap, rank=16, probe_every=10, refresh='extract', norm_blend=0., signals=False,
                   calibration='lr', strength=64., top_rhs='gradient', seed=0)


def cell(directory, name):
    common.verify_archive(directory)
    spec = read(directory/'protocol.json')['cells'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    raw, labels, manifest = cifar.cifar10()
    if manifest != read(directory/'cifar_split_manifest.json'):
        raise RuntimeError('CIFAR-10 split differs from the run manifest')
    device = torch.device('cuda')
    train = torch.tensor(manifest['training'])
    x = cifar.normalize(raw[train], manifest).to(device)
    y = labels[train].to(device)
    del raw
    model = cifar.make_cifar_model(spec['model'], seed=0, device=device)
    parameters = Registry(model).size
    optimizer = make_optimizer(spec['optimizer'], model)
    order = DataOrder(list(range(len(train))), common.data_order_seed(0))
    probe_generator = torch.Generator().manual_seed(PROBE_SEED)
    torch.cuda.reset_peak_memory_stats()
    rows, outcome, step = [], {'status': 'completed'}, 0
    try:
        for step in range(1, WARMUP+TIMED+1):
            ids = order.take(BATCH)
            torch.cuda.synchronize()
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            if isinstance(optimizer, PARSECL):
                probes = (ggn_probes(model, x[ids], samples=spec['samples'], generator=probe_generator)[0]
                          if optimizer.wants_probes() else None)
                optimizer.step(probes=probes)
                del probes
            else:
                optimizer.step()
            torch.cuda.synchronize()
            rows.append({'step': step, 'seconds': time.perf_counter()-began, 'loss': float(loss),
                         'probe': isinstance(optimizer, PARSECL) and optimizer.last_info.get('probe') is not None})
    except torch.cuda.OutOfMemoryError as error:
        outcome = {'status': 'out_of_memory', 'step': step, 'error': str(error)[:1000]}
    except Exception as error:
        write_json(unit/'failure.json', {'cell': name, 'step': step, 'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    timed = [r for r in rows if r['step'] > WARMUP]
    probe, plain = [r['seconds'] for r in timed if r['probe']], [r['seconds'] for r in timed if not r['probe']]
    summary = {'cell': name, 'spec': spec, 'parameters': parameters, 'outcome': outcome,
               'median_seconds_plain_step': statistics.median(plain) if plain else None,
               'median_seconds_probe_step': statistics.median(probe) if probe else None,
               'mean_seconds_per_step': statistics.mean(r['seconds'] for r in timed) if timed else None,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'environment': common.environment()}
    write_json(unit/'steps.json', rows)
    write_json(unit/'cost.json', summary)
    common.finish_unit(unit, {'outcome': outcome})
    print(f"{name}: {outcome['status']} D={parameters} plain={summary['median_seconds_plain_step']} "
          f"probe={summary['median_seconds_probe_step']} peak={summary['peak_allocated_bytes']/2**30:.2f} GiB", flush=True)


def create():
    c = common.study_config()
    raw, labels, manifest = cifar.cifar10()
    protocol = {'study': 'gpu_port_20261012', 'phase': 'cost', 'document': str(common.PROTOCOL), 'cells': cells(),
                'batch': BATCH, 'warmup': WARMUP, 'timed': TIMED, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
                'models': cifar.MODELS, 'environment': common.environment()}
    directory = common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'run_cost.sh')])
    write_json(directory/'cifar_split_manifest.json', manifest)
    print(directory)


def finalize(directory):
    names = read(directory/'protocol.json')['cells']
    missing = [n for n in names if not common.unit_complete(directory/n)]
    if missing:
        raise RuntimeError(f'Incomplete cells: {missing}')
    write_json(directory/'cost_summary.json', [read(directory/n/'cost.json') for n in names])
    write_json(directory/'complete.json', {'completed': True, 'phase': 'cost', 'cells': len(names)})
    print(f'mirrored to {common.mirror(directory)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'cell', 'finalize'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--cell')
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        create()
    elif args.command == 'cell':
        cell(args.run, args.cell)
    else:
        finalize(args.run)


if __name__ == '__main__':
    sys.exit(main())
