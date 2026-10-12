import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .curvature.operators import allocation_estimate, guard
from .models import Registry, make_model


METHODS = ['native_soap', 'native_ray', 'frozen_minus_Pg', 'exact_diagonal', 'exact_full_output_dual',
           'soap_preconditioned_CG', 'allocation_diagonal', 'allocation_cone', 'allocation_span',
           'sketch_diagonal', 'sketch_coupled', 'sketch_one_sweep']


def estimate(c, command='snapshots'):
    if c['model']['kind'] == 'transformer':
        raise NotImplementedError('Stage 4 Transformer is deferred; configuration is a status marker, not a runnable model')
    model = make_model(c)
    registry = Registry(model)
    outputs = c['data']['batch_size'] * c['model']['classes']
    memory = allocation_estimate(registry.size, outputs, max(c['sketch']['ranks']), 8 if c['dtype'] == 'float64' else 4)
    guard(registry.size, outputs, max(c['sketch']['ranks']), c['resources'], getattr(torch, c['dtype']))
    return {'status': 'validated; no data read or training performed', 'model': c['model'],
            'parameters': registry.size, 'stored_tensors': len(registry.specs), 'outputs': outputs,
            'maximum_GGN_rank': c['data']['batch_size'] * (c['model']['classes'] - 1),
            'allocations': memory, 'resources': c['resources'],
            'exact_reference': 'output-dual; dense additionally audited below parameter cap',
            'planned_methods': [c['training']['method']] if command == 'train' else METHODS,
            'training_steps': c['training']['steps'], 'snapshot_steps': c['training']['snapshots'] if command != 'train' else [],
            'gammas': c['damping']['gammas'], 'sketch_ranks': c['sketch']['ranks'], 'repeats': c['sketch']['repeats'],
            'operations_per_snapshot': {'jacobian_reverse_rows': outputs,
                                        'probe_vjps': max(c['sketch']['ranks']) * c['sketch']['repeats'],
                                        'microbatch_audit_vjps': max(c['sketch']['ranks']) * c['sketch']['repeats'] * ((c['data']['batch_size'] + c['sketch']['microbatch'] - 1) // c['sketch']['microbatch']),
                                        'CG_max_iterations_per_gamma': c['solver']['max_iterations'],
                                        'CG_products_per_iteration': 2},
            'data': {'dataset': c['data']['dataset'], 'root': c['data']['root'], 'automatic_download': False,
                     'task': f"downsampled {c['data']['resolution']}x{c['data']['resolution']} MNIST" if c['data']['dataset'] == 'mnist' else 'network-free synthetic'},
            'runtime_estimate': 'not measured; no toy-to-MNIST extrapolation asserted'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Residual curvature laboratory')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['smoke', 'snapshots', 'train', 'estimate']:
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--dry-run', action='store_true')
        if name in ['train', 'snapshots']:
            p.add_argument('--resume')
    p = sub.add_parser('data')
    p.add_argument('--dataset', choices=['mnist'], required=True)
    p.add_argument('--download', action='store_true', required=True)
    p.add_argument('--root', default='datasets')
    p = sub.add_parser('summarize')
    p.add_argument('--run-dir', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'data':
            from torchvision.datasets import MNIST
            MNIST(args.root, train=True, download=True)
            MNIST(args.root, train=False, download=True)
            print(f'MNIST downloaded to {args.root}; no experiment launched')
            return
        if args.command == 'summarize':
            from .analysis.report import summarize
            print(summarize(args.run_dir))
            return
        c = load_config(args.config)
        plan = estimate(c, args.command)
        if args.command == 'estimate' or args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        if args.command == 'smoke' and c['data']['dataset'] != 'synthetic':
            raise ValueError('smoke is reserved for network-free synthetic data')
        from .experiments.training import run
        from .analysis.report import summarize
        directory, _, _ = run(c, analyze=args.command != 'train', resume=getattr(args, 'resume', None))
        summarize(directory)
        print(directory.resolve())
    except (ValueError, RuntimeError, MemoryError, NotImplementedError) as e:
        parser.exit(2, f'{type(e).__name__}: {e}\n')


if __name__ == '__main__':
    main()
