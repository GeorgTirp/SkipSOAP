"""SN curvature study runner: one unit per (model, SN condition, seed).

Each unit trains 200 consumed searched-SOAP steps and runs the A-C diagnostics after steps
10, 50 and 200 (excluded from training time). Usage (from residual-curvature-lab/):
  run.py create | work --run DIR --unit NAME | pending --run DIR | finalize --run DIR
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

import common
from common import read, write_compact
import diagnostics
import sn
from residual_curvature import checkpoint
from residual_curvature.config import load_config
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry
from residual_curvature.provenance import checksum, write_json
from run_exact_training_followup import evaluate
from run_mini_vit_allocation import propose as mini_propose

CONFIGS = {'vit': 'configs/mini_vit_allocation.yaml', 'mlp': 'runs/20261002T121635-0dac3337/resolved.yaml'}
BOUNDS = [None, .95, 2., 6.]
SEEDS = [0, 1]
STEPS, SNAPSHOTS, VALIDATION = 200, [10, 50, 200], [0, 10, 25, 50, 100, 150, 200]
VIT_GATE = common.GAP_AUDIT_RUNS/'20261007T125950-63edd820'/'b2_w16__s0'
MLP_GATE = {10: 0.9028134971683025, 25: 0.4536378557680572, 50: 0.4911033240812833}
PANEL_SOURCE = Path('runs/20261005T143029-9b582ed5/searched_soap-0010/diagnostics.json')


def condition(bound):
    return 'nosn' if bound is None else f'sn{bound:g}'


def units():
    return {f'{model}__{condition(bound)}__s{seed}': {'model': model, 'bound': bound, 'seed': seed}
            for model in ['vit', 'mlp'] for bound in BOUNDS for seed in SEEDS}


def unit_config(spec):
    c = load_config(CONFIGS[spec['model']])
    c['output'] = str(common.RUNS)
    c['seed'] = spec['seed']
    c['training'].update(steps=STEPS, snapshots=SNAPSHOTS, method='soap', policy='native')
    return c


def work(directory, name):
    common.verify_archive(directory)
    spec = read(directory/'protocol.json')['units'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    c = unit_config(spec)
    write_json(unit/'resolved_arm.json', {'config': c, 'spec': spec})
    x, y, manifest = dataset(c)
    reference = read(Path('runs/20261005T135920-d7ce29b2/split_manifest.json'))
    if manifest['training'] != reference['training'] or manifest['validation'] != reference['validation'] or manifest['analysis'] != reference['analysis']:
        raise RuntimeError('Split differs from the main lab split')
    panel_ids = read(PANEL_SOURCE)['panel_ids']
    model = sn.make_sn_model(c, spec['bound'])
    registry = Registry(model)
    order = DataOrder(manifest['training'], common.data_order_seed(spec['seed']))
    state, seconds, accepted, history, gate = None, 0., 0, [], {}
    validation = [dict(step=0, training_seconds=0., **evaluate(model, x, y, manifest['validation']))]
    try:
        for step in range(1, STEPS+1):
            ids = order.take(512)
            flat = registry.flatten(dict(model.named_parameters())).detach()
            started = time.perf_counter()
            delta, state, contexts, row = mini_propose('searched_soap', model, registry, flat, state, x[ids], y[ids], c, step)
            if not torch.isfinite(delta).all():
                raise FloatingPointError('Nonfinite update')
            commit(model, registry, flat+delta)
            elapsed = time.perf_counter()-started
            seconds += elapsed
            accepted += int(float(delta.norm()) > 0)
            search = row['search'] or {}
            history.append({'step': step, 'loss_before': row['loss_before'], 'loss_after': row['loss_after'], 'norm': row['norm'],
                            'step_seconds': elapsed, 'batch_sha256': common.ids_sha256(ids),
                            'search': {k: search.get(k) for k in ['selected_alpha', 'evaluations', 'bracketed', 'rejected']}})
            if step in VALIDATION:
                validation.append(dict(step=step, training_seconds=seconds, **evaluate(model, x, y, manifest['validation'])))
                write_json(unit/'validation.json', validation)
            if spec['bound'] is None and spec['seed'] == 0:
                if spec['model'] == 'vit' and step in [10, 50]:
                    saved = torch.load(VIT_GATE/f'state-{step:04d}'/'arrays.pt', map_location='cpu', weights_only=False)['parameters']
                    gate[f'parameters_step{step}'] = float((registry.flatten(dict(model.named_parameters())).detach()-saved).abs().max())
                if spec['model'] == 'mlp' and step in MLP_GATE:
                    gate[f'validation_step{step}'] = abs(validation[-1]['validation_loss']-MLP_GATE[step])
                if gate and max(gate.values()) > 1e-6:
                    raise RuntimeError(f'Reproduction gate failed by more than 1e-6: {gate}')
            if step in SNAPSHOTS:
                checkpoint.save(unit/f'checkpoint-{step:04d}.pt', model, state, contexts, step, accepted, order, c)
                flat = registry.flatten(dict(model.named_parameters())).detach()
                out, arrays = diagnostics.diagnose(model, c, spec['bound'], registry, flat, state, x, y, manifest, panel_ids, step)
                snapshot = unit/f'state-{step:04d}'
                snapshot.mkdir()
                torch.save(arrays, snapshot/'arrays.pt')
                write_json(snapshot/'diagnostics.json', out)
                write_json(snapshot/'artifacts.json', {'arrays.pt': checksum(snapshot/'arrays.pt')})
                print(f"{name} step {step}: val={validation[-1]['validation_loss']:.4f} headroom={out['estimation']['headroom']:.5f} "
                      f"kron_err={out['raw']['nearest_kronecker_error']:.3f} diag {out['seconds']:.0f}s", flush=True)
    except Exception as error:
        write_json(unit/'failure.json', {'unit': name, 'error': repr(error), 'traceback': traceback.format_exc()})
        write_json(directory/'failure.json', {'unit': name, 'error': repr(error)})
        raise
    write_compact(unit/'training.json', history)
    write_json(unit/'validation.json', validation)
    write_json(unit/'runtime.json', {'training_seconds': seconds, 'process_peak_rss_bytes': common.peak_rss_bytes()})
    if gate:
        write_json(unit/'reproduction_gate.json', {'differences': gate, 'within_1e-9': max(gate.values()) <= 1e-9})
    common.finish_unit(unit, {'final_validation_loss': validation[-1]['validation_loss'], 'gate': gate or None})


def create():
    c = unit_config({'model': 'vit', 'seed': 0})
    protocol = {'study': 'sn_curvature_20261008', 'document': str(common.PROTOCOL), 'units': units(), 'steps': STEPS,
                'snapshots': SNAPSHOTS, 'validation_steps': VALIDATION, 'configs': CONFIGS, 'bounds': BOUNDS, 'seeds': SEEDS,
                'sn': 'W_eff = W min(1, c/sigma(W)), exact sigma; residual-branch linear weights only',
                'optimizer': 'searched SOAP, default settings', 'diagnostics_budget_mib': diagnostics.BUDGET_MIB}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'queue_runner.sh')]))


def finalize(directory):
    names = list(read(directory/'protocol.json')['units'])
    missing = [n for n in names if not common.unit_complete(directory/n)]
    if missing or (directory/'failure.json').exists():
        raise RuntimeError(f'Incomplete or failed units: {missing}')
    write_json(directory/'units_summary.json', {n: read(directory/n/'complete.json')['summary'] for n in names})
    write_json(directory/'complete.json', {'completed': True, 'units': len(names), 'states': 3*len(names)})
    print(f'mirrored to {common.mirror(directory)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'finalize'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--unit')
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        create()
    elif args.command == 'work':
        work(args.run, args.unit)
    elif args.command == 'pending':
        names = read(args.run/'protocol.json')['units']
        pending = [n for n in names if not common.unit_complete(args.run/n)]
        print('\n'.join(sorted(pending, key=lambda n: (names[n]['model'] != 'vit', n))))
    else:
        finalize(args.run)


if __name__ == '__main__':
    sys.exit(main())
