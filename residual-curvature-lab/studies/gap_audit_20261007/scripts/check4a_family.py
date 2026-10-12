"""Check 4a: architecture-family dataset of saved searched-SOAP states.

Nine ViTs (blocks {1,2,3} x width {8,16,24}, heads 2, expansion 2 x width, patch 7,
MNIST 28x28, batch 512). Each (architecture, seed) runs 50 consumed searched-SOAP
steps (default settings, section 3 seed rule) and snapshots after steps 10, 25, 50.
Usage: check4a_family.py create | work --run DIR --unit NAME | pending --run DIR | finalize --run DIR
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import torch

import common
from common import read, write_compact
from residual_curvature import checkpoint
from residual_curvature.curvature.layerwise import streamed_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import SOAPMetric, soap_propose
from residual_curvature.provenance import checksum, write_json
from run_exact_training_followup import evaluate
from run_large_batch_oracle import state_equal
from run_mini_vit_allocation import propose as mini_propose

PARENT = Path('runs/20261005T143029-9b582ed5')
TRAINING = Path('runs/20261005T135920-d7ce29b2')
SNAPSHOTS = [10, 25, 50]
TRAIN_ARCHS = [(b, w) for b in [1, 2] for w in [8, 16]]
SPLITS = {'held_out_width': [(1, 24), (2, 24)], 'held_out_depth': [(3, 8), (3, 16)], 'held_out_both': [(3, 24)]}


def split_of(blocks, width, seed):
    if (blocks, width) in TRAIN_ARCHS:
        return 'validation' if seed == 2 else 'train'
    return next(name for name, archs in SPLITS.items() if (blocks, width) in archs)


def units():
    result = {}
    for blocks in [1, 2, 3]:
        for width in [8, 16, 24]:
            seeds = [0, 1, 2] if (blocks, width) in TRAIN_ARCHS else [0, 1]
            for seed in seeds:
                result[f'b{blocks}_w{width}__s{seed}'] = {'blocks': blocks, 'width': width, 'seed': seed,
                                                          'split': split_of(blocks, width, seed)}
    return result


def arch_config(spec):
    return common.study_config(seed=spec['seed'], model={'width': spec['width'], 'expansion': 2*spec['width'],
                                                         'blocks': spec['blocks']})


def snapshot(model, registry, state, x, y, construction, panel_ids, c, step, directory):
    """Saved state as in run_jacobian_information.audit, with exact streamed tensor GGN blocks."""
    directory.mkdir()
    started = time.perf_counter()
    before, optimizer, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(state), torch.get_rng_state().clone()
    flat = registry.flatten(dict(model.named_parameters())).detach()
    xb, yb = x[construction], y[construction]
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    g = op.gradient
    proposal, _, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), state, c['soap'])
    native = registry.flatten(proposal)
    metric = SOAPMetric(registry, contexts)
    momentum = -metric.map(native, inverse=True)
    momentum *= g.norm()/momentum.norm()
    recovered = -metric.map(momentum)
    momentum_check = relative_error(recovered/recovered.norm(), native/native.norm())
    blocks, curvature = streamed_blocks(model, registry, flat, xb, 8, c['resources']['budget_mib'])
    gen = torch.Generator().manual_seed(c['analysis_seed']+step)
    actions = {}
    for spec, block in zip(registry.specs, blocks):
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=flat.dtype)
        actions[spec.name] = relative_error(block@v[spec.slice], op.action(v)[spec.slice])
    tolerance = c['solver']['tolerance64']
    if momentum_check > tolerance or max(actions.values()) > tolerance:
        raise RuntimeError(f'Snapshot checks failed: momentum {momentum_check}, action {max(actions.values())}')
    if not (state_equal(before, model.state_dict()) and state_equal(optimizer, state) and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('Snapshot mutated training state')
    arrays = {'parameters': flat, 'gradient': g, 'momentum_rhs': momentum, 'contexts': contexts, 'optimizer': optimizer,
              'raw_soap': native, 'ggn_blocks': blocks}
    torch.save(arrays, directory/'arrays.pt')
    info = {'step': step, 'construction_ids': construction, 'panel_ids': panel_ids, 'curvature': curvature,
            'exact_action_checks': actions, 'momentum_ray_error': momentum_check, 'mu_rule': 'max(.01 trace(C)/D, 1e-8)',
            'trace': curvature['trace'], 'parameters': registry.size, 'registry': registry.json(),
            'nonmutation_passed': True, 'seconds': time.perf_counter()-started}
    write_json(directory/'diagnostics.json', info)
    write_json(directory/'artifacts.json', {'arrays.pt': checksum(directory/'arrays.pt')})
    return info


def work(directory, name):
    common.verify_archive(directory)
    spec = read(directory/'protocol.json')['units'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    c = arch_config(spec)
    c['training'].update(method='searched_soap', policy='broad_direction_search')
    write_json(unit/'resolved_arm.json', {'config': c, 'spec': spec})
    x, y, manifest = dataset(c)
    if manifest != read(TRAINING/'split_manifest.json'):
        raise RuntimeError('Split manifest differs from the main lab split')
    parent = read(PARENT/'searched_soap-0010'/'diagnostics.json')
    construction, panel_ids = parent['construction_ids'], parent['panel_ids']
    assert construction == manifest['analysis'][:512]
    assert panel_ids == [manifest['analysis'][2560+512*i:2560+512*(i+1)] for i in range(4)]
    model = make_model(c)
    registry = Registry(model)
    order = DataOrder(manifest['training'], common.data_order_seed(spec['seed']))
    state, seconds, history, gate, accepted = None, 0., [], {}, 0
    validation = [dict(step=0, **evaluate(model, x, y, manifest['validation']))]
    try:
        for step in range(1, 51):
            ids = order.take(512)
            flat = registry.flatten(dict(model.named_parameters())).detach()
            started = time.perf_counter()
            delta, state, contexts, row = mini_propose('searched_soap', model, registry, flat, state, x[ids], y[ids], c, step)
            commit(model, registry, flat+delta)
            elapsed = time.perf_counter()-started
            seconds += elapsed
            accepted += int(float(delta.norm()) > 0)
            search = row['search'] or {}
            history.append({'step': step, 'loss_before': row['loss_before'], 'loss_after': row['loss_after'],
                            'norm': row['norm'], 'step_seconds': elapsed, 'batch_sha256': common.ids_sha256(ids),
                            'search': {k: search.get(k) for k in ['selected_alpha', 'evaluations', 'bracketed', 'rejected']}})
            if step in SNAPSHOTS:
                validation.append(dict(step=step, training_seconds=seconds, **evaluate(model, x, y, manifest['validation'])))
                checkpoint.save(unit/f'checkpoint-{step:04d}.pt', model, state, contexts, step, accepted, order, c)
                info = snapshot(model, registry, state, x, y, construction, panel_ids, c, step, unit/f'state-{step:04d}')
                print(f"{name} step {step}: val={validation[-1]['validation_loss']:.6f} snapshot {info['seconds']:.1f}s", flush=True)
                if (spec['blocks'], spec['width'], spec['seed']) == (2, 16, 0):
                    reference = torch.load(PARENT/f'searched_soap-{step:04d}'/'arrays.pt', map_location='cpu',
                                           weights_only=False)['parameters']
                    current = registry.flatten(dict(model.named_parameters())).detach()
                    gate[step] = float((current-reference).abs().max())
                    if gate[step] > 1e-12:
                        raise RuntimeError(f'4a reproduction gate failed at step {step}: {gate[step]}')
    except Exception as error:
        write_json(unit/'failure.json', {'unit': name, 'error': repr(error)})
        write_json(directory/'failure.json', {'unit': name, 'error': repr(error)})
        raise
    write_compact(unit/'training.json', history)
    write_json(unit/'validation.json', validation)
    write_json(unit/'runtime.json', {'training_seconds': seconds, 'process_peak_rss_bytes': common.peak_rss_bytes()})
    if gate:
        write_json(unit/'reproduction_gate.json', {'max_abs_parameter_difference': gate, 'tolerance': 1e-12,
                                                    'reference': str(PARENT/'searched_soap-*/arrays.pt'), 'passed': True})
    common.finish_unit(unit, {'final_validation_loss': validation[-1]['validation_loss'], 'parameters': registry.size,
                              'split': spec['split'], 'reproduction_gate': gate or None})


def create():
    c = common.study_config()
    protocol = {'study': 'gap_audit_20261007', 'check': '4a', 'document': str(common.PROTOCOL), 'units': units(),
                'snapshots': SNAPSHOTS, 'optimizer': 'searched SOAP, default settings', 'heads': 2, 'patch': 7,
                'expansion': '2 x width', 'construction': 'first 512 analysis IDs', 'panels': 'analysis[2560+512i:2560+512(i+1)], i<4',
                'splits': {'train': 'width {8,16} x blocks {1,2}, seeds 0,1', 'validation': 'same architectures, seed 2', **{
                    k: [f'b{b}_w{w}' for b, w in v] for k, v in SPLITS.items()}}}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'run_pending.sh')]))


def finalize(directory):
    names = list(read(directory/'protocol.json')['units'])
    missing = [n for n in names if not common.unit_complete(directory/n)]
    if missing:
        raise RuntimeError(f'Incomplete units: {missing}')
    if (directory/'failure.json').exists():
        raise RuntimeError('Run has a failure record')
    summary = {n: read(directory/n/'complete.json')['summary'] for n in names}
    write_json(directory/'units_summary.json', summary)
    gate = summary['b2_w16__s0']['reproduction_gate']
    write_json(directory/'complete.json', {'completed': True, 'units': len(names), 'states': 3*len(names),
                                          'reproduction_gate': gate, 'reproduction_gate_passed': max(gate.values()) <= 1e-12})
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
        # Largest architectures first, for better packing across workers.
        pending = [n for n in names if not common.unit_complete(args.run/n)]
        print('\n'.join(sorted(pending, key=lambda n: -names[n]['blocks']*names[n]['width'])))
    else:
        finalize(args.run)


if __name__ == '__main__':
    sys.exit(main())
