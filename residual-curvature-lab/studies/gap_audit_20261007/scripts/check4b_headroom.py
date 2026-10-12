"""Check 4b: exact-GGN local headroom and family capacity across the architecture family.

Every 4a state is scored with the SOAP current-gradient map, exact tensor GGN,
soap_basis_diag, the best cheap family from 2c, pcg_full_2 and pcg_full_5, on the
current gradient and the effective momentum, with the 2c search, panel and mu rules.
Usage: check4b_headroom.py create --family DIR --check2c DIR | work --run DIR --unit NAME
       | pending --run DIR | finalize --run DIR
"""
import argparse
import copy
import statistics
import sys
import time
from pathlib import Path

import torch

import common
from common import read, write_compact
import families as fam
from arms import empirical_fisher_blocks
from residual_curvature.curvature.layerwise import solve_blocks
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import capacity
from residual_curvature.optim.soap import SOAPMetric
from residual_curvature.provenance import checksum, write_json
from run_large_batch_oracle import measurement, state_equal
from run_mini_vit_allocation import search_candidate

TRAINING = Path('runs/20261005T135920-d7ce29b2')
STEPS = [10, 25, 50]


def candidate_names(best):
    return ['identity', 'exact', 'soap_basis_diag', best, 'pcg_full_2', 'pcg_full_5']


def score_state(source, unit_state, best, x, y, tolerance):
    """One saved 4a state; returns candidate rows, operator rows and oracle parameters."""
    for name, digest in read(source/'artifacts.json').items():
        if checksum(source/name) != digest:
            raise RuntimeError(f'4a artifact checksum failed: {source/name}')
    arrays = torch.load(source/'arrays.pt', map_location='cpu', weights_only=False)
    info = read(source/'diagnostics.json')
    config = read(source.parent/'resolved_arm.json')['config']
    model = make_model(config)
    registry = Registry(model)
    flat, g, momentum, contexts = [arrays[k] for k in ['parameters', 'gradient', 'momentum_rhs', 'contexts']]
    ggn, native = arrays['ggn_blocks'], arrays['raw_soap']
    commit(model, registry, flat)
    model_before, contexts_before, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(contexts), torch.get_rng_state().clone()
    construction, panel_ids = info['construction_ids'], info['panel_ids']
    fn = functional_logits(model, registry, x[construction])
    objective = lambda w: loss(fn(w), y[construction])
    op = GGN(fn, flat, y[construction])
    if relative_error(op.gradient, g) > tolerance:
        raise RuntimeError('Stored gradient does not match the rebuilt objective')
    metric = SOAPMetric(registry, contexts)
    mu = max(.01*sum(float(b.trace()) for b in ggn)/registry.size, config['damping']['floor'])
    ef = None
    if best.endswith('_ef'):
        ef, mean_gradient, _ = empirical_fisher_blocks(model, registry, flat, x[construction], y[construction])
        if relative_error(mean_gradient, g) > tolerance:
            raise RuntimeError('Per-example gradient mean check failed')
    teacher = fam.DampedBlocks(ggn, registry, metric, mu, tolerance)
    gen = torch.Generator().manual_seed(270100+info['step'])
    zs = torch.randn(registry.size, 16, dtype=flat.dtype, generator=gen)
    factor_rhs = torch.empty_like(zs)
    for spec in registry.specs:
        factor_rhs[spec.slice] = capacity.factor(zs[spec.slice].reshape(spec.shape+(16,)), contexts[spec.name],
                                                 transpose=True, inverse=True).reshape(spec.count, 16)
    groups = {'gradient': g[:, None], 'momentum': momentum[:, None], 'factor': factor_rhs}
    exact_values = {k: teacher.solve(v) for k, v in groups.items()}
    panels = [lambda w, f=functional_logits(model, registry, x[ids]), t=y[ids]: loss(f(w), t) for ids in panel_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    rows, operators, oracle = [], [], {}
    for name in candidate_names(best):
        started = time.perf_counter()
        if name.startswith('pcg_'):
            iterations = int(name.split('_')[2])
            action = fam.pcg_operator('full', op, ggn, registry, metric, mu)
            proposals = {k: torch.stack([fam.pcg(action, metric.map, v[:, j], iterations)[0] for j in range(v.shape[1])], 1)
                         for k, v in groups.items()}
            candidates = {'gradient': -proposals['gradient'][:, 0], 'momentum': -proposals['momentum'][:, 0]}
            extra = {'matvecs_per_rhs': iterations, 'storage_numbers': 0}
        else:
            blocks, parameters, extra = fam.family_blocks(name, registry, ggn, ef, contexts)
            if name == best:
                oracle = parameters
            damped = fam.DampedBlocks(blocks, registry, metric, mu, tolerance)
            proposals = {k: damped.solve(v) for k, v in groups.items()}
            candidates = {}
            for rhs_name, rhs in [('gradient', g), ('momentum', momentum)]:
                values, _ = solve_blocks(blocks, registry, rhs, metric, [mu], tolerance)
                if relative_error(values[mu], -proposals[rhs_name][:, 0]) > 1e-12:
                    raise RuntimeError(f'Cached and shared solves disagree for {name}')
                candidates[rhs_name] = values[mu]
            extra['matvecs_per_rhs'] = 0
        errors = {k: capacity.energy_errors(groups[k], proposals[k], exact_values[k], teacher.systems, registry) for k in groups}
        operators.append({'name': name, 'errors': errors, 'seconds': time.perf_counter()-started,
                          'storage_numbers': extra['storage_numbers'], 'storage_model_vectors': extra['storage_numbers']/registry.size,
                          'matvecs_per_rhs': extra['matvecs_per_rhs'], 'tensors': extra.get('tensors')})
        for rhs_name in ['gradient', 'momentum']:
            selected, search = search_candidate(flat, candidates[rhs_name], native, objective)
            values = measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry)
            values['block_teacher_quadratic_decrease'] = -float(g@selected)-.5*float(selected@teacher.action(selected))
            values.pop('tensor_update_shares')
            rows.append({'name': name, 'rhs': rhs_name, 'mu': mu, 'selected': values,
                         'search': {k: search[k] for k in ['selected_alpha', 'evaluations', 'bracketed', 'rejected', 'boundary']}})
    if not (state_equal(model_before, model.state_dict()) and state_equal(contexts_before, contexts) and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('State mutated during 4b scoring')
    return rows, operators, oracle, {'mu': mu, 'parameters': registry.size, 'metric_scale': metric.scale}


def work(directory, name):
    common.verify_archive(directory)
    protocol = read(directory/'protocol.json')
    spec = protocol['units'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    family = Path(protocol['family_run'])
    c = common.study_config()
    x, y, manifest = dataset(c)
    assert manifest == read(TRAINING/'split_manifest.json')
    states = {}
    try:
        for step in STEPS:
            source = family/name/f'state-{step:04d}'
            rows, operators, oracle, info = score_state(source, unit, protocol['best_cheap_family'], x, y,
                                                         c['solver']['tolerance64'])
            states[step] = info
            write_compact(unit/f'candidates-{step:04d}.json', rows)
            write_compact(unit/f'operators-{step:04d}.json', operators)
            torch.save(oracle, unit/f'oracle-{step:04d}.pt')
            pick = lambda n, r='gradient': next(v for v in rows if v['name'] == n and v['rhs'] == r)['selected']['independent_mean_decrease']
            print(f"{name} step {step}: headroom={pick('exact')-pick('identity'):.6f} "
                  f"best={pick(protocol['best_cheap_family'])-pick('identity'):.6f}", flush=True)
    except Exception as error:
        write_json(unit/'failure.json', {'unit': name, 'error': repr(error)})
        write_json(directory/'failure.json', {'unit': name, 'error': repr(error)})
        raise
    write_json(unit/'states.json', {'states': states, 'split': spec['split'], 'process_peak_rss_bytes': common.peak_rss_bytes()})
    common.finish_unit(unit, {'split': spec['split']})


def recovery(rows, name, rhs):
    means = {n: statistics.mean(r['selected']['independent_mean_decrease'] for r in rows if r['name'] == n and r['rhs'] == rhs)
             for n in ['identity', 'exact', name]}
    gap = means['exact']-means['identity']
    return (means[name]-means['identity'])/gap if gap > 1e-4 else None, gap, means


def finalize(directory):
    protocol = read(directory/'protocol.json')
    names = list(protocol['units'])
    missing = [n for n in names if not common.unit_complete(directory/n)]
    if missing or (directory/'failure.json').exists():
        raise RuntimeError(f'Incomplete or failed units: {missing}')
    best = protocol['best_cheap_family']
    families = candidate_names(best)[2:]
    records = []
    for n in names:
        spec = protocol['units'][n]
        for step in STEPS:
            for row in read(directory/n/f'candidates-{step:04d}.json'):
                records.append({**row, 'unit': n, 'step': step, 'blocks': spec['blocks'], 'width': spec['width'],
                                'seed': spec['seed'], 'split': spec['split']})
    summary = {'best_cheap_family': best, 'per_state': {}, 'per_architecture': {}, 'per_split': {}}
    for n in names:
        for step in STEPS:
            subset = [r for r in records if r['unit'] == n and r['step'] == step]
            entry = {}
            for rhs in ['gradient', 'momentum']:
                rec = {f: recovery(subset, f, rhs)[0] for f in families}
                entry[rhs] = {'headroom': recovery(subset, 'exact', rhs)[1], 'recovery': rec,
                              'identity': recovery(subset, 'exact', rhs)[2]['identity']}
            summary['per_state'][f'{n}-{step}'] = entry
    for key, group in [('per_architecture', lambda r: f"b{r['blocks']}_w{r['width']}"), ('per_split', lambda r: r['split'])]:
        for label in sorted({group(r) for r in records}):
            subset = [r for r in records if group(r) == label]
            summary[key][label] = {rhs: {'aggregate_headroom': recovery(subset, 'exact', rhs)[1],
                                         'recovery': {f: recovery(subset, f, rhs)[0] for f in families},
                                         'states': len(subset)//(2*len(candidate_names(best)))} for rhs in ['gradient', 'momentum']}
    summary['train_split_best_cheap_recovery'] = summary['per_split']['train']['gradient']['recovery'][best]
    write_json(directory/'summary.json', summary)
    write_json(directory/'complete.json', {'completed': True, 'units': len(names), 'states': 3*len(names),
                                          'best_cheap_family': best,
                                          'train_split_best_cheap_recovery': summary['train_split_best_cheap_recovery']})
    print(f'mirrored to {common.mirror(directory)}')


def create(family, check2c):
    if not read(family/'complete.json')['completed'] or not read(check2c/'complete.json')['completed']:
        raise RuntimeError('4b needs completed 4a and 2c runs')
    best = read(check2c/'summary.json')['best_cheap_family']
    units = read(family/'protocol.json')['units']
    c = common.study_config()
    protocol = {'study': 'gap_audit_20261007', 'check': '4b', 'document': str(common.PROTOCOL), 'family_run': str(family),
                'check2c_run': str(check2c), 'best_cheap_family': best, 'candidates': candidate_names(best),
                'rhs': ['gradient', 'momentum'], 'units': units, 'steps': STEPS,
                'rules': 'mu = max(.01 trace(C)/D, 1e-8); search_candidate with native radius; four analysis panels'}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'run_pending.sh')]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'finalize'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--unit')
    parser.add_argument('--family', type=Path)
    parser.add_argument('--check2c', type=Path)
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        create(args.family, args.check2c)
    elif args.command == 'work':
        work(args.run, args.unit)
    elif args.command == 'pending':
        units = read(args.run/'protocol.json')['units']
        pending = [n for n in units if not common.unit_complete(args.run/n)]
        print('\n'.join(sorted(pending, key=lambda n: -units[n]['blocks']*units[n]['width'])))
    else:
        finalize(args.run)


if __name__ == '__main__':
    sys.exit(main())
