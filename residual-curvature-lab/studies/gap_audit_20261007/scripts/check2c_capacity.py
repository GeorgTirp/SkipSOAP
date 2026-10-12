"""Check 2c: offline capacity of structured per-tensor families at six saved ViT states.

No training. Objective, panels, GGN, SOAPMetric, mu and Omega are rebuilt as in
scripts/run_learnable_capacity.py::audit; `measurement` and `search_candidate` are reused.
Usage: check2c_capacity.py [--resume RUN_DIR]
"""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

import common
from common import read, write_compact
import families as fam
from arms import empirical_fisher_action
from residual_curvature.config import load_config
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

PARENT = Path('runs/20261005T143029-9b582ed5')
TRAINING = Path('runs/20261005T135920-d7ce29b2')
SCREEN = Path('runs/20261007T093903-8d7d86f1')
STATES = [(t, s) for t in ['searched_soap', 'exact_tensor_ggn'] for s in [10, 25, 50]]
OPERATOR_FAMILIES = ['identity', 'exact', 'soap_basis_diag', 'kron_basis_diag', 'ks_1', 'ks_2', 'ks_3', 'ks_2_ef',
                     'exact_1d_only']
FAMILIES = OPERATOR_FAMILIES[:-1]+[f'pcg_block_{k}' for k in fam.PCG_STEPS]+[f'pcg_full_{k}' for k in fam.PCG_STEPS]+['exact_1d_only']


def references(trajectory, step):
    screen = read(SCREEN/f'damping-{trajectory}-{step:04d}'/'candidates.json')
    pick = lambda name: next(r for r in screen if r['name'] == name and r['rhs'] == 'gradient')['selected']['independent_mean_decrease']
    information = read(PARENT/'all_candidates.json')
    diag = next(r for r in information if r['trajectory'] == trajectory and r['step'] == step and r['name'] == 'ggn_diagonal_gradient')
    return {'identity': pick('identity'), 'exact': pick('exact'), 'soap_basis_diag': diag['selected']['independent_mean_decrease']}


def audit_state(trajectory, step, unit, c, x, y, model, registry):
    source = PARENT/f'{trajectory}-{step:04d}'
    for name, digest in read(source/'artifacts.json').items():
        if checksum(source/name) != digest:
            raise RuntimeError(f'Parent artifact checksum failed: {source/name}')
    arrays = torch.load(source/'arrays.pt', map_location='cpu', weights_only=False)
    info = read(source/'diagnostics.json')
    assert info['trajectory'] == trajectory and info['step'] == step
    flat, g, momentum, contexts = [arrays[k] for k in ['parameters', 'gradient', 'momentum_rhs', 'contexts']]
    ggn, ef, native = arrays['ggn_blocks'], arrays['empirical_fisher_blocks'], arrays['raw_soap']
    commit(model, registry, flat)
    model_before, contexts_before, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(contexts), torch.get_rng_state().clone()
    construction, panel_ids = info['construction_ids'], info['panel_ids']
    fn = functional_logits(model, registry, x[construction])
    objective = lambda w: loss(fn(w), y[construction])
    op = GGN(fn, flat, y[construction])
    tolerance = c['solver']['tolerance64']
    checks = {'gradient_error': relative_error(op.gradient, g)}
    metric = SOAPMetric(registry, contexts)
    mu = max(.01*sum(float(b.trace()) for b in ggn)/registry.size, c['damping']['floor'])
    gen = torch.Generator().manual_seed(270010+step)
    ggn_actions, ef_actions = [], []
    for spec, block, fblock in zip(registry.specs, ggn, ef):
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, dtype=flat.dtype, generator=gen)
        ggn_actions.append(relative_error(block@v[spec.slice], op.action(v)[spec.slice]))
        ef_actions.append(relative_error(fblock@v[spec.slice], empirical_fisher_action(op, v)[spec.slice]))
    checks.update(max_ggn_action_error=max(ggn_actions), max_ef_action_error=max(ef_actions))
    teacher = fam.DampedBlocks(ggn, registry, metric, mu, tolerance)
    # Evaluation RHS seeds as in the capacity screen; factor RHS without beta (OT/GGN convention).
    gen = torch.Generator().manual_seed(270100+step+(0 if trajectory == 'searched_soap' else 1000))
    torch.randn(registry.size, 16, dtype=flat.dtype, generator=gen)  # physical draw, kept for stream parity
    zs = torch.randn(registry.size, 16, dtype=flat.dtype, generator=gen)
    factor_rhs = torch.empty_like(zs)
    for spec in registry.specs:
        factor_rhs[spec.slice] = capacity.factor(zs[spec.slice].reshape(spec.shape+(16,)), contexts[spec.name],
                                                 transpose=True, inverse=True).reshape(spec.count, 16)
    groups = {'gradient': g[:, None], 'momentum': momentum[:, None], 'factor': factor_rhs}
    exact_values = {k: teacher.solve(v) for k, v in groups.items()}
    teacher_energy = capacity.energy_errors(g, exact_values['gradient'][:, 0], -arrays['raw_ggn_gradient'],
                                            teacher.systems, registry)['raw'][0]
    checks['cached_teacher_energy_error'] = max(teacher_energy, 0.)**.5
    if max(checks.values()) > tolerance:
        raise RuntimeError(f'State checks failed: {checks}')
    panels = [lambda w, f=functional_logits(model, registry, x[ids]), t=y[ids]: loss(f(w), t) for ids in panel_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    rows, operators, deltas = [], [], {}

    def measure(name, rhs_name, delta):
        selected, search = search_candidate(flat, delta, native, objective)
        values = measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry)
        block = float(selected@teacher.action(selected))
        values['block_teacher_quadratic_decrease'] = -float(g@selected)-.5*block
        rows.append({'trajectory': trajectory, 'step': step, 'name': name, 'rhs': rhs_name, 'mu': mu,
                     'selected': values, 'raw_norm': float(delta.norm()), 'search': search})
        deltas[f'{name}_{rhs_name}'] = selected
        write_compact(unit/'candidates.json', rows)
        return values['independent_mean_decrease']

    for name in FAMILIES:
        started = time.perf_counter()
        if name.startswith('pcg_'):
            kind, iterations = name.split('_')[1], int(name.split('_')[2])
            action = fam.pcg_operator(kind, op, ggn, registry, metric, mu)
            def apply(matrix):
                columns, counts = [], []
                for j in range(matrix.shape[1]):
                    value, matvecs = fam.pcg(action, metric.map, matrix[:, j], iterations)
                    columns.append(value)
                    counts.append(matvecs)
                assert set(counts) == {iterations}
                return torch.stack(columns, 1)
            proposals = {k: apply(v) for k, v in groups.items()}
            solved_g = proposals['gradient'][:, 0]
            residual = relative_error(action(solved_g), g)
            extra = {'kind': kind, 'iterations': iterations, 'matvecs_per_rhs': iterations,
                     'matvec_unit': 'tensor-block GGN action' if kind == 'block' else 'one JVP + one VJP (full GGN)',
                     'gradient_relative_residual_diagnostic': residual, 'storage_numbers': 0,
                     'workspace_model_vectors': 4}
            candidates = {'gradient': -solved_g, 'momentum': -proposals['momentum'][:, 0]}
        else:
            blocks, parameters, extra = fam.family_blocks(name, registry, ggn, ef, contexts)
            damped = fam.DampedBlocks(blocks, registry, metric, mu, tolerance)
            proposals = {k: damped.solve(v) for k, v in groups.items()}
            candidates = {}
            for rhs_name, rhs in [('gradient', g), ('momentum', momentum)]:
                values, solve = solve_blocks(blocks, registry, rhs, metric, [mu], tolerance)
                candidates[rhs_name] = values[mu]
                agreement = relative_error(values[mu], -proposals[rhs_name][:, 0])
                if agreement > 1e-12:
                    raise RuntimeError(f'Cached and shared solves disagree for {name}: {agreement}')
                extra[f'{rhs_name}_solve_residual'] = solve[mu]['residual']
            extra['matvecs_per_rhs'] = 0
        errors = {k: capacity.energy_errors(groups[k], proposals[k], exact_values[k], teacher.systems, registry) for k in groups}
        fit_seconds = time.perf_counter()-started
        decreases = {rhs: measure(name, rhs, candidates[rhs]) for rhs in ['gradient', 'momentum']}
        operators.append({'trajectory': trajectory, 'step': step, 'name': name, 'errors': errors,
                          'fit_and_apply_seconds': fit_seconds, **extra,
                          'storage_model_vectors': extra['storage_numbers']/registry.size,
                          'storage_relative_to_soap_factors': extra['storage_numbers']/fam.soap_factor_storage(registry)})
        write_compact(unit/'operators.json', operators)
        print(f"{trajectory}-{step} {name}: gradient={decreases['gradient']:.6f} momentum={decreases['momentum']:.6f} "
              f"gradient ray error={errors['gradient']['ray'][0]:.3e}", flush=True)
    ref = references(trajectory, step)
    reproduction = {}
    for name, stored in ref.items():
        ours = next(r for r in rows if r['name'] == name and r['rhs'] == 'gradient')['selected']['independent_mean_decrease']
        reproduction[name] = {'reproduced': ours, 'stored': stored, 'absolute_difference': abs(ours-stored)}
    if not (state_equal(model_before, model.state_dict()) and state_equal(contexts_before, contexts)
            and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('State mutated during the audit')
    torch.save({'deltas': deltas, 'mu': mu}, unit/'arrays.pt')
    state = {'trajectory': trajectory, 'step': step, 'mu': mu, 'metric_scale': metric.scale, 'checks': checks,
             'reproduction': reproduction, 'nonmutation_passed': True, 'parent': str(source),
             'process_peak_rss_bytes': common.peak_rss_bytes()}
    write_json(unit/'state.json', state)
    return rows, operators, state


def aggregate(directory):
    rows, operators, states = [], [], []
    for trajectory, step in STATES:
        unit = directory/f'{trajectory}-{step:04d}'
        rows += read(unit/'candidates.json')
        operators += read(unit/'operators.json')
        states.append(read(unit/'state.json'))
    def mean(name, rhs):
        return statistics.mean(r['selected']['independent_mean_decrease'] for r in rows if r['name'] == name and r['rhs'] == rhs)
    summary = {'families': {}, 'states': [f'{t}-{s}' for t, s in STATES]}
    for rhs in ['gradient', 'momentum']:
        gap = mean('exact', rhs)-mean('identity', rhs)
        summary[f'{rhs}_aggregate_exact_headroom'] = gap
        for name in FAMILIES:
            entry = summary['families'].setdefault(name, {})
            entry[f'{rhs}_mean_decrease'] = mean(name, rhs)
            entry[f'{rhs}_recovery'] = (mean(name, rhs)-mean('identity', rhs))/gap if gap > 1e-4 else None
            per = {}
            for t, s in STATES:
                pick = lambda n: next(r for r in rows if r['name'] == n and r['rhs'] == rhs and r['trajectory'] == t and r['step'] == s)['selected']['independent_mean_decrease']
                local = pick('exact')-pick('identity')
                per[f'{t}-{s}'] = {'decrease': pick(name), 'recovery': (pick(name)-pick('identity'))/local if abs(local) > 1e-4 else None}
            entry[f'{rhs}_per_state'] = per
    for name in FAMILIES:
        selected = [o for o in operators if o['name'] == name]
        entry = summary['families'][name]
        entry['pooled_ray_errors'] = {k: sum(o['errors'][k]['ray_numerator_sum'] for o in selected)/sum(o['errors'][k]['denominator_sum'] for o in selected)
                                      for k in ['gradient', 'momentum', 'factor']}
        entry['pooled_raw_errors'] = {k: sum(o['errors'][k]['raw_numerator_sum'] for o in selected)/sum(o['errors'][k]['denominator_sum'] for o in selected)
                                      for k in ['gradient', 'momentum', 'factor']}
        entry['storage_numbers'] = max(o['storage_numbers'] for o in selected)
        entry['storage_model_vectors'] = max(o['storage_model_vectors'] for o in selected)
        entry['storage_relative_to_soap_factors'] = max(o['storage_relative_to_soap_factors'] for o in selected)
        entry['matvecs_per_rhs'] = selected[0]['matvecs_per_rhs']
        if name in fam.CHEAP:
            r = entry['gradient_recovery']
            entry['tier'] = 'meets_bar' if r >= .8 else 'promising' if r >= .6 else 'insufficient'
    cheap = sorted(fam.CHEAP, key=lambda n: (-summary['families'][n]['gradient_recovery'], summary['families'][n]['storage_numbers']))
    summary['best_cheap_family'] = cheap[0]
    summary['reproduction'] = {f"{s['trajectory']}-{s['step']}": s['reproduction'] for s in states}
    worst = max(v['absolute_difference'] for s in states for v in s['reproduction'].values())
    summary['reproduction_max_absolute_difference'] = worst
    summary['reproduction_within_1e-9'] = worst <= 1e-9
    summary['checks_max'] = {k: max(s['checks'][k] for s in states) for k in states[0]['checks']}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    c = load_config(TRAINING/'resolved.yaml')
    c['output'] = str(common.RUNS)
    if args.resume:
        directory = args.resume
        common.verify_archive(directory)
    else:
        directory = common.create_run(c, {'study': 'gap_audit_20261007', 'check': '2c', 'document': str(common.PROTOCOL),
                                          'parent': str(PARENT), 'screen_reference': str(SCREEN), 'families': FAMILIES,
                                          'rhs': ['gradient', 'momentum'], 'gamma': .01, 'damping': 'exact trace, normalized inverse frozen SOAP metric',
                                          'training_launched': False})
    print(directory, flush=True)
    x, y, manifest = dataset(c)
    assert manifest == read(TRAINING/'split_manifest.json')
    model = make_model(c)
    registry = Registry(model)
    for trajectory, step in STATES:
        unit = directory/f'{trajectory}-{step:04d}'
        if common.unit_complete(unit):
            print(f'{unit.name}: verified complete; skipped', flush=True)
            continue
        common.prepare_unit(unit)
        try:
            _, _, state = audit_state(trajectory, step, unit, c, x, y, model, registry)
        except Exception as error:
            write_json(directory/'failure.json', {'state': unit.name, 'error': repr(error)})
            raise
        common.finish_unit(unit, {'reproduction': state['reproduction']})
    summary = aggregate(directory)
    write_json(directory/'summary.json', summary)
    if summary['reproduction_max_absolute_difference'] > 1e-6:
        write_json(directory/'failure.json', {'reason': 'sanity reproduction differs by more than 1e-6'})
        raise RuntimeError('Sanity reproduction failed by more than 1e-6')
    write_json(directory/'complete.json', {'completed': True, 'states': 6, 'families': len(FAMILIES),
                                          'candidate_steps': 6*len(FAMILIES)*2,
                                          'best_cheap_family': summary['best_cheap_family'],
                                          'reproduction_within_1e-9': summary['reproduction_within_1e-9']})
    print(json.dumps({k: v for k, v in summary.items() if k not in ['families', 'reproduction']}, indent=2), flush=True)
    print(f'mirrored to {common.mirror(directory)}', flush=True)


if __name__ == '__main__':
    main()
