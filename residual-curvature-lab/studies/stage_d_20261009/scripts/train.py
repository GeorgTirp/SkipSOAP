"""Stage D runner (PROTOCOL.md): one unit per (arm, settings, seed).

Phases: `tune` (seed 0: SOAP x 3 lr, brake 4 calibrations x 3 lr, Early-Newton RHS x hand-over),
`eval_fixed` (seeds 1-3: SOAP x 3 lr, GGN with selected step; independent of tuning) and `eval_tuned`
(seeds 1-3: the selected brake configuration and Early-Newton with the selected RHS, hand-over off/on).

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  train.py create --phase tune|eval_fixed | create --phase eval_tuned --tuning DIR
  train.py work --run DIR --unit NAME | pending --run DIR | finalize --run DIR | dryrun
"""
import argparse
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

import common
from common import read, write_compact
import ggn_selected
from parsec_l import PARSECL, ggn_probes, per_example_gradients
from residual_curvature.curvature.operators import GGN
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.provenance import write_json
from run_exact_training_followup import evaluate

STEPS = 500
VALIDATION = [0, 10, *range(25, STEPS+1, 25)]
BATCH, SELECTION_BATCH, SELECTION_SEED, PROBE_SEED = 512, 128, 5102, 9000
LRS = [1e-2, 2e-2, 3e-2]
CALIBRATIONS = [('lr', 1.), ('lr', 4.), ('lr', 30.), ('damping', 1.)]
GGN_GAMMA, GGN_AUDITS, DIAGNOSE_EVERY = .001, [2, 50, 200, 500], 10
COST = {'ggn_selected': 0, 'early_newton': 1, 'brake': 2, 'soap': 3}


def name_of(spec):
    tag = {'soap': lambda s: f"lr{s['lr']:g}",
           'brake': lambda s: f"{s['calibration']}{s['strength']:g}_lr{s['lr']:g}",
           'early_newton': lambda s: f"{s['rhs']}_handover{int(s['handover'])}",
           'ggn_selected': lambda s: 'gamma0.001'}[spec['arm']](spec)
    return f"{spec['arm']}__{tag}__s{spec['seed']}"


def units_for(specs):
    return {name_of(s): s for s in specs}


def tune_units():
    specs = [{'arm': 'soap', 'lr': lr, 'seed': 0} for lr in LRS]
    specs += [{'arm': 'brake', 'calibration': cal, 'strength': c, 'lr': lr, 'seed': 0} for cal, c in CALIBRATIONS for lr in LRS]
    specs += [{'arm': 'early_newton', 'rhs': rhs, 'handover': h, 'seed': 0} for rhs in ['momentum', 'gradient'] for h in [False, True]]
    return units_for(specs)


def eval_fixed_units():
    specs = [{'arm': 'soap', 'lr': lr, 'seed': s} for s in [1, 2, 3] for lr in LRS]
    specs += [{'arm': 'ggn_selected', 'seed': s} for s in [1, 2, 3]]
    return units_for(specs)


def eval_tuned_units(selection):
    b = selection['brake']
    specs = [{'arm': 'brake', 'calibration': b['calibration'], 'strength': b['strength'], 'lr': b['lr'], 'seed': s} for s in [1, 2, 3]]
    specs += [{'arm': 'early_newton', 'rhs': selection['early_newton_rhs'], 'handover': h, 'seed': s} for s in [1, 2, 3] for h in [False, True]]
    return units_for(specs)


def unit_config(seed):
    c = common.study_config(seed=seed)
    if c['model']['branch_scale'] != 1.0 or c['model']['blocks'] != 2:
        raise RuntimeError('Protocol: depth 2, branch scaling off')
    c['training']['steps'] = STEPS
    return c


def make_optimizer(spec, model, c):
    soap = dict(c['soap'], lr=spec.get('lr', .01))
    core = dict(rank=16, gamma=.01, probe_every=10, refresh='extract', norm_blend=0., batched=False, signals=False, seed=spec['seed'])
    if spec['arm'] == 'soap':
        settings = dict(rank=0, norm_blend=0., batched=False, signals=False)
    elif spec['arm'] == 'brake':
        settings = dict(core, calibration=spec['calibration'], strength=spec['strength'])
    else:
        settings = dict(core, calibration='damping', rhs=spec['rhs'], select_grid=(-4, 8), select_refine=12,
                        select_compare_soap=spec['handover'], select_handover=10 if spec['handover'] else None)
    return PARSECL(model.parameters(), **soap, **settings)


def peek(order, batch_size):
    """The next batch `order.take` would return, without advancing `order`."""
    twin = DataOrder([0], 0)
    state = order.state_dict()
    twin.load_state_dict({'ids': state['ids'], 'order': state['order'], 'position': state['position'], 'rng': state['rng']})
    return twin.take(batch_size)


def batch_loss(model, x, y):
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(model(x), y))


def finite(value):
    return value is not None and value == value and abs(value) != float('inf')


def json_safe(value):
    """Non-finite floats (e.g. r when the cross-fitted signal is not positive) are recorded as null."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not finite(value):
        return None
    return value


def summarize_info(info):
    """Compact per-step record of the optimizer's diagnostics."""
    out = {'correction_energy': info['correction_energy']}
    probe = info.get('probe')
    if probe and 'mu' in probe:
        kappas = [t['kappa_max'] for t in probe['tensors'].values()]
        out['probe'] = {'mu': probe['mu'], 'beta_damping': probe['beta_damping'], 'eta_lambda_max': probe['eta_lambda_max'],
                        'kappa_max': max(kappas) if kappas else 0.,
                        'residual_held_max': max((t.get('residual_held', 0.) for t in probe['tensors'].values()), default=0.)}
    if info.get('select'):
        out['select'] = {k: info['select'][k] for k in ['alpha', 'value', 'soap_value', 'winner', 'soap_wins', 'evaluations', 'handover']}
    if info.get('optimism'):
        out['optimism'] = {k: v for k, v in info['optimism'].items() if k != 'seconds'}
    out['handover'] = bool(info.get('handover') or (info.get('select') or {}).get('handover'))
    return out


def work(directory, name):
    common.verify_archive(directory)
    spec = read(directory/'protocol.json')['units'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    seed, arm = spec['seed'], spec['arm']
    c = unit_config(seed)
    write_json(unit/'resolved_arm.json', {'config': c, 'spec': spec})
    x, y, manifest = dataset(c)
    if manifest != read(Path('runs/20261005T135920-d7ce29b2/split_manifest.json')):
        raise RuntimeError('Split manifest differs from the main lab split')
    model = make_model(c)
    registry = Registry(model)
    order = DataOrder(manifest['training'], common.data_order_seed(seed))
    selection_order = DataOrder(manifest['training'], SELECTION_SEED+1000*seed)
    reference = None
    if arm == 'soap' and spec['lr'] == .01 and seed in [1, 2, 3]:
        reference = {r['step']: r['validation_loss'] for r in read(common.GAP_EVAL/f'native_soap__lr0.01__s{seed}'/'validation.json')}
    optimizer = None if arm == 'ggn_selected' else make_optimizer(spec, model, c)
    probe_generator = torch.Generator().manual_seed(PROBE_SEED+seed)
    state, seconds, examples_gradient, examples_selection, gate = None, 0., 0, 0, {}
    validation = [json_safe(dict(step=0, training_seconds=0., **evaluate(model, x, y, manifest['validation'])))]
    history, outcome, step = [], {'status': 'completed'}, 0
    started_unit = time.perf_counter()
    try:
        for step in range(1, STEPS+1):
            ids = order.take(BATCH)
            following = peek(order, BATCH)
            own_before = batch_loss(model, x[ids], y[ids])
            next_before = batch_loss(model, x[following], y[following])
            diagnose = step % DIAGNOSE_EVERY == 0
            wants_selection = arm == 'ggn_selected' or (arm == 'early_newton' and optimizer.wants_selection())
            selected = selection_order.take(SELECTION_BATCH) if wants_selection else None
            true = per_example_gradients(model, x[ids], y[ids]) if diagnose and optimizer is not None else None
            row = {'step': step}
            began = time.perf_counter()
            if arm == 'ggn_selected':
                flat = registry.flatten(dict(model.named_parameters())).detach()
                try:
                    delta, state, _, info, diagnostic = ggn_selected.propose(GGN_GAMMA, model, registry, flat, state, x[ids], y[ids],
                                                                             x[selected], y[selected], c, step,
                                                                             audit=step in GGN_AUDITS, diagnose=diagnose)
                except FloatingPointError as error:
                    outcome = {'status': 'diverged', 'step': step, 'reason': repr(error)}
                    break
                commit(model, registry, flat+delta)
                elapsed = time.perf_counter()-began-info['audit_seconds']-info['diagnostic_seconds']
                algorithm, search = info['algorithm'], info['search'] or {}
                row.update(mu=algorithm.get('mu'), trace=(algorithm.get('curvature') or {}).get('trace'),
                           solve_residual=(algorithm.get('solve') or {}).get('residual'),
                           max_action_check=max(algorithm['action_checks'].values()) if algorithm.get('action_checks') else None,
                           search={k: search.get(k) for k in ['selected_alpha', 'evaluations', 'bracketed', 'rejected']} if search else None,
                           selection_before=info['selection_before'], selection_after=info['selection_after'], norm=info['norm'],
                           optimism=diagnostic)
            else:
                flat = registry.flatten(dict(model.named_parameters())).detach()
                gradient = registry.unflatten(GGN(functional_logits(model, registry, x[ids]), flat, y[ids]).gradient)
                for pname, p in model.named_parameters():
                    p.grad = gradient[pname].detach().clone()
                probes = ggn_probes(model, x[ids], samples=4, generator=probe_generator)[0] if optimizer.wants_probes() else None
                selection = (lambda: batch_loss(model, x[selected], y[selected])) if selected is not None else None
                optimizer.step(probes=probes, selection=selection, true_gradients=true)
                elapsed = time.perf_counter()-began-((optimizer.last_info.get('optimism') or {}).get('seconds', 0.))
                row.update(summarize_info(optimizer.last_info))
                row['norm'] = float((registry.flatten(dict(model.named_parameters())).detach()-flat).norm())
            seconds += elapsed
            examples_gradient += BATCH
            examples_selection += SELECTION_BATCH if selected is not None else 0
            own_after = batch_loss(model, x[ids], y[ids])
            row.update(own_before=own_before, own_after=own_after, next_before=next_before,
                       next_after=batch_loss(model, x[following], y[following]), step_seconds=elapsed, training_seconds=seconds,
                       examples_gradient=examples_gradient, examples_selection=examples_selection)
            history.append(json_safe(row))
            if not finite(own_after):
                outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite training-batch loss'}
                break
            if step in VALIDATION:
                measured = dict(step=step, training_seconds=seconds, examples_gradient=examples_gradient,
                                examples_selection=examples_selection, **evaluate(model, x, y, manifest['validation']))
                validation.append(json_safe(measured))
                write_json(unit/'validation.json', validation)
                write_compact(unit/'training.json', history)
                if reference is not None:
                    gate[step] = abs(measured['validation_loss']-reference[step])
                    if gate[step] > 1e-6:
                        raise RuntimeError(f'SOAP reproduction gate failed at step {step}: {gate[step]}')
                print(f"{name} step {step}: val={measured['validation_loss']:.4f} acc={measured['validation_accuracy']:.2%} "
                      f"train_s={seconds:.1f}", flush=True)
                if measured['validation_loss'] is None or not finite(measured['validation_loss']):
                    outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite validation loss'}
                    break
    except Exception as error:
        write_json(unit/'failure.json', {'unit': name, 'step': step, 'error': repr(error), 'traceback': traceback.format_exc()})
        write_json(directory/'failure.json', {'unit': name, 'step': step, 'error': repr(error)})
        raise
    write_json(unit/'validation.json', validation)
    write_compact(unit/'training.json', history)
    write_json(unit/'outcome.json', outcome)
    torch.save({'parameters': registry.flatten(dict(model.named_parameters())).detach(), 'step': step}, unit/'final_parameters.pt')
    if reference is not None:
        write_json(unit/'reproduction_gate.json', {'reference': str(common.GAP_EVAL/f'native_soap__lr0.01__s{seed}'),
                                                   'differences': gate, 'max_difference': max(gate.values()) if gate else None})
    write_json(unit/'runtime.json', {'training_seconds': seconds, 'unit_wall_seconds': time.perf_counter()-started_unit,
                                     'examples_gradient': examples_gradient, 'examples_selection': examples_selection,
                                     'process_peak_rss_bytes': common.peak_rss_bytes(),
                                     'note': 'training seconds exclude validation, own/next-batch measurements, numerical audits and r diagnostics'})
    common.finish_unit(unit, {'outcome': outcome, 'final_step': validation[-1]['step'],
                              'final_validation_loss': validation[-1]['validation_loss'], 'training_seconds': seconds})
    print(f"{name}: {outcome['status']}", flush=True)


def create(phase, tuning=None):
    c = common.study_config()
    extra = {}
    if phase == 'tune':
        units = tune_units()
    elif phase == 'eval_fixed':
        units = eval_fixed_units()
    else:
        if tuning is None or not read(tuning/'complete.json')['completed']:
            raise RuntimeError('eval_tuned needs a completed tuning run')
        selection = read(tuning/'selection.json')
        units, extra = eval_tuned_units(selection), {'tuning_run': str(tuning), 'selection': selection}
    protocol = {'study': 'stage_d_20261009', 'phase': phase, 'document': str(common.PROTOCOL), 'units': units, 'steps': STEPS,
                'validation_steps': VALIDATION, 'batch': BATCH, 'selection_batch': SELECTION_BATCH, **extra}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'queue_runner.sh')]))


def pending(directory):
    units = read(directory/'protocol.json')['units']
    left = [n for n in units if not common.unit_complete(directory/n)]
    return sorted(left, key=lambda n: (COST[units[n]['arm']], n))


def select_settings(directory):
    """Seed-0 choices (PROTOCOL.md section 4): brake by step-500 validation loss; Early-Newton RHS by step-100 loss
    averaged over the two hand-over variants."""
    units = read(directory/'protocol.json')['units']
    loss_at = lambda n, s: next(r['validation_loss'] for r in read(directory/n/'validation.json') if r['step'] == s)
    final = lambda n: read(directory/n/'validation.json')[-1]
    brake = {}
    for n, spec in units.items():
        if spec['arm'] == 'brake':
            last = final(n)
            brake[n] = last['validation_loss'] if last['step'] == STEPS and finite(last['validation_loss']) else float('inf')
    best = min(brake, key=brake.get)
    rhs = {}
    for r in ['momentum', 'gradient']:
        names = [n for n, s in units.items() if s['arm'] == 'early_newton' and s['rhs'] == r]
        values = [loss_at(n, 100) if any(v['step'] == 100 for v in read(directory/n/'validation.json')) else None for n in names]
        values = [v if finite(v) else float('inf') for v in values]
        rhs[r] = statistics.mean(values)
    chosen = min(rhs, key=rhs.get)
    return {'brake': {k: units[best][k] for k in ['calibration', 'strength', 'lr']} | {'unit': best, 'final_validation_loss': brake[best]},
            'brake_grid': brake, 'early_newton_rhs': chosen, 'early_newton_rhs_step100': rhs}


def finalize(directory):
    missing = pending(directory)
    if missing or (directory/'failure.json').exists():
        raise RuntimeError(f'Incomplete or failed units: {missing}')
    protocol = read(directory/'protocol.json')
    if protocol['phase'] == 'tune':
        selection = select_settings(directory)
        write_json(directory/'selection.json', selection)
        print(selection)
    write_json(directory/'units_summary.json', {n: read(directory/n/'complete.json')['summary'] for n in protocol['units']})
    write_json(directory/'complete.json', {'completed': True, 'phase': protocol['phase'], 'units': len(protocol['units'])})
    print(f'mirrored to {common.mirror(directory)}')


def dryrun(steps=5):
    """Sequential seconds per step of every arm type (steps 2..5), written to logs/dryrun.json."""
    specs = [{'arm': 'soap', 'lr': .01, 'seed': 1}, {'arm': 'brake', 'calibration': 'lr', 'strength': 1., 'lr': .01, 'seed': 1},
             {'arm': 'brake', 'calibration': 'damping', 'strength': 1., 'lr': .01, 'seed': 1},
             {'arm': 'early_newton', 'rhs': 'momentum', 'handover': True, 'seed': 1}, {'arm': 'ggn_selected', 'seed': 1}]
    out = {}
    for spec in specs:
        c = unit_config(1)
        x, y, manifest = dataset(c)
        model = make_model(c)
        registry = Registry(model)
        order, selection_order = DataOrder(manifest['training'], 1102), DataOrder(manifest['training'], 6102)
        optimizer = None if spec['arm'] == 'ggn_selected' else make_optimizer(spec, model, c)
        generator, state, times = torch.Generator().manual_seed(1), None, []
        for step in range(1, steps+1):
            ids, selected = order.take(BATCH), selection_order.take(SELECTION_BATCH)
            began = time.perf_counter()
            flat = registry.flatten(dict(model.named_parameters())).detach()
            if optimizer is None:
                delta, state, _, info, _ = ggn_selected.propose(GGN_GAMMA, model, registry, flat, state, x[ids], y[ids], x[selected], y[selected], c, step)
                commit(model, registry, flat+delta)
            else:
                gradient = registry.unflatten(GGN(functional_logits(model, registry, x[ids]), flat, y[ids]).gradient)
                for pname, p in model.named_parameters():
                    p.grad = gradient[pname].detach().clone()
                probes = ggn_probes(model, x[ids], samples=4, generator=generator)[0] if optimizer.wants_probes() else None
                selection = (lambda: batch_loss(model, x[selected], y[selected])) if spec['arm'] == 'early_newton' else None
                optimizer.step(probes=probes, selection=selection)
            times.append(time.perf_counter()-began)
        out[name_of(spec)] = {'seconds_per_step_2_on': statistics.mean(times[1:]), 'all': times}
        print(name_of(spec), out[name_of(spec)], flush=True)
    write_json(common.STUDY/'logs'/'dryrun.json', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'finalize', 'dryrun'])
    parser.add_argument('--phase', choices=['tune', 'eval_fixed', 'eval_tuned'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--unit')
    parser.add_argument('--tuning', type=Path)
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        create(args.phase, args.tuning)
    elif args.command == 'work':
        work(args.run, args.unit)
    elif args.command == 'pending':
        print('\n'.join(pending(args.run)))
    elif args.command == 'finalize':
        finalize(args.run)
    else:
        dryrun()


if __name__ == '__main__':
    sys.exit(main())
