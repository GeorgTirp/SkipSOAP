"""Check 1 (with 2a/2b): reproduction gate, seed-0 tuning and fresh-seed evaluation.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  check1_train.py create --phase gate|tune       create a run and print its path
  check1_train.py create --phase eval --tuning DIR [--steps 300]
  check1_train.py work --run DIR --unit NAME       execute one (arm, setting, seed) unit
  check1_train.py pending --run DIR                list units without a verified marker
  check1_train.py extend --run DIR                 tuning: add one edge-extension point, once
  check1_train.py finalize --run DIR               aggregate, gate/select, complete.json, mirror
"""
import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import torch

import common
from common import read, write_compact, write_gz
import arms
from residual_curvature import checkpoint
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.provenance import write_json
from run_exact_training_followup import evaluate

GATE_REFERENCES = {
    'native_soap': ('runs/20261005T135920-d7ce29b2/validation.json', 'native_soap'),
    'searched_soap': ('runs/20261005T135920-d7ce29b2/validation.json', 'searched_soap'),
    'adam_native': ('runs/20261006T100557-f1efb932/validation.json', 'adam_native'),
    'ggn_exact_trace': ('runs/20261006T100557-f1efb932/validation.json', 'ggn_exact_trace'),
}
LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
GAMMA_GRID = [.001, .01, .1]
SOAP_GRID = [{'betas': [.95, .95], 'precondition_frequency': 1}, {'betas': [.95, .95], 'precondition_frequency': 10},
             {'betas': [.9, .99], 'precondition_frequency': 1}, {'betas': [.9, .99], 'precondition_frequency': 10}]
# One grid point beyond each edge (log spacing of the declared grids).
LR_EXTENSION = {3e-4: 1e-4, 3e-2: 1e-1}
GAMMA_EXTENSION = {.001: .0001, .1: 1.}


def schedule(steps):
    return sorted({0, 10, *range(25, steps+1, 25), steps})


def tag(settings):
    if not settings:
        return 'default'
    parts = []
    for key, value in sorted(settings.items()):
        if key == 'betas':
            parts.append('b' + '-'.join(f'{v:g}' for v in value))
        elif key == 'precondition_frequency':
            parts.append(f'pf{value}')
        else:
            parts.append(f'{key}{value:g}')
    return '_'.join(parts)


def unit_spec(arm, settings, seed, steps, checkpoints, audits, implementation='study', phase=None):
    name = f'{arm}__{tag(settings)}__s{seed}'
    return name, {'arm': arm, 'settings': settings, 'seed': seed, 'steps': steps,
                  'validation_steps': [0, 10, 25, 50] if steps == 50 else schedule(steps),
                  'checkpoint_steps': checkpoints, 'audit_steps': audits if arm in ['ggn_exact_trace', 'dense_ef'] else [],
                  'implementation': implementation, 'phase': phase}


def gate_units():
    units = {}
    for arm, settings, implementation in [('native_soap', {'lr': .003}, 'original'), ('searched_soap', {}, 'original'),
                                          ('adam_native', {'lr': .003}, 'original'),
                                          ('ggn_exact_trace', {'gamma': .01}, 'original')]:
        name, spec = unit_spec(arm, settings, 0, 50, [50], [2, 50], implementation, 'gate')
        units[name] = spec
    return units


def tune_units():
    units = {}
    for arm in ['native_soap', 'adam_native']:
        for lr in LR_GRID:
            name, spec = unit_spec(arm, {'lr': lr}, 0, 500, [500], [], phase='tune')
            units[name] = spec
    for settings in SOAP_GRID:
        name, spec = unit_spec('searched_soap_tuned', settings, 0, 500, [500], [], phase='tune')
        units[name] = spec
    for gamma in GAMMA_GRID:
        name, spec = unit_spec('ggn_exact_trace', {'gamma': gamma}, 0, 200, [200], [2, 200], phase='tune')
        units[name] = spec
    return units


def eval_units(selection):
    units = {}
    settings = {'native_soap': {'lr': selection['native_soap']['lr']},
                'adam_native': {'lr': selection['adam_native']['lr']},
                'searched_soap': {},
                'searched_soap_tuned': {k: selection['searched_soap_tuned'][k] for k in ['betas', 'precondition_frequency']},
                'ggn_exact_trace': {'gamma': selection['ggn_exact_trace']['gamma']},
                'dense_ef': {'gamma': selection['ggn_exact_trace']['gamma']},
                'soap_gradient_map': {}}
    steps = selection.get('evaluation_steps', 500)
    checkpoints = [s for s in [50, 200, 500] if s <= steps] + ([steps] if steps not in [50, 200, 500] else [])
    for seed in [1, 2, 3]:
        for arm in arms.ARMS:
            name, spec = unit_spec(arm, settings[arm], seed, steps, checkpoints,
                                   [s for s in [2, 50, 200, steps] if s <= steps], phase='eval')
            units[name] = spec
    return units


def all_units(directory):
    units = dict(read(directory/'protocol.json')['units'])
    if (directory/'extension.json').exists():
        units.update(read(directory/'extension.json')['units'])
    return units


def finite(value):
    return value is not None and math.isfinite(value)


def diverged(error, flat):
    """A divergent configuration (failed setting), as opposed to a failed numerical check."""
    nonfinite_message = isinstance(error, FloatingPointError) or 'Nonfinite' in str(error)
    return nonfinite_message or not bool(torch.isfinite(flat).all())


def work(directory, name):
    common.verify_archive(directory)
    spec = all_units(directory)[name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    base = common.study_config(seed=spec['seed'])
    c = arms.arm_config(base, spec['arm'], spec['settings'])
    c['training']['steps'] = spec['steps']
    write_json(unit/'resolved_arm.json', {'config': c, 'spec': spec})
    x, y, manifest = dataset(c)
    if manifest != read(Path('runs/20261005T135920-d7ce29b2/split_manifest.json')):
        raise RuntimeError('Split manifest differs from the main lab split')
    model = make_model(c)
    registry = Registry(model)
    order = DataOrder(manifest['training'], common.data_order_seed(spec['seed']))
    state, contexts, seconds, audit_seconds, accepted, cpu_seconds = None, None, 0., 0., 0, 0.
    validation = [dict(step=0, training_seconds=0., **evaluate(model, x, y, manifest['validation']))]
    history, full, batches, outcome, step = [], [], [], {'status': 'completed'}, 0
    started_unit = time.perf_counter()
    try:
        for step in range(1, spec['steps']+1):
            ids = order.take(c['data']['batch_size'])
            flat = registry.flatten(dict(model.named_parameters())).detach()
            started, cpu_started = time.perf_counter(), time.process_time()
            try:
                delta, state, contexts, row = arms.propose_step(spec['arm'], spec['settings'], spec['implementation'],
                                                                model, registry, flat, state, x[ids], y[ids], c, step,
                                                                audit=step in spec['audit_steps'])
            except Exception as error:
                if not diverged(error, flat):
                    raise
                outcome = {'status': 'diverged', 'step': step, 'reason': f'nonfinite state or proposal: {error!r}'}
                break
            if not torch.isfinite(delta).all() or not finite(row['loss_after']):
                outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite update or train-batch loss'}
                break
            commit(model, registry, flat+delta)
            elapsed = time.perf_counter()-started-row.get('audit_seconds', 0.)
            cpu_elapsed = time.process_time()-cpu_started-row.get('audit_seconds', 0.)
            seconds += elapsed
            cpu_seconds += cpu_elapsed
            audit_seconds += row.get('audit_seconds', 0.)
            accepted += int(float(delta.norm()) > 0)
            summary = arms.compact_row(row)
            summary.update(step=step, step_seconds=elapsed, training_seconds=seconds, accepted_updates=accepted,
                           step_cpu_seconds=cpu_elapsed, training_cpu_seconds=cpu_seconds,
                           batch_sha256=common.ids_sha256(ids))
            history.append(summary)
            row.update(step=step, step_seconds=elapsed)
            full.append(row)
            batches.append(ids)
            if step in spec['validation_steps']:
                measured = dict(step=step, training_seconds=seconds, training_cpu_seconds=cpu_seconds,
                                **evaluate(model, x, y, manifest['validation']))
                nonfinite = not finite(measured['validation_loss'])
                if nonfinite:
                    measured.update(validation_loss=None, nonfinite=True)
                validation.append(measured)
                write_json(unit/'validation.json', validation)
                write_compact(unit/'training.json', history)
                print(f"{name} step {step}: val={measured['validation_loss']} acc={measured['validation_accuracy']:.2%} "
                      f"train_s={seconds:.1f}", flush=True)
                if nonfinite:
                    outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite validation loss'}
                    break
            if step in spec['checkpoint_steps']:
                checkpoint.save(unit/f'checkpoint-{step:04d}.pt', model, state, contexts, step, accepted, order, c)
    except Exception as error:
        write_json(unit/'failure.json', {'unit': name, 'step': step, 'error': repr(error), 'traceback': traceback.format_exc()})
        write_json(directory/'failure.json', {'unit': name, 'step': step, 'error': repr(error)})
        raise
    write_json(unit/'validation.json', validation)
    write_compact(unit/'training.json', history)
    write_gz(unit/'history_full.json.gz', full)
    torch.save(torch.tensor(batches, dtype=torch.long), unit/'batch_ids.pt')
    write_json(unit/'outcome.json', outcome)
    write_json(unit/'runtime.json', {'training_seconds': seconds, 'training_cpu_seconds': cpu_seconds,
                                     'numerical_audit_seconds': audit_seconds,
                                     'unit_wall_seconds': time.perf_counter()-started_unit,
                                     'process_peak_rss_bytes': common.peak_rss_bytes(),
                                     'note': 'training seconds exclude validation, I/O and numerical audits'})
    final = validation[-1]
    common.finish_unit(unit, {'outcome': outcome, 'final_step': final['step'], 'final_validation_loss': final['validation_loss'],
                              'training_seconds': seconds})
    print(f'{name}: {outcome["status"]}', flush=True)


def create(phase, tuning=None, steps=None):
    common.guard()
    c = common.study_config()
    if phase == 'gate':
        units = gate_units()
        extra = {}
    elif phase == 'tune':
        units = tune_units()
        extra = {}
    else:
        if tuning is None or not read(tuning/'complete.json')['completed']:
            raise RuntimeError('Evaluation needs a completed tuning run (--tuning)')
        selection = read(tuning/'selection.json')
        if steps is not None:
            selection['evaluation_steps'] = steps
        units = eval_units(selection)
        extra = {'tuning_run': str(tuning), 'selection': selection}
    protocol = {'study': 'gap_audit_20261007', 'check': 1, 'phase': phase, 'document': str(common.PROTOCOL),
                'workload': 'configs/mini_vit_allocation.yaml (ViT width 16, expansion 32, 2 blocks, 2 heads, MNIST 28x28, batch 512)',
                'validation': 'reserved 5,000 validation examples', 'search': 'search_candidate: zero + 2^[-16..16] x native radius, 18 golden refinements',
                'units': units, **extra}
    directory = common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'run_units.sh')])
    print(directory)


COST_ORDER = ['ggn_exact_trace', 'dense_ef', 'searched_soap_tuned', 'soap_gradient_map', 'searched_soap', 'native_soap', 'adam_native']


def pending(directory):
    """Pending units, most expensive arms first (better packing across workers)."""
    units = all_units(directory)
    names = [name for name in units if not common.unit_complete(directory/name)]
    return sorted(names, key=lambda n: COST_ORDER.index(units[n]['arm']))


def tuning_selection(directory):
    units = all_units(directory)
    results = {}
    for name, spec in units.items():
        done = read(directory/name/'complete.json')['summary']
        results[name] = {'arm': spec['arm'], 'settings': spec['settings'], 'status': done['outcome']['status'],
                         'final_validation_loss': done['final_validation_loss'] if done['outcome']['status'] == 'completed' else None}
    best = {}
    for arm in ['native_soap', 'adam_native', 'searched_soap_tuned', 'ggn_exact_trace']:
        candidates = {n: r for n, r in results.items() if r['arm'] == arm and r['final_validation_loss'] is not None}
        name = min(candidates, key=lambda n: candidates[n]['final_validation_loss'])
        best[arm] = {'unit': name, **candidates[name]['settings'], 'final_validation_loss': candidates[name]['final_validation_loss']}
    return results, best


def extend(directory):
    if (directory/'extension.json').exists():
        raise RuntimeError('The edge extension is applied once only')
    missing = pending(directory)
    if missing:
        raise RuntimeError(f'Tuning units incomplete: {missing}')
    _, best = tuning_selection(directory)
    units, reasons = {}, {}
    for arm in ['native_soap', 'adam_native']:
        if best[arm]['lr'] in LR_EXTENSION:
            name, spec = unit_spec(arm, {'lr': LR_EXTENSION[best[arm]['lr']]}, 0, 500, [500], [], phase='tune_extension')
            units[name], reasons[name] = spec, f"best lr {best[arm]['lr']:g} on grid edge"
    if best['ggn_exact_trace']['gamma'] in GAMMA_EXTENSION:
        gamma = GAMMA_EXTENSION[best['ggn_exact_trace']['gamma']]
        name, spec = unit_spec('ggn_exact_trace', {'gamma': gamma}, 0, 200, [200], [2, 200], phase='tune_extension')
        units[name], reasons[name] = spec, f"best gamma {best['ggn_exact_trace']['gamma']:g} on grid edge"
    if best['searched_soap_tuned']['precondition_frequency'] == 10:
        settings = {'betas': best['searched_soap_tuned']['betas'], 'precondition_frequency': 100}
        name, spec = unit_spec('searched_soap_tuned', settings, 0, 500, [500], [], phase='tune_extension')
        units[name], reasons[name] = spec, 'best precondition_frequency 10 on upper grid edge'
    write_json(directory/'extension.json', {'units': units, 'reasons': reasons, 'best_before_extension': best})
    print('\n'.join(units) or 'no extension needed')


def finalize(directory):
    missing = pending(directory)
    if missing:
        raise RuntimeError(f'Incomplete units: {missing}')
    protocol = read(directory/'protocol.json')
    units = all_units(directory)
    validation = []
    for name, spec in units.items():
        for row in read(directory/name/'validation.json'):
            validation.append({'unit': name, 'arm': spec['arm'], 'seed': spec['seed'], 'settings': spec['settings'], **row})
    write_json(directory/'validation.json', validation)
    result = {}
    if protocol['phase'] == 'gate':
        rows = []
        for name, spec in units.items():
            path, method = GATE_REFERENCES[spec['arm']]
            stored = {r['step']: r for r in read(path) if r['method'] == method}
            ours = {r['step']: r for r in read(directory/name/'validation.json')}
            differences = {step: abs(ours[step]['validation_loss']-stored[step]['validation_loss']) for step in stored if step in ours}
            difference = differences[50]
            rows.append({'unit': name, 'arm': spec['arm'], 'reference': path, 'stored_step50': stored[50]['validation_loss'],
                         'reproduced_step50': ours[50]['validation_loss'], 'absolute_difference_step50': difference,
                         'differences_all_steps': differences,
                         'accuracy_equal_step50': ours[50]['validation_accuracy'] == stored[50]['validation_accuracy'],
                         'within_1e-9': difference <= 1e-9, 'stop_above_1e-6': difference > 1e-6})
        result = {'rows': rows, 'gate_passed': all(r['within_1e-9'] for r in rows),
                  'study_stops': any(r['stop_above_1e-6'] for r in rows)}
        write_json(directory/'gate.json', result)
    elif protocol['phase'] == 'tune':
        results, best = tuning_selection(directory)
        result = {'results': results, 'selected': best, 'evaluation_steps': 500,
                  'note': 'Selected by validation loss at the end of the tuning horizon (seed 0 only).'}
        write_json(directory/'selection.json', {**best, 'evaluation_steps': 500})
        write_json(directory/'tuning_results.json', result)
    if (directory/'failure.json').exists():
        raise RuntimeError('Run has a failure record; not marking complete')
    write_json(directory/'complete.json', {'completed': True, 'phase': protocol['phase'], 'units': len(units), **(
        {'gate_passed': result['gate_passed']} if protocol['phase'] == 'gate' else {})})
    target = common.mirror(directory)
    print(f'Finalized {directory}; mirrored to {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'extend', 'finalize'])
    parser.add_argument('--phase', choices=['gate', 'tune', 'eval'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--unit')
    parser.add_argument('--tuning', type=Path, help='eval: completed tuning run')
    parser.add_argument('--steps', type=int, help='eval: horizon (500, or the 300-step budget fallback)')
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        create(args.phase, args.tuning, args.steps)
    elif args.command == 'work':
        work(args.run, args.unit)
    elif args.command == 'pending':
        print('\n'.join(pending(args.run)))
    elif args.command == 'extend':
        extend(args.run)
    else:
        finalize(args.run)


if __name__ == '__main__':
    sys.exit(main())
