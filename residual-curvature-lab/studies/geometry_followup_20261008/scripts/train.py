"""Geometry follow-up runner: reproduction gates, staged seed-0 tuning, fresh-seed evaluation.

`work()` is copied verbatim from studies/gap_audit_20261007/scripts/check1_train.py
(SHA-256 8a70c33e...). Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  train.py create --phase gate|tune
  train.py create --phase eval --arms a,b,... [--tuning DIR] [--steps N]
  train.py work --run DIR --unit NAME
  train.py pending --run DIR
  train.py advance --run DIR      tuning: add the next stage of every arm (staged protocol)
  train.py finalize --run DIR
"""
import argparse
import math
import statistics
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

PILOT = 'runs/20261006T100557-f1efb932/validation.json'
GATE_REFERENCES = {
    'parsec_default': (PILOT, 'parsec_native'),
    'ot_tensor': (PILOT, 'wasserstein_tensor'),
    'ggn_exact_trace': (str(common.GAP_EVAL/'ggn_exact_trace__gamma0.001__s1'/'validation.json'), None),
}
LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
LR_EXTENSION = {3e-4: 1e-4, 3e-2: 1e-1}
GAMMA_GRID = [.001, .01, .1]
GAMMA_EXTENSION = {.001: .0001, .1: 1.}
AUDITED = ['ggn_exact_trace', 'ot_tensor', 'ggn_full', 'ot_full']
COST_ORDER = ['ot_full', 'ggn_full', 'ot_tensor', 'ggn_exact_trace', 'parsec_split', 'parsec_default']


def schedule(steps):
    return sorted({0, 10, *range(25, steps+1, 25), steps})


def tag(settings):
    if not settings:
        return 'default'
    return '_'.join(f'{key}{value:g}' for key, value in sorted(settings.items()))


def unit_spec(arm, settings, seed, steps, checkpoints, audits, implementation='study', phase=None):
    name = f'{arm}__{tag(settings)}__s{seed}'
    return name, {'arm': arm, 'settings': settings, 'seed': seed, 'steps': steps,
                  'validation_steps': [0, 10, 25, 50] if steps == 50 else schedule(steps),
                  'checkpoint_steps': checkpoints, 'audit_steps': audits if arm in AUDITED else [],
                  'implementation': implementation, 'phase': phase}


def all_units(directory):
    units = dict(read(directory/'protocol.json')['units'])
    if (directory/'stages.json').exists():
        for stage in read(directory/'stages.json'):
            units.update(stage['units'])
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




def gate_units():
    units = {}
    for arm, settings, seed, implementation in [
            ('parsec_default', {'lr': .003, 'refresh_every': 10, 'rank': 16}, 0, 'original'),
            ('ot_tensor', {'gamma': .01}, 0, 'original'),
            ('ggn_exact_trace', {'gamma': .001}, 1, 'study')]:
        name, spec = unit_spec(arm, settings, seed, 50, [50], [2, 50], implementation, 'gate')
        units[name] = spec
    return units


def tune_units():
    """Stage A of every tuned arm (PROTOCOL.md step 2)."""
    units = {}
    for lr in LR_GRID:
        for refresh in [10, 100]:
            name, spec = unit_spec('parsec_default', {'lr': lr, 'refresh_every': refresh, 'rank': 16}, 0, 500, [500], [], phase='A')
            units[name] = spec
        name, spec = unit_spec('parsec_split', {'lr_perp': lr, 'lr_top': lr, 'refresh_every': 10, 'rank': 16}, 0, 500, [500], [], phase='A')
        units[name] = spec
    for gamma in GAMMA_GRID:
        name, spec = unit_spec('ot_tensor', {'gamma': gamma}, 0, 200, [200], [2, 200], phase='A')
        units[name] = spec
    return units


def results(directory):
    out = {}
    for name, spec in all_units(directory).items():
        if not common.unit_complete(directory/name):
            continue
        done = read(directory/name/'complete.json')['summary']
        ok = done['outcome']['status'] == 'completed'
        out[name] = {**spec, 'status': done['outcome']['status'], 'loss': done['final_validation_loss'] if ok else None}
    return out


def best(rows, arm, phases=None):
    candidates = {n: r for n, r in rows.items() if r['arm'] == arm and r['loss'] is not None
                  and (phases is None or r['phase'] in phases)}
    name = min(candidates, key=lambda n: candidates[n]['loss'])
    return name, candidates[name]


def advance(directory):
    """Add the next stage per arm. Stage order: parsec_default A -> [ext] -> B; parsec_split
    A -> [ext] -> B -> C; ot_tensor A -> [ext]. A stage starts only when the previous is complete."""
    units, rows = all_units(directory), results(directory)
    done = lambda arm, phases: all(n in rows for n, s in units.items() if s['arm'] == arm and s['phase'] in phases)
    added = {s['phase']+':'+s['arm'] for s in units.values()}
    new, reasons = {}, {}
    def add(arm, settings, steps, phase, reason, audits=()):
        name, spec = unit_spec(arm, settings, 0, steps, [steps], list(audits), phase=phase)
        if name not in units and name not in new:
            new[name], reasons[name] = spec, reason
    for arm, key, grid, ext in [('parsec_default', 'lr', LR_GRID, LR_EXTENSION), ('parsec_split', 'lr_perp', LR_GRID, LR_EXTENSION),
                                ('ot_tensor', 'gamma', GAMMA_GRID, GAMMA_EXTENSION)]:
        if not done(arm, ['A']) or f'ext:{arm}' in added or f'B:{arm}' in added or (arm == 'ot_tensor' and 'done:ot_tensor' in added):
            continue
        _, top = best(rows, arm, ['A'])
        value = top['settings'][key]
        if value in ext:
            settings = dict(top['settings'])
            settings[key] = ext[value]
            if arm == 'parsec_split':
                settings['lr_top'] = settings['lr_perp']
            steps = 200 if arm == 'ot_tensor' else 500
            add(arm, settings, steps, 'ext', f'{key} {value:g} on grid edge', [2, 200] if arm == 'ot_tensor' else [])
    for arm in ['parsec_default', 'parsec_split']:
        if f'B:{arm}' in added or not done(arm, ['A', 'ext']) or (f'ext:{arm}' not in added and _edge(rows, arm)):
            continue
        _, top = best(rows, arm, ['A', 'ext'])
        if arm == 'parsec_default':
            for rank in [8, 32]:
                add(arm, {**top['settings'], 'rank': rank}, 500, 'B', 'rank at the stage-A optimum')
        else:
            for lr_top in [1e-2, 1e-1, 1.]:
                add(arm, {**top['settings'], 'lr_top': lr_top}, 500, 'B', 'lr_top at the stage-A lr_perp')
    if 'C:parsec_split' not in added and 'B:parsec_split' in added and done('parsec_split', ['B']):
        _, top = best(rows, 'parsec_split', ['A', 'ext', 'B'])
        add('parsec_split', {**top['settings'], 'refresh_every': 100}, 500, 'C', 'refresh 100 at the stage-B optimum')
    stages = read(directory/'stages.json') if (directory/'stages.json').exists() else []
    if new:
        stages.append({'units': new, 'reasons': reasons, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        write_json(directory/'stages.json', stages)
    print('\n'.join(new) if new else 'nothing to add')


def _edge(rows, arm):
    key = 'lr' if arm == 'parsec_default' else 'lr_perp'
    _, top = best(rows, arm, ['A'])
    return top['settings'][key] in LR_EXTENSION


def tuning_complete(directory):
    units, rows = all_units(directory), results(directory)
    phases = {arm: {s['phase'] for s in units.values() if s['arm'] == arm} for arm in ['parsec_default', 'parsec_split', 'ot_tensor']}
    expected = (('B' in phases['parsec_default']) and ('C' in phases['parsec_split'])
                and (('ext' in phases['ot_tensor']) or best(rows, 'ot_tensor', ['A'])[1]['settings']['gamma'] not in GAMMA_EXTENSION))
    return expected and all(n in rows for n in units)


def selection(directory):
    rows = results(directory)
    return {arm: {'unit': best(rows, arm)[0], **best(rows, arm)[1]['settings'], 'final_validation_loss': best(rows, arm)[1]['loss']}
            for arm in ['parsec_default', 'parsec_split', 'ot_tensor']}


def eval_units(arm_list, chosen, steps):
    units = {}
    checkpoints = [s for s in [50, 200, 500] if s <= steps] + ([steps] if steps not in [50, 200, 500] else [])
    gap = read(common.GAP_EVAL/'protocol.json')['selection']
    for seed in [1, 2, 3]:
        for arm in arm_list:
            if arm == 'ggn_full':
                settings = {'gamma': gap['ggn_exact_trace']['gamma']}
            elif arm == 'ot_full':
                settings = {'gamma': chosen['ot_tensor']['gamma']}
            else:
                keys = ['gamma'] if arm == 'ot_tensor' else ['lr', 'refresh_every', 'rank'] if arm == 'parsec_default' else ['lr_perp', 'lr_top', 'refresh_every', 'rank']
                settings = {k: chosen[arm][k] for k in keys}
            name, spec = unit_spec(arm, settings, seed, steps, checkpoints, [s for s in [2, 50, 200, steps] if s <= steps], phase='eval')
            units[name] = spec
    return units


def create(phase, arm_list=None, tuning=None, steps=500):
    common.guard()
    c = common.study_config()
    extra = {}
    if phase == 'gate':
        units = gate_units()
    elif phase == 'tune':
        units = tune_units()
    else:
        chosen = None
        if any(a != 'ggn_full' for a in arm_list):
            if tuning is None or not read(tuning/'complete.json')['completed']:
                raise RuntimeError('This evaluation needs a completed tuning run')
            chosen = read(tuning/'selection.json')
            extra['tuning_run'] = str(tuning)
        units = eval_units(arm_list, chosen, steps)
        extra.update(selection=chosen, evaluation_steps=steps, baselines=str(common.GAP_EVAL))
    protocol = {'study': 'geometry_followup_20261008', 'phase': phase, 'document': str(common.PROTOCOL), 'units': units, **extra}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'queue_runner.sh')]))


def pending(directory):
    units = all_units(directory)
    names = [n for n in units if not common.unit_complete(directory/n)]
    return sorted(names, key=lambda n: COST_ORDER.index(units[n]['arm']))


def finalize(directory):
    missing = pending(directory)
    if missing:
        raise RuntimeError(f'Incomplete units: {missing}')
    if (directory/'failure.json').exists():
        raise RuntimeError('Run has a failure record; not marking complete')
    protocol = read(directory/'protocol.json')
    units = all_units(directory)
    validation = [{'unit': n, 'arm': s['arm'], 'seed': s['seed'], 'settings': s['settings'], **row}
                  for n, s in units.items() for row in read(directory/n/'validation.json')]
    write_json(directory/'validation.json', validation)
    extra = {}
    if protocol['phase'] == 'gate':
        rows = []
        for name, spec in units.items():
            path, method = GATE_REFERENCES[spec['arm']]
            stored = {r['step']: r for r in read(path) if method is None or r['method'] == method}
            ours = {r['step']: r for r in read(directory/name/'validation.json')}
            differences = {step: abs(ours[step]['validation_loss']-stored[step]['validation_loss']) for step in stored if step in ours}
            rows.append({'unit': name, 'reference': path, 'differences': differences, 'max_difference': max(differences.values()),
                         'within_1e-9': max(differences.values()) <= 1e-9, 'stop_above_1e-6': max(differences.values()) > 1e-6})
        gate = {'rows': rows, 'gate_passed': all(r['within_1e-9'] for r in rows), 'study_stops': any(r['stop_above_1e-6'] for r in rows)}
        write_json(directory/'gate.json', gate)
        extra = {'gate_passed': gate['gate_passed']}
    elif protocol['phase'] == 'tune':
        if not tuning_complete(directory):
            raise RuntimeError('Tuning stages incomplete; run advance first')
        chosen = selection(directory)
        write_json(directory/'selection.json', chosen)
        write_json(directory/'tuning_results.json', results(directory))
        extra = {'selection': chosen}
    write_json(directory/'complete.json', {'completed': True, 'phase': protocol['phase'], 'units': len(units), **extra})
    print(f'Finalized {directory}; mirrored to {common.mirror(directory)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'advance', 'finalize'])
    parser.add_argument('--phase', choices=['gate', 'tune', 'eval'])
    parser.add_argument('--arms')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--unit')
    parser.add_argument('--tuning', type=Path)
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        create(args.phase, args.arms.split(',') if args.arms else None, args.tuning, args.steps)
    elif args.command == 'work':
        work(args.run, args.unit)
    elif args.command == 'pending':
        print('\n'.join(pending(args.run)))
    elif args.command == 'advance':
        advance(args.run)
    else:
        finalize(args.run)


if __name__ == '__main__':
    sys.exit(main())
