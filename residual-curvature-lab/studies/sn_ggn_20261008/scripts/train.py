"""SN x exact per-tensor GGN training: one unit per (optimizer, SN condition, seed).

Each unit trains 250 consumed steps of the gap audit's `ggn_exact_trace` (gamma .001) or native SOAP
(lr 1e-2) on the main ViT, with SN at c = 0.95 or without SN (PROTOCOL.md). Next-batch losses before
and after every update are measured outside the timed region.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  train.py create | work --run DIR --unit NAME | pending --run DIR | finalize --run DIR | dryrun
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

import common
from common import read, write_compact, write_gz
import arms
import sn
from residual_curvature import checkpoint
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry
from residual_curvature.provenance import write_json
from run_exact_training_followup import evaluate

OPTIMIZERS = {'ggn': ('ggn_exact_trace', {'gamma': .001}), 'soap': ('native_soap', {'lr': .01})}
BOUNDS = [None, .95]
SEEDS = [1, 2, 3]
STEPS = 250
VALIDATION = [0, 10, *range(25, STEPS+1, 25)]
AUDITS = [2, 50, 200, 250]
CHECKPOINTS = [50, 250]
REFERENCE = {'ggn': 'ggn_exact_trace__gamma0.001__s{seed}', 'soap': 'native_soap__lr0.01__s{seed}'}


def condition(bound):
    return 'nosn' if bound is None else f'sn{bound:g}'


def units():
    return {f'{optimizer}__{condition(bound)}__s{seed}': {'optimizer': optimizer, 'bound': bound, 'seed': seed}
            for optimizer in OPTIMIZERS for bound in BOUNDS for seed in SEEDS}


def unit_config(spec):
    arm, settings = OPTIMIZERS[spec['optimizer']]
    c = arms.arm_config(common.study_config(seed=spec['seed']), arm, settings)
    c['training']['steps'] = STEPS
    return c, arm, settings


def peek(order, batch_size):
    """The next batch `order.take` would return, without advancing `order`."""
    twin = DataOrder([0], 0)
    state = order.state_dict()
    twin.load_state_dict({'ids': state['ids'], 'order': state['order'], 'position': state['position'], 'rng': state['rng']})
    return twin.take(batch_size)


def batch_loss(model, x, y):
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(model(x), y))


def spectra(model, bound):
    return {name: {'sigma_raw': row['sigma_raw'], 'active': row['active']} for name, row in sn.spectral_report(model, bound).items()}


def work(directory, name):
    common.verify_archive(directory)
    spec = read(directory/'protocol.json')['units'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    c, arm, settings = unit_config(spec)
    write_json(unit/'resolved_arm.json', {'config': c, 'spec': spec, 'arm': arm, 'settings': settings})
    x, y, manifest = dataset(c)
    if manifest != read(Path('runs/20261005T135920-d7ce29b2/split_manifest.json')):
        raise RuntimeError('Split manifest differs from the main lab split')
    reference = None
    if spec['bound'] is None:
        reference = {r['step']: r['validation_loss'] for r in read(common.GAP_EVAL/REFERENCE[spec['optimizer']].format(seed=spec['seed'])/'validation.json')}
    model = sn.make_sn_model(c, spec['bound'])
    registry = Registry(model)
    order = DataOrder(manifest['training'], common.data_order_seed(spec['seed']))
    batch = c['data']['batch_size']
    state, contexts, seconds, cpu_seconds, audit_seconds, accepted, step = None, None, 0., 0., 0., 0, 0
    validation = [dict(step=0, training_seconds=0., **evaluate(model, x, y, manifest['validation']))]
    history, full, sigma, gate, outcome = [], [], {0: spectra(model, spec['bound'])}, {}, {'status': 'completed'}
    started_unit = time.perf_counter()
    try:
        for step in range(1, STEPS+1):
            ids = order.take(batch)
            following = peek(order, batch)
            flat = registry.flatten(dict(model.named_parameters())).detach()
            next_before = batch_loss(model, x[following], y[following])
            started, cpu_started = time.perf_counter(), time.process_time()
            try:
                delta, state, contexts, row = arms.propose_step(arm, settings, 'study', model, registry, flat, state, x[ids], y[ids],
                                                                c, step, audit=arm == 'ggn_exact_trace' and step in AUDITS)
            except Exception as error:
                if not (isinstance(error, FloatingPointError) or 'Nonfinite' in str(error) or not bool(torch.isfinite(flat).all())):
                    raise
                outcome = {'status': 'diverged', 'step': step, 'reason': repr(error)}
                break
            if not torch.isfinite(delta).all() or not torch.isfinite(torch.tensor(row['loss_after'])):
                outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite update or train-batch loss'}
                break
            commit(model, registry, flat+delta)
            elapsed = time.perf_counter()-started-row.get('audit_seconds', 0.)
            cpu_elapsed = time.process_time()-cpu_started-row.get('audit_seconds', 0.)
            seconds, cpu_seconds = seconds+elapsed, cpu_seconds+cpu_elapsed
            audit_seconds += row.get('audit_seconds', 0.)
            accepted += int(float(delta.norm()) > 0)
            summary = arms.compact_row(row)
            summary.update(step=step, step_seconds=elapsed, training_seconds=seconds, step_cpu_seconds=cpu_elapsed,
                           accepted_updates=accepted, next_loss_before=next_before,
                           next_loss_after=batch_loss(model, x[following], y[following]),
                           batch_sha256=common.ids_sha256(ids), next_batch_sha256=common.ids_sha256(following))
            history.append(summary)
            row.update(step=step, step_seconds=elapsed)
            full.append(row)
            if step in VALIDATION:
                measured = dict(step=step, training_seconds=seconds, training_cpu_seconds=cpu_seconds,
                                **evaluate(model, x, y, manifest['validation']))
                validation.append(measured)
                sigma[step] = spectra(model, spec['bound'])
                write_json(unit/'validation.json', validation)
                write_compact(unit/'training.json', history)
                if reference is not None:
                    gate[step] = abs(measured['validation_loss']-reference[step])
                    if gate[step] > 1e-9:
                        raise RuntimeError(f'Reproduction gate failed at step {step}: {gate[step]}')
                active = sum(r['active'] for r in sigma[step].values())
                print(f"{name} step {step}: val={measured['validation_loss']:.4f} acc={measured['validation_accuracy']:.2%} "
                      f"active_sn={active} train_s={seconds:.1f}", flush=True)
                if not torch.isfinite(torch.tensor(measured['validation_loss'])):
                    outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite validation loss'}
                    break
            if step in CHECKPOINTS:
                checkpoint.save(unit/f'checkpoint-{step:04d}.pt', model, state, contexts, step, accepted, order, c)
    except Exception as error:
        write_json(unit/'failure.json', {'unit': name, 'step': step, 'error': repr(error), 'traceback': traceback.format_exc()})
        write_json(directory/'failure.json', {'unit': name, 'step': step, 'error': repr(error)})
        raise
    write_json(unit/'validation.json', validation)
    write_compact(unit/'training.json', history)
    write_gz(unit/'history_full.json.gz', full)
    write_json(unit/'spectra.json', sigma)
    write_json(unit/'outcome.json', outcome)
    if reference is not None:
        write_json(unit/'reproduction_gate.json', {'reference': str(common.GAP_EVAL/REFERENCE[spec['optimizer']].format(seed=spec['seed'])),
                                                   'differences': gate, 'max_difference': max(gate.values()) if gate else None})
    write_json(unit/'runtime.json', {'training_seconds': seconds, 'training_cpu_seconds': cpu_seconds,
                                     'numerical_audit_seconds': audit_seconds, 'unit_wall_seconds': time.perf_counter()-started_unit,
                                     'process_peak_rss_bytes': common.peak_rss_bytes(),
                                     'note': 'training seconds exclude validation, next-batch measurements, I/O and numerical audits'})
    common.finish_unit(unit, {'outcome': outcome, 'final_step': validation[-1]['step'],
                              'final_validation_loss': validation[-1]['validation_loss'], 'training_seconds': seconds,
                              'max_gate_difference': max(gate.values()) if gate else None})
    print(f"{name}: {outcome['status']}", flush=True)


def create():
    c = common.study_config()
    protocol = {'study': 'sn_ggn_20261008', 'document': str(common.PROTOCOL), 'units': units(), 'steps': STEPS,
                'validation_steps': VALIDATION, 'audit_steps': AUDITS, 'checkpoint_steps': CHECKPOINTS,
                'optimizers': {k: {'arm': a, 'settings': s} for k, (a, s) in OPTIMIZERS.items()}, 'bounds': BOUNDS, 'seeds': SEEDS,
                'sn': 'W_eff = W min(1, c/sigma(W)), exact sigma; residual-branch linear weights (12 tensors)',
                'workload': 'configs/mini_vit_allocation.yaml (ViT width 16, 2 blocks, MNIST 28x28, batch 512, CPU float64)',
                'reproduction_references': {k: str(common.GAP_EVAL/v) for k, v in REFERENCE.items()}}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'queue_runner.sh')]))


def pending(directory):
    names = read(directory/'protocol.json')['units']
    left = [n for n in names if not common.unit_complete(directory/n)]
    return sorted(left, key=lambda n: (names[n]['optimizer'] != 'ggn', names[n]['bound'] is None, n))


def finalize(directory):
    names = list(read(directory/'protocol.json')['units'])
    missing = [n for n in names if not common.unit_complete(directory/n)]
    if missing or (directory/'failure.json').exists():
        raise RuntimeError(f'Incomplete or failed units: {missing}')
    write_json(directory/'units_summary.json', {n: read(directory/n/'complete.json')['summary'] for n in names})
    write_json(directory/'complete.json', {'completed': True, 'units': len(names)})
    print(f'mirrored to {common.mirror(directory)}')


def dryrun(steps=4):
    """Sequential cost of a few GGN steps with and without SN (seed 1); nothing is kept but the timings."""
    out = {}
    for bound in BOUNDS:
        c, arm, settings = unit_config({'optimizer': 'ggn', 'seed': 1})
        x, y, manifest = dataset(c)
        model = sn.make_sn_model(c, bound)
        registry = Registry(model)
        order = DataOrder(manifest['training'], common.data_order_seed(1))
        state, times = None, []
        for step in range(1, steps+1):
            ids = order.take(512)
            flat = registry.flatten(dict(model.named_parameters())).detach()
            started = time.perf_counter()
            delta, state, _, row = arms.propose_step(arm, settings, 'study', model, registry, flat, state, x[ids], y[ids], c, step,
                                                     audit=step == 2)
            commit(model, registry, flat+delta)
            times.append(time.perf_counter()-started-row.get('audit_seconds', 0.))
            if step == 2:
                out[f'{condition(bound)}_max_action_check'] = arms.compact_row(row)['max_action_check']
        out[condition(bound)] = {'seconds_per_step_steps_2_on': sum(times[1:])/len(times[1:]), 'all': times}
    write_json(common.STUDY/'logs'/'dryrun.json', out)
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'finalize', 'dryrun'])
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
        print('\n'.join(pending(args.run)))
    elif args.command == 'finalize':
        finalize(args.run)
    else:
        dryrun()


if __name__ == '__main__':
    sys.exit(main())
