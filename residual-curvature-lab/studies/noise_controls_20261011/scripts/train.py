"""Noise-controls runner (PROTOCOL.md): one unit per (arm, schedule, peak lr, strength, seed).

Phases: `tune` (seed 0, 17 units), `eval_fixed` (seeds 1-3: constant SOAP and brake, 6 units) and
`eval_tuned` (seeds 1-3: the selected schedule arms and top-free variant, 15 units).

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  train.py create --phase tune|eval_fixed | create --phase eval_tuned --tuning DIR
  train.py work --run DIR --unit NAME | pending --run DIR | finalize --run DIR
"""
import argparse
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.func import grad, jvp

import common
from common import read, write_compact
from parsec_l import PARSECL, ggn_probes, per_example_gradients
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.provenance import write_json
from run_exact_training_followup import evaluate

STEPS = 500
VALIDATION = [0, 10, *range(25, STEPS+1, 25)]
BATCH, PROBE_SEED, DIAGNOSE_EVERY = 512, 9000, 10
PEAKS, TOPFREE_STRENGTHS, EMA_DECAYS = [1e-2, 2e-2, 3e-2], [1., 4., 16.], [.9, .98, .995]
SHARPNESS_STEPS, LANCZOS = [10, 25, 50, 100, 200, 300, 400, 500], 30
BETA2 = .95
STAGE_D = Path.home()/'skipsoap-study-data'/'stage_d_20261009'/'runs'
STAGE_D_RUNS = {'tune': STAGE_D/'20261010T171813-415c766d', 'fixed': STAGE_D/'20261010T171814-46f283ad',
                'tuned': STAGE_D/'20261010T180135-101694d8'}
COST = {'topfree': 0, 'brake': 1, 'soap': 2}


def lr_at(schedule, peak, t):
    if schedule == 'constant':
        return peak
    if schedule == 'cosine':
        return peak*.5*(1+math.cos(math.pi*(t-1)/STEPS))
    if schedule == 'wsd':
        return peak if t <= 400 else peak*(501-t)/100
    raise ValueError(schedule)


def name_of(spec):
    strength = f"_c{spec['strength']:g}" if spec['arm'] == 'topfree' else ''
    return f"{spec['arm']}__{spec['schedule']}__lr{spec['lr']:g}{strength}__s{spec['seed']}"


def spec_of(arm, schedule, lr, seed, strength=None, sharpness=False):
    spec = {'arm': arm, 'schedule': schedule, 'lr': lr, 'seed': seed, 'sharpness': sharpness}
    if arm != 'soap':
        spec['strength'] = 4. if strength is None else strength
    return spec


def units_for(specs):
    return {name_of(s): s for s in specs}


def tune_units():
    specs = [spec_of('soap', 'constant', .01, 0, sharpness=True), spec_of('brake', 'constant', .01, 0, sharpness=True)]
    specs += [spec_of(arm, schedule, peak, 0) for arm in ['soap', 'brake'] for schedule in ['cosine', 'wsd'] for peak in PEAKS]
    specs += [spec_of('topfree', 'constant', .01, 0, strength=c) for c in TOPFREE_STRENGTHS]
    return units_for(specs)


def eval_fixed_units():
    return units_for([spec_of(arm, 'constant', .01, s, sharpness=True) for s in [1, 2, 3] for arm in ['soap', 'brake']])


def eval_tuned_units(selection):
    specs = []
    for s in [1, 2, 3]:
        for key, choice in selection['schedules'].items():
            arm, schedule = key.split('|')
            specs.append(spec_of(arm, schedule, choice['lr'], s))
        specs.append(spec_of('topfree', 'constant', .01, s, strength=selection['topfree']['strength']))
    return units_for(specs)


def unit_config(seed):
    c = common.study_config(seed=seed)
    if c['model']['branch_scale'] != 1.0 or c['model']['blocks'] != 2:
        raise RuntimeError('Protocol: depth 2, branch scaling off')
    c['training']['steps'] = STEPS
    return c


def make_optimizer(spec, model, c):
    """As stage D's make_optimizer for SOAP and the brake (bitwise reproduction gate); `topfree` adds top_rhs."""
    soap = dict(c['soap'], lr=spec['lr'])
    core = dict(rank=16, gamma=.01, probe_every=10, refresh='extract', norm_blend=0., batched=False, signals=False, seed=spec['seed'])
    if spec['arm'] == 'soap':
        return PARSECL(model.parameters(), **soap, rank=0, norm_blend=0., batched=False, signals=False)
    settings = dict(core, calibration='lr', strength=spec['strength'])
    if spec['arm'] == 'topfree':
        settings['top_rhs'] = 'gradient'
    return PARSECL(model.parameters(), **soap, **settings)


def reference_unit(spec):
    """Stage-D unit that the constant SOAP/brake arm at lr 1e-2 must reproduce bitwise, or None."""
    if spec['schedule'] != 'constant' or spec['lr'] != .01 or spec['arm'] == 'topfree':
        return None
    seed = spec['seed']
    if spec['arm'] == 'soap':
        return (STAGE_D_RUNS['tune'] if seed == 0 else STAGE_D_RUNS['fixed'])/f'soap__lr0.01__s{seed}'
    return (STAGE_D_RUNS['tune'] if seed == 0 else STAGE_D_RUNS['tuned'])/f'brake__lr4_lr0.01__s{seed}'


def peek(order, batch_size):
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
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not finite(value):
        return None
    return value


def effective_step(optimizer, lr):
    """lr * sqrt(1 - beta2^t) / (sqrt(v) + eps) over all rotated coordinates: SOAP's per-coordinate step on m_hat."""
    values = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if state.get('step', 0) > 0 and 'exp_avg_sq' in state:
                values.append((lr*(1-group['betas'][1]**state['step'])**.5/(state['exp_avg_sq'].sqrt()+group['eps'])).reshape(-1))
    if not values:
        return None
    e = torch.cat(values)
    return {'median': float(e.median()), 'geometric_mean': float(e.log().mean().exp()), 'mean': float(e.mean())}


def top_eigenvalue(matvec, n, iterations, seed, dtype):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(n, generator=gen, dtype=dtype)
    basis, images = [], []
    for _ in range(iterations):
        for _ in range(2):
            if basis:
                b = torch.stack(basis, 1)
                q = q-b@(b.T@q)
        q = q/q.norm()
        basis.append(q)
        images.append(matvec(q))
        q = images[-1].clone()
    b, a = torch.stack(basis, 1), torch.stack(images, 1)
    t = b.T@a
    return float(torch.linalg.eigvalsh((t+t.T)/2).max())


def factor_maps(model, registry, optimizer):
    """F0 = Q diag(d^-1/2) and its transpose from the current SOAP state, d = sqrt(v) + eps."""
    params = list(model.parameters())
    states = [optimizer.state[p] for p in params]
    eps = optimizer.param_groups[0]['eps']
    roots = [(st['exp_avg_sq'].sqrt()+eps).sqrt() for st in states]

    def forward(v):
        out = torch.empty_like(v)
        for spec, st, root in zip(registry.specs, states, roots):
            out[spec.slice] = optimizer.project_back(v[spec.slice].reshape(spec.shape)/root, st).reshape(-1)
        return out

    def transpose(w):
        out = torch.empty_like(w)
        for spec, st, root in zip(registry.specs, states, roots):
            out[spec.slice] = (optimizer.project(w[spec.slice].reshape(spec.shape), st)/root).reshape(-1)
        return out

    return forward, transpose, states[0]['step']


def sharpness(model, registry, optimizer, xb, yb, lr, seed):
    """lr * lambda_max(P_hat H) with P_hat = sqrt(1 - beta2^t) Q diag(1/d) Q^T from the current SOAP state."""
    flat = registry.flatten(dict(model.named_parameters())).detach()
    fn = functional_logits(model, registry, xb)
    gradient_of = grad(lambda w: loss(fn(w), yb))
    forward, transpose, t = factor_maps(model, registry, optimizer)
    matvec = lambda v: transpose(jvp(gradient_of, (flat,), (forward(v),))[1].detach())
    return lr*(1-BETA2**t)**.5*top_eigenvalue(matvec, registry.size, LANCZOS, 11+seed, flat.dtype)


def work(directory, name):
    common.verify_archive(directory)
    spec = read(directory/'protocol.json')['units'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    seed = spec['seed']
    c = unit_config(seed)
    write_json(unit/'resolved_arm.json', {'config': c, 'spec': spec})
    x, y, manifest = dataset(c)
    if manifest != read(Path('runs/20261005T135920-d7ce29b2/split_manifest.json')):
        raise RuntimeError('Split manifest differs from the main lab split')
    model, evaluator = make_model(c), make_model(c)
    registry, evaluator_registry = Registry(model), Registry(evaluator)
    order = DataOrder(manifest['training'], common.data_order_seed(seed))
    optimizer = make_optimizer(spec, model, c)
    probe_generator = torch.Generator().manual_seed(PROBE_SEED+seed)
    analysis = manifest['analysis'][:512]
    reference_dir = reference_unit(spec)
    reference = {r['step']: r['validation_loss'] for r in read(reference_dir/'validation.json')} if reference_dir else None
    ema = {d: torch.zeros(registry.size, dtype=torch.float64) for d in EMA_DECAYS}
    seconds, gate, history, outcome, step = 0., {}, [], {'status': 'completed'}, 0
    validation = [json_safe(dict(step=0, training_seconds=0., lr=lr_at(spec['schedule'], spec['lr'], 1),
                                 **evaluate(model, x, y, manifest['validation'])))]
    try:
        for step in range(1, STEPS+1):
            lr = lr_at(spec['schedule'], spec['lr'], step)
            for group in optimizer.param_groups:
                group['lr'] = lr
            ids = order.take(BATCH)
            following = peek(order, BATCH)
            own_before = batch_loss(model, x[ids], y[ids])
            next_before = batch_loss(model, x[following], y[following])
            true = per_example_gradients(model, x[ids], y[ids]) if step % DIAGNOSE_EVERY == 0 else None
            began = time.perf_counter()
            flat = registry.flatten(dict(model.named_parameters())).detach()
            gradient = registry.unflatten(GGN(functional_logits(model, registry, x[ids]), flat, y[ids]).gradient)
            for pname, p in model.named_parameters():
                p.grad = gradient[pname].detach().clone()
            probes = ggn_probes(model, x[ids], samples=4, generator=probe_generator)[0] if optimizer.wants_probes() else None
            optimizer.step(probes=probes, true_gradients=true)
            elapsed = time.perf_counter()-began-((optimizer.last_info.get('optimism') or {}).get('seconds', 0.))
            seconds += elapsed
            after = registry.flatten(dict(model.named_parameters())).detach()
            for d in EMA_DECAYS:
                ema[d].mul_(d).add_(after, alpha=1-d)
            info = optimizer.last_info
            row = {'step': step, 'lr': lr, 'own_before': own_before, 'own_after': batch_loss(model, x[ids], y[ids]),
                   'next_before': next_before, 'next_after': batch_loss(model, x[following], y[following]),
                   'step_seconds': elapsed, 'training_seconds': seconds, 'norm': float((after-flat).norm()),
                   'correction_energy': info['correction_energy']}
            probe = info.get('probe')
            if probe and 'mu' in probe:
                row['eta_lambda_max'] = probe['eta_lambda_max']
            if info.get('optimism'):
                row['optimism'] = {k: v for k, v in info['optimism'].items() if k != 'seconds'}
            history.append(json_safe(row))
            if not finite(row['own_after']):
                outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite training-batch loss'}
                break
            if step in VALIDATION:
                measured = dict(step=step, training_seconds=seconds, lr=lr, **evaluate(model, x, y, manifest['validation']))
                for d in EMA_DECAYS:
                    commit(evaluator, evaluator_registry, ema[d]/(1-d**step))
                    measured[f'ema{d:g}'] = evaluate(evaluator, x, y, manifest['validation'])
                measured['effective_step'] = effective_step(optimizer, lr)
                measured['train_loss_last25'] = statistics.mean(r['own_before'] for r in history[-25:])
                if spec.get('sharpness') and step in SHARPNESS_STEPS:
                    measured['sharpness'] = sharpness(model, registry, optimizer, x[analysis], y[analysis], lr, seed)
                validation.append(json_safe(measured))
                write_json(unit/'validation.json', validation)
                write_compact(unit/'training.json', history)
                if reference is not None and step in reference and reference[step] is not None:
                    gate[step] = abs(measured['validation_loss']-reference[step])
                    if gate[step] > 1e-12:
                        raise RuntimeError(f'Stage-D reproduction gate failed at step {step}: {gate[step]}')
                print(f"{name} step {step}: val={measured['validation_loss']:.4f} ema.98={measured['ema0.98']['validation_loss']:.4f} "
                      f"lr={lr:.2e} train_s={seconds:.1f}" + (f" S={measured['sharpness']:.2f}" if 'sharpness' in measured else ''), flush=True)
                if not finite(measured['validation_loss']):
                    outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite validation loss'}
                    break
    except Exception as error:
        write_json(unit/'failure.json', {'unit': name, 'step': step, 'error': repr(error), 'traceback': traceback.format_exc()})
        write_json(directory/'failure.json', {'unit': name, 'step': step, 'error': repr(error)})
        raise
    write_json(unit/'validation.json', validation)
    write_compact(unit/'training.json', history)
    write_json(unit/'outcome.json', outcome)
    if reference is not None:
        write_json(unit/'reproduction_gate.json', {'reference': str(reference_dir), 'differences': gate,
                                                   'max_difference': max(gate.values()) if gate else None})
    write_json(unit/'runtime.json', {'training_seconds': seconds, 'process_peak_rss_bytes': common.peak_rss_bytes()})
    common.finish_unit(unit, {'outcome': outcome, 'final_validation_loss': validation[-1]['validation_loss'], 'training_seconds': seconds})
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
    protocol = {'study': 'noise_controls_20261011', 'phase': phase, 'document': str(common.PROTOCOL), 'units': units,
                'steps': STEPS, 'validation_steps': VALIDATION, 'ema_decays': EMA_DECAYS, **extra}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'queue_runner.sh')]))


def pending(directory):
    units = read(directory/'protocol.json')['units']
    left = [n for n in units if not common.unit_complete(directory/n)]
    return sorted(left, key=lambda n: (COST[units[n]['arm']], not units[n].get('sharpness', False), n))


def select_settings(directory):
    """Seed-0 choices (PROTOCOL.md section 3): peaks per (arm, schedule), top-free strength, EMA decay."""
    units = read(directory/'protocol.json')['units']
    final = lambda n: {r['step']: r for r in read(directory/n/'validation.json')}.get(STEPS)
    loss500 = lambda n: (final(n) or {}).get('validation_loss') if finite((final(n) or {}).get('validation_loss')) else float('inf')
    schedules = {}
    for arm in ['soap', 'brake']:
        for schedule in ['cosine', 'wsd']:
            names = [n for n, s in units.items() if s['arm'] == arm and s['schedule'] == schedule]
            best = min(names, key=loss500)
            schedules[f'{arm}|{schedule}'] = {'lr': units[best]['lr'], 'unit': best, 'loss500': loss500(best),
                                              'grid': {units[n]['lr']: loss500(n) for n in names}}
    topfree = [n for n, s in units.items() if s['arm'] == 'topfree']
    best = min(topfree, key=loss500)
    soap_constant = name_of(spec_of('soap', 'constant', .01, 0))
    ema = {d: final(soap_constant)[f'ema{d:g}']['validation_loss'] for d in EMA_DECAYS}
    return {'schedules': schedules,
            'topfree': {'strength': units[best]['strength'], 'unit': best, 'grid': {units[n]['strength']: loss500(n) for n in topfree}},
            'ema_decay': min(ema, key=ema.get), 'ema_grid_soap_constant': ema}


def finalize(directory):
    missing = pending(directory)
    if missing or (directory/'failure.json').exists():
        raise RuntimeError(f'Incomplete or failed units: {missing}')
    protocol = read(directory/'protocol.json')
    if protocol['phase'] == 'tune':
        selection = select_settings(directory)
        write_json(directory/'selection.json', selection)
        print(selection)
    write_json(directory/'complete.json', {'completed': True, 'phase': protocol['phase'], 'units': len(protocol['units'])})
    print(f'mirrored to {common.mirror(directory)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'finalize'])
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
    else:
        finalize(args.run)


if __name__ == '__main__':
    sys.exit(main())
