"""Damping-sweep runner (PROTOCOL.md part 1): top-free brake at c in {256, 1024, 4096, 1e9}, seeds 4-6,
plus the c = 64 seed-4 reproduction gate. Copied from studies/topfree_confirm_20261011/scripts/train.py.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  train.py create --phase sweep | work --run DIR --unit NAME | pending --run DIR | finalize --run DIR
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
SWEEP_STRENGTHS, EMA_DECAYS, EVAL_SEEDS, CHECKPOINT_EVERY = [256., 1024., 4096., 1e9], [.9, .98, .995], [4, 5, 6], 10
SHARPNESS_STEPS, LANCZOS = [10, 25, 50, 100, 200, 300, 400, 500], 30
BETA2 = .95
CONFIRM_TUNED = Path.home()/'skipsoap-study-data'/'topfree_confirm_20261011'/'runs'/'20261011T140034-192dd041'
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


def spec_of(arm, schedule, lr, seed, strength=None, sharpness=False, checkpoints=False):
    spec = {'arm': arm, 'schedule': schedule, 'lr': lr, 'seed': seed, 'sharpness': sharpness, 'checkpoints': checkpoints}
    if arm != 'soap':
        spec['strength'] = 4. if strength is None else strength
    return spec


def units_for(specs):
    return {name_of(s): s for s in specs}


def sweep_units():
    specs = [spec_of('topfree', 'constant', .01, s, strength=c, sharpness=True) for s in EVAL_SEEDS for c in SWEEP_STRENGTHS]
    specs.append(spec_of('topfree', 'constant', .01, 4, strength=64.))
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
    """The existing c = 64 seed-4 unit of topfree_confirm, which the rerun must reproduce bitwise, or None."""
    if spec['arm'] == 'topfree' and spec['strength'] == 64. and spec['seed'] == 4:
        return CONFIRM_TUNED/name_of(spec)
    return None


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
            if spec.get('checkpoints') and step % CHECKPOINT_EVERY == 0:
                (unit/'checkpoints').mkdir(exist_ok=True)
                torch.save({'step': step, 'parameters': after.clone(), 'optimizer': optimizer.state_dict(), 'lr': lr,
                            'ema': {d: (ema[d]/(1-d**step)).clone() for d in EMA_DECAYS}}, unit/'checkpoints'/f'step-{step:04d}.pt')
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
                        raise RuntimeError(f'Confirmation reproduction gate failed at step {step}: {gate[step]}')
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
    protocol = {'study': 'damping_sweep_20261011', 'phase': 'sweep', 'document': str(common.PROTOCOL), 'units': sweep_units(),
                'steps': STEPS, 'validation_steps': VALIDATION, 'ema_decays': EMA_DECAYS}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'queue_runner.sh')]))


def pending(directory):
    units = read(directory/'protocol.json')['units']
    left = [n for n in units if not common.unit_complete(directory/n)]
    return sorted(left, key=lambda n: (COST[units[n]['arm']], not units[n].get('sharpness', False), n))


def finalize(directory):
    missing = pending(directory)
    if missing or (directory/'failure.json').exists():
        raise RuntimeError(f'Incomplete or failed units: {missing}')
    protocol = read(directory/'protocol.json')
    write_json(directory/'complete.json', {'completed': True, 'phase': protocol['phase'], 'units': len(protocol['units'])})
    print(f'mirrored to {common.mirror(directory)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'finalize'])
    parser.add_argument('--phase', choices=['sweep'])
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
