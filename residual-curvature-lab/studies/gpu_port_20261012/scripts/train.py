"""GPU-port runner (PROTOCOL.md, sections 4-6): the top-free confirmation's MNIST workload with an implementation,
device and dtype switch; full states in the windows for the one-step replay (G3, R1); the replay itself.

The training loop, gradient route and optimizer settings are those of studies/damping_sweep_20261011/scripts/train.py.
The optimism and sharpness diagnostics are dropped; they never changed a trajectory.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  train.py create | pending --run DIR --device cpu|cuda | work --run DIR --unit NAME
           | replay --run DIR --unit NAME --device cpu|cuda | finalize --run DIR
"""
import argparse
import copy
import statistics
import sys
import time
import traceback
from pathlib import Path

import common
from common import read, write_compact

import torch  # noqa: E402

import gpu_parsec  # noqa: E402
import parsec_l  # noqa: E402
from residual_curvature.curvature.operators import GGN  # noqa: E402
from residual_curvature.data import DataOrder, dataset  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry, make_model  # noqa: E402
from residual_curvature.models.mlp import functional_logits  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402
from run_exact_training_followup import evaluate  # noqa: E402

STEPS, BATCH, PROBE_SEED, LR, STRENGTH = 500, 512, 9000, 1e-2, 64.
VALIDATION = [0, 10, *range(25, STEPS+1, 25)]
EMA_DECAYS = [.9, .98, .995]
WINDOWS = [(50, 62), (250, 262), (488, 500)]
SAVED = sorted({t for a, b in WINDOWS for t in range(a, b+1)})
ARMS = ('soap', 'topfree')
CONFIGURATIONS = {  # name: (implementation, device, dtype, seeds, repeat of)
    'cpu64orig': ('original', 'cpu', 'float64', [0], None),
    'cpu64': ('copy', 'cpu', 'float64', [0, 4, 5, 6], None),
    'gpu64': ('copy', 'cuda', 'float64', [0], None),
    'gpu64rep': ('copy', 'cuda', 'float64', [0], 'gpu64'),
    'gpu32': ('copy', 'cuda', 'float32', [0, 4, 5, 6], None),
    'gpu32rep': ('copy', 'cuda', 'float32', [0], 'gpu32'),
}
SPLIT_MANIFEST = Path('runs/20261005T135920-d7ce29b2/split_manifest.json')


def name_of(arm, configuration, seed):
    return f'{arm}__{configuration}__s{seed}'


def units():
    out = {}
    for arm in ARMS:
        for configuration, (implementation, device, dtype, seeds, repeat) in CONFIGURATIONS.items():
            for seed in seeds:
                out[name_of(arm, configuration, seed)] = {
                    'arm': arm, 'configuration': configuration, 'implementation': implementation, 'device': device,
                    'dtype': dtype, 'seed': seed, 'windows': configuration == 'cpu64' and seed == 0,
                    'repeat_of': name_of(arm, repeat, seed) if repeat else None}
    return out


def unit_config(seed):
    c = common.study_config(seed=seed)
    if c['model']['branch_scale'] != 1.0 or c['model']['blocks'] != 2 or c['dtype'] != 'float64':
        raise RuntimeError('Protocol: depth 2, branch scaling off, float64 initialization')
    c['training']['steps'] = STEPS
    return c


def implementation(name):
    return gpu_parsec if name == 'copy' else parsec_l


def make_optimizer(spec, model, c):
    """The top-free confirmation's `soap` and `topfree` (c = 64) settings."""
    PARSECL = implementation(spec['implementation']).PARSECL
    soap = dict(c['soap'], lr=LR)
    if spec['arm'] == 'soap':
        return PARSECL(model.parameters(), **soap, rank=0, norm_blend=0., batched=False, signals=False)
    return PARSECL(model.parameters(), **soap, rank=16, gamma=.01, probe_every=10, refresh='extract', norm_blend=0.,
                   batched=False, signals=False, seed=spec['seed'], calibration='lr', strength=STRENGTH,
                   top_rhs='gradient')


def build(spec):
    """Data and models on the unit's device and dtype. Float32 rounds the float64 data and initialization."""
    c = unit_config(spec['seed'])
    x, y, manifest = dataset(c)
    if manifest != read(SPLIT_MANIFEST):
        raise RuntimeError('Split manifest differs from the main lab split')
    device, dtype = torch.device(spec['device']), getattr(torch, spec['dtype'])
    x, y = x.to(device=device, dtype=dtype), y.to(device)
    model = make_model(c).to(device=device, dtype=dtype)
    evaluator = make_model(c).to(device=device, dtype=dtype)
    return c, x, y, manifest, model, evaluator


def train_step(model, registry, optimizer, probe_fn, probe_generator, x, y, ids):
    """One training step exactly as in the earlier runners: GGN-route gradient, s = 4 probes when scheduled."""
    flat = registry.flatten(dict(model.named_parameters())).detach()
    gradient = registry.unflatten(GGN(functional_logits(model, registry, x[ids]), flat, y[ids]).gradient)
    for pname, p in model.named_parameters():
        p.grad = gradient[pname].detach().clone()
    probes = probe_fn(model, x[ids], samples=4, generator=probe_generator)[0] if optimizer.wants_probes() else None
    optimizer.step(probes=probes)
    return flat


def batch_loss(model, x, y):
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(model(x), y))


def finite(value):
    return value is not None and value == value and abs(value) != float('inf')


def flat_cpu(model, registry):
    return registry.flatten(dict(model.named_parameters())).detach().cpu().clone()


def save_state(path, step, model, registry, optimizer, ema, order, probe_generator):
    torch.save({'step': step, 'parameters': flat_cpu(model, registry), 'optimizer': optimizer.state_dict(),
                'ema': {d: v.detach().cpu().clone() for d, v in ema.items()}, 'order': order.state_dict(),
                'probe_generator': probe_generator.get_state()}, path)


def work(directory, name):
    common.verify_archive(directory)
    spec = read(directory/'protocol.json')['units'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    device = torch.device(spec['device'])
    c, x, y, manifest, model, evaluator = build(spec)
    write_json(unit/'resolved_arm.json', {'config': c, 'spec': spec, 'environment': common.environment()})
    registry, evaluator_registry = Registry(model), Registry(evaluator)
    order = DataOrder(manifest['training'], common.data_order_seed(spec['seed']))
    optimizer = make_optimizer(spec, model, c)
    probe_fn = implementation(spec['implementation']).ggn_probes
    probe_generator = torch.Generator().manual_seed(PROBE_SEED+spec['seed'])
    ema = {d: torch.zeros(registry.size, dtype=model.head.weight.dtype, device=device) for d in EMA_DECAYS}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    seconds, history, outcome, step = 0., [], {'status': 'completed'}, 0
    validation = [dict(step=0, training_seconds=0., lr=LR, **evaluate(model, x, y, manifest['validation']))]
    if spec['windows']:
        (unit/'states').mkdir()
    try:
        for step in range(1, STEPS+1):
            for group in optimizer.param_groups:
                group['lr'] = LR
            ids = order.take(BATCH)
            own_before = batch_loss(model, x[ids], y[ids])
            common.synchronize(device)
            began = time.perf_counter()
            flat = train_step(model, registry, optimizer, probe_fn, probe_generator, x, y, ids)
            common.synchronize(device)
            elapsed = time.perf_counter()-began
            seconds += elapsed
            after = registry.flatten(dict(model.named_parameters())).detach()
            for d in EMA_DECAYS:
                ema[d].mul_(d).add_(after, alpha=1-d)
            if spec['windows'] and step in SAVED:
                save_state(unit/'states'/f'step-{step:04d}.pt', step, model, registry, optimizer, ema, order, probe_generator)
            info = optimizer.last_info
            history.append({'step': step, 'own_before': own_before, 'step_seconds': elapsed, 'training_seconds': seconds,
                            'norm': float((after-flat).norm()), 'probe': info.get('probe') is not None,
                            'correction_energy': info['correction_energy']})
            if not finite(own_before):
                outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite training-batch loss'}
                break
            if step in VALIDATION:
                measured = dict(step=step, training_seconds=seconds, lr=LR, **evaluate(model, x, y, manifest['validation']))
                for d in EMA_DECAYS:
                    commit(evaluator, evaluator_registry, ema[d]/(1-d**step))
                    measured[f'ema{d:g}'] = evaluate(evaluator, x, y, manifest['validation'])
                measured['train_loss_last25'] = statistics.mean(r['own_before'] for r in history[-25:])
                validation.append(measured)
                write_json(unit/'validation.json', validation)
                write_compact(unit/'training.json', history)
                print(f"{name} step {step}: val={measured['validation_loss']:.6f} "
                      f"ema.98={measured['ema0.98']['validation_loss']:.6f} train_s={seconds:.1f}", flush=True)
                if not finite(measured['validation_loss']):
                    outcome = {'status': 'diverged', 'step': step, 'reason': 'nonfinite validation loss'}
                    break
    except Exception as error:
        write_json(unit/'failure.json', {'unit': name, 'step': step, 'error': repr(error), 'traceback': traceback.format_exc()})
        write_json(directory/'failure.json', {'unit': name, 'step': step, 'error': repr(error)})
        raise
    torch.save(flat_cpu(model, registry), unit/'final_parameters.pt')
    write_json(unit/'validation.json', validation)
    write_compact(unit/'training.json', history)
    write_json(unit/'outcome.json', outcome)
    runtime = {'training_seconds': seconds, 'process_peak_rss_bytes': common.peak_rss_bytes()}
    if device.type == 'cuda':
        runtime['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    write_json(unit/'runtime.json', runtime)
    common.finish_unit(unit, {'outcome': outcome, 'final_validation_loss': validation[-1]['validation_loss'],
                              'training_seconds': seconds})
    print(f"{name}: {outcome['status']}", flush=True)


def state_differences(mine, reference):
    """Largest absolute difference per state kind (SOAP bases Q, moments) between two optimizer state dicts of the
    same parameters; mine may live on another device. The held PARSEC-L factor is compared through its operator
    M = W~^T W~, which is invariant to the sign and rotation conventions of the SVD: largest absolute and relative
    Frobenius difference over tensors, with the tensor index of the largest relative one."""
    out = {}
    for pid, state in reference['state'].items():
        other = mine['state'][pid]
        pairs = [('Q', q, o) for q, o in zip(state['Q'], other['Q']) if len(q)]
        pairs += [(k, state[k], other[k]) for k in ('exp_avg', 'exp_avg_sq')]
        for kind, a, b in pairs:
            out[kind] = max(out.get(kind, 0.), float((a-b.cpu()).abs().max()))
        held = state.get('parsec', {}).get('wt')
        if held is not None:
            a, b = held.reshape(len(held), -1), other['parsec']['wt'].cpu().reshape(len(held), -1)
            difference, scale = float((a.T@a-b.T@b).norm()), float((a.T@a).norm())
            relative = difference/scale if scale > 0 else (0. if difference == 0 else float('inf'))
            out['held_operator_abs'] = max(out.get('held_operator_abs', 0.), difference)
            if relative >= out.get('held_operator_rel', -1.):
                out['held_operator_rel'], out['held_operator_rel_tensor'] = relative, pid
    return out


def replay_step(spec, c, model, registry, x, y, before, after):
    """Load the saved state `before` into the model and a fresh optimizer on the model's device, take the next
    step, and compare it with the saved state `after`. The state is copied first: load_state_dict keeps tensors
    that need no cast, and the step would then change `before` in place."""
    commit(model, registry, before['parameters'].to(model.head.weight.device))
    optimizer = make_optimizer(spec, model, c)
    optimizer.load_state_dict(copy.deepcopy(before['optimizer']))
    order = DataOrder([0], 0)
    order.load_state_dict(before['order'])
    probe_generator = torch.Generator()
    probe_generator.set_state(before['probe_generator'])
    bases = {id(p): [q.clone() if len(q) else q for q in optimizer.state[p]['Q']] for p in model.parameters()}
    ids = order.take(BATCH)
    train_step(model, registry, optimizer, implementation(spec['implementation']).ggn_probes, probe_generator, x, y, ids)
    expected = after['parameters']-before['parameters']
    got = flat_cpu(model, registry)-before['parameters']
    refreshed = any(len(a) and not torch.equal(a, b) for p in model.parameters()
                    for a, b in zip(bases[id(p)], optimizer.state[p]['Q']))
    return {'probe': optimizer.last_info['probe'] is not None, 'soap_refresh': refreshed,
            'relative_update_error': float((got-expected).norm()/expected.norm()),
            'state_max_abs_difference': state_differences(optimizer.state_dict(), after['optimizer'])}


def verify_amended(directory):
    """verify_archive, except that a file listed in the run's amendments.json (DEVIATIONS.md) must match its
    amended checksum and its archived amended copy instead of the original archive."""
    amended = read(directory/'amendments.json')['files'] if (directory/'amendments.json').exists() else {}
    sources = read(directory/'source_archive_checksums.json')
    for name, digest in sources.items():
        expected, copy_ = (amended[name]['sha256'], directory/'source_amended'/name) if name in amended else (digest, directory/'source'/name)
        if common.checksum(name) != expected or common.checksum(copy_) != expected:
            raise RuntimeError(f'Source changed since the run was created: {name}')
    missing = [name for name in common.loaded_sources() if name not in sources]
    if missing:
        raise RuntimeError(f'Loaded sources missing from the run archive: {missing}')


def replay(directory, name, device):
    """G3 (device cpu) and R1 (device cuda): one step from every saved state whose successor is saved."""
    spec = read(directory/'protocol.json')['units'][name]
    if not spec['windows'] or spec['implementation'] != 'copy' or spec['dtype'] != 'float64':
        raise RuntimeError('Replay needs the float64 copy unit with saved windows')
    if not common.unit_complete(directory/name):
        raise RuntimeError(f'{name} is not complete')
    verify_amended(directory)
    replayed = dict(spec, device=device)
    c, x, y, manifest, model, _ = build(replayed)
    registry = Registry(model)
    rows = []
    for t in SAVED:
        if t+1 not in SAVED:
            continue
        before = torch.load(directory/name/'states'/f'step-{t:04d}.pt', weights_only=False)
        after = torch.load(directory/name/'states'/f'step-{t+1:04d}.pt', weights_only=False)
        rows.append(dict(step=t+1, **replay_step(replayed, c, model, registry, x, y, before, after)))
        print(f"{name} {device} step {t+1}: rel={rows[-1]['relative_update_error']:.3e} probe={rows[-1]['probe']} "
              f"refresh={rows[-1]['soap_refresh']}", flush=True)
    write_json(directory/name/f'replay_{device}.json', {'unit': name, 'device': device,
                                                        'environment': common.environment(), 'rows': rows})


def create():
    c = common.study_config()
    protocol = {'study': 'gpu_port_20261012', 'phase': 'port', 'document': str(common.PROTOCOL), 'units': units(),
                'steps': STEPS, 'validation_steps': VALIDATION, 'ema_decays': EMA_DECAYS, 'windows': WINDOWS,
                'environment': common.environment()}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/'run_all.sh')]))


def pending(directory, device):
    specs = read(directory/'protocol.json')['units']
    left = [n for n, s in specs.items() if s['device'] == device and not common.unit_complete(directory/n)]
    return sorted(left, key=lambda n: (not specs[n]['windows'], specs[n]['repeat_of'] is not None, n))


def finalize(directory):
    specs = read(directory/'protocol.json')['units']
    missing = [n for n in specs if not common.unit_complete(directory/n)]
    missing += [f'{n} replay {d}' for n, s in specs.items() if s['windows'] for d in ('cpu', 'cuda')
                if not (directory/n/f'replay_{d}.json').exists()]
    if missing or (directory/'failure.json').exists():
        raise RuntimeError(f'Incomplete or failed units: {missing}')
    write_json(directory/'complete.json', {'completed': True, 'phase': 'port', 'units': len(specs)})
    print(f'mirrored to {common.mirror(directory)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'pending', 'work', 'replay', 'finalize'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--unit')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        create()
    elif args.command == 'pending':
        print('\n'.join(pending(args.run, args.device)))
    elif args.command == 'work':
        work(args.run, args.unit)
    elif args.command == 'replay':
        replay(args.run, args.unit, args.device)
    else:
        finalize(args.run)


if __name__ == '__main__':
    sys.exit(main())
