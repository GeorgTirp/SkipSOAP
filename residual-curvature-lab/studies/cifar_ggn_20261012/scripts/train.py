"""CIFAR-10 runner (PROTOCOL.md): tuned SOAP, top-free PARSEC-L, and exact per-tensor GGN with and without SN on the
depth-6 ViT M1d6, batch 2048, GPU float32; distance preservation and quadratic-model fidelity along the steps taken.

Arms:
  soap     SOAP as PARSEC-L with rank 0 (the lab's `soap` arm), peak lr from the tuning grid;
  topfree  PARSEC-L, rank 16, s = 1 probe per example every 10 steps (lean_probes), learning-rate calibration c = 64,
           top_rhs='gradient', norm blend 0, per-tensor extraction;
  ggn      stage D's exact per-tensor GGN step (gamma 1e-2, not tuned; owner decision): damping mu = max(gamma tr C / D, 1e-8) in the normalized inverse
           metric of a shadow SOAP, current-gradient RHS, step length chosen on a second (selection) batch by the
           lab's broad direction search;
  ggn_sn   ggn with SN (c = 1) on the residual-branch weights, same gamma.
Schedules: soap and topfree use linear warm-up over 25 steps, then constant, then linear decay over steps 401-500
to 1% (WSD), with decoupled weight decay 0.05 (upstream SOAP). The ggn arms multiply the selected step by the same
decay factor (no warm-up), and apply decoupled weight decay at the reference lr 3e-3 with the same schedule.
Every arm tracks weight EMAs (d in 0.9, 0.98, 0.995).

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  train.py create --phase tune | create --phase extend|eval --tuning DIR [DIR] | pending --run DIR
           | work --run DIR --unit NAME | finalize --run DIR
  train.py dryrun --arm ARM --steps N
"""
import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import common
from common import read, write_compact

import torch  # noqa: E402

import cifar  # noqa: E402
import ggn  # noqa: E402
import measure  # noqa: E402
import micro  # noqa: E402
from gpu_parsec import PARSECL  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402
from residual_curvature.models.mlp import functional_logits  # noqa: E402
from residual_curvature.optim.soap import SOAPMetric, soap_propose  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402
from run_mini_vit_allocation import search_candidate  # noqa: E402

MODEL, STEPS, BATCH, MICRO = 'M1d6', 500, 2048, 512
VALIDATION = [0, 10, *range(25, STEPS+1, 25)]
MEASURE_EVERY, JACOBIAN_EVERY, CHECKPOINTS = 10, 50, [50, 100, 250, 500]
EMA_DECAYS = [.9, .98, .995]
WARMUP, DECAY_START, WEIGHT_DECAY, REFERENCE_LR = 25, 400, .05, 3e-3
SN_BOUND, STRENGTH, RANK, PROBE_EVERY, SAMPLES = 1., 64., 16, 10, 1
LR_GRID, GAMMA, GGN_EMA_DECAY = [1e-3, 3e-3, 1e-2, 3e-2], 1e-2, .98  # owner decision: no GGN tuning
GGN_BUDGET, GGN_CHUNK, FLOOR, TOLERANCE = int(2.5*2**30), 64, 1e-8, 1e-4
AUDIT_STEPS = [2, 100, 250, 500]
EVAL_SEEDS = [1, 2, 3]
SEEDS = {'order': 102, 'augment': 5000, 'selection': 7102, 'selection_augment': 6000, 'probe': 9000, 'distance': 9001}
COST = {'soap': 0, 'topfree': 1, 'ggn': 2, 'ggn_sn': 3}  # queue order: cheap arms first


def lr_factor(t, warmup=True):
    """Linear warm-up over WARMUP steps (soap, topfree), constant, then linear decay over steps 401-500 to 1%."""
    if warmup and t <= WARMUP:
        return t/WARMUP
    return 1. if t <= DECAY_START else (STEPS+1-t)/(STEPS-DECAY_START)


def name_of(spec):
    setting = f"gamma{spec['gamma']:g}" if spec['arm'].startswith('ggn') else f"lr{spec['lr']:g}"
    return f"{spec['arm']}__{setting}__s{spec['seed']}"


def spec_of(arm, seed, lr=None, gamma=None, measure_=False):
    spec = {'arm': arm, 'seed': seed, 'measure': measure_}
    spec.update({'gamma': gamma} if arm.startswith('ggn') else {'lr': lr})
    return spec


def tune_units():
    specs = [spec_of(arm, 0, lr=lr) for arm in ('soap', 'topfree') for lr in LR_GRID]
    return {name_of(s): s for s in specs}


def selection(tunings):
    """Seed-0 selection (PROTOCOL.md, section 4) over one or more tuning runs: for soap and topfree the lr and EMA
    decay with the lowest step-500 EMA loss (a diverged unit counts as worse than any finite one), and whether that
    lr lies at an edge of the searched grid. The GGN arms use gamma 1e-2 and EMA decay 0.98 (no tuning)."""
    out = {}
    for arm in ('soap', 'topfree'):
        best, grid = None, set()
        for tuning in tunings:
            for name, spec in read(tuning/'protocol.json')['units'].items():
                if spec['arm'] != arm:
                    continue
                grid.add(spec['lr'])
                final = {r['step']: r for r in read(tuning/name/'validation.json')}.get(STEPS)
                if final is None:
                    continue
                for d in EMA_DECAYS:
                    value = final[f'ema{d:g}']['validation_loss']
                    if value == value and (best is None or value < best[0]):
                        best = (value, spec['lr'], d)
        edge = None if best[1] not in (min(grid), max(grid)) else ('low' if best[1] == min(grid) else 'high')
        out[arm] = {'lr': best[1], 'ema_decay': best[2], 'value': best[0], 'grid': sorted(grid), 'edge': edge}
    out['ggn'] = {'gamma': GAMMA, 'ema_decay': GGN_EMA_DECAY, 'value': None, 'fixed': True}
    return out


def extension_units(chosen):
    """One more lr beyond a selected grid edge (x3 or /3), seed 0, per arm whose selection lies at an edge."""
    specs = []
    for arm in ('soap', 'topfree'):
        row = chosen[arm]
        if row['edge'] == 'low':
            specs.append(spec_of(arm, 0, lr=min(row['grid'])/3))
        elif row['edge'] == 'high':
            specs.append(spec_of(arm, 0, lr=max(row['grid'])*3))
    return {name_of(s): s for s in specs}


def eval_units(chosen):
    specs = []
    for seed in EVAL_SEEDS:
        specs.append(spec_of('soap', seed, lr=chosen['soap']['lr'], measure_=True))
        specs.append(spec_of('topfree', seed, lr=chosen['topfree']['lr'], measure_=True))
        for arm in ('ggn', 'ggn_sn'):
            specs.append(spec_of(arm, seed, gamma=chosen['ggn']['gamma'], measure_=True))
    return {name_of(s): s for s in specs}


class Data:
    """CIFAR-10 on the GPU as uint8; batches are normalized (and augmented for training) on demand."""

    def __init__(self, device):
        raw, labels, self.manifest = cifar.cifar10()
        self.device = device
        self.mean = torch.tensor(self.manifest['normalization']['mean']).reshape(1, 3, 1, 1).to(device)
        self.std = torch.tensor(self.manifest['normalization']['std']).reshape(1, 3, 1, 1).to(device)
        self.train = raw[self.manifest['training']].to(device)
        self.train_y = labels[self.manifest['training']].to(device)
        self.validation = self.normalized(raw[self.manifest['validation']].to(device))
        self.validation_y = labels[self.manifest['validation']].to(device)

    def normalized(self, images):
        """(uint8 / 255 - mean) / std per channel in float32 on the device, for training and validation alike."""
        return (images.float()/255-self.mean)/self.std

    def batch(self, positions, generator=None):
        """Training examples at `positions` (indices into the training split), normalized, augmented if a generator."""
        index = torch.tensor(positions, device=self.device)
        x = self.normalized(self.train[index])
        return (cifar.augment(x, generator) if generator is not None else x), self.train_y[index]

    def distance_pairs(self):
        """Fixed validation pairs: 1,000 random global pairs from the first 2,000 and 512 local perturbations."""
        gen = torch.Generator().manual_seed(SEEDS['distance'])
        i = torch.randint(0, 2000, (1000,), generator=gen)
        j = (i+1+torch.randint(0, 1999, (1000,), generator=gen)) % 2000
        local = self.validation[2000:2512]
        xi = torch.randn(local.shape, generator=gen, dtype=local.dtype).to(self.device)
        xi = xi/xi.flatten(1).norm(dim=1).reshape(-1, 1, 1, 1)
        perturbed = local+1e-3*local.flatten(1).norm(dim=1).reshape(-1, 1, 1, 1)*xi
        return {'global': (self.validation[i.to(self.device)], self.validation[j.to(self.device)]),
                'local': (local, perturbed)}


def evaluate(model, x, y, chunk=1000):
    total, correct = 0., 0
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            z = model(x[start:start+chunk])
            total += float(torch.nn.functional.cross_entropy(z, y[start:start+chunk], reduction='sum'))
            correct += int((z.argmax(-1) == y[start:start+chunk]).sum())
    return {'validation_loss': total/len(x), 'validation_accuracy': correct/len(x), 'validation_examples': len(x)}


def build_model(spec, device):
    return cifar.make_cifar_model(MODEL, spec['seed'], device, sn_bound=SN_BOUND if spec['arm'] == 'ggn_sn' else None)


def make_optimizer(spec, model):
    soap = dict(lr=spec['lr'], betas=(.95, .95), weight_decay=WEIGHT_DECAY, precondition_frequency=10)
    if spec['arm'] == 'soap':
        return PARSECL(model.parameters(), **soap, rank=0, norm_blend=0., batched=False, signals=False)
    return PARSECL(model.parameters(), **soap, rank=RANK, probe_every=PROBE_EVERY, refresh='extract', norm_blend=0.,
                   batched=False, signals=False, calibration='lr', strength=STRENGTH, top_rhs='gradient', seed=spec['seed'])


def shadow_config():
    """The shadow SOAP that supplies the GGN arms' damping metric and search radius (no weight decay: soap_propose)."""
    c = common.study_config()
    return dict(c['soap'], lr=REFERENCE_LR, betas=[.95, .95], precondition_frequency=10, weight_decay=0.)


class Arm:
    """One unit's model, optimizer state and step."""

    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self.model = build_model(spec, device)
        self.registry = Registry(self.model)
        self.ggn = spec['arm'].startswith('ggn')
        seed = spec['seed']
        if self.ggn:
            self.shadow, self.shadow_cfg = None, shadow_config()
        else:
            self.optimizer = make_optimizer(spec, self.model)
        self.probe_generator = torch.Generator().manual_seed(SEEDS['probe']+seed)
        self.info = {}

    def flat(self):
        return self.registry.flatten(dict(self.model.named_parameters())).detach()

    def step(self, t, x, y, selection_batch):
        if self.ggn:
            return self.ggn_step(t, x, y, selection_batch)
        factor = lr_factor(t)
        for group in self.optimizer.param_groups:
            group['lr'] = self.spec['lr']*factor
        self.optimizer.zero_grad(set_to_none=True)
        total, value = len(x), 0.
        for part in micro.parts(total, MICRO):
            part_loss = torch.nn.functional.cross_entropy(self.model(x[part]), y[part])*((part.stop-part.start)/total)
            part_loss.backward()
            value += float(part_loss)
        probes = None
        if self.spec['arm'] == 'topfree' and self.optimizer.wants_probes():
            probes = micro.lean_probes(self.model, x, samples=SAMPLES, generator=self.probe_generator)
        self.optimizer.step(probes=probes)
        del probes
        info = self.optimizer.last_info
        self.info = {'loss': value, 'lr': self.spec['lr']*factor, 'probe': info.get('probe') is not None,
                     'correction_energy': info.get('correction_energy')}
        return self.info

    def ggn_step(self, t, x, y, selection_batch):
        model, registry = self.model, self.registry
        flat = self.flat()
        value, g = micro.gradient(model, registry, flat, x, y, MICRO)
        proposal, self.shadow, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), self.shadow, self.shadow_cfg)
        native = registry.flatten(proposal)
        delta, solve, search = native, None, None
        if all(v is not None for v in contexts.values()):
            metric = SOAPMetric(registry, contexts)
            audit = t in AUDIT_STEPS
            delta, solve = ggn.tensor_ggn_step(
                model, registry, flat, x, g, metric, self.spec['gamma'], FLOOR, TOLERANCE, GGN_BUDGET, GGN_CHUNK,
                action=micro.ggn_action(model, registry, flat, x, MICRO) if audit else None,
                audit_generator=torch.Generator().manual_seed(303+t) if audit else None)
        xs, ys = selection_batch
        if float(native.norm()) > 0:
            # One no-grad forward pass of the whole selection batch per evaluation (no activations are kept).
            delta, search = search_candidate(flat, delta, native, lambda w: micro.batch_loss(model, registry, w, xs, ys, BATCH))
        factor = lr_factor(t, warmup=False)
        new = flat+factor*delta
        new = new-REFERENCE_LR*lr_factor(t)*WEIGHT_DECAY*new
        commit(model, registry, new)
        self.info = {'loss': value, 'decay_factor': factor, 'native_norm': float(native.norm()),
                     'mu': solve and solve['mu'], 'trace': solve and solve['trace'], 'solve_residual': solve and solve['residual'],
                     'solve_residual_original': solve and solve['residual_original'],
                     'kappa_max': max(solve['kappa_max'].values()) if solve and solve['kappa_max'] else None,
                     'solve_seconds': solve and solve['seconds'], 'groups': solve and solve['groups'],
                     'max_action_check': max(solve['action_checks'].values()) if solve and solve['action_checks'] else None,
                     'alpha': search and search['selected_alpha'], 'search_evaluations': search and search['evaluations'],
                     'search_bracketed': search and search['bracketed']}
        return self.info

    def state(self):
        return {'shadow': self.shadow} if self.ggn else {'optimizer': self.optimizer.state_dict()}


def peek(order, size):
    twin = DataOrder([0], 0)
    twin.load_state_dict({k: v for k, v in order.state_dict().items()})
    return twin.take(size)


def finite(value):
    return value is not None and value == value and abs(value) not in (float('inf'),)


def work(directory, name):
    common.verify_archive(directory)
    spec = read(directory/'protocol.json')['units'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    write_json(unit/'resolved_arm.json', {'spec': spec, 'environment': common.environment()})
    run(spec, unit, STEPS)


def run(spec, unit, steps):
    device = torch.device('cuda')
    data = Data(device)
    arm = Arm(spec, device)
    seed = spec['seed']
    order = DataOrder(list(range(len(data.manifest['training']))), SEEDS['order']+1000*seed)
    augment = torch.Generator().manual_seed(SEEDS['augment']+seed)
    selection_order = DataOrder(list(range(len(data.manifest['training']))), SEEDS['selection']+1000*seed)
    selection_augment = torch.Generator().manual_seed(SEEDS['selection_augment']+seed)
    evaluator = build_model(spec, device)
    evaluator_registry = Registry(evaluator)
    pairs = data.distance_pairs() if spec.get('measure') else None
    jacobian_generator = torch.Generator().manual_seed(SEEDS['distance']+1)
    ema = {d: torch.zeros(arm.registry.size, device=device) for d in EMA_DECAYS}
    torch.cuda.reset_peak_memory_stats()
    seconds, history, measurements, outcome, t = 0., [], [], {'status': 'completed'}, 0
    validation = [dict(step=0, training_seconds=0., **evaluate(arm.model, data.validation, data.validation_y))]
    try:
        for t in range(1, steps+1):
            ids = order.take(BATCH)
            following = peek(order, BATCH) if spec.get('measure') and t % MEASURE_EVERY == 0 else None
            x, y = data.batch(ids, augment)
            selection_batch = data.batch(selection_order.take(BATCH), selection_augment) if arm.ggn else None
            before = arm.flat().clone()
            torch.cuda.synchronize()
            began = time.perf_counter()
            info = arm.step(t, x, y, selection_batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-began
            seconds += elapsed
            after = arm.flat()
            for d in EMA_DECAYS:
                ema[d].mul_(d).add_(after, alpha=1-d)
            change = after-before
            live = torch.cat([change[spec.slice] for spec in arm.registry.specs if not spec.name.endswith('attention.key.bias')])
            history.append({'step': t, 'step_seconds': elapsed, 'training_seconds': seconds, 'norm': float(change.norm()),
                            'norm_without_key_bias': float(live.norm()), **info})
            if not finite(info['loss']) or not torch.isfinite(after).all():
                outcome = {'status': 'diverged', 'step': t, 'reason': 'nonfinite loss or parameters'}
                break
            if following is not None:
                started = time.perf_counter()
                xn, yn = data.batch(following)
                row = {'step': t, 'own': measure.quadratic_check(arm.model, arm.registry, before, after-before, x, y, MICRO),
                       'next': measure.quadratic_check(arm.model, arm.registry, before, after-before, xn, yn, MICRO),
                       'distance': measure.distance_preservation(arm.model, pairs),
                       'spectral': measure.spectral_report(arm.model)}
                if t % JACOBIAN_EVERY == 0:
                    row['block_jacobian'] = measure.block_jacobian_deviation(arm.model, pairs['local'][0], jacobian_generator)
                row['seconds'] = time.perf_counter()-started
                measurements.append(row)
            if t in CHECKPOINTS and spec.get('measure'):
                (unit/'checkpoints').mkdir(exist_ok=True)
                torch.save({'step': t, 'parameters': after.cpu(), 'ema': {d: v.cpu() for d, v in ema.items()}, **arm.state()},
                           unit/'checkpoints'/f'step-{t:04d}.pt')
            if t in VALIDATION:
                measured = dict(step=t, training_seconds=seconds, **evaluate(arm.model, data.validation, data.validation_y))
                for d in EMA_DECAYS:
                    commit(evaluator, evaluator_registry, ema[d]/(1-d**t))
                    measured[f'ema{d:g}'] = evaluate(evaluator, data.validation, data.validation_y)
                measured['train_loss_last25'] = statistics.mean(r['loss'] for r in history[-25:])
                validation.append(measured)
                write_json(unit/'validation.json', validation)
                write_compact(unit/'training.json', history)
                write_compact(unit/'measurements.json', measurements)
                print(f"{unit.name} step {t}: val={measured['validation_loss']:.4f} acc={measured['validation_accuracy']:.4f} "
                      f"ema.98={measured['ema0.98']['validation_loss']:.4f} train_s={seconds:.1f}", flush=True)
                if not finite(measured['validation_loss']):
                    outcome = {'status': 'diverged', 'step': t, 'reason': 'nonfinite validation loss'}
                    break
    except Exception as error:
        write_json(unit/'failure.json', {'unit': unit.name, 'step': t, 'error': repr(error), 'traceback': traceback.format_exc()})
        write_json(unit.parent/'failure.json', {'unit': unit.name, 'step': t, 'error': repr(error)})
        raise
    write_json(unit/'validation.json', validation)
    write_compact(unit/'training.json', history)
    write_compact(unit/'measurements.json', measurements)
    write_json(unit/'outcome.json', outcome)
    write_json(unit/'runtime.json', {'training_seconds': seconds, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                     'process_peak_rss_bytes': common.peak_rss_bytes()})
    common.finish_unit(unit, {'outcome': outcome, 'final_validation_loss': validation[-1]['validation_loss'],
                              'training_seconds': seconds})
    print(f"{unit.name}: {outcome['status']}", flush=True)


def create(phase, tunings=()):
    c = common.study_config()
    if phase == 'tune':
        units, chosen = tune_units(), None
    elif phase == 'extend':
        chosen = selection(tunings)
        units = extension_units(chosen)
        if not units:
            print('none')
            return
    else:
        chosen = selection(tunings)
        if any(chosen[arm]['edge'] for arm in ('soap', 'topfree')) and len(tunings) < 2:
            raise RuntimeError(f'Selected lr at a grid edge; run the extension phase first: {chosen}')
        units = eval_units(chosen)
    protocol = {'study': 'cifar_ggn_20261012', 'phase': phase, 'document': str(common.PROTOCOL), 'units': units,
                'tuning_runs': [str(t) for t in tunings], 'chosen': chosen, 'model': MODEL, 'steps': STEPS,
                'batch': BATCH, 'validation_steps': VALIDATION, 'ema_decays': EMA_DECAYS, 'environment': common.environment()}
    print(common.create_run(c, protocol, [str(common.STUDY_REL/'scripts'/f) for f in ('run_queue.sh', 'orchestrate.sh')]))


def pending(directory):
    units = read(directory/'protocol.json')['units']
    left = [n for n in units if not common.unit_complete(directory/n)]
    return sorted(left, key=lambda n: (COST[units[n]['arm']], n))


def finalize(directory):
    missing = pending(directory)
    if missing or (directory/'failure.json').exists():
        raise RuntimeError(f'Incomplete or failed units: {missing}')
    protocol = read(directory/'protocol.json')
    write_json(directory/'complete.json', {'completed': True, 'phase': protocol['phase'], 'units': len(protocol['units'])})
    print(f'mirrored to {common.mirror(directory)}')


def dryrun(arm, steps, measure_):
    """A few steps of one arm (seed 0) into a scratch unit under $CIFAR_GGN_DATA/dryrun, for the cost table."""
    spec = spec_of(arm, 0, lr=3e-3, gamma=1e-2, measure_=measure_)
    unit = common.DATA/'dryrun'/f"{name_of(spec)}{'__measure' if measure_ else ''}"
    common.prepare_unit(unit)
    run(spec, unit, steps)
    history = json.loads((unit/'training.json').read_text())
    runtime = read(unit/'runtime.json')
    m = json.loads((unit/'measurements.json').read_text())
    print(json.dumps({'arm': arm, 'steps': steps, 'median_step_seconds': statistics.median(r['step_seconds'] for r in history[1:]),
                      'max_step_seconds': max(r['step_seconds'] for r in history), 'gpu_peak_GiB': runtime['gpu_peak_allocated_bytes']/2**30,
                      'measurement_seconds': [r['seconds'] for r in m]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'pending', 'work', 'finalize', 'dryrun'])
    parser.add_argument('--phase', choices=['tune', 'extend', 'eval'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--tuning', type=Path, nargs='*', default=[])
    parser.add_argument('--unit')
    parser.add_argument('--arm', choices=list(COST))
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--measure', action='store_true')
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        create(args.phase, args.tuning)
    elif args.command == 'pending':
        print('\n'.join(pending(args.run)))
    elif args.command == 'work':
        work(args.run, args.unit)
    elif args.command == 'finalize':
        finalize(args.run)
    else:
        dryrun(args.arm, args.steps, args.measure)


if __name__ == '__main__':
    sys.exit(main())
