"""CIFAR-10 runner (PROTOCOL.md): SOAP with and without SN (c = 1) on the depth-6 ViT M1d6, batch 2048, GPU float32;
distance preservation and quadratic-model fidelity along the steps taken.

Derived from studies/cifar_ggn_20261012/scripts/train.py. The `soap` arm, the data path, the evaluation and the
diagnostics are that study's code unchanged (the model, measurement and microbatch modules are imported from it), so
its `soap` units reproduce that study's `soap` units bitwise in every field that study recorded (gate G1). The GGN and
top-free arms are removed. Each diagnostic row adds measures A-D (`nonlinear.py`) under the key 'nonlinear'.

Arms:
  soap     SOAP as PARSEC-L with rank 0 (the lab's `soap` arm), lr 1e-2 and EMA decay 0.98 as selected on seed 0 in
           cifar_ggn_20261012 (not retuned);
  soap_sn  the same SOAP on CifarViTSN: W_eff = W min(1, c/sigma(W)), c = 1, on the 36 residual-branch weights; the
           optimizer acts on the raw weights; peak lr tuned on seed 0 over the same grid.
Schedules (both arms): linear warm-up over 25 steps, constant, linear decay over steps 401-500 to 1% (WSD), decoupled
weight decay 0.05 (upstream SOAP), weight EMAs d in 0.9, 0.98, 0.995.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  train.py create --phase tune | create --phase extend|eval --tuning DIR [DIR] | pending --run DIR
           | work --run DIR --unit NAME | finalize --run DIR
  train.py dryrun --arm ARM --steps N [--measure]
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
import measure  # noqa: E402
import micro  # noqa: E402
import nonlinear  # noqa: E402
from gpu_parsec import PARSECL  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

MODEL, STEPS, BATCH, MICRO = 'M1d6', 500, 2048, 512
VALIDATION = [0, 10, *range(25, STEPS+1, 25)]
MEASURE_EVERY, JACOBIAN_EVERY, CHECKPOINTS = 10, 50, [50, 100, 250, 500]
EMA_DECAYS = [.9, .98, .995]
WARMUP, DECAY_START, WEIGHT_DECAY = 25, 400, .05
SN_BOUND = 1.
LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2]
SOAP_LR, SOAP_EMA_DECAY = 1e-2, .98  # cifar_ggn_20261012 seed-0 selection
EVAL_SEEDS = [1, 2, 3]
SEEDS = {'order': 102, 'augment': 5000, 'distance': 9001}
COST = {'soap': 0, 'soap_sn': 1}  # queue order
# Measure B: fixed function-space step sizes, the RMS over examples of |J_i d| (logits), set from the dry runs.
MATCHED_RMS = (.1, 1.)


def lr_factor(t, warmup=True):
    """Linear warm-up over WARMUP steps, constant, then linear decay over steps 401-500 to 1%."""
    if warmup and t <= WARMUP:
        return t/WARMUP
    return 1. if t <= DECAY_START else (STEPS+1-t)/(STEPS-DECAY_START)


def name_of(spec):
    return f"{spec['arm']}__lr{spec['lr']:g}__s{spec['seed']}"


def spec_of(arm, seed, lr, measure_=False):
    return {'arm': arm, 'seed': seed, 'measure': measure_, 'lr': lr}


def tune_units():
    specs = [spec_of('soap_sn', 0, lr) for lr in LR_GRID]
    return {name_of(s): s for s in specs}


def selection(tunings):
    """Seed-0 selection (PROTOCOL.md, section 4) for soap_sn over one or more tuning runs: the lr and EMA decay with
    the lowest step-500 EMA loss, and whether that lr lies at an edge of the searched grid. soap is fixed."""
    best, grid = None, set()
    for tuning in tunings:
        for name, spec in read(tuning/'protocol.json')['units'].items():
            grid.add(spec['lr'])
            final = {r['step']: r for r in read(tuning/name/'validation.json')}.get(STEPS)
            if final is None:
                continue
            for d in EMA_DECAYS:
                value = final[f'ema{d:g}']['validation_loss']
                if value == value and (best is None or value < best[0]):
                    best = (value, spec['lr'], d)
    edge = None if best[1] not in (min(grid), max(grid)) else ('low' if best[1] == min(grid) else 'high')
    return {'soap_sn': {'lr': best[1], 'ema_decay': best[2], 'value': best[0], 'grid': sorted(grid), 'edge': edge},
            'soap': {'lr': SOAP_LR, 'ema_decay': SOAP_EMA_DECAY, 'fixed': 'cifar_ggn_20261012 seed-0 selection'}}


def extension_units(chosen):
    """One more lr beyond a selected grid edge (x3 or /3), seed 0."""
    row = chosen['soap_sn']
    if row['edge'] == 'low':
        specs = [spec_of('soap_sn', 0, min(row['grid'])/3)]
    elif row['edge'] == 'high':
        specs = [spec_of('soap_sn', 0, max(row['grid'])*3)]
    else:
        specs = []
    return {name_of(s): s for s in specs}


def eval_units(chosen):
    """Seeds 1-3: soap (G1 reproduction), soap_sn at its tuned lr, and soap_sn at SOAP's lr if that differs."""
    specs = []
    for seed in EVAL_SEEDS:
        specs.append(spec_of('soap', seed, SOAP_LR, measure_=True))
        specs.append(spec_of('soap_sn', seed, chosen['soap_sn']['lr'], measure_=True))
        if chosen['soap_sn']['lr'] != SOAP_LR:
            specs.append(spec_of('soap_sn', seed, SOAP_LR, measure_=True))
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
    return cifar.make_cifar_model(MODEL, spec['seed'], device, sn_bound=SN_BOUND if spec['arm'] == 'soap_sn' else None)


def make_optimizer(spec, model):
    soap = dict(lr=spec['lr'], betas=(.95, .95), weight_decay=WEIGHT_DECAY, precondition_frequency=10)
    return PARSECL(model.parameters(), **soap, rank=0, norm_blend=0., batched=False, signals=False)


class Arm:
    """One unit's model, optimizer state and step (cifar_ggn's soap path)."""

    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self.model = build_model(spec, device)
        self.registry = Registry(self.model)
        self.optimizer = make_optimizer(spec, self.model)
        self.info = {}

    def flat(self):
        return self.registry.flatten(dict(self.model.named_parameters())).detach()

    def step(self, t, x, y):
        factor = lr_factor(t)
        for group in self.optimizer.param_groups:
            group['lr'] = self.spec['lr']*factor
        self.optimizer.zero_grad(set_to_none=True)
        total, value = len(x), 0.
        for part in micro.parts(total, MICRO):
            part_loss = torch.nn.functional.cross_entropy(self.model(x[part]), y[part])*((part.stop-part.start)/total)
            part_loss.backward()
            value += float(part_loss)
        self.optimizer.step(probes=None)
        info = self.optimizer.last_info
        self.info = {'loss': value, 'lr': self.spec['lr']*factor, 'probe': info.get('probe') is not None,
                     'correction_energy': info.get('correction_energy')}
        return self.info

    def state(self):
        return {'optimizer': self.optimizer.state_dict()}


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
    evaluator = build_model(spec, device)
    evaluator_registry = Registry(evaluator)
    pairs = data.distance_pairs() if spec.get('measure') else None
    jacobian_generator = torch.Generator().manual_seed(SEEDS['distance']+1)
    reference_before, reference_states = None, None  # measures C and D: the previous measurement's state
    ema = {d: torch.zeros(arm.registry.size, device=device) for d in EMA_DECAYS}
    torch.cuda.reset_peak_memory_stats()
    seconds, history, measurements, outcome, t = 0., [], [], {'status': 'completed'}, 0
    validation = [dict(step=0, training_seconds=0., **evaluate(arm.model, data.validation, data.validation_y))]
    try:
        for t in range(1, steps+1):
            ids = order.take(BATCH)
            following = peek(order, BATCH) if spec.get('measure') and t % MEASURE_EVERY == 0 else None
            x, y = data.batch(ids, augment)
            before = arm.flat().clone()
            torch.cuda.synchronize()
            began = time.perf_counter()
            info = arm.step(t, x, y)
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
                row['nonlinear'] = added = {
                    'next': nonlinear.error_split(arm.model, arm.registry, before, after-before, xn, yn, MICRO, MATCHED_RMS),
                    'own': nonlinear.error_split(arm.model, arm.registry, before, after-before, x, y, MICRO)}
                added['persistence'] = nonlinear.persistence(arm.model, arm.registry, before, after-before, xn, MICRO,
                                                             added['next']['step']['full'], reference_before)
                added['representation'], reference_states = nonlinear.representation(
                    arm.model, arm.registry, before, after-before, pairs['local'][0], reference_states)
                reference_before = before
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
        if chosen['soap_sn']['edge'] and len(tunings) < 2:
            raise RuntimeError(f'Selected lr at a grid edge; run the extension phase first: {chosen}')
        units = eval_units(chosen)
    protocol = {'study': 'cifar_sn_soap_20261012', 'phase': phase, 'document': str(common.PROTOCOL), 'units': units,
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
    """A few steps of one arm (seed 0, lr 1e-2) into a scratch unit under $CIFAR_SN_DATA/dryrun, for the cost table."""
    spec = spec_of(arm, 0, SOAP_LR, measure_=measure_)
    unit = common.DATA/'dryrun'/f"{name_of(spec)}{'__measure' if measure_ else ''}"
    common.prepare_unit(unit)
    run(spec, unit, steps)
    history = json.loads((unit/'training.json').read_text())
    runtime = read(unit/'runtime.json')
    m = json.loads((unit/'measurements.json').read_text())
    print(json.dumps({'arm': arm, 'steps': steps, 'median_step_seconds': statistics.median(r['step_seconds'] for r in history[1:]),
                      'max_step_seconds': max(r['step_seconds'] for r in history), 'gpu_peak_GiB': runtime['gpu_peak_allocated_bytes']/2**30,
                      'measurement_seconds': [r['seconds'] for r in m],
                      'sn_active': [sum(v['active'] for v in r['spectral'].values()) for r in m]}))


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
