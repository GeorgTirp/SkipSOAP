"""CIFAR-10 runner for the tuned 1,000-step comparison (PROTOCOL.md): SOAP and top-free PARSEC-L with OT-metric
probes, both tuned on seed 0 and evaluated on seeds 1-3. Depth-6 ViT M1d6, batch 2048, GPU float32.

Derived from studies/ot_1000_20261013/scripts/train.py (1,000-step schedule; `soap` is cifar_ggn_20261012's path and
`topfree_ot` probe_variants_20261013's arm with the arm's c), with tuning, extension and evaluation phases added.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  train.py create --phase tune | create --phase extend|eval --tuning DIR [DIR] | pending --run DIR
           | work --run DIR --unit NAME | finalize --run DIR
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
import micro  # noqa: E402
import n9  # noqa: E402
import probes_ot  # noqa: E402
from gpu_parsec import PARSECL  # noqa: E402
from gpu_parsec.optimizer import NEVER  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

MODEL, STEPS, BATCH, MICRO = 'M1d6', 1000, 2048, 512
VALIDATION = [0, 10, 25, 50, *range(100, STEPS+1, 100)]
EMA_DECAYS = [.98, .995, .998]
WARMUP, DECAY_START, WEIGHT_DECAY = 50, 800, .05
STRENGTH, RANK, PROBE_EVERY, SAMPLES = 64., 16, 10, 1
ARMS = ['soap', 'topfree_ot']
LR_GRIDS = {'soap': [3e-2, 9e-2, .27], 'topfree_ot': [3e-2, 9e-2]}
C_GRIDS = {'soap': [None], 'topfree_ot': [1024., 4096.]}
EVAL_SEEDS = [1, 2, 3]
SEEDS = {'order': 102, 'augment': 5000, 'probe': 9000}
COST = {'soap': 0, 'topfree_ot': 1}


def lr_factor(t, warmup=True):
    """Linear warm-up over WARMUP steps, constant, then linear decay over steps 801-1000 to 1%."""
    if warmup and t <= WARMUP:
        return t/WARMUP
    # Linear from 1 at step DECAY_START + 1 to 1% at step STEPS. For a 100-step decay this is exactly the 500-step
    # runners' (STEPS + 1 - t) / 100; their formula would end at 1/400 for a 400-step decay.
    return 1. if t <= DECAY_START else 1-.99*(t-DECAY_START-1)/(STEPS-DECAY_START-1)


def rebuild_step(t):
    """The top-free probe schedule (PARSECL.wants_probes): steps 2, 12, 22, ..."""
    return t >= 2 and (t-2) % PROBE_EVERY == 0


def name_of(spec):
    extra = f"__c{spec['strength']:g}" if spec.get('strength') is not None else ''
    return f"{spec['arm']}__lr{spec['lr']:g}{extra}__s{spec['seed']}"


def spec_of(arm, seed, lr):
    return {'arm': arm, 'seed': seed, 'lr': lr, 'measure': False}


def unit(arm, seed, lr, c=None):
    spec = spec_of(arm, seed, lr)
    if c is not None:
        spec['strength'] = c
    return spec


def tune_units():
    return {name_of(s): s for s in (unit(arm, 0, lr, c) for arm in ARMS for lr in LR_GRIDS[arm] for c in C_GRIDS[arm])}


def selection(tunings):
    """Per arm: the (lr, c, EMA decay) with the lowest step-1000 EMA loss (a diverged unit counts as worse), and
    whether its lr lies at an edge of the arm's lr grid (including extensions)."""
    out = {}
    for arm in ARMS:
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
                        best = (value, spec['lr'], spec.get('strength'), d)
        edge = None if best[1] not in (min(grid), max(grid)) else ('low' if best[1] == min(grid) else 'high')
        out[arm] = {'lr': best[1], 'strength': best[2], 'ema_decay': best[3], 'value': best[0], 'grid': sorted(grid),
                    'edge': edge}
    return out


def extension_units(chosen):
    """One unit beyond a selected lr edge (x3 or /3), at the arm's selected c."""
    specs = []
    for arm in ARMS:
        row = chosen[arm]
        if row['edge'] == 'low':
            specs.append(unit(arm, 0, min(row['grid'])/3, row['strength']))
        elif row['edge'] == 'high':
            specs.append(unit(arm, 0, max(row['grid'])*3, row['strength']))
    return {name_of(s): s for s in specs}


def eval_units(chosen):
    return {name_of(s): s for s in (unit(arm, seed, chosen[arm]['lr'], chosen[arm]['strength'])
                                    for seed in EVAL_SEEDS for arm in ARMS)}


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


def evaluate(model, x, y, chunk=1000):
    total, correct = 0., 0
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            z = model(x[start:start+chunk])
            total += float(torch.nn.functional.cross_entropy(z, y[start:start+chunk], reduction='sum'))
            correct += int((z.argmax(-1) == y[start:start+chunk]).sum())
    return {'validation_loss': total/len(x), 'validation_accuracy': correct/len(x), 'validation_examples': len(x)}


def build_model(spec, device):
    return cifar.make_cifar_model(MODEL, spec['seed'], device)


def make_optimizer(spec, model):
    """cifar_ggn's soap and topfree optimizers; topfree_n9 switches upstream's QR refresh off."""
    frequency = NEVER if spec['arm'] == 'topfree_n9' else 10
    soap = dict(lr=spec['lr'], betas=(.95, .95), weight_decay=WEIGHT_DECAY, precondition_frequency=frequency)
    if spec['arm'].startswith('soap'):
        return PARSECL(model.parameters(), **soap, rank=0, norm_blend=0., batched=False, signals=False)
    return PARSECL(model.parameters(), **soap, rank=RANK, probe_every=PROBE_EVERY, refresh='extract', norm_blend=0.,
                   batched=False, signals=False, calibration='lr', strength=spec.get('strength', STRENGTH), top_rhs='gradient',
                   seed=spec['seed'])


class Arm:
    """One unit's model, optimizer state and step."""

    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self.model = build_model(spec, device)
        self.registry = Registry(self.model)
        self.optimizer = make_optimizer(spec, self.model)
        self.probe_generator = torch.Generator().manual_seed(SEEDS['probe']+spec['seed'])
        self.info = {}

    def flat(self):
        return self.registry.flatten(dict(self.model.named_parameters())).detach()

    def step(self, t, x, y):
        arm = self.spec['arm']
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
        if arm == 'topfree' and self.optimizer.wants_probes():
            probes = micro.lean_probes(self.model, x, samples=SAMPLES, generator=self.probe_generator)
        elif arm == 'topfree_ot' and self.optimizer.wants_probes():
            probes = probes_ot.ot_probes(self.model, x, samples=SAMPLES, generator=self.probe_generator)
        self.optimizer.step(probes=probes)
        info = self.optimizer.last_info
        self.info = {'loss': value, 'lr': self.spec['lr']*factor, 'probe': info.get('probe') is not None,
                     'correction_energy': info.get('correction_energy')}
        if arm == 'topfree_n9' and rebuild_step(t):
            if probes is None:
                raise RuntimeError(f'{arm}: no probes at rebuild step {t}')
            began = time.perf_counter()
            self.info['rebuild'] = {**n9.rebuild(self.optimizer, 'probe', probes), 'seconds': time.perf_counter()-began}
        del probes
        return self.info


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
    ema = {d: torch.zeros(arm.registry.size, device=device) for d in EMA_DECAYS}
    torch.cuda.reset_peak_memory_stats()
    seconds, history, outcome, t = 0., [], {'status': 'completed'}, 0
    validation = [dict(step=0, training_seconds=0., **evaluate(arm.model, data.validation, data.validation_y))]
    try:
        for t in range(1, steps+1):
            ids = order.take(BATCH)
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
            if t in VALIDATION:
                measured = dict(step=t, training_seconds=seconds, **evaluate(arm.model, data.validation, data.validation_y))
                for d in EMA_DECAYS:
                    commit(evaluator, evaluator_registry, ema[d]/(1-d**t))
                    measured[f'ema{d:g}'] = evaluate(evaluator, data.validation, data.validation_y)
                measured['train_loss_last25'] = statistics.mean(r['loss'] for r in history[-25:])
                validation.append(measured)
                write_json(unit/'validation.json', validation)
                write_compact(unit/'training.json', history)
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
        if any(chosen[arm]['edge'] for arm in ARMS) and len(tunings) < 2:
            raise RuntimeError(f'Selected lr at a grid edge; run the extension phase first: {chosen}')
        units = eval_units(chosen)
    protocol = {'study': 'ot_tuned_1000_20261014', 'phase': phase, 'document': str(common.PROTOCOL), 'units': units,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'pending', 'work', 'finalize'])
    parser.add_argument('--phase', choices=['tune', 'extend', 'eval'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--tuning', type=Path, nargs='*', default=[])
    parser.add_argument('--unit')
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        create(args.phase, args.tuning)
    elif args.command == 'pending':
        print('\n'.join(pending(args.run)))
    elif args.command == 'work':
        work(args.run, args.unit)
    else:
        finalize(args.run)


if __name__ == '__main__':
    sys.exit(main())
