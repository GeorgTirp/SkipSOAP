"""CIFAR-10 runner for the c control (PROTOCOL.md): top-free PARSEC-L with GGN probes and c tuned on seed 0
(`topfree_c`), and top-free with OT probes at c = 4096 (`topfree_ot`). Depth-6 ViT M1d6, batch 2048, lr 3e-2.

Derived from studies/probe_variants_20261013/scripts/train.py: `topfree_ot` is that study's arm unchanged (gate G1);
`topfree_c` is its `topfree` path (GGN probes of the step's batch) with the arm's own c. The cross-batch arm is
removed.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  train.py create --phase tune | create --phase extend|eval --tuning DIR [DIR] | pending --run DIR
           | work --run DIR --unit NAME | finalize --run DIR | dryrun --arm ARM --steps N
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
import probes_ot  # noqa: E402
from gpu_parsec import PARSECL  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

MODEL, STEPS, BATCH, MICRO = 'M1d6', 500, 2048, 512
VALIDATION = [0, 10, *range(25, STEPS+1, 25)]
EMA_DECAYS = [.9, .98, .995]
WARMUP, DECAY_START, WEIGHT_DECAY = 25, 400, .05
STRENGTH, RANK, PROBE_EVERY, SAMPLES = 64., 16, 10, 1
LR = 3e-2  # cifar_ggn_20261012's top-free selection (EMA decay 0.98)
C_GRID = [64., 256., 1024., 4096.]  # topfree_c only
OT_C = 4096.  # probe_variants_20261013's selection for topfree_ot
ARMS = ['topfree_c', 'topfree_ot']
EVAL_SEEDS = [1, 2, 3]
SEEDS = {'order': 102, 'augment': 5000, 'probe': 9000, 'probe_order': 7102, 'probe_augment': 6000}
COST = {'topfree_c': 0, 'topfree_ot': 1}
REFERENCE = Path.home()/'skipsoap-study-data'/'cifar_ggn_20261012'/'runs'/'20261012T110529-d747f9fc'


def lr_factor(t, warmup=True):
    """Linear warm-up over WARMUP steps, constant, then linear decay over steps 401-500 to 1%."""
    if warmup and t <= WARMUP:
        return t/WARMUP
    return 1. if t <= DECAY_START else (STEPS+1-t)/(STEPS-DECAY_START)


def name_of(spec):
    extra = f"__c{spec['strength']:g}" if spec['arm'] in ARMS else ''
    return f"{spec['arm']}__lr{spec['lr']:g}{extra}__s{spec['seed']}"


def spec_of(arm, seed, strength=STRENGTH):
    spec = {'arm': arm, 'seed': seed, 'lr': LR, 'measure': False}
    if arm in ARMS:
        spec['strength'] = strength
    return spec


def tune_units():
    return {name_of(s): s for s in (spec_of('topfree_c', 0, c) for c in C_GRID)}


def selection(tunings):
    """topfree_c's seed-0 selection of c (lowest step-500 EMA loss over the decays) with its grid edge; topfree_ot fixed."""
    out = {'topfree_ot': {'strength': OT_C, 'ema_decay': .98, 'fixed': 'probe_variants_20261013 selection'}}
    best, grid = None, set()
    for tuning in tunings:
        for name, spec in read(tuning/'protocol.json')['units'].items():
            grid.add(spec['strength'])
            final = {r['step']: r for r in read(tuning/name/'validation.json')}.get(STEPS)
            if final is None:
                continue
            for d in EMA_DECAYS:
                value = final[f'ema{d:g}']['validation_loss']
                if value == value and (best is None or value < best[0]):
                    best = (value, spec['strength'], d)
    edge = None if best[1] not in (min(grid), max(grid)) else ('low' if best[1] == min(grid) else 'high')
    out['topfree_c'] = {'strength': best[1], 'ema_decay': best[2], 'value': best[0], 'grid': sorted(grid), 'edge': edge}
    return out


def extension_units(chosen):
    row = chosen['topfree_c']
    if row['edge'] == 'low':
        specs = [spec_of('topfree_c', 0, min(row['grid'])/4)]
    elif row['edge'] == 'high':
        specs = [spec_of('topfree_c', 0, max(row['grid'])*4)]
    else:
        specs = []
    return {name_of(s): s for s in specs}


def eval_units(chosen):
    return {name_of(s): s for s in (spec_of(arm, seed, chosen[arm]['strength']) for seed in EVAL_SEEDS for arm in ARMS)}


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
    """cifar_ggn's top-free optimizer, with the arm's brake strength c."""
    soap = dict(lr=spec['lr'], betas=(.95, .95), weight_decay=WEIGHT_DECAY, precondition_frequency=10)
    return PARSECL(model.parameters(), **soap, rank=RANK, probe_every=PROBE_EVERY, refresh='extract', norm_blend=0.,
                   batched=False, signals=False, calibration='lr', strength=spec.get('strength', STRENGTH),
                   top_rhs='gradient', seed=spec['seed'])


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

    def step(self, t, x, y, probe_batch=None):
        """cifar_ggn's top-free step; the probe rows depend on the arm (`probe_batch` draws the independent batch)."""
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
        if self.optimizer.wants_probes():
            if arm == 'topfree_c':
                probes = micro.lean_probes(self.model, x, samples=SAMPLES, generator=self.probe_generator)
            else:
                probes = probes_ot.ot_probes(self.model, x, samples=SAMPLES, generator=self.probe_generator)
        self.optimizer.step(probes=probes)
        del probes
        info = self.optimizer.last_info
        self.info = {'loss': value, 'lr': self.spec['lr']*factor, 'probe': info.get('probe') is not None,
                     'correction_energy': info.get('correction_energy')}
        if info.get('probe') is not None and arm != 'topfree':
            self.info['eta_lambda_max'] = info['probe'].get('eta_lambda_max')
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
    probe_order = DataOrder(list(range(len(data.manifest['training']))), SEEDS['probe_order']+1000*seed)
    probe_augment = torch.Generator().manual_seed(SEEDS['probe_augment']+seed)
    probe_batch = lambda: data.batch(probe_order.take(BATCH), probe_augment)
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
            info = arm.step(t, x, y, probe_batch)
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
        if chosen['topfree_c']['edge'] and len(tunings) < 2:
            raise RuntimeError(f'Selected lr at a grid edge; run the extension phase first: {chosen}')
        units = eval_units(chosen)
    protocol = {'study': 'ggn_c_control_20261013', 'phase': phase, 'document': str(common.PROTOCOL), 'units': units,
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


def dryrun(arm, steps):
    """A few steps of one arm (seed 0, c = 64) into a scratch unit under $CC_DATA/dryrun."""
    spec = spec_of(arm, 0)
    unit = common.DATA/'dryrun'/name_of(spec)
    common.prepare_unit(unit)
    run(spec, unit, steps)
    history = json.loads((unit/'training.json').read_text())
    runtime = read(unit/'runtime.json')
    print(json.dumps({'arm': arm, 'steps': steps, 'median_step_seconds': statistics.median(r['step_seconds'] for r in history[1:]),
                      'probe_step_seconds': [r['step_seconds'] for r in history if r['probe']],
                      'eta_lambda_max': [r.get('eta_lambda_max') for r in history if r['probe']],
                      'gpu_peak_GiB': runtime['gpu_peak_allocated_bytes']/2**30}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create', 'pending', 'work', 'finalize', 'dryrun'])
    parser.add_argument('--phase', choices=['tune', 'extend', 'eval'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--tuning', type=Path, nargs='*', default=[])
    parser.add_argument('--unit')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--steps', type=int, default=5)
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
        dryrun(args.arm, args.steps)


if __name__ == '__main__':
    sys.exit(main())
