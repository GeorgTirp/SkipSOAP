"""CIFAR-10 runner for the N9 training test (PROTOCOL.md): SOAP and top-free PARSEC-L with SOAP bases rebuilt from the
probe GGN, the `soap_eigh` control, and the two baselines. Depth-6 ViT M1d6, batch 2048, GPU float32.

Derived from studies/cifar_ggn_20261012/scripts/train.py. Its `soap` and `topfree` steps, data path, schedules,
evaluation and run loop are copied unchanged, so this runner's `soap` and `topfree` units reproduce that study's
evaluation units bitwise (gate G1). The GGN arms and the diagnostics are removed.

Arms (scripts/n9.py rebuilds the bases at steps 2, 12, 22, ..., the top-free probe schedule, after the step):
  soap        upstream SOAP (QR refresh every 10 steps), lr 1e-2 (cifar_ggn selection);
  soap_eigh   SOAP, upstream refresh off; bases = exact eigenvectors of SOAP's own factors GG at the rebuild steps;
  soap_n9     SOAP, upstream refresh off; s = 1 probes drawn at the rebuild steps, bases = eigenvectors of their
              factors S_L = sum G G^T, S_R = sum G^T G;
  topfree     top-free PARSEC-L (rank 16, c = 64, top_rhs='gradient'), lr 3e-2 (cifar_ggn selection);
  topfree_n9  top-free PARSEC-L, upstream refresh off; bases rebuilt from the probes it already draws.

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
import n9  # noqa: E402
from gpu_parsec import PARSECL  # noqa: E402
from gpu_parsec.optimizer import NEVER  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

MODEL, STEPS, BATCH, MICRO = 'M1d6', 500, 2048, 512
VALIDATION = [0, 10, *range(25, STEPS+1, 25)]
EMA_DECAYS = [.9, .98, .995]
WARMUP, DECAY_START, WEIGHT_DECAY = 25, 400, .05
STRENGTH, RANK, PROBE_EVERY, SAMPLES = 64., 16, 10, 1
LR_GRID = [3e-3, 1e-2, 3e-2]
FIXED = {'soap': (1e-2, .98), 'topfree': (3e-2, .98)}  # cifar_ggn_20261012 seed-0 selections
TUNED = ['soap_eigh', 'soap_n9', 'topfree_n9']
ARMS = ['soap', 'soap_eigh', 'soap_n9', 'topfree', 'topfree_n9']
EVAL_SEEDS = [1, 2, 3]
SEEDS = {'order': 102, 'augment': 5000, 'probe': 9000}
COST = {'soap': 0, 'soap_eigh': 1, 'topfree': 2, 'topfree_n9': 3, 'soap_n9': 4}
REFERENCE = Path.home()/'skipsoap-study-data'/'cifar_ggn_20261012'/'runs'/'20261012T110529-d747f9fc'


def lr_factor(t, warmup=True):
    """Linear warm-up over WARMUP steps, constant, then linear decay over steps 401-500 to 1%."""
    if warmup and t <= WARMUP:
        return t/WARMUP
    return 1. if t <= DECAY_START else (STEPS+1-t)/(STEPS-DECAY_START)


def rebuild_step(t):
    """The top-free probe schedule (PARSECL.wants_probes): steps 2, 12, 22, ..."""
    return t >= 2 and (t-2) % PROBE_EVERY == 0


def name_of(spec):
    return f"{spec['arm']}__lr{spec['lr']:g}__s{spec['seed']}"


def spec_of(arm, seed, lr):
    return {'arm': arm, 'seed': seed, 'lr': lr, 'measure': False}


def tune_units():
    return {name_of(s): s for s in (spec_of(arm, 0, lr) for arm in TUNED for lr in LR_GRID)}


def selection(tunings):
    """Seed-0 selection per tuned arm (lowest step-500 EMA loss over the decays) with its grid edge; fixed baselines."""
    out = {arm: {'lr': lr, 'ema_decay': d, 'fixed': 'cifar_ggn_20261012'} for arm, (lr, d) in FIXED.items()}
    for arm in TUNED:
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
    return out


def extension_units(chosen):
    specs = []
    for arm in TUNED:
        row = chosen[arm]
        if row['edge'] == 'low':
            specs.append(spec_of(arm, 0, min(row['grid'])/3))
        elif row['edge'] == 'high':
            specs.append(spec_of(arm, 0, max(row['grid'])*3))
    return {name_of(s): s for s in specs}


def eval_units(chosen):
    return {name_of(s): s for s in (spec_of(arm, seed, chosen[arm]['lr']) for seed in EVAL_SEEDS for arm in ARMS)}


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
    """cifar_ggn's soap and topfree optimizers; the rebuild arms switch upstream's QR refresh off."""
    frequency = 10 if spec['arm'] in FIXED else NEVER
    soap = dict(lr=spec['lr'], betas=(.95, .95), weight_decay=WEIGHT_DECAY, precondition_frequency=frequency)
    if spec['arm'].startswith('soap'):
        return PARSECL(model.parameters(), **soap, rank=0, norm_blend=0., batched=False, signals=False)
    return PARSECL(model.parameters(), **soap, rank=RANK, probe_every=PROBE_EVERY, refresh='extract', norm_blend=0.,
                   batched=False, signals=False, calibration='lr', strength=STRENGTH, top_rhs='gradient', seed=spec['seed'])


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
        if arm.startswith('topfree') and self.optimizer.wants_probes():
            probes = micro.lean_probes(self.model, x, samples=SAMPLES, generator=self.probe_generator)
        elif arm == 'soap_n9' and rebuild_step(t):
            probes = micro.lean_probes(self.model, x, samples=SAMPLES, generator=self.probe_generator)
        self.optimizer.step(probes=probes if arm.startswith('topfree') else None)
        info = self.optimizer.last_info
        self.info = {'loss': value, 'lr': self.spec['lr']*factor, 'probe': info.get('probe') is not None,
                     'correction_energy': info.get('correction_energy')}
        if arm not in FIXED and rebuild_step(t):
            if arm != 'soap_eigh' and probes is None:
                raise RuntimeError(f'{arm}: no probes at rebuild step {t}')
            began = time.perf_counter()
            checks = n9.rebuild(self.optimizer, 'own' if arm == 'soap_eigh' else 'probe', probes)
            self.info['rebuild'] = {**checks, 'seconds': time.perf_counter()-began}
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
        if any(chosen[arm]['edge'] for arm in TUNED) and len(tunings) < 2:
            raise RuntimeError(f'Selected lr at a grid edge; run the extension phase first: {chosen}')
        units = eval_units(chosen)
    protocol = {'study': 'n9_training_20261013', 'phase': phase, 'document': str(common.PROTOCOL), 'units': units,
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
    """A few steps of one arm (seed 0, its baseline lr or 1e-2) into a scratch unit under $N9_DATA/dryrun."""
    spec = spec_of(arm, 0, FIXED.get(arm, (3e-2 if arm.startswith('topfree') else 1e-2,))[0])
    unit = common.DATA/'dryrun'/name_of(spec)
    common.prepare_unit(unit)
    run(spec, unit, steps)
    history = json.loads((unit/'training.json').read_text())
    runtime = read(unit/'runtime.json')
    rebuilds = [r['rebuild'] for r in history if 'rebuild' in r]
    print(json.dumps({'arm': arm, 'steps': steps, 'median_step_seconds': statistics.median(r['step_seconds'] for r in history[1:]),
                      'rebuild_step_seconds': [r['step_seconds'] for r in history if 'rebuild' in r],
                      'rebuild_checks': rebuilds, 'gpu_peak_GiB': runtime['gpu_peak_allocated_bytes']/2**30}))


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
