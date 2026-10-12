"""B1 (I2): probe-estimated versus exact PARSEC-L (damping-calibrated teacher_r) at saved states.

Usage: b1_extraction.py create | work --run DIR --unit NAME | pending --run DIR | finalize --run DIR
"""
import argparse
import copy
import sys
from pathlib import Path

import torch

import common
from common import read, write_compact
import probes
from residual_curvature.config import load_config
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import capacity
from residual_curvature.provenance import checksum, write_json
from run_large_batch_oracle import measurement, state_equal
from run_mini_vit_allocation import search_candidate

PARENT = Path('runs/20261005T143029-9b582ed5')
TRAINING = Path('runs/20261005T135920-d7ce29b2')
SCREEN = Path('runs/20261007T093903-8d7d86f1')
RANKS, DRAWS = [4, 8, 16], [1, 4]


def units():
    out = {f'main__{t}-{s:04d}': {'kind': 'main', 'trajectory': t, 'step': s}
           for t in ['searched_soap', 'exact_tensor_ggn'] for s in [10, 25, 50]}
    for name, spec in read(common.FAMILY/'protocol.json')['units'].items():
        for step in [10, 25, 50]:
            out[f'fam__{name}__{step:04d}'] = {'kind': 'family', 'unit': name, 'step': step, 'split': spec['split'],
                                               'blocks': spec['blocks'], 'width': spec['width']}
    return out


def source(spec):
    if spec['kind'] == 'main':
        return PARENT/f"{spec['trajectory']}-{spec['step']:04d}", load_config(TRAINING/'resolved.yaml')
    unit = common.FAMILY/spec['unit']
    return unit/f"state-{spec['step']:04d}", read(unit/'resolved_arm.json')['config']


def work(directory, name):
    common.verify_archive(directory)
    spec = read(directory/'protocol.json')['units'][name]
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    src, config = source(spec)
    for artifact, digest in read(src/'artifacts.json').items():
        if checksum(src/artifact) != digest:
            raise RuntimeError(f'Artifact checksum failed: {src/artifact}')
    arrays = torch.load(src/'arrays.pt', map_location='cpu', weights_only=False)
    info = read(src/'diagnostics.json')
    x, y, manifest = dataset(config)
    model = make_model(config)
    registry = Registry(model)
    flat, g, momentum, contexts = [arrays[k] for k in ['parameters', 'gradient', 'momentum_rhs', 'contexts']]
    blocks, native = arrays['ggn_blocks'], arrays['raw_soap']
    commit(model, registry, flat)
    before, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    construction, panel_ids = info['construction_ids'], info['panel_ids']
    fn = functional_logits(model, registry, x[construction])
    objective = lambda w: loss(fn(w), y[construction])
    op = GGN(fn, flat, y[construction])
    checks = {'gradient_error': relative_error(op.gradient, g)}
    metric, mu, beta, factors, ks = probes.setup(registry, contexts, blocks)
    seed = 303+7919*spec['step']+(0 if spec['kind'] == 'main' and spec['trajectory'] == 'searched_soap' else 1)
    u = probes.sampled_factor(model, registry, flat, x[construction], max(DRAWS), seed)
    trace = sum(float(b.trace()) for b in blocks)
    state = {'mu': mu, 'beta': beta, 'trace': trace, 'probe_trace_relative_error': {}, 'angles': {}, 'kappa1_relative_error': {}}
    cores = {'identity': [{'kind': 'identity', 'storage': 0, 'clipped': 0} for _ in registry.specs]}
    for r in RANKS:
        cores[f'exact_r{r}'] = [capacity.fit_core(k, 'teacher', r) for k in ks]
        exact = [probes.exact_top(k, r) for k in ks]
        for s in DRAWS:
            probe_cores, pairs, kappa_errors = [], [], []
            for index, (spec_b, f) in enumerate(zip(registry.specs, factors)):
                z = probes.whitened_probe(u, spec_b, f, beta, s)
                w, kappa = probes.randomized_top(z, r, 1000*index+r)
                probe_cores.append(probes.subspace_core(w, kappa))
                ew, ek = exact[index]
                pairs.append((ew[:, :w.shape[1]], w, float(ek.sum())))
                if float(ek[0]) > 0:
                    kappa_errors.append(abs(float(kappa[0])-float(ek[0]))/float(ek[0]))
            cores[f'probe_s{s}_r{r}'] = probe_cores
            state['angles'][f's{s}_r{r}'] = probes.weighted_angle(pairs)
            state['kappa1_relative_error'][f's{s}_r{r}'] = sum(kappa_errors)/len(kappa_errors)
    for s in DRAWS:
        sub = u.reshape(512, max(DRAWS), -1)[:, :s].reshape(-1, registry.size)*(max(DRAWS)/s)**.5
        state['probe_trace_relative_error'][f's{s}'] = abs(float(sub.square().sum())-trace)/trace
    panels = [lambda w, f=functional_logits(model, registry, x[ids]), t=y[ids]: loss(f(w), t) for ids in panel_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    rows = []
    for core_name, core in cores.items():
        for rhs_name, rhs in [('gradient', g), ('momentum', momentum)]:
            delta = -capacity.apply(core, registry, contexts, beta, rhs)
            selected, search = search_candidate(flat, delta, native, objective)
            values = measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry)
            values.pop('tensor_update_shares')
            rows.append({'name': core_name, 'rhs': rhs_name, 'selected': values,
                         'search': {k: search[k] for k in ['selected_alpha', 'evaluations', 'bracketed', 'rejected']}})
        write_compact(unit/'candidates.json', rows)
    if spec['kind'] == 'main':
        screen = read(SCREEN/f"damping-{spec['trajectory']}-{spec['step']:04d}"/'candidates.json')
        gate = {}
        for core_name, screen_name in [('identity', 'identity')]+[(f'exact_r{r}', f'teacher_{r}') for r in RANKS]:
            for rhs_name in ['gradient', 'momentum']:
                ours = next(r for r in rows if r['name'] == core_name and r['rhs'] == rhs_name)['selected']['independent_mean_decrease']
                theirs = next(r for r in screen if r['name'] == screen_name and r['rhs'] == rhs_name)['selected']['independent_mean_decrease']
                gate[f'{core_name}_{rhs_name}'] = abs(ours-theirs)
        state['gate'] = gate
        if max(gate.values()) > 1e-6:
            write_json(unit/'state.json', state)
            raise RuntimeError(f'Capacity-screen reproduction gate failed by more than 1e-6: {gate}')
    if max(checks.values()) > 1e-8:
        raise RuntimeError(f'Checks failed: {checks}')
    if not (state_equal(before, model.state_dict()) and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('State mutated')
    state['checks'] = checks
    write_json(unit/'state.json', state)
    common.finish_unit(unit, {'gate': state.get('gate')})
    print(f"{name}: angles s4_r16={state['angles']['s4_r16']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'finalize'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--unit')
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        c = load_config(TRAINING/'resolved.yaml')
        c['output'] = str(common.RUNS)
        print(common.create_run(c, {'study': 'parsec_offline_20261008', 'part': 'B1 (I2)', 'document': str(common.PROTOCOL),
                                    'units': units(), 'ranks': RANKS, 'draws': DRAWS},
                                [str(common.STUDY_REL/'scripts'/'queue_runner.sh')]))
    elif args.command == 'work':
        try:
            work(args.run, args.unit)
        except Exception as error:
            write_json(args.run/'failure.json', {'unit': args.unit, 'error': repr(error)})
            raise
    elif args.command == 'pending':
        names = read(args.run/'protocol.json')['units']
        print('\n'.join(n for n in names if not common.unit_complete(args.run/n)))
    else:
        names = list(read(args.run/'protocol.json')['units'])
        missing = [n for n in names if not common.unit_complete(args.run/n)]
        if missing or (args.run/'failure.json').exists():
            raise RuntimeError(f'Incomplete or failed: {missing}')
        write_json(args.run/'complete.json', {'completed': True, 'states': len(names)})
        print(common.mirror(args.run))


if __name__ == '__main__':
    sys.exit(main())
