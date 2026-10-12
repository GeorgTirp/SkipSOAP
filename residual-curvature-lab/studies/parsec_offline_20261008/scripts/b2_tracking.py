"""B2 (I3): subspace drift and one-step tracking of the PARSEC-L subspaces along a trajectory.

Usage: b2_tracking.py create | work --run DIR --unit trajectory | pending --run DIR | finalize --run DIR
"""
import argparse
import copy
import statistics
import sys
from pathlib import Path

import torch

import common
from common import read, write_compact
import probes
from residual_curvature.config import load_config
from residual_curvature.curvature.layerwise import streamed_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import soap_propose
from residual_curvature.provenance import write_json
from run_large_batch_oracle import state_equal
from run_mini_vit_allocation import propose as mini_propose

TRAINING = Path('runs/20261005T135920-d7ce29b2')
RANKS, FIRST, LAST = [4, 8, 16], 20, 50


def analyse_state(model, registry, state, x, y, construction, step, c):
    """Exact blocks, SOAP factors, exact top-r subspaces and an s = 4 probe matrix at one state."""
    before, optimizer, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(state), torch.get_rng_state().clone()
    flat = registry.flatten(dict(model.named_parameters())).detach()
    xb, yb = x[construction], y[construction]
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    _, _, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(op.gradient), state, c['soap'])
    blocks, _ = streamed_blocks(model, registry, flat, xb, 8, 768)
    gen = torch.Generator().manual_seed(303+step)
    actions = []
    for spec, block in zip(registry.specs, blocks):
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=flat.dtype)
        actions.append(relative_error(block@v[spec.slice], op.action(v)[spec.slice]))
    if max(actions) > c['solver']['tolerance64']:
        raise RuntimeError(f'Action check failed at step {step}: {max(actions)}')
    _, _, beta, factors, ks = probes.setup(registry, contexts, blocks)
    u = probes.sampled_factor(model, registry, flat, xb, 4, 303+7919*step+5)
    z = [probes.whitened_probe(u, spec, f, beta, 4) for spec, f in zip(registry.specs, factors)]
    exact = {r: [probes.exact_top(k, r) for k in ks] for r in RANKS}
    if not (state_equal(before, model.state_dict()) and state_equal(optimizer, state) and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('Analysis mutated the training state')
    return {'factors': factors, 'ks': ks, 'z': z, 'exact': exact, 'max_action_error': max(actions)}



def work(directory, name):
    common.verify_archive(directory)
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    c = load_config(TRAINING/'resolved.yaml')
    x, y, manifest = dataset(c)
    construction = manifest['analysis'][:512]
    model = make_model(c)
    registry = Registry(model)
    order = DataOrder(manifest['training'], common.data_order_seed(0))
    state, gate, rows = None, {}, []
    previous, tracked_exact, tracked_probe = None, None, None
    for step in range(1, LAST+1):
        ids = order.take(512)
        flat = registry.flatten(dict(model.named_parameters())).detach()
        delta, state, _, _ = mini_propose('searched_soap', model, registry, flat, state, x[ids], y[ids], c, step)
        commit(model, registry, flat+delta)
        if step in [25, 50]:
            saved = torch.load(common.FAMILY/'b2_w16__s0'/f'state-{step:04d}'/'arrays.pt', map_location='cpu', weights_only=False)['parameters']
            gate[f'parameters_step{step}'] = float((registry.flatten(dict(model.named_parameters())).detach()-saved).abs().max())
            if gate[f'parameters_step{step}'] > 1e-12:
                raise RuntimeError(f'Trajectory gate failed: {gate}')
        if step < FIRST:
            continue
        current = analyse_state(model, registry, state, x, y, construction, step, c)
        row = {'step': step, 'max_action_error': current['max_action_error']}
        if previous is None:
            tracked_exact = {r: [w for w, _ in current['exact'][r]] for r in RANKS}
            tracked_probe = {r: [w for w, _ in current['exact'][r]] for r in RANKS}
        else:
            for r in RANKS:
                drift, exact_track, probe_track, fresh = [], [], [], []
                new_exact, new_probe = [], []
                for index in range(len(registry.specs)):
                    w_now, kappa = current['exact'][r][index]
                    weight = float(kappa.sum())
                    f_prev, f_now = previous['factors'][index], current['factors'][index]
                    moved = probes.transport(previous['exact'][r][index][0], f_prev, f_now)
                    drift.append((moved, w_now, weight))
                    e = probes.transport(tracked_exact[r][index], f_prev, f_now)
                    e, _ = torch.linalg.qr(current['ks'][index]@e-e) if e.shape[1] else (e, None)
                    new_exact.append(e)
                    exact_track.append((e, w_now, weight))
                    p = probes.transport(tracked_probe[r][index], f_prev, f_now)
                    zz = current['z'][index]
                    p, _ = torch.linalg.qr(zz@(zz.T@p)) if p.shape[1] else (p, None)
                    new_probe.append(p)
                    probe_track.append((p, w_now, weight))
                    w_fresh, _ = probes.randomized_top(zz, r, 1000*index+r+7*step)
                    fresh.append((w_fresh[:, :w_now.shape[1]], w_now, weight))
                tracked_exact[r], tracked_probe[r] = new_exact, new_probe
                row[f'r{r}'] = {'drift': probes.weighted_angle(drift), 'exact_tracking': probes.weighted_angle(exact_track),
                                'probe_tracking': probes.weighted_angle(probe_track), 'fresh_extraction': probes.weighted_angle(fresh)}
        rows.append(row)
        previous = current
        write_compact(unit/'tracking.json', rows)
        print(f"step {step}: " + ', '.join(f"r{r} drift={row[f'r{r}']['drift']:.3f} probe_track={row[f'r{r}']['probe_tracking']:.3f} fresh={row[f'r{r}']['fresh_extraction']:.3f}"
                                           for r in RANKS if f'r{r}' in row), flush=True)
    summary = {}
    for r in RANKS:
        series = {k: [row[f'r{r}'][k] for row in rows if f'r{r}' in row] for k in ['drift', 'exact_tracking', 'probe_tracking', 'fresh_extraction']}
        summary[f'r{r}'] = {k: {'median': statistics.median(v), 'mean': statistics.mean(v), 'max': max(v)} for k, v in series.items()}
        summary[f'r{r}']['kill'] = summary[f'r{r}']['probe_tracking']['median'] > summary[f'r{r}']['fresh_extraction']['median']
    summary['kill_I3'] = any(summary[f'r{r}']['kill'] for r in RANKS)
    write_json(unit/'summary.json', {'gate': gate, 'summary': summary})
    common.finish_unit(unit, {'gate': gate, 'kill_I3': summary['kill_I3']})


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
        print(common.create_run(c, {'study': 'parsec_offline_20261008', 'part': 'B2 (I3)', 'document': str(common.PROTOCOL),
                                    'units': {'trajectory': {}}, 'ranks': RANKS, 'steps': [FIRST, LAST]},
                                [str(common.STUDY_REL/'scripts'/'queue_runner.sh')]))
    elif args.command == 'work':
        try:
            work(args.run, args.unit)
        except Exception as error:
            write_json(args.run/'failure.json', {'unit': args.unit, 'error': repr(error)})
            raise
    elif args.command == 'pending':
        print('\n'.join(n for n in read(args.run/'protocol.json')['units'] if not common.unit_complete(args.run/n)))
    else:
        if not common.unit_complete(args.run/'trajectory') or (args.run/'failure.json').exists():
            raise RuntimeError('Incomplete or failed')
        write_json(args.run/'complete.json', {'completed': True})
        print(common.mirror(args.run))


if __name__ == '__main__':
    sys.exit(main())
