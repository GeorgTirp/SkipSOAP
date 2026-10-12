import time

import torch
from torch.func import grad

from .. import checkpoint
from ..curvature.operators import guard, loss
from ..curvature.sketch import categorical_probes, factors_from_probes
from ..curvature.solves import IdentityMetric, diagonal_solve, factor_solve
from ..data import DataOrder, dataset
from ..models import Registry, make_model
from ..models.mlp import functional_logits
from ..optim.soap import SOAPMetric, soap_propose
from ..provenance import new_run, write_json
from .search import training_backtracking
from .snapshot import snapshot


def commit(model, registry, flat):
    with torch.no_grad():
        for name, value in registry.unflatten(flat).items():
            dict(model.named_parameters())[name].copy_(value)


def training_step(model, registry, state, x, y, c, step):
    flat = registry.flatten(dict(model.named_parameters())).detach()
    fn = functional_logits(model, registry, x)
    objective = lambda p: loss(fn(p), y)
    g = grad(objective)(flat).detach()
    native, next_state, context = soap_propose(registry.unflatten(flat), registry.unflatten(g), state, c['soap'])
    delta = registry.flatten(native)
    method, settings = c['training']['method'], c['training']
    metadata = {'method': method, 'policy': settings['policy']}
    if method != 'soap' and all(v is not None for v in context.values()):
        # Training uses a labelled sketch trace, avoiding an exact GGN per step.
        rank = settings['rank']
        guard(registry.size, rank, rank, c['resources'], flat.dtype)
        xi = categorical_probes(fn(flat), rank, c['probe_seed'] + step)
        factor = factors_from_probes(fn, flat, xi)
        metric = SOAPMetric(registry, context) if c['damping']['metric'] == 'soap' else IdentityMetric()
        mu = max(settings['gamma'] * float(factor.square().sum()) / registry.size, c['damping']['floor'])
        delta = factor_solve(g, factor, mu, metric.inverse) if method == 'sketch_coupled' else diagonal_solve(g, factor, mu, metric.inverse, registry.groups(c['partition']))
        metadata.update(mu=mu, trace_source='current shared-probe estimate', rank=rank)
    if settings['policy'] == 'training_backtracking':
        delta, search = training_backtracking(flat, delta, g, objective, c['search'])
        metadata.update(search)
    else:
        metadata['accepted'] = bool(delta.norm() > 0)
    before = float(objective(flat))
    commit(model, registry, flat + delta)
    metadata.update(loss_before=before, loss_after=float(objective(flat + delta)), norm=float(delta.norm()), step=step)
    # Statistics advance exactly once, even if parameters did not move.
    return next_state, context, metadata


def run(c, analyze=True, resume=None, stop_after=None, monitor=None):
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    if c['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model, (x, y, manifest) = make_model(c), dataset(c)
    registry = Registry(model)
    directory = new_run(c)
    print(f'Run directory: {directory.resolve()}', flush=True)
    write_json(directory / 'split_manifest.json', manifest)
    write_json(directory / 'registry.json', registry.json())
    order = DataOrder(manifest['training'], c['data_seed'] + 1)
    state, context, first, accepted = None, None, 0, 0
    if resume:
        restored = checkpoint.load(resume, model, order, c)
        state, context, first, accepted = restored['optimizer'], restored['used_context'], restored['step'], restored['accepted']
    b = c['data']['batch_size']
    analysis_ids = manifest['analysis']
    construction_ids = analysis_ids[:b]
    panels = [analysis_ids[(i+1)*b:(i+2)*b] for i in range(c['data']['panel_batches'])]
    rows = []
    started = time.perf_counter()
    for step in range(first + 1, min(c['training']['steps'], stop_after or c['training']['steps']) + 1):
        ids = order.take(b)
        before = time.perf_counter()
        state, context, row = training_step(model, registry, state, x[ids].to(c['device']), y[ids].to(c['device']), c, step)
        accepted += int(row['accepted'])
        row.update(batch_ids=ids, accepted_updates=accepted, step_seconds=time.perf_counter()-before, wall_seconds=time.perf_counter()-started)
        rows.append(row)
        # Persist the consumed training step before potentially expensive audits.
        # A failed reference calculation must not erase the state that exposed it.
        write_json(directory / 'training.json', rows)
        if (step % c['training']['checkpoint_every'] == 0 or step == c['training']['steps']
                or step == stop_after or (analyze and step in c['training']['snapshots'])):
            checkpoint.save(directory / 'checkpoint.pt', model, state, context, step, accepted, order, c)
        if monitor:
            monitor(model, registry, state, x, y, c, step)
        if analyze and step in c['training']['snapshots']:
            print(f'Iteration {step}: starting snapshot measurements', flush=True)
            try:
                snapshot(model, registry, state, x, y, construction_ids, panels, c, step, directory / f'snapshot-{step:04d}')
            except Exception as error:
                write_json(directory / 'failure.json', {'step': step, 'phase': 'snapshot',
                           'exception': type(error).__name__, 'message': str(error)})
                raise
        if step % 10 == 0 or step == c['training']['steps']:
            print(f'Iteration {step}: accepted={accepted}, minibatch loss={row["loss_after"]:.6f}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    return directory, model, state
