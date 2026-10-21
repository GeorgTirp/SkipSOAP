"""Parts V and S (PROTOCOL.md, sections 3-4) at one top-free state.

`analyse` receives a snapshot of state t (weights, copied SOAP contexts, metric scale, ages) together with the weights
and batch positions around it (t-8 ... t+8) and returns the visit-phase aggregates, the capture fractions of every
source on both targets, and the scored steps. Probe rows are recomputed per tensor group, so that one group's pooled
rows (8 x 2,048 columns) fit the memory budget.
"""
import torch

import ggn
import micro
import probes
import spectral
import states
from gpu_parsec.optimizer import randomized_top

SOURCES = ['cur', 'next8', 'last8', 'last8own', 'rand8', 'rand8_rad']
STEPS = SOURCES+['identity']
TARGETS = ['queue', 'held']
RANK, OVERSAMPLE, POWER, CG_STEPS = 64, 16, 2, 8
KS = [16, 64]
GAMMA, FLOOR, CHUNK = 1e-2, 1e-8, 64
BUDGET = int(2*2**30)
MULTIPLIERS = [2.**j for j in range(-14, 3)]
POOL = 8
RECENT, STALE, MAX_AGE = 1, 20, 50


def matrix_specs(registry):
    return [s for s in registry.specs if len(s.shape) == 2 and min(s.shape) > 1]


def groups(specs, columns, budget_bytes, element_bytes=4):
    """Consecutive specs whose (count x columns) row blocks fit the budget together."""
    out, current, used = [], [], 0
    for spec in specs:
        size = spec.count*columns*element_bytes
        if current and used+size > budget_bytes:
            out.append(current)
            current, used = [], 0
        current.append(spec)
        used += size
    if current:
        out.append(current)
    return out


def squared_norm(z, width=1024):
    """||z||_F^2 in float64, accumulated over blocks of columns (no full-size float64 copy; DEVIATIONS.md D1)."""
    return sum(float(z[:, i:i+width].double().square().sum()) for i in range(0, z.shape[1], width))


def capture_terms(rows_t, basis, ks=KS):
    """{k: ||Z_T^T U_k||_F^2} for target rows Z_T (n, D) and an orthonormal basis U (D, >= max k)."""
    projected = rows_t@basis
    return {k: float(projected[:, :k].double().square().sum()) for k in ks}


def age_profile(ages, values):
    """{age: [count, mean]} for ages 0..MAX_AGE (ages < 0: never consumed, excluded)."""
    out = {}
    for a in range(MAX_AGE+1):
        mask = ages == a
        n = int(mask.sum())
        if n:
            out[a] = [n, float(values[mask].double().mean())]
    return out


def band_difference(ages, values):
    stale, recent = ages >= STALE, (ages >= 0) & (ages <= RECENT)
    return float(values[stale].double().mean()-values[recent].double().mean())


def visit_phase(model, registry, data, flat, ages, placebo, next_positions, last_positions, generator):
    """Part V at weights `flat`: per-example losses and trace estimates of all training and validation examples."""
    losses, traces = [], []
    total = len(data.manifest['training'])
    for start in range(0, total, states.BATCH):
        positions = list(range(start, min(start+states.BATCH, total)))
        x, y = data.batch(positions)
        loss, trace = probes.example_statistics(model, registry, flat, x, y, generator)
        losses.append(loss)
        traces.append(trace)
    losses, traces = torch.cat(losses).cpu(), torch.cat(traces).cpu()
    vloss, vtrace = probes.example_statistics(model, registry, flat, data.validation, data.validation_y, generator)
    ages, placebo = torch.as_tensor(ages), torch.as_tensor(placebo)
    out = {'D_loss': band_difference(ages, losses), 'D_trace': band_difference(ages, traces),
           'D_loss_placebo': band_difference(placebo, losses), 'D_trace_placebo': band_difference(placebo, traces),
           'profile_loss': age_profile(ages, losses), 'profile_trace': age_profile(ages, traces),
           'profile_loss_placebo': age_profile(placebo, losses), 'profile_trace_placebo': age_profile(placebo, traces),
           'next_batch': {'loss': float(losses[next_positions].double().mean()), 'trace': float(traces[next_positions].double().mean())},
           'last_batch': {'loss': float(losses[last_positions].double().mean()), 'trace': float(traces[last_positions].double().mean())},
           'train_mean': {'loss': float(losses.double().mean()), 'trace': float(traces.double().mean())},
           'validation_mean': {'loss': float(vloss.double().mean()), 'trace': float(vtrace.double().mean())}}
    return out, {'loss': losses, 'trace': traces, 'ages': ages, 'placebo': placebo}


def pooled_columns(model, registry, data, group, contexts, batches, kind, generator):
    """{name: Z (D, n_total)}: whitened probe columns of the batches [(positions or (x, y), flat)], weighted so that
    Z Z^T is the per-example-average GGN estimate."""
    total = sum(len(b[0]) if not isinstance(b[0], tuple) else len(b[0][0]) for b in batches)
    names = [s.name for s in group]
    columns = {s.name: None for s in group}
    filled = 0
    for source, flat in batches:
        x = data.batch(source)[0] if not isinstance(source, tuple) else source[0]
        rows = probes.probe_rows(model, registry, flat, x, names, kind, generator, total**-.5)
        for spec in group:
            z = ggn.whiten(rows.pop(spec.name), contexts[spec.name]).reshape(len(x), -1)
            if columns[spec.name] is None:
                columns[spec.name] = torch.empty(spec.count, total, dtype=z.dtype, device=z.device)
            columns[spec.name][:, filled:filled+len(x)] = z.T
            del z
        filled += len(x)
        del rows
    return columns


def analyse(model, registry, data, snap, seed):
    """Part S at state t = snap['t'] (and Part V via visit_phase, called by the driver)."""
    t, flat, contexts, scale = snap['t'], snap['flat'], snap['contexts'], snap['scale']
    flats, positions, rand_positions = snap['flats'], snap['positions'], snap['rand_positions']
    base_seed = 50000+1000*seed+t
    x_next, y_next = data.batch(positions[t+1])
    _, trace = ggn.accumulate(model, registry, flat, x_next, [], CHUNK, all_parameters=True, contexts=contexts)
    mu = max(GAMMA*trace/registry.size, FLOOR)
    delta = mu*scale
    _, g = micro.gradient(model, registry, flat, x_next, y_next, states.MICRO)
    specs = matrix_specs(registry)
    zg = {s.name: ggn.whiten(g[s.slice].reshape(s.shape)[None], contexts[s.name])[0].reshape(-1) for s in specs}
    validation = data.validation, data.validation_y
    held = (validation[0][1000:], validation[1][1000:])
    selection = (validation[0][:1000], validation[1][:1000])
    sources = {
        'cur': ([(positions[t+1], flat)], 'label'),
        'next8': ([(positions[t+s], flat) for s in range(1, POOL+1)], 'label'),
        'last8': ([(positions[t-s], flat) for s in range(POOL)], 'label'),
        'last8own': ([(positions[t-s], flats[t-s-1]) for s in range(POOL)], 'label'),
        'rand8': ([(p, flat) for p in rand_positions], 'label'),
        'rand8_rad': ([(p, flat) for p in rand_positions], 'rademacher')}
    targets = {'queue': [(positions[t+s], flats[t+s-1]) for s in range(1, POOL+1)],
               'held': [((held[0][i:i+states.BATCH], None), flat) for i in range(0, len(held[0]), states.BATCH)]}
    steps = {name: torch.zeros_like(flat) for name in STEPS}
    capture = {target: {source: {} for source in SOURCES} for target in TARGETS}
    spectra = {source: {} for source in SOURCES}
    for index, group in enumerate(groups(specs, POOL*states.BATCH, BUDGET)):
        bases = {source: {} for source in SOURCES}
        for k, (source, (batches, kind)) in enumerate(sources.items()):
            generator = torch.Generator().manual_seed(base_seed*100+k)
            sketch = torch.Generator().manual_seed(base_seed*100+50+k)
            columns = pooled_columns(model, registry, data, group, contexts, batches, kind, generator)
            for spec in group:
                z = columns.pop(spec.name)
                u, kappa = randomized_top(z, RANK, sketch, OVERSAMPLE, POWER)
                bases[source][spec.name] = u
                spectra[source][spec.name] = kappa[:RANK].double().cpu().tolist()
                y = spectral.conjugate_gradient(lambda v: z@(z.T@v)+delta*v, -zg[spec.name], CG_STEPS)
                steps[source][spec.slice] = ggn.unwhiten(y.reshape(spec.shape), contexts[spec.name]).reshape(-1)
                del z
        for k, (target, batches) in enumerate(targets.items()):
            generator = torch.Generator().manual_seed(base_seed*100+90+k)
            columns = pooled_columns(model, registry, data, group, contexts, batches, 'label', generator)
            for spec in group:
                z = columns.pop(spec.name)
                total = squared_norm(z)
                rows_t = z.T
                for source in SOURCES:
                    terms = capture_terms(rows_t, bases[source][spec.name])
                    capture[target][source][spec.name] = {str(k_): terms[k_]/total for k_ in KS}
                del z, rows_t
        del bases
    for spec in specs:
        steps['identity'][spec.slice] = ggn.unwhiten(-zg[spec.name].reshape(spec.shape), contexts[spec.name]).reshape(-1)
    queue_sets = [data.batch(positions[t+s]) for s in range(1, POOL+1)]

    def losses(point):
        queue = sum(micro.batch_loss(model, registry, point, xq, yq, states.MICRO) for xq, yq in queue_sets)/POOL
        return {'queue': queue, 'held': micro.batch_loss(model, registry, point, held[0], held[1], states.MICRO),
                'selection': micro.batch_loss(model, registry, point, selection[0], selection[1], states.MICRO)}
    base = losses(flat)
    eta = snap['eta']
    scored = {}
    for name in STEPS:
        alpha0 = eta if name == 'identity' else 1.
        grid = {m: losses(flat+alpha0*m*steps[name]) for m in MULTIPLIERS}
        chosen = max(grid, key=lambda m: base['selection']-grid[m]['selection'])
        scored[name] = {'alpha0': alpha0, 'multiplier': chosen, 'interior': MULTIPLIERS[0] < chosen < MULTIPLIERS[-1],
                        'decrease': {k: base[k]-grid[chosen][k] for k in ['queue', 'held', 'selection']},
                        'oracle': {k: max(base[k]-grid[m][k] for m in MULTIPLIERS) for k in ['queue', 'held']}}
    return {'t': t, 'mu': mu, 'scale': scale, 'delta': delta, 'trace': trace, 'base_losses': base,
            'capture': capture, 'steps': scored, 'source_spectra': spectra}
