"""One state of the noise-scale study (PROTOCOL.md, section 3).

In SOAP-whitened coordinates, per matrix tensor b, with A_b = K_b + δI (K_b from 4,096 reference examples):
- quadratic forms zᵀA^q z of the corpus gradient (mean of 43 partition-batch gradients of 1,024) and the mean
  deviation forms (1/m) Σ_j ξ_jᵀA^q ξ_j, for q on a 0.05 grid in [−1, 1] (batched Lanczos quadrature);
- the fractional-Newton steps y_ν = −A^{−ν} z_B of the step batch, and their block-model scale terms;
- gate G2 on block 0's attention weights at the declared states (float64 eigendecomposition reference).
The per-state step scores (selection and evaluation batches, validation) are computed by `score_steps`.
"""
import copy

import torch

import ggn
import lanczos
import micro
import states
from residual_curvature.data import DataOrder
from residual_curvature.optim.soap import SOAPMetric

GAMMA, FLOOR, CHUNK, GROUP_CHUNK, BUDGET = 1e-2, 1e-8, 64, 128, int(2.5*2**30)
STEPS = 300
POWERS = [round(-1+0.05*i, 2) for i in range(41)]
NUS = [0., .25, .5, .75, 1.]
PARTITION_SEED, PARTITION_BATCH, PARTITION_COUNT = 20261021, 1024, 43
REFERENCE_SEED = 20261020  # spectral_rotation_20261020's batches A and B
G2_TENSORS = ['blocks.0.attention.query.weight', 'blocks.0.attention.key.weight', 'blocks.0.attention.value.weight',
              'blocks.0.attention.output.weight']
G2_TOLERANCE = 1e-2
MULTIPLIERS = [2.**j for j in range(-14, 3)]


def matrix_specs(registry):
    return [s for s in registry.specs if len(s.shape) == 2 and min(s.shape) > 1]


def groups(specs, budget_bytes, element_bytes):
    out, current, used = [], [], 0
    for spec in specs:
        size = spec.count**2*element_bytes
        if current and used+size > budget_bytes:
            out.append(current)
            current, used = [], 0
        current.append(spec)
        used += size
    if current:
        out.append(current)
    return out


def peek(replay, k):
    """Positions of the batch k + 1 places ahead (k = 0: the next one), without consuming the replay's order."""
    twin = DataOrder([0], 0)
    twin.load_state_dict(copy.deepcopy(replay.order.state_dict()))
    taken = []
    for _ in range(k+1):
        taken = twin.take(states.BATCH)
    return taken


def peek_step_batch(replay):
    """The next training batch exactly as Replay.next_batch would return it (augmented), without consuming."""
    positions = peek(replay, 0)
    generator = torch.Generator()
    generator.set_state(replay.augment.get_state())
    return replay.data.batch(positions, generator)


def partition(data):
    order = DataOrder(list(range(len(data.manifest['training']))), PARTITION_SEED)
    return [order.take(PARTITION_BATCH) for _ in range(PARTITION_COUNT)]


def reference_inputs(data):
    order = DataOrder(list(range(len(data.manifest['training']))), REFERENCE_SEED)
    first, second = order.take(states.BATCH), order.take(states.BATCH)
    return torch.cat([data.batch(first)[0], data.batch(second)[0]])


def whiten_all(vector, specs, contexts):
    return {s.name: ggn.whiten(vector[s.slice].reshape(s.shape)[None], contexts[s.name])[0].reshape(-1) for s in specs}


def exact_forms(matrix, vectors, powers):
    """Reference: float64 eigendecomposition; (len(powers), p) forms vᵀA^q v."""
    values, basis = torch.linalg.eigh(matrix.double())
    projected = (basis.T@vectors.double()).square()
    return torch.stack([((values.clamp_min(torch.finfo(torch.float64).tiny)**q)[:, None]*projected).sum(0) for q in powers])


def analyse(replay, data, partition_positions, reference_x, check_g2):
    model, registry, optimizer = replay.arm.model, replay.arm.registry, replay.arm.optimizer
    flat = replay.flat().detach().clone()
    contexts = copy.deepcopy(states.contexts(optimizer, registry))
    scale = SOAPMetric(registry, contexts).scale
    x_step, y_step = peek_step_batch(replay)
    _, trace = ggn.accumulate(model, registry, flat, x_step, [], CHUNK, all_parameters=True, contexts=contexts)
    mu = max(GAMMA*trace/registry.size, FLOOR)
    delta = mu*scale
    specs = matrix_specs(registry)
    gradients = []
    for positions in partition_positions:
        x, y = data.batch(positions)
        gradients.append(whiten_all(micro.gradient(model, registry, flat, x, y, states.MICRO)[1], specs, contexts))
    step_gradient = whiten_all(micro.gradient(model, registry, flat, x_step, y_step, states.MICRO)[1], specs, contexts)
    per_tensor, steps = {}, {nu: torch.zeros_like(flat) for nu in NUS}
    scale_terms = {nu: {'lin': 0., 'cur': 0.} for nu in NUS}
    g2 = {}
    for group in groups(specs, BUDGET, flat.element_size()):
        blocks, _ = ggn.accumulate(model, registry, flat, reference_x, group, GROUP_CHUNK, all_parameters=False,
                                   contexts=contexts)
        for spec in group:
            matrix = ggn.symmetrize_(blocks.pop(spec.name))
            matrix.diagonal().add_(delta)
            stack = torch.stack([g[spec.name] for g in gradients], 1)  # (D, m)
            corpus = stack.mean(1)
            deviations = stack-corpus[:, None]
            start = torch.cat([corpus[:, None], step_gradient[spec.name][:, None], deviations], 1)
            run = lanczos.lanczos(lambda v: matrix@v, start, STEPS)
            forms = run.forms(POWERS)  # (q, 2 + m)
            record = {'count': spec.count, 'shape': list(spec.shape),
                      'corpus': forms[:, 0].tolist(), 'step': forms[:, 1].tolist(),
                      'noise_mean': forms[:, 2:].mean(1).tolist(),
                      'min_ritz_over_delta': float(run.min_ritz().min())/delta}
            zb = step_gradient[spec.name]
            for nu in NUS:
                y = -zb if nu == 0 else -run.apply(1, -nu)
                steps[nu][spec.slice] = ggn.unwhiten(y.reshape(spec.shape), contexts[spec.name]).reshape(-1)
                scale_terms[nu]['lin'] += float((zb.double()*y.double()).sum())
                scale_terms[nu]['cur'] += float((y.double()*(matrix@y).double()).sum())
            if check_g2 and spec.name in G2_TENSORS:
                reference = exact_forms(matrix, start, POWERS)
                approximation = forms
                relative = ((approximation-reference).abs()/reference.abs().clamp_min(torch.finfo(torch.float64).tiny))
                g2[spec.name] = float(relative.max())
            per_tensor[spec.name] = record
            del matrix, run, stack, deviations, start
        del blocks
    if g2 and max(g2.values()) > G2_TOLERANCE:
        raise RuntimeError(f'Gate G2 failed at step {replay.t}: {g2}')
    return {'t': replay.t, 'mu': mu, 'scale': scale, 'delta': delta, 'trace': trace, 'tensors': per_tensor,
            'g2': g2, 'scale_terms': scale_terms}, steps, flat


def score_steps(replay, data, flat, steps, scale_terms):
    """Selection on the batch after the step batch, evaluation on the next one (unaugmented); validation decrease
    at the selected multiplier (descriptive)."""
    model, registry = replay.arm.model, replay.arm.registry
    xs, ys = data.batch(peek(replay, 1))
    xe, ye = data.batch(peek(replay, 2))
    xv, yv = data.validation, data.validation_y
    loss = lambda point, x, y: micro.batch_loss(model, registry, point, x, y, states.MICRO)
    base = {'selection': loss(flat, xs, ys), 'evaluation': loss(flat, xe, ye), 'validation': loss(flat, xv, yv)}
    out = {}
    for nu, direction in steps.items():
        lin, cur = scale_terms[nu]['lin'], scale_terms[nu]['cur']
        alpha0 = -lin/cur if lin < 0 and cur > 0 else 0.
        selection = {m: base['selection']-loss(flat+alpha0*m*direction, xs, ys) for m in MULTIPLIERS}
        chosen = max(selection, key=selection.get)
        point = flat+alpha0*chosen*direction
        out[str(nu)] = {'alpha0': alpha0, 'multiplier': chosen, 'interior': MULTIPLIERS[0] < chosen < MULTIPLIERS[-1],
                        'selection': selection[chosen], 'evaluation': base['evaluation']-loss(point, xe, ye),
                        'validation': base['validation']-loss(point, xv, yv)}
    return {'base': base, 'steps': out}
