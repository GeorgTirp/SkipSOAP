"""Part A (PROTOCOL.md, section 4): the basis ladder for N9 at one SOAP state.

Candidates per tensor: SOAP's map; exact diagonals in SOAP's held basis, in the eigenbasis of SOAP's current factors,
in the probe-GGN Kronecker basis (N9) and in the nearest-Kronecker basis of the exact block (ceiling); the probe basis
with the probe diagonal (deployable N9); and the exact per-tensor GGN step. Each is scored by its ray reduction on
the damped per-tensor quadratic (dense blocks), and globally by the loss decrease on an evaluation batch at a step
size chosen on a selection batch.
"""
import time

import torch

import ggn
import micro
import quad
import states
from residual_curvature.optim.soap import SOAPMetric

GAMMA, FLOOR, TOLERANCE, BUDGET, CHUNK, GROUP_CHUNK = 1e-2, 1e-8, 1e-4, int(2.5*2**30), 64, 128
CONSISTENCY = 1e-3  # gate A0
CANDIDATES = ['soap_map', 'soap_exact', 'soapeigh_exact', 'probe_exact', 'probe_probe', 'kron_exact', 'ggn']
MULTIPLIERS = [2.**k for k in range(-3, 4)]
POSTHOC_MULTIPLIERS = [2.**k for k in range(-14, 1)]  # DEVIATIONS.md D2
PROBE_SEED = 9100


def run_state(replay, probe_seed, multipliers=MULTIPLIERS):
    """Ladder at the replay's current state (after step t), with the batch of step t + 1."""
    started = time.perf_counter()
    model, registry, optimizer = replay.arm.model, replay.arm.registry, replay.arm.optimizer
    x, y = replay.next_batch()
    xs, ys = replay.following(0)
    xe, ye = replay.following(1)
    flat = replay.flat()
    _, g = micro.gradient(model, registry, flat, x, y, states.MICRO)
    contexts = states.soap_contexts(optimizer, registry)
    metric = SOAPMetric(registry, contexts)
    delta, solve = ggn.tensor_ggn_step(model, registry, flat, x, g, metric, GAMMA, FLOOR, TOLERANCE, BUDGET, CHUNK,
                                       group_chunk=GROUP_CHUNK)
    mu, scale = solve['mu'], solve['scale']
    params = {spec.name: p for p, spec in zip(model.parameters(), registry.specs)}

    # Probe-derived bases and diagonals first, then free the probe rows (2,048 x D).
    probes = micro.lean_probes(model, x, samples=1, generator=torch.Generator().manual_seed(probe_seed))
    probe_bases, probe_diagonals = {}, {}
    for spec in registry.specs:
        q = contexts[spec.name]['basis']
        modes = {k for k, m in enumerate(q) if len(m)}
        rows = probes[params[spec.name]]
        bases = quad.factor_bases(quad.probe_factors(rows, modes))
        probe_bases[spec.name] = bases
        probe_diagonals[spec.name] = quad.probe_diagonal(rows, bases)
    del probes

    vectors = {name: torch.zeros_like(flat) for name in CANDIDATES}
    rows, failures = {}, []
    for specs in ggn.groups_for(registry, BUDGET, flat.element_size()):
        blocks, _ = ggn.accumulate(model, registry, flat, x, specs, GROUP_CHUNK, all_parameters=False, contexts=None)
        for spec in specs:
            block = ggn.symmetrize_(blocks.pop(spec.name))
            context = contexts[spec.name]
            soap_bases, den = context['basis'], context['denominator']
            gb = g[spec.slice].reshape(spec.shape)
            state = optimizer.state[params[spec.name]]
            bases = {'soap': soap_bases, 'soapeigh': quad.factor_bases(state['GG']), 'probe': probe_bases[spec.name]}
            if len(spec.shape) == 2 and all(len(m) for m in soap_bases):
                bases['kron'] = quad.factor_bases(quad.nearest_kronecker(block, spec.shape))
            else:
                bases['kron'] = soap_bases if not any(len(m) for m in soap_bases) else bases['soapeigh']
            steps = {'soap_map': quad.soap_map(gb, soap_bases, den), 'ggn': delta[spec.slice].reshape(spec.shape)}
            for key, basis in bases.items():
                diagonal = quad.exact_diagonal(block, spec.shape, basis)+mu*scale*quad.damping_diagonal(den, soap_bases, basis)
                steps[f'{key}_exact'] = quad.apply_diagonal(gb, basis, diagonal)
            probe_diag = probe_diagonals.pop(spec.name)+mu*scale*quad.damping_diagonal(den, soap_bases, probe_bases[spec.name])
            steps['probe_probe'] = quad.apply_diagonal(gb, probe_bases[spec.name], probe_diag)
            optimum = -.5*float((gb*steps['ggn']).sum())
            row = {'shape': list(spec.shape), 'optimum': optimum, 'candidates': {}}
            for name in CANDIDATES:
                d = steps[name]
                linear = float((gb*d).sum())
                flat_d = d.reshape(-1)
                curvature = float(flat_d@(block@flat_d))
                damping = quad.damping_quadratic(d, context, mu, scale)
                reduction = quad.ray(linear, curvature+damping)
                row['candidates'][name] = {'linear': linear, 'curvature': curvature, 'damping': damping, 'ray': reduction,
                                           'as_is': -(linear+.5*(curvature+damping))}
                vectors[name][spec.slice] = flat_d
                if reduction > optimum*(1+CONSISTENCY)+1e-12:
                    failures.append({'tensor': spec.name, 'candidate': name, 'ray': reduction, 'optimum': optimum})
            rows[spec.name] = row
            del block
        del blocks
    if failures:
        raise RuntimeError(f'Gate A0 failed: {failures[:5]}')

    # Transfer: global step alpha0 * 2^k, k chosen on the selection batch, decrease on the evaluation batch.
    base_selection = micro.batch_loss(model, registry, flat, xs, ys, states.MICRO)
    base_evaluation = micro.batch_loss(model, registry, flat, xe, ye, states.MICRO)
    transfer = {}
    for name in CANDIDATES:
        linear = sum(r['candidates'][name]['linear'] for r in rows.values())
        curvature = sum(r['candidates'][name]['curvature']+r['candidates'][name]['damping'] for r in rows.values())
        alpha0 = -linear/curvature if linear < 0 and curvature > 0 else 0.
        selection = {m: base_selection-micro.batch_loss(model, registry, flat+alpha0*m*vectors[name], xs, ys, states.MICRO)
                     for m in multipliers}
        chosen = max(selection, key=selection.get)
        transfer[name] = {'alpha0': alpha0, 'multiplier': chosen, 'selection_decrease': selection[chosen],
                          'evaluation_decrease': base_evaluation-micro.batch_loss(model, registry, flat+alpha0*chosen*vectors[name],
                                                                                  xe, ye, states.MICRO)}
    return {'step': replay.t, 'seed': replay.seed, 'mu': mu, 'scale': scale, 'solve_residual': solve['residual'],
            'tensors': rows, 'transfer': transfer, 'seconds': time.perf_counter()-started}
