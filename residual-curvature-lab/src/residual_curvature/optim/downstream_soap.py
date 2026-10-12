"""Frozen SOAP coordinates with output-curvature-derived diagonal scaling.

This is a local quadratic candidate, not an unchanged SOAP optimizer. Q denotes
the actual saved tensor-product SOAP basis, including its small nonorthogonality.
We use C_hat = Q^-T diag(c) Q^-1 and Omega = scale Q^-T diag(d) Q^-1.
"""
import time

import torch

from .soap import mode


def basis_map(values, registry, contexts, operation):
    result = torch.empty_like(values)
    for spec in registry.specs:
        local = values[spec.slice].reshape(spec.shape + values.shape[1:])
        for axis, q in enumerate(contexts[spec.name]['basis']):
            local = mode(local, q, axis, operation)
        result[spec.slice] = local.reshape((spec.count,) + values.shape[1:])
    return result


def exact_coordinate_diagonal(blocks, registry, contexts):
    started = time.perf_counter()
    diagonal = blocks[0].new_empty(registry.size)
    for spec, block in zip(registry.specs, blocks):
        def rotate(matrix):
            value = matrix.reshape(spec.shape + (spec.count,))
            for axis, q in enumerate(contexts[spec.name]['basis']):
                value = mode(value, q, axis, 'transpose')
            return value.reshape(spec.count, spec.count)
        # rotate(rotate(C).T) = Q^T C^T Q; C is symmetric.
        local = rotate(rotate(block).T).diag()
        tolerance = 100 * torch.finfo(local.dtype).eps * max(1., float(local.abs().max()))
        if float(local.min()) < -tolerance:
            raise RuntimeError('Negative exact curvature diagonal beyond roundoff')
        diagonal[spec.slice] = local.clamp_min(0)
    return diagonal, {'seconds': time.perf_counter()-started, 'representation': 'diag(Q^T C_bb Q), exact each tensor'}


def probe_coordinate_diagonal(factors, registry, contexts):
    return basis_map(factors, registry, contexts, 'transpose').square().sum(1)


def coordinate_action(v, diagonal, registry, contexts):
    weights = diagonal.reshape(diagonal.shape + (1,) * (v.ndim - 1))
    return basis_map(weights * basis_map(v, registry, contexts, 'solve'), registry, contexts, 'solve_transpose')


def coordinate_candidate(g, diagonal, metric, mu):
    if mu <= 0 or (diagonal < 0).any() or not torch.isfinite(diagonal).all():
        raise ValueError('Finite nonnegative curvature and positive damping required')
    registry, contexts = metric.registry, metric.contexts
    denom = torch.cat([contexts[s.name]['denominator'].reshape(-1) for s in registry.specs])
    rhs = basis_map(g, registry, contexts, 'transpose')
    coordinates = -rhs / (diagonal + mu * metric.scale * denom)
    return basis_map(coordinates, registry, contexts, 'forward')


def transfer_tensor_norms(directions, norm_source, registry):
    """Keep unit directions but assign another update's complete tensor norms."""
    result = torch.zeros_like(directions)
    for spec in registry.specs:
        norm, target = directions[spec.slice].norm(), norm_source[spec.slice].norm()
        if norm == 0 and target != 0:
            raise ValueError(f'Cannot allocate nonzero norm to absent direction {spec.name}')
        if norm > 0:
            result[spec.slice] = directions[spec.slice] * (target / norm)
    return result
