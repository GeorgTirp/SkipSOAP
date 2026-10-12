"""Nonnegative tensor amplitudes for frozen SOAP tensor directions."""
import torch

from ..curvature.reduced import reduced_ggn
from .allocation import direction_matrix, nonnegative_qp


def reduced_allocation(operator, native, registry, metric, mu, tolerance=1e-8):
    if mu <= 0:
        raise ValueError('Positive damping required')
    e, names, excluded = direction_matrix(native, registry)
    if e.shape[1] == 0:
        return {'allocation_diagonal': native*0, 'allocation_full': native*0}, {
            'available': False, 'excluded': excluded}, {}
    k, cost = reduced_ggn(operator, e)
    phi = -e.T @ operator.gradient
    h = k + mu * e.T @ metric.apply(e)
    h = (h+h.T)/2
    full, kkt = nonnegative_qp(h, phi, tolerance)
    diagonal = (phi/h.diag()).clamp_min(0)
    native_amplitudes = e.T @ native
    ray_denominator = float(native_amplitudes @ h @ native_amplitudes)
    ray_slope = float(phi @ native_amplitudes)
    ray_gain = .5*max(ray_slope, 0.)**2/ray_denominator if ray_denominator > 0 else 0.
    full_gain = float(phi@full - .5*full@h@full)
    if full_gain < ray_gain - tolerance*max(1., abs(ray_gain)):
        raise RuntimeError('Full restricted allocation is worse than its feasible native ray')
    candidates = {'allocation_diagonal': e @ diagonal, 'allocation_full': e @ full}
    metadata = {'available': True, 'names': names, 'excluded': excluded, 'kkt': kkt,
        'directional_curvature': cost, 'dimension': len(names), 'native_quadratic_ray_gain': ray_gain,
        'full_quadratic_gain': full_gain,
        'diagonal_candidate_full_quadratic_gain': float(phi@diagonal - .5*diagonal@h@diagonal),
        'uphill_directions_retained_in_full': [n for n, value in zip(names, phi) if value < 0],
        'active_full': int((full > 0).sum()), 'active_diagonal': int((diagonal > 0).sum()),
        'amplitudes_full': full.tolist(), 'amplitudes_diagonal': diagonal.tolist(),
        'shares_full': (full.square()/full.square().sum()).tolist() if full.norm() else None,
        'shares_diagonal': (diagonal.square()/diagonal.square().sum()).tolist() if diagonal.norm() else None}
    return candidates, metadata, {'E': e, 'K': k, 'H': h, 'phi': phi,
                                   'amplitudes_full': full, 'amplitudes_diagonal': diagonal}
