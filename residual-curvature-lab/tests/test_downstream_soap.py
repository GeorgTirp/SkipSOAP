import pytest
import torch

from residual_curvature.optim.downstream_soap import (basis_map, coordinate_action, coordinate_candidate,
    exact_coordinate_diagonal, probe_coordinate_diagonal, transfer_tensor_norms)
from residual_curvature.optim.soap import SOAPMetric, soap_propose


def test_curvature_coordinates_and_actual_nonorthogonal_metric(tiny, config):
    _, registry, flat, _, _, op, _, factor = tiny
    params, grads = registry.unflatten(flat), registry.unflatten(op.gradient)
    _, state, _ = soap_propose(params, grads, None, config['soap'])
    _, _, contexts = soap_propose(params, grads, state, config['soap'])
    metric = SOAPMetric(registry, contexts)
    dense = factor @ factor.T
    blocks = [dense[s.slice, s.slice] for s in registry.specs]
    diagonal, _ = exact_coordinate_diagonal(blocks, registry, contexts)
    torch.testing.assert_close(diagonal, probe_coordinate_diagonal(factor, registry, contexts), rtol=1e-9, atol=1e-12)
    q = basis_map(torch.eye(registry.size), registry, contexts, 'forward')
    torch.testing.assert_close(diagonal, (q.T @ dense @ q).diag(), rtol=1e-9, atol=1e-12)
    qi = torch.linalg.inv(q)
    approximation = qi.T @ torch.diag(diagonal) @ qi
    v = torch.linspace(-1., 1., registry.size)
    torch.testing.assert_close(coordinate_action(v, diagonal, registry, contexts), approximation @ v, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(coordinate_action(torch.eye(registry.size), diagonal, registry, contexts),
                               approximation, rtol=1e-9, atol=1e-12)
    mu = .01
    system = approximation + mu * metric.apply(torch.eye(registry.size))
    proposal = coordinate_candidate(op.gradient, diagonal, metric, mu)
    reference = torch.linalg.solve(system, -op.gradient)
    # Early SOAP metrics yield condition numbers around 1e8. Use the declared
    # normwise solve tolerance, not elementwise relative errors on tiny entries.
    assert float((proposal-reference).norm()/reference.norm()) < 1e-8
    assert float((system @ proposal + op.gradient).norm()/op.gradient.norm()) < 1e-8
    # Curvature is the only new information: zero curvature recovers -(mu Omega)^-1 g.
    torch.testing.assert_close(coordinate_candidate(op.gradient, diagonal*0, metric, mu),
                               -metric.inverse(op.gradient)/mu, rtol=1e-9, atol=1e-10)


def test_share_transfer_preserves_directions_and_norms(tiny):
    _, registry, flat, _, _, op, _, _ = tiny
    with pytest.raises(ValueError, match='absent direction'):
        transfer_tensor_norms(flat, op.gradient, registry)
    directions = flat + 2.0
    transferred = transfer_tensor_norms(directions, op.gradient, registry)
    for spec in registry.specs:
        assert abs(float(transferred[spec.slice].norm()-op.gradient[spec.slice].norm())) < 1e-12
        if op.gradient[spec.slice].norm() > 0:
            torch.testing.assert_close(transferred[spec.slice]/transferred[spec.slice].norm(),
                                       directions[spec.slice]/directions[spec.slice].norm(), rtol=1e-10, atol=1e-12)
