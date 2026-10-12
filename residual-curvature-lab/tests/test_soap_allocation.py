import copy

import torch

from third_party.soap.soap import SOAP
from residual_curvature.models import Registry, make_model
from residual_curvature.optim.soap import SOAPMetric, soap_propose, default_dtype, frozen_apply
from residual_curvature.optim.allocation import allocation_candidates, direction_matrix, ray_value
from residual_curvature.curvature.solves import IdentityMetric, factor_solve


def assert_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree(x, y)
    else:
        assert a == b


def test_soap_reference_state_and_context(config):
    model = make_model(config)
    registry = Registry(model)
    params = dict(model.named_parameters())
    settings = dict(config['soap'], precondition_frequency=2)
    reference = SOAP(params.values(), **settings)
    state = None
    gen = torch.Generator().manual_seed(234)
    for iteration in range(8):
        before = {n: p.detach().clone() for n, p in params.items()}
        grads = {n: torch.randn(p.shape, generator=gen) for n, p in params.items()}
        frozen = copy.deepcopy(state)
        delta, next_state, context = soap_propose(before, grads, state, settings)
        assert_tree(frozen, state)
        for name, p in params.items():
            p.grad = grads[name]
        with default_dtype(torch.float64):
            reference.step()
        for name, p in params.items():
            assert torch.equal(delta[name], p.detach() - before[name])
            if iteration:
                c = context[name]
                # Reconstruct update from used momentum, basis, denominator and scalar.
                update = reference.project_back(c['momentum'] / c['denominator'], {'Q': c['basis']})
                torch.testing.assert_close(delta[name], -c['lr'] * c['bias_correction'] * update, rtol=1e-9, atol=1e-14)
            else:
                assert context[name] is None and torch.count_nonzero(delta[name]) == 0
        assert_tree(next_state, reference.state_dict())
        state = next_state
    metric = SOAPMetric(registry, context)
    v, w = torch.randn(registry.size, generator=gen), torch.randn(registry.size, generator=gen)
    torch.testing.assert_close(v @ metric.map(w), w @ metric.map(v), rtol=1e-9, atol=1e-11)
    assert v @ metric.map(v) > 0
    torch.testing.assert_close(metric.apply(metric.inverse(v)), v, rtol=1e-9, atol=1e-11)
    omega = metric.apply(torch.eye(registry.size))
    torch.testing.assert_close(omega.trace()/registry.size, torch.tensor(1.), rtol=1e-9, atol=1e-11)
    assert torch.linalg.eigvalsh(omega).min() > 0


def test_allocation_kkt_ordering_and_zero_uphill(tiny):
    _, registry, _, _, _, op, _, u = tiny
    g, mu = op.gradient, .1
    native = -g.clone()
    native[registry.specs[0].slice] = 0
    native[registry.specs[1].slice] *= -1
    candidates, details, e, k = allocation_candidates(g, u, IdentityMetric(), mu, native, registry)
    torch.testing.assert_close(e.T @ e, torch.eye(e.shape[1]), rtol=1e-9, atol=1e-11)
    assert details['excluded'] and details['uphill_retained']
    assert max(details['kkt'].values()) < 1e-8
    action = lambda v: u @ (u.T @ v) + mu * v
    values = [ray_value(g, d, action)['value'] for d in [native, candidates['allocation_cone'], candidates['allocation_span'], factor_solve(g, u, mu, lambda v: v)]]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))
    phi = -e.T @ g
    span = .5 * phi @ torch.linalg.solve(k + mu * torch.eye(k.shape[0]), phi)
    assert abs(float(span)-values[2]) < 1e-9
    empty, detail, _, _ = allocation_candidates(g, u, IdentityMetric(), mu, native*0, registry)
    assert empty == {} and detail['unavailable']
