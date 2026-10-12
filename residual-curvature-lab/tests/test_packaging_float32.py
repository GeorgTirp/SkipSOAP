import copy

import torch

from residual_curvature.curvature.operators import GGN
from residual_curvature.curvature.solves import cg, factor_solve
from residual_curvature.data import synthetic
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits


def test_float32_operator_and_solve_with_separate_tolerances(config):
    config['dtype'] = 'float32'
    model = make_model(config)
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    x, y, _ = synthetic(config)
    op = GGN(functional_logits(model, registry, x[:4]), flat, y[:4])
    _, u, _ = op.factors()
    v = torch.linspace(-1, 1, registry.size, dtype=torch.float32)
    torch.testing.assert_close(op.action(v), u @ (u.T @ v), rtol=1e-4, atol=1e-6)
    mu = .1
    delta = factor_solve(op.gradient, u, mu, lambda value: value)
    assert (op.action(delta) + mu * delta + op.gradient).norm()/op.gradient.norm() < config['solver']['tolerance32']


def test_initialization_independent_of_callers_default_dtype(config):
    first = make_model(config).state_dict()
    torch.set_default_dtype(torch.float32)
    second = make_model(config).state_dict()
    assert torch.get_default_dtype() == torch.float32
    for key in first:
        assert torch.equal(first[key], second[key])
