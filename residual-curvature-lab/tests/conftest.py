import copy

import pytest
import torch

from residual_curvature.config import DEFAULT
from residual_curvature.data import synthetic
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.curvature.operators import GGN


@pytest.fixture(autouse=True)
def cpu_defaults():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def config():
    return copy.deepcopy(DEFAULT)


@pytest.fixture(params=[0, 7, 19])
def tiny(request, config):
    config['seed'] = request.param
    model = make_model(config)
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    x, y, _ = synthetic(config)
    x, y = x[:4], y[:4]
    fn = functional_logits(model, registry, x)
    op = GGN(fn, flat, y)
    j, u, _ = op.factors()
    return model, registry, flat, x, y, op, j, u
