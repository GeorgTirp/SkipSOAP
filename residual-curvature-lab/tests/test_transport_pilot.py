import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from residual_curvature.models import make_model, Registry
from residual_curvature.optim.ni_soap import matrix_routes

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
from run_transport_parsec_noise import METHODS, NI_LR_METHODS, optimizer_settings, propose
from run_large_batch_oracle import state_equal


@pytest.mark.parametrize('method', METHODS + NI_LR_METHODS)
def test_pilot_proposal_nonmutation_and_replay(config, method):
    config['model'].update(kind='vit', input_dim=16, width=4, expansion=6, blocks=1, classes=3)
    config['vit'].update(image_size=4, patch_size=2, heads=2)
    model = make_model(config)
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    routes = matrix_routes(model)
    settings = optimizer_settings(method)
    if method.startswith('parsec'):
        settings.update(rank=2, lanczos_iters=4, refresh_every=2)
    x, y = torch.linspace(-1, 1, 48).reshape(3, 16), torch.tensor([0, 2, 1])
    state = None
    for step in [1, 2]:
        before = copy.deepcopy(state)
        before_model, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
        a, next_state, _, info = propose(method, model, registry, flat, state, x, y, config, step, settings, routes)
        b, replay, _, _ = propose(method, model, registry, flat, state, x, y, config, step, settings, routes)
        assert torch.equal(a, b) and state_equal(next_state, replay)
        assert state_equal(state, before) and state_equal(model.state_dict(), before_model)
        assert torch.equal(rng, torch.get_rng_state())
        if info['search']:
            assert info['search']['bracketed'] or info['search']['rejected']
        state = next_state


@pytest.mark.parametrize('alpha', [0., 1.])
def test_noise_lr_path_matches_direct_optimizer_settings(config, alpha):
    from residual_curvature.optim.ni_soap import decompose
    from residual_curvature.experiments.search import noise_learning_rate_search, broad_direction_search
    model = make_model(config)
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    routes = matrix_routes(model)
    gradient = torch.linspace(-.2, .3, registry.size)
    settings = optimizer_settings('ni_soap_alpha1')
    settings['alpha'] = alpha
    _, _, state, _ = decompose(gradient, flat, registry, routes, None, settings)
    drift, noise, next_state, _ = decompose(gradient, flat, registry, routes, state, settings)
    for multiplier in [.01, .5, 3.]:
        changed = {**settings, 'lr': settings['lr']*multiplier}
        d, n, direct_state, _ = decompose(gradient, flat, registry, routes, state, changed)
        torch.testing.assert_close(d+n, multiplier*drift+multiplier**.5*noise, atol=1e-14, rtol=1e-12)
        assert state_equal(direct_state, next_state)
    target = flat+drift+noise
    objective = lambda value: (value-target).square().sum()
    delta, info = noise_learning_rate_search(flat, drift, noise, objective, settings['lr'])
    assert info['selected_alpha'] == 1. and info['bracketed']
    torch.testing.assert_close(delta, drift+noise)
    if alpha == 0:
        ray, ray_info = broad_direction_search(flat, drift, float(drift.norm()), objective)
        assert info['selected_alpha'] == ray_info['selected_alpha']
        torch.testing.assert_close(delta, ray)
