import copy
import runpy
import sys

import torch

from residual_curvature.models import Registry, make_model


def test_vit_allocation_step_statistics_nonmutation_and_replay(config):
    config['model'].update(kind='vit', input_dim=16, width=4, expansion=6, blocks=1, classes=3)
    config['vit'].update(image_size=4, patch_size=2, heads=2)
    sys.path.insert(0, 'scripts')
    try:
        propose = runpy.run_path('scripts/run_mini_vit_allocation.py')['propose']
        equal = runpy.run_path('scripts/run_large_batch_oracle.py')['state_equal']
    finally:
        sys.path.pop(0)
    model = make_model(config)
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    x = torch.linspace(-1, 1, 48).reshape(3, 16)
    y = torch.tensor([0, 1, 2])
    _, state, _, _ = propose('native_soap', model, registry, flat, None, x, y, config, 1)
    before, saved, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(state), torch.get_rng_state().clone()
    _, expected_state, _, _ = propose('native_soap', model, registry, flat, state, x, y, config, 2)
    for method in ['allocation_diagonal', 'allocation_full', 'exact_tensor_ggn']:
        delta, next_state, _, info = propose(method, model, registry, flat, state, x, y, config, 2)
        assert equal(expected_state, next_state)
        assert info['loss_after'] <= info['loss_before'] + 1e-12
        assert equal(state, saved) and equal(model.state_dict(), before)
        assert torch.equal(rng, torch.get_rng_state())
        replay, _, _, replay_info = propose(method, model, registry, flat, state, x, y, config, 2)
        torch.testing.assert_close(delta, replay, rtol=0, atol=0)
        assert info['search']['trials'] == replay_info['search']['trials']
