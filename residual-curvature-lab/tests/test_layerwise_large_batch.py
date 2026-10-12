import copy
import runpy

import pytest
import torch

from residual_curvature.curvature.layerwise import streamed_blocks, block_action, solve_blocks
from residual_curvature.curvature.solves import IdentityMetric, diagonal_solve
from residual_curvature.experiments.search import broad_direction_search
from residual_curvature.optim.soap import SOAPMetric, soap_propose


def test_streamed_exact_blocks_uneven_microbatch_and_solves(tiny, config):
    model, registry, flat, x, _, operator, _, factor = tiny
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    # Four examples in chunks of three catches mean-normalization errors.
    blocks, metadata = streamed_blocks(model, registry, flat, x, 3, 16)
    dense = factor @ factor.T
    for spec, block in zip(registry.specs, blocks):
        torch.testing.assert_close(block, dense[spec.slice, spec.slice], rtol=1e-9, atol=1e-12)
    assert abs(metadata['trace'] - float(dense.trace())) < 1e-10
    params, gradients = registry.unflatten(flat), registry.unflatten(operator.gradient)
    _, state, _ = soap_propose(params, gradients, None, config['soap'])
    _, _, contexts = soap_propose(params, gradients, state, config['soap'])
    for metric in [IdentityMetric(), SOAPMetric(registry, contexts)]:
        solutions, checks = solve_blocks(blocks, registry, operator.gradient, metric, [.01, .1], 1e-8)
        for mu, solution in solutions.items():
            reference = diagonal_solve(operator.gradient, factor, mu, metric.inverse, registry.groups())
            torch.testing.assert_close(solution, reference, rtol=1e-7, atol=1e-8)
            torch.testing.assert_close(block_action(blocks, registry, solution) + mu * metric.apply(solution),
                                       -operator.gradient, rtol=1e-8, atol=1e-10)
            assert checks[mu]['residual'] < 1e-8
    assert torch.equal(rng, torch.get_rng_state())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
    with pytest.raises(MemoryError):
        streamed_blocks(model, registry, flat, x, 3, .0001)


def test_broad_search_reaches_beyond_old_limit_and_preserves_state():
    flat, delta = torch.tensor([0.]), torch.tensor([7.])
    before = flat.clone()
    selected, metadata = broad_direction_search(flat, delta, 1., lambda p: (p[0]-37.3)**2)
    assert abs(float(selected) - 37.3) < .01
    assert metadata['bracketed'] and not metadata['boundary']
    assert metadata['evaluations'] == metadata['evaluation_budget'] == 52
    assert torch.equal(flat, before)
    rejected, check = broad_direction_search(flat, -delta, 1., lambda p: (p[0]-37.3)**2)
    assert check['rejected'] and torch.equal(rejected, flat)
    _, check = broad_direction_search(flat, delta, 1., lambda p: -p[0])
    assert check['boundary'] and not check['bracketed']


def test_exact_training_proposal_advances_statistics_once_and_replays(tiny, config):
    propose = runpy.run_path('scripts/run_exact_training_followup.py')['propose']
    equal = runpy.run_path('scripts/run_large_batch_oracle.py')['state_equal']
    model, registry, flat, x, y, _, _, _ = tiny
    before, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    zero, state, _, _ = propose('exact_tensor_ggn', model, registry, flat, None, x, y, config)
    assert zero.norm() == 0
    saved = copy.deepcopy(state)
    delta, next_state, _, measurements = propose('exact_tensor_ggn', model, registry, flat, state, x, y, config)
    replay, replay_state, _, replay_measurements = propose('exact_tensor_ggn', model, registry, flat, saved, x, y, config)
    _, soap_state, _, _ = propose('native_soap', model, registry, flat, saved, x, y, config)
    assert equal(next_state, soap_state) and equal(next_state, replay_state)
    assert equal(state, saved) and equal(before, model.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(delta, replay, rtol=0, atol=0)
    assert measurements['loss_after'] <= measurements['loss_before']
    assert measurements['solve']['residual'] < 1e-8
    assert measurements['search']['trials'] == replay_measurements['search']['trials']
