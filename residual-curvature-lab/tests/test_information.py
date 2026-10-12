import copy

import torch

from residual_curvature.curvature.information import information_blocks, match_traces, coordinate_diagonal_blocks
from residual_curvature.optim.downstream_soap import exact_coordinate_diagonal, coordinate_action
from residual_curvature.optim.soap import SOAPMetric, soap_propose


def test_information_targets_samples_and_nonmutation(tiny):
    model, registry, flat, x, y, op, jac, factor = tiny
    before, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    blocks, gradient, labels, metadata = information_blocks(model, registry, flat, x, y, ranks=(1, 4), repeats=2, microbatch=3)
    p = op.logits.detach().softmax(-1)
    jac = jac.reshape(len(x), p.shape[-1], registry.size)
    mean = (p[..., None]*jac).sum(1)
    scores = mean-jac[torch.arange(len(x)), y]
    ggn, ef = factor@factor.T, scores.T@scores/len(x)
    torch.testing.assert_close(gradient, op.gradient, rtol=1e-10, atol=1e-12)
    for spec, c, f in zip(registry.specs, blocks['ggn'], blocks['empirical_fisher']):
        torch.testing.assert_close(c, ggn[spec.slice, spec.slice], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(f, ef[spec.slice, spec.slice], rtol=1e-10, atol=1e-12)
    for repeat in range(2):
        for rank in (1, 4):
            sampled = mean[:, None]-jac[torch.arange(len(x))[:, None], labels[repeat, :, :rank]]
            sampled = sampled.reshape(-1, registry.size)
            expected = sampled.T@sampled/(len(x)*rank)
            for spec, block in zip(registry.specs, blocks[f'sampled_{rank}_repeat_{repeat}']):
                torch.testing.assert_close(block, expected[spec.slice, spec.slice], rtol=1e-10, atol=1e-12)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[n], v) for n, v in model.state_dict().items())
    assert metadata['jacobian_rows'] == len(x)*p.shape[1]


def test_coordinate_trace_control_and_momentum_rhs(tiny, config):
    _, registry, flat, _, _, op, _, factor = tiny
    params, grads = registry.unflatten(flat), registry.unflatten(op.gradient)
    _, state, _ = soap_propose(params, grads, None, config['soap'])
    proposal, _, contexts = soap_propose(params, grads, state, config['soap'])
    native = registry.flatten(proposal)
    metric = SOAPMetric(registry, contexts)
    rhs = -metric.map(native, inverse=True)
    torch.testing.assert_close(-metric.map(rhs), native, rtol=1e-9, atol=1e-11)
    dense = factor@factor.T
    blocks = [dense[s.slice, s.slice] for s in registry.specs]
    diagonal, _ = exact_coordinate_diagonal(blocks, registry, contexts)
    approximation = coordinate_diagonal_blocks(diagonal, registry, contexts)
    vector = torch.linspace(-1, 1, registry.size)
    actual = torch.cat([b@vector[s.slice] for s, b in zip(registry.specs, approximation)])
    torch.testing.assert_close(actual, coordinate_action(vector, diagonal, registry, contexts), rtol=1e-10, atol=1e-12)
    matched, info = match_traces(approximation, blocks, registry)
    for spec, b, c in zip(registry.specs, matched, blocks):
        if spec.name not in info['numerically_null_tensors']:
            torch.testing.assert_close(b.trace(), c.trace(), rtol=1e-10, atol=1e-12)
        assert torch.linalg.eigvalsh(b).min() >= -1e-10
