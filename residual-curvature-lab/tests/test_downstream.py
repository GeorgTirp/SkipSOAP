import copy

import pytest
import torch
from torch import nn

from residual_curvature.curvature.downstream import nested_probe_prefix, propagate_metric_probes
from residual_curvature.curvature.operators import GGN, output_factor
from residual_curvature.curvature.sketch import categorical_probes, factors_from_probes
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from test_soap_allocation import assert_tree


@pytest.mark.parametrize('branch_scale', [0., .7])
def test_stage_vjp_shared_factors_all_parameters_and_uneven_microbatches(config, branch_scale):
    config['model']['branch_scale'] = branch_scale
    model = make_model(config)
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    generator = torch.Generator().manual_seed(912)
    x = torch.randn(5, config['model']['input_dim'], generator=generator)
    fn = functional_logits(model, registry, x)
    logits = fn(flat)
    weights = torch.tensor([2., 0., 1., 3., 4.])
    xi16 = categorical_probes(logits, 16, 812, weights)
    for rank in [1, 4, 8, 16]:
        xi = nested_probe_prefix(xi16, rank)
        torch.testing.assert_close(xi, categorical_probes(logits, rank, 812, weights), rtol=1e-12, atol=1e-12)
        direct = factors_from_probes(fn, flat, xi)
        whole = propagate_metric_probes(model, registry, flat, x, xi, token_mask=weights > 0)
        micro = propagate_metric_probes(model, registry, flat, x, xi, microbatch=2, token_mask=weights > 0)
        torch.testing.assert_close(whole.factors, direct, rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(micro.factors, direct, rtol=1e-9, atol=1e-11)
        coupled = whole.factors @ whole.factors.T
        for spec, response in zip(registry.specs, whole.by_tensor(registry).values()):
            torch.testing.assert_close(response @ response.T, coupled[spec.slice, spec.slice], rtol=1e-9, atol=1e-11)
            if branch_scale == 0 and spec.name.startswith('blocks.'):
                assert response.count_nonzero() == 0
        assert micro.counts['microbatches'] == 3
        assert micro.counts['vjp'] == 3 * rank * (len(model.blocks) + 2)
        assert micro.counts['forward'] == 3 * (2 * len(model.blocks) + 3)
        assert whole.metadata['factor_bytes'] == direct.numel() * direct.element_size()


def test_exact_output_factor_transport_matches_ggn_blocks_and_actions(tiny):
    model, registry, flat, x, targets, op, _, direct = tiny
    root, _ = output_factor(op.logits)
    xi = root.reshape(*op.logits.shape, -1)
    factors = propagate_metric_probes(model, registry, flat, x, xi, microbatch=3).factors
    torch.testing.assert_close(factors, direct, rtol=1e-9, atol=1e-11)
    generator = torch.Generator().manual_seed(645)
    for _ in range(2):
        vector = torch.randn(flat.shape, generator=generator)
        torch.testing.assert_close(factors @ (factors.T @ vector), op.action(vector), rtol=1e-9, atol=1e-11)
        for spec in registry.specs:
            restricted = torch.zeros_like(vector)
            restricted[spec.slice] = vector[spec.slice]
            block_action = factors[spec.slice] @ (factors[spec.slice].T @ vector[spec.slice])
            torch.testing.assert_close(block_action, op.action(restricted)[spec.slice], rtol=1e-9, atol=1e-11)


class ProjectedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip = nn.Linear(3, 4)
        self.norm = nn.LayerNorm(3)
        self.branch = nn.Linear(3, 4)

    def forward(self, h):
        return self.skip(h) + .7 * self.branch(self.norm(h)).tanh()


class ProjectedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 3)
        self.blocks = nn.ModuleList([ProjectedResidual()])
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.blocks[0](self.input(x)))


def test_projected_skip_uses_actual_jacobian_with_width_change():
    with torch.random.fork_rng():
        torch.manual_seed(22)
        model = ProjectedMLP()
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    x = torch.arange(10).reshape(5, 2) / 11
    fn = functional_logits(model, registry, x)
    xi = categorical_probes(fn(flat), 8, 73)
    response = propagate_metric_probes(model, registry, flat, x, xi, microbatch=2)
    torch.testing.assert_close(response.factors, factors_from_probes(fn, flat, xi), rtol=1e-9, atol=1e-11)


def test_transport_nonmutation_and_metadata(config):
    model = make_model(config)
    model.register_buffer('audit_buffer', torch.tensor([3.]))
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    x = torch.arange(4 * config['model']['input_dim']).reshape(4, -1).to(flat) / 20
    xi = categorical_probes(functional_logits(model, registry, x)(flat), 4, 993)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    model(x).square().sum().backward()
    optimizer.step()
    original_model = copy.deepcopy(model.state_dict())
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_gradients = [p.grad.detach().clone() for p in model.parameters()]
    original_inputs = [v.clone() for v in (flat, x, xi)]
    rng = torch.get_rng_state().clone()
    response = propagate_metric_probes(model, registry, flat, x, xi, microbatch=3,
                                       batch_id='fixed-batch', state_step=17,
                                       probe_seed=993, loss_normalization='global_weighted_mean')
    assert_tree(model.state_dict(), original_model)
    assert_tree(optimizer.state_dict(), original_optimizer)
    assert torch.equal(torch.get_rng_state(), rng)
    for old, current in zip(original_gradients, model.parameters()):
        assert torch.equal(old, current.grad)
    for old, current in zip(original_inputs, (flat, x, xi)):
        assert torch.equal(old, current)
    assert not response.factors.requires_grad
    assert response.metadata['batch_id'] == 'fixed-batch'
    assert response.metadata['state_step'] == 17
    assert response.metadata['probe_seed'] == 993
    assert response.metadata['loss_normalization'] == 'global_weighted_mean'
    assert response.metadata['exact_for_supplied_probes']
    assert response.metadata['curvature_representation'] == 'approximate_shared_GGN_sketch'
    direct = factors_from_probes(functional_logits(model, registry, x), flat, xi)
    torch.testing.assert_close(response.factors, direct, rtol=1e-9, atol=1e-11)


def test_transport_rejects_invalid_scope_shape_and_storage(tiny):
    model, registry, flat, x, _, op, _, _ = tiny
    xi = categorical_probes(op.logits, 4, 12)
    with pytest.raises(MemoryError, match='storage budget'):
        propagate_metric_probes(model, registry, flat, x, xi, max_factor_bytes=1)
    with pytest.raises(ValueError, match='Microbatch'):
        propagate_metric_probes(model, registry, flat, x, xi, microbatch=0)
    with pytest.raises(ValueError, match='Masked'):
        propagate_metric_probes(model, registry, flat, x, xi, token_mask=[False] * len(x))
    with pytest.raises(ValueError, match='Probe rank'):
        nested_probe_prefix(xi, 5)
    response = propagate_metric_probes(model, registry, flat, x, xi)
    incorrect = copy.deepcopy(registry)
    incorrect.specs = incorrect.specs[:-1]
    with pytest.raises(ValueError, match='differs'):
        response.by_tensor(incorrect)
