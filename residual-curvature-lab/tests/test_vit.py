import copy

import pytest
import torch

from residual_curvature.config import validate
from residual_curvature.curvature.layerwise import streamed_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.models import Registry, SmallViT, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import SOAPMetric, mode, soap_propose


def tiny_vit_config(config):
    config['model'].update(kind='vit', input_dim=16, width=4, expansion=6,
                           blocks=1, classes=3)
    config['vit'].update(image_size=4, patch_size=2, heads=2)
    return config


def test_vit_seed_registry_and_functional_nonmutation(config):
    c = tiny_vit_config(config)
    rng = torch.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    model = make_model(c)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.get_default_dtype() == default_dtype
    replica = make_model(c)
    registry = Registry(model)
    assert registry.size == sum(p.numel() for p in model.parameters())
    assert {s.name for s in registry.specs} == set(dict(model.named_parameters()))
    flat = registry.flatten(dict(model.named_parameters())).detach()
    torch.testing.assert_close(flat, Registry(replica).flatten(dict(replica.named_parameters())), rtol=0, atol=0)
    assert torch.equal(flat, registry.flatten(registry.unflatten(flat)))
    groups = registry.groups('block')
    assert len(groups) == len(model.blocks) + 2
    assert groups[0].start == 0 and groups[-1].stop == registry.size
    assert all(left.stop == right.start for left, right in zip(groups, groups[1:]))
    assert sum(group.stop - group.start for group in groups) == registry.size
    assert all(s.role == 'normalization' for s in registry.specs if 'norm' in s.name)
    assert {s.name for s in registry.specs if s.role == 'embedding'} == {
        'input.cls_token', 'input.position_embedding'}
    x = torch.linspace(-1, 1, 48).reshape(3, 16)
    before = copy.deepcopy(model.state_dict())
    fn = functional_logits(model, registry, x)
    assert fn(flat).shape == (3, 3)
    torch.testing.assert_close(fn(flat), model(x), rtol=0, atol=0)
    assert not torch.equal(fn(flat + .001 * torch.arange(len(flat))), fn(flat))
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_vit_patch_layout_and_shared_token_parameters(config):
    model = make_model(tiny_vit_config(config))
    x = torch.arange(16, dtype=torch.float64).reshape(1, 16)
    expected_patches = torch.tensor([[0., 1., 4., 5.], [2., 3., 6., 7.],
                                     [8., 9., 12., 13.], [10., 11., 14., 15.]])
    tokens = model.input(x)[0] - model.input.position_embedding
    torch.testing.assert_close(tokens[0], model.input.cls_token[0])
    torch.testing.assert_close(tokens[1:], model.input.projection(expected_patches))
    # Token count changes position parameters, not the shared projection shapes.
    assert model.input.projection.weight.shape == (4, 4)
    assert model.blocks[0].attention.query.weight.shape == (4, 4)
    with pytest.raises(ValueError, match='flattened'):
        model(torch.zeros(1, 1, 4, 4))


def test_vit_ggn_matrixfree_and_streamed_blocks(config):
    model = make_model(tiny_vit_config(config))
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    x = torch.linspace(-1, 1, 48).reshape(3, 16)
    y = torch.tensor([0, 2, 1])
    op = GGN(functional_logits(model, registry, x), flat, y)
    _, factor, _ = op.factors()
    dense = factor @ factor.T
    vector = torch.linspace(-.5, .5, registry.size)
    torch.testing.assert_close(op.action(vector), dense @ vector, rtol=1e-9, atol=1e-11)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    # An uneven last microbatch checks full-batch mean normalization.
    blocks, metadata = streamed_blocks(model, registry, flat, x, 2, 16)
    for spec, block in zip(registry.specs, blocks):
        torch.testing.assert_close(block, dense[spec.slice, spec.slice], rtol=1e-9, atol=1e-11)
    assert abs(metadata['trace'] - float(dense.trace())) < 1e-10
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_vit_soap_proposal_and_frozen_metric(config):
    model = make_model(tiny_vit_config(config))
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    x = torch.linspace(-1, 1, 48).reshape(3, 16)
    op = GGN(functional_logits(model, registry, x), flat, torch.tensor([0, 2, 1]))
    params, gradients = registry.unflatten(flat), registry.unflatten(op.gradient)
    before = copy.deepcopy(model.state_dict())
    _, state, _ = soap_propose(params, gradients, None, config['soap'])
    delta, _, contexts = soap_propose(params, gradients, state, config['soap'])
    assert registry.flatten(delta).norm() > 0
    for spec in registry.specs:
        context = contexts[spec.name]
        # Captured momentum already lives in SOAP's projected coordinates.
        update = context['momentum'] / context['denominator']
        for axis, basis in enumerate(context['basis']):
            update = mode(update, basis, axis, 'forward')
        expected = -context['lr'] * context['bias_correction'] * update
        torch.testing.assert_close(delta[spec.name], expected, rtol=1e-7, atol=1e-10)
    metric = SOAPMetric(registry, contexts)
    vector = torch.linspace(-1, 1, registry.size)
    torch.testing.assert_close(metric.inverse(metric.apply(vector)), vector, rtol=1e-8, atol=1e-9)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_vit_configuration_and_legacy_mlp(config):
    c = tiny_vit_config(copy.deepcopy(config))
    validate(c)
    for section, field, value in [('vit', 'heads', 3), ('vit', 'patch_size', 3),
                                  ('vit', 'image_size', 8), ('vit', 'heads', 0)]:
        bad = copy.deepcopy(c)
        bad[section][field] = value
        with pytest.raises(ValueError):
            validate(bad)
    old = copy.deepcopy(config)
    old.pop('vit')
    assert make_model(old)(torch.zeros(2, 6)).shape == (2, 3)
    production = SmallViT()
    assert sum(p.numel() for p in production.parameters()) == 5738
    assert production(torch.zeros(2, 784)).shape == (2, 10)
