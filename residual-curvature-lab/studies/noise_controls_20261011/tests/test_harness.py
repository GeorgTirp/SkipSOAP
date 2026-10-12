"""Noise-controls harness tests (PROTOCOL.md)."""
import copy

import torch

import train
from residual_curvature.curvature.operators import GGN
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits


def test_schedules_units_and_settings():
    assert train.lr_at('constant', .01, 7) == .01 and train.lr_at('wsd', .02, 400) == .02 and train.lr_at('wsd', .02, 401) == .02
    assert abs(train.lr_at('wsd', .02, 500)-.0002) < 1e-15 and train.lr_at('cosine', .03, 1) == .03
    assert abs(train.lr_at('cosine', .03, 251)-.015) < 1e-12 and train.lr_at('cosine', .03, 500) < 1e-6
    assert len(train.tune_units()) == 17 and len(train.eval_fixed_units()) == 6
    selection = {'schedules': {f'{a}|{s}': {'lr': .02} for a in ['soap', 'brake'] for s in ['cosine', 'wsd']}, 'topfree': {'strength': 4.}}
    assert len(train.eval_tuned_units(selection)) == 15
    c = train.unit_config(1)
    model = make_model(c)
    brake = train.make_optimizer(train.spec_of('brake', 'constant', .01, 1), model, c)
    expected = dict(rank=16, gamma=.01, probe_every=10, refresh='extract', norm_blend=0., batched=False, signals=False, seed=1,
                    calibration='lr', strength=4., top_rhs='momentum', rhs='momentum')
    assert all(brake.parsec[k] == v for k, v in expected.items())
    topfree = train.make_optimizer(train.spec_of('topfree', 'constant', .01, 1, strength=16.), model, c)
    assert topfree.parsec['top_rhs'] == 'gradient' and topfree.parsec['strength'] == 16.
    soap = train.make_optimizer(train.spec_of('soap', 'cosine', .03, 1), model, c)
    assert soap.parsec['rank'] == 0 and soap.param_groups[0]['lr'] == .03
    assert train.reference_unit(train.spec_of('brake', 'constant', .01, 2)).name == 'brake__lr4_lr0.01__s2'
    assert train.reference_unit(train.spec_of('brake', 'wsd', .01, 2)) is None


def test_factor_maps_are_adjoint_and_sharpness_is_finite(mnist):
    c0, x, y, manifest = mnist
    c = train.unit_config(1)
    model = make_model(c)
    registry = Registry(model)
    optimizer = train.make_optimizer(train.spec_of('soap', 'constant', .01, 1), model, c)
    for step in range(4):
        ids = manifest['training'][64*step:64*(step+1)]
        flat = registry.flatten(dict(model.named_parameters())).detach()
        gradient = registry.unflatten(GGN(functional_logits(model, registry, x[ids]), flat, y[ids]).gradient)
        for name, p in model.named_parameters():
            p.grad = gradient[name].detach().clone()
        optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    forward, transpose, _ = train.factor_maps(model, registry, optimizer)
    gen = torch.Generator().manual_seed(0)
    v, w = torch.randn(registry.size, generator=gen), torch.randn(registry.size, generator=gen)
    assert abs(float(transpose(w)@v)-float(w@forward(v))) <= 1e-10*float(w.norm()*v.norm())
    xb, yb = x[manifest['analysis'][:64]], y[manifest['analysis'][:64]]
    value = train.sharpness(model, registry, optimizer, xb, yb, .01, 1)
    assert value == value and value > 0
    after = optimizer.state_dict()
    assert all(torch.equal(a, b) for k in before['state'] for a, b in
               [(before['state'][k]['exp_avg_sq'], after['state'][k]['exp_avg_sq'])])
