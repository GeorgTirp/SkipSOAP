"""SN x GGN training study tests (PROTOCOL.md)."""
import copy

import torch

import arms
import common
import sn
import train
from residual_curvature.config import load_config
from residual_curvature.curvature.operators import GGN
from residual_curvature.data import DataOrder
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry
from residual_curvature.models.mlp import functional_logits


def test_config_and_units():
    c, arm, settings = train.unit_config({'optimizer': 'ggn', 'seed': 2})
    main = load_config('configs/mini_vit_allocation.yaml')
    assert (arm, settings) == ('ggn_exact_trace', {'gamma': .001})
    assert c['seed'] == 2 and c['training']['steps'] == 250 and c['training']['policy'] == 'broad_direction_search'
    assert {k: v for k, v in c.items() if k not in ['output', 'seed', 'training']} == {k: v for k, v in main.items() if k not in ['output', 'seed', 'training']}
    assert len(train.units()) == 12 and train.VALIDATION == [0, 10]+list(range(25, 251, 25))
    soap, _, _ = train.unit_config({'optimizer': 'soap', 'seed': 1})
    assert soap['soap']['lr'] == .01 and soap['training']['policy'] == 'native'


def test_peek_returns_next_batch_without_advancing(mnist):
    c, x, y, manifest = mnist
    order = DataOrder(manifest['training'], common.data_order_seed(1))
    for _ in range(116):  # crosses an epoch boundary (50,000 / 512)
        expected = copy.deepcopy(order.state_dict())
        nxt = train.peek(order, 512)
        state = order.state_dict()
        assert state['position'] == expected['position'] and torch.equal(state['order'], expected['order'])
        assert torch.equal(state['rng'], expected['rng'])
        assert order.take(512) == nxt


def test_sn_ggn_step_passes_action_audit_and_has_scale_null(mnist):
    c, x, y, manifest = mnist
    c, arm, settings = train.unit_config({'optimizer': 'ggn', 'seed': 1})
    model = sn.make_sn_model(c, .95)
    registry = Registry(model)
    ids = manifest['training'][:64]
    state = None
    for step in [1, 2]:
        flat = registry.flatten(dict(model.named_parameters())).detach()
        delta, state, _, row = arms.propose_step(arm, settings, 'study', model, registry, flat, state, x[ids], y[ids], c, step,
                                                 audit=step == 2)
        commit(model, registry, flat+delta)
    assert row['algorithm']['action_checks'] and max(row['algorithm']['action_checks'].values()) <= 1e-8
    assert row['algorithm']['solve']['residual'] <= 1e-8
    # SN is active from the start at c = 0.95, so every SN tensor's gradient is orthogonal to its raw weight.
    flat = registry.flatten(dict(model.named_parameters())).detach()
    gradient = GGN(functional_logits(model, registry, x[ids]), flat, y[ids]).gradient
    report = sn.spectral_report(model, .95)
    assert all(r['active'] for r in report.values())
    for spec in registry.specs:
        if 'parametrizations' in spec.name:
            w, g = flat[spec.slice], gradient[spec.slice]
            assert abs(float(w@g)) <= 1e-12*float(w.norm()*g.norm())


def test_inactive_sn_training_step_matches_plain_model(mnist):
    c, x, y, manifest = mnist
    c, arm, settings = train.unit_config({'optimizer': 'ggn', 'seed': 1})
    ids = manifest['training'][:64]
    losses = []
    for bound in [None, 1e9]:
        model = sn.make_sn_model(c, bound)
        registry = Registry(model)
        state = None
        for step in [1, 2]:
            flat = registry.flatten(dict(model.named_parameters())).detach()
            delta, state, _, row = arms.propose_step(arm, settings, 'study', model, registry, flat, state, x[ids], y[ids], c, step)
            commit(model, registry, flat+delta)
        losses.append(train.batch_loss(model, x[ids], y[ids]))
    assert abs(losses[0]-losses[1]) <= 1e-10*abs(losses[0])
