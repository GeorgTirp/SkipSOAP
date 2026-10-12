"""Check 1/2a/2b arm tests (section 6, 2a tests 1-3, plus copy and map checks)."""
import copy

import torch

import arms
import check1_train
import common
from residual_curvature.config import load_config
from residual_curvature.curvature.information import information_blocks
from residual_curvature.curvature.layerwise import block_action, solve_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.curvature.solves import relative_error
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import SOAPMetric, soap_propose
from run_large_batch_oracle import state_equal
from run_mini_vit_allocation import propose as mini_propose
from run_transport_parsec_noise import optimizer_settings, propose as transport_propose


def setup(c, x, y, manifest, n, offset=0):
    model = make_model(c)
    registry = Registry(model)
    ids = manifest['training'][offset:offset+n]
    flat = registry.flatten(dict(model.named_parameters())).detach()
    return model, registry, flat, x[ids], y[ids]


def test_study_config_matches_main_workload_except_output():
    c = common.study_config()
    main = load_config('configs/mini_vit_allocation.yaml')
    assert c['output'] == str(common.RUNS) and not common.RUNS.is_relative_to(common.LAB)
    c['output'] = main['output']
    assert c == main
    assert common.data_order_seed(0) == main['data_seed']+1 == 102
    assert common.data_order_seed(3) == 3102


def test_validation_schedule():
    assert check1_train.schedule(500) == [0, 10]+list(range(25, 501, 25))
    assert len(check1_train.schedule(500)) == 22


def test_per_example_gradient_mean_equals_ggn_gradient(mnist):
    c, x, y, manifest = mnist
    model, registry, flat, xb, yb = setup(c, x, y, manifest, 64)
    rows = arms.per_example_gradients(model, registry, flat, xb, yb)
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    assert rows.shape == (64, registry.size)
    assert relative_error(rows.mean(0), op.gradient) <= 1e-12


def test_empirical_fisher_blocks_match_information_blocks(mnist):
    c, x, y, manifest = mnist
    model, registry, flat, xb, yb = setup(c, x, y, manifest, 16)
    commit(model, registry, flat+.05*torch.randn(registry.size, generator=torch.Generator().manual_seed(5), dtype=flat.dtype))
    flat = registry.flatten(dict(model.named_parameters())).detach()
    blocks, mean, _ = arms.empirical_fisher_blocks(model, registry, flat, xb, yb)
    reference = information_blocks(model, registry, flat, xb, yb, ranks=(1,), repeats=1)[0]['empirical_fisher']
    total = sum(float((a-b).square().sum()) for a, b in zip(blocks, reference))/sum(float(b.square().sum()) for b in reference)
    assert total**.5 <= 1e-10
    assert max(relative_error(a, b) for a, b in zip(blocks, reference)) <= 1e-10
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    gen = torch.Generator().manual_seed(1)
    for spec, block in zip(registry.specs, blocks):
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=flat.dtype)
        assert relative_error(block@v[spec.slice], arms.empirical_fisher_action(op, v)[spec.slice]) <= 1e-10


def two_soap_states(c, model, registry, flat, xb, yb):
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    _, state, _ = soap_propose(registry.unflatten(flat), registry.unflatten(op.gradient), None, c['soap'])
    return state


def test_dense_ef_solve_residual(mnist):
    c, x, y, manifest = mnist
    model, registry, flat, xb, yb = setup(c, x, y, manifest, 64)
    state = two_soap_states(c, model, registry, flat, xb, yb)
    delta, _, contexts, row = arms.dense_ef_propose(.01, model, registry, flat, state, xb, yb, c, 2, audit=True)
    assert row['algorithm']['solve']['residual'] <= 1e-8
    assert max(row['algorithm']['solve']['tensor_residuals'].values()) <= 1e-8
    assert max(row['algorithm']['action_checks'].values()) <= 1e-8
    blocks, _, info = arms.empirical_fisher_blocks(model, registry, flat, xb, yb)
    metric = SOAPMetric(registry, contexts)
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    mu = max(.01*info['trace']/registry.size, c['damping']['floor'])
    assert mu == row['algorithm']['mu']
    value = solve_blocks(blocks, registry, op.gradient, metric, [mu], 1e-8)[0][mu]
    assert relative_error(block_action(blocks, registry, value)+mu*metric.apply(value), -op.gradient) <= 1e-8
    assert torch.isfinite(delta).all() and row['search'] is not None


def test_study_ggn_copy_is_bitwise_identical_at_gamma_01(mnist):
    c, x, y, manifest = mnist
    model, registry, flat, xb, yb = setup(c, x, y, manifest, 32)
    first = [None, None]
    for step in [1, 2]:
        a = transport_propose('ggn_exact_trace', model, registry, flat, copy.deepcopy(first[0]), xb, yb, c, step,
                              optimizer_settings('ggn_exact_trace'), None, audit=True)
        b = arms.study_ggn_propose(.01, model, registry, flat, copy.deepcopy(first[1]), xb, yb, c, step, audit=True)
        assert torch.equal(a[0], b[0]) and state_equal(a[1], b[1]) and state_equal(a[2], b[2])
        for key in ['loss_before', 'loss_after', 'norm', 'raw_norm', 'slope', 'native_radius', 'gradient_norm']:
            assert a[3][key] == b[3][key]
        assert a[3]['search'] == b[3]['search']
        if step == 2:
            assert a[3]['algorithm']['mu'] == b[3]['algorithm']['mu']
            assert a[3]['algorithm']['action_checks'] == b[3]['algorithm']['action_checks']
        first = [a[1], b[1]]
        commit(model, registry, flat+a[0])
        flat = registry.flatten(dict(model.named_parameters())).detach()


def test_soap_gradient_map_advances_statistics_once(mnist):
    c, x, y, manifest = mnist
    model, registry, flat, xb, yb = setup(c, x, y, manifest, 64)
    state = two_soap_states(c, model, registry, flat, xb, yb)
    native, native_state, native_contexts, _ = mini_propose('native_soap', model, registry, flat, copy.deepcopy(state), xb, yb, c, 2)
    delta, mapped_state, contexts, row = arms.soap_gradient_map_propose(model, registry, flat, copy.deepcopy(state), xb, yb, c, 2)
    assert state_equal(native_state, mapped_state) and state_equal(native_contexts, contexts)
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    candidate = -SOAPMetric(registry, contexts).map(op.gradient)
    assert row['raw_norm'] == float(candidate.norm())
    assert row['search']['radius'] == float(native.norm()) == row['native_radius']
    direction = candidate*(float(native.norm())/float(candidate.norm()))
    assert relative_error(delta, direction*row['search']['selected_alpha']) <= 1e-12
