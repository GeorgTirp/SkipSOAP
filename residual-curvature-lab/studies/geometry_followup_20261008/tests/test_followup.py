"""Geometry follow-up tests (PROTOCOL.md, Checks)."""
import copy
from types import SimpleNamespace

import torch

import arms
import common
from residual_curvature.config import load_config
from residual_curvature.curvature import wasserstein
from residual_curvature.curvature.layerwise import streamed_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.curvature.solves import relative_error
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from run_large_batch_oracle import state_equal
from run_transport_parsec_noise import optimizer_settings, propose as transport_propose


def setup(c, x, y, manifest, n, perturb=False):
    model = make_model(c)
    registry = Registry(model)
    if perturb:
        flat = registry.flatten(dict(model.named_parameters())).detach()
        commit(model, registry, flat+.05*torch.randn(registry.size, generator=torch.Generator().manual_seed(4), dtype=flat.dtype))
    ids = manifest['training'][:n]
    return model, registry, registry.flatten(dict(model.named_parameters())).detach(), x[ids], y[ids]


def test_config_and_gate_settings():
    c = common.study_config()
    main = load_config('configs/mini_vit_allocation.yaml')
    assert not common.RUNS.is_relative_to(common.LAB)
    c['output'] = main['output']
    assert c == main
    assert arms.parsec_settings({'lr': .003, 'refresh_every': 10, 'rank': 16}) == optimizer_settings('parsec_native')


def test_study_ot_copy_is_bitwise_identical_at_gamma_01(mnist):
    c, x, y, manifest = mnist
    model, registry, flat, xb, yb = setup(c, x, y, manifest, 32)
    states = [None, None]
    for step in [1, 2]:
        a = transport_propose('wasserstein_tensor', model, registry, flat, copy.deepcopy(states[0]), xb, yb, c, step,
                              optimizer_settings('wasserstein_tensor'), None, audit=True)
        b = arms.propose_step('ot_tensor', {'gamma': .01}, 'study', model, registry, flat, copy.deepcopy(states[1]), xb, yb, c, step, audit=True)
        assert torch.equal(a[0], b[0]) and state_equal(a[1], b[1]) and state_equal(a[2], b[2])
        assert a[3]['search'] == b[3]['search'] and a[3]['loss_after'] == b[3]['loss_after']
        if step == 2:
            assert a[3]['algorithm']['mu'] == b[3]['algorithm']['mu']
            assert a[3]['algorithm']['action_checks'] == b[3]['algorithm']['action_checks']
        states = [a[1], b[1]]
        commit(model, registry, flat+a[0])
        flat = registry.flatten(dict(model.named_parameters())).detach()


def split_settings(**kw):
    s = arms.parsec_settings({'lr_perp': .01, 'lr_top': .5, 'refresh_every': 0, 'rank': 2})
    s.update(kw)
    return s


def test_split_port_before_curvature_is_source_adam():
    g = torch.tensor([.3, -2., 1e-3], dtype=torch.float64)
    s = split_settings()
    delta, state, info = arms.parsec_split_propose(g, g*0, None, s, None)
    b1, b2 = s['betas']
    t = 2  # source: incremented step + 1
    m, v = (1-b1)*g, (1-b2)*g.square()
    expected = -s['lr_perp']*(m/(1-b1**t))/((v/(1-b2**t)).sqrt()+s['eps'])
    torch.testing.assert_close(delta, expected, rtol=1e-15, atol=0)
    assert info['bias_correction_index'] == 2 and info['top_step_norm'] == 0


def test_split_port_full_rank_top_step_is_damped_newton():
    gen = torch.Generator().manual_seed(2)
    a = torch.randn(5, 5, generator=gen, dtype=torch.float64)
    a = a@a.T+.5*torch.eye(5, dtype=torch.float64)
    g = torch.randn(5, generator=gen, dtype=torch.float64)
    s = split_settings(refresh_every=1, rank=5, lanczos_iters=10, lr_perp=0.)
    delta, state, info = arms.parsec_split_propose(g, g*0, None, s, SimpleNamespace(action=lambda v: a@v))
    expected = -s['lr_top']*torch.linalg.solve(a+s['damping']*torch.eye(5, dtype=torch.float64), g)
    torch.testing.assert_close(delta, expected, rtol=1e-10, atol=1e-12)
    assert info['active_rank'] == 5 and state['last_refresh'] == 1


def test_full_matrices_act_like_operators_and_contain_blocks(mnist):
    c, x, y, manifest = mnist
    model, registry, flat, xb, yb = setup(c, x, y, manifest, 16, perturb=True)
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    v = torch.randn(registry.size, generator=torch.Generator().manual_seed(9), dtype=flat.dtype)
    for kind, action, block_fn in [('ggn', op.action, streamed_blocks),
                                   ('ot', lambda u: wasserstein.action(model, registry, flat, xb, u), wasserstein.streamed_wasserstein_blocks)]:
        full, info = arms.full_matrix(kind, model, registry, flat, xb)
        blocks = block_fn(model, registry, flat, xb, 8, 768)[0]
        assert relative_error(full@v, action(v)) <= 1e-8
        assert relative_error(full, full.T) <= 1e-12
        diag = (sum(float((full[s.slice, s.slice]-b).square().sum()) for s, b in zip(registry.specs, blocks))
                / sum(float(b.square().sum()) for b in blocks))**.5
        assert diag <= 1e-8
        assert abs(info['trace']-sum(float(b.trace()) for b in blocks)) <= 1e-10*info['trace']
        assert 0 < arms.off_diagonal_energy(full, registry) < 1


def test_full_step_solves_and_is_finite(mnist):
    c, x, y, manifest = mnist
    model, registry, flat, xb, yb = setup(c, x, y, manifest, 32)
    state = None
    for step in [1, 2]:
        delta, state, contexts, row = arms.propose_step('ggn_full', {'gamma': .001}, 'study', model, registry, flat, state, xb, yb, c, step, audit=step == 2)
    assert row['algorithm']['solve']['residual'] <= 1e-8
    assert max(row['algorithm']['audit_checks'].values()) <= 1e-8
    assert torch.isfinite(delta).all() and row['search'] is not None
