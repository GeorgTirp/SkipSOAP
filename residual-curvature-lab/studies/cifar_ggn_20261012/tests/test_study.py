"""CIFAR GGN study tests (PROTOCOL.md): lean probes, microbatched quantities, quadratic checks, SN, augmentation,
distance preservation and the unit plan."""
import torch

import cifar
import gpu_parsec
import measure
import micro
import train
from residual_curvature.curvature.layerwise import streamed_blocks
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits


def mnist_state(mnist, n=12):
    c, x, y, manifest = mnist
    model = make_model(c)
    registry = Registry(model)
    ids = manifest['training'][:n]
    return model, registry, registry.flatten(dict(model.named_parameters())).detach(), x[ids], y[ids]


def test_lean_probes_equal_ggn_probes_bitwise(mnist):
    model, _, _, x, _ = mnist_state(mnist, 10)
    lean = micro.lean_probes(model, x, samples=2, generator=torch.Generator().manual_seed(4), chunk=7)
    reference, _ = gpu_parsec.ggn_probes(model, x, samples=2, generator=torch.Generator().manual_seed(4), chunk=7)
    assert lean.keys() == reference.keys() and all(torch.equal(lean[p], reference[p]) for p in lean)


def test_microbatched_gradient_action_and_hessian(mnist):
    model, registry, flat, x, y = mnist_state(mnist, 13)
    op = GGN(functional_logits(model, registry, x), flat, y)
    value, g = micro.gradient(model, registry, flat, x, y, 5)
    assert (g-op.gradient).norm() <= 1e-12*op.gradient.norm()
    assert abs(value-float(loss(op.logits, y))) <= 1e-12
    v = torch.randn(registry.size, generator=torch.Generator().manual_seed(1), dtype=flat.dtype)
    assert (micro.ggn_action(model, registry, flat, x, 5)(v)-op.action(v)).norm() <= 1e-12*op.action(v).norm()
    w = flat.clone().requires_grad_(True)
    first = torch.autograd.grad(loss(functional_logits(model, registry, x)(w), y), w, create_graph=True)[0]
    reference = torch.autograd.grad(first@v, w)[0]
    assert (micro.hessian_action(model, registry, flat, x, y, 5)(v)-reference).norm() <= 1e-10*reference.norm()


def test_quadratic_check_terms_match_dense_references(mnist):
    model, registry, flat, x, y = mnist_state(mnist, 11)
    d = 1e-3*torch.randn(registry.size, generator=torch.Generator().manual_seed(2), dtype=flat.dtype)
    out = measure.quadratic_check(model, registry, flat, d, x, y, 4)
    op = GGN(functional_logits(model, registry, x), flat, y)
    blocks, _ = streamed_blocks(model, registry, flat, x, 4, 768)
    block = sum(float(d[s.slice]@(b@d[s.slice])) for s, b in zip(registry.specs, blocks))
    w = flat.clone().requires_grad_(True)
    first = torch.autograd.grad(loss(functional_logits(model, registry, x)(w), y), w, create_graph=True)[0]
    hessian = float(torch.autograd.grad(first@d, w)[0]@d)
    objective = lambda v: float(loss(functional_logits(model, registry, x)(v), y))
    assert abs(out['full']-float(d@op.action(d))) <= 1e-10*abs(out['full'])
    assert abs(out['block']-block) <= 1e-10*abs(block)
    assert abs(out['hessian']-hessian) <= 1e-10*abs(hessian)
    # Independent route: the forward-over-reverse HVP (float64; LayerNorm's forward-over-forward is wrong in torch 2.14.1).
    assert abs(out['hessian']-float(micro.hessian_action(model, registry, flat, x, y, 4)(d)@d)) <= 1e-10*abs(hessian)
    assert abs(out['gd']-float(op.gradient@d)) <= 1e-12 and abs(out['actual']-(objective(flat+d)-objective(flat))) <= 1e-12
    assert abs(out['ratio_hessian']-1) < abs(out['ratio_full']-1)+1e-3  # a tiny step: the Hessian model is accurate


def cifar_model(arm='ggn', seed=0):
    return train.build_model({'arm': arm, 'seed': seed}, 'cpu').double()


def test_sn_inactive_bound_is_the_plain_model_and_active_sn_is_scale_invariant():
    x = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    plain = cifar_model('ggn')
    loose = cifar.make_cifar_model('M1d6', 0, 'cpu', sn_bound=1e9).double()  # same rounding as cifar_model
    assert [n for n, _ in plain.named_parameters()] == [n for n, _ in loose.named_parameters()]
    assert torch.equal(plain(x), loose(x))
    model = cifar_model('ggn_sn')
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    y = torch.tensor([0, 1, 2, 3])
    _, g = micro.gradient(model, registry, flat, x, y, 3)
    report = measure.spectral_report(model)
    active = [n for n, r in report.items() if r['active']]
    assert len(active) >= 24 and all(abs(r['sigma_effective']-1) < 1e-10 for r in report.values() if r['active'])
    for spec in registry.specs:
        if spec.name in active:
            cosine = float(g[spec.slice]@flat[spec.slice])/float(g[spec.slice].norm()*flat[spec.slice].norm())
            assert abs(cosine) <= 1e-10
    out = measure.quadratic_check(model, registry, flat, 1e-4*g, x, y, 3)
    assert out['full'] > 0 and out['block'] > 0
    # Second derivative through SN (double backward) against central finite differences.
    d = 1e-3*torch.randn(registry.size, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    objective = lambda v: float(loss(functional_logits(model, registry, x)(v), y))
    h = 1e-3
    finite = (objective(flat+h*d)-2*objective(flat)+objective(flat-h*d))/h**2
    assert abs(micro.hessian_quadratic(model, registry, flat, d, x, y, 3)-finite) <= 1e-5*abs(finite)
    rows = measure.block_jacobian_deviation(model, x[:2], torch.Generator().manual_seed(0), probes=2)
    assert len(rows) == 6 and all(r['rms'] > 0 for r in rows)


def test_batched_spectral_norms_equal_svd():
    gen = torch.Generator().manual_seed(5)
    weights = [torch.randn(*shape, generator=gen, dtype=torch.float64) for shape in [(64, 64)]*3+[(256, 64), (64, 256)]]
    sigma = cifar.spectral_norms(weights)
    for w, s in zip(weights, sigma):
        assert abs(float(s)-float(torch.linalg.matrix_norm(w, ord=2))) <= 1e-12*float(s)


def test_augment_crops_and_flips_deterministically():
    x = torch.randn(64, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    a = cifar.augment(x, torch.Generator().manual_seed(5))
    b = cifar.augment(x, torch.Generator().manual_seed(5))
    assert torch.equal(a, b) and a.shape == x.shape
    gen = torch.Generator().manual_seed(5)
    offsets = torch.randint(0, 9, (64, 2), generator=gen)
    flips = torch.rand(64, generator=gen) < .5
    padded = torch.nn.functional.pad(x, (4,)*4)
    for i in range(64):
        r, s = offsets[i].tolist()
        expected = padded[i, :, r:r+32, s:s+32]
        assert torch.equal(a[i], expected.flip(-1) if flips[i] else expected)


def test_distance_preservation_and_block_jacobian_shapes():
    model = cifar_model('soap')
    x = torch.randn(20, 3, 32, 32, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    out = measure.distance_preservation(model, {'global': (x[:10], x[10:]), 'local': (x[:10], x[:10]+1e-3*x[10:])})
    assert set(out['local']) == ({f'depth{k}_stream' for k in range(7)} | {f'depth{k}_cls' for k in range(1, 7)}
                                 | {'head'})
    assert all(v['quantiles_1_5_50_95_99'][0] > 0 for kind in out.values() for v in kind.values())
    rows = measure.block_jacobian_deviation(model, x[:4], torch.Generator().manual_seed(0), probes=2)
    assert [r['block'] for r in rows] == [1, 2, 3, 4, 5, 6] and all(r['rms'] > 0 for r in rows)


def test_unit_plan_schedules_and_settings():
    units = train.tune_units()
    assert len(units) == 8 and {s['seed'] for s in units.values()} == {0} and {s['arm'] for s in units.values()} == {'soap', 'topfree'}
    chosen = {'soap': {'lr': 3e-3}, 'topfree': {'lr': 1e-2}, 'ggn': {'gamma': 1e-2}}
    edges = train.extension_units({'soap': {'edge': 'low', 'grid': train.LR_GRID}, 'topfree': {'edge': 'high', 'grid': train.LR_GRID}})
    assert sorted((s['arm'], s['lr']) for s in edges.values()) == [('soap', 1e-3/3), ('topfree', 3e-2*3)]
    evaluation = train.eval_units(chosen)
    assert len(evaluation) == 12 and all(s['measure'] for s in evaluation.values())
    assert sorted({s['arm'] for s in evaluation.values()}) == ['ggn', 'ggn_sn', 'soap', 'topfree']
    assert train.lr_factor(1) == 1/25 and train.lr_factor(25) == 1. and train.lr_factor(400) == 1.
    assert train.lr_factor(500) == .01 and train.lr_factor(1, warmup=False) == 1.
    model = cifar_model('soap').float()
    topfree = train.make_optimizer({'arm': 'topfree', 'lr': 1e-2, 'seed': 1}, model)
    assert {k: topfree.parsec[k] for k in ['rank', 'strength', 'calibration', 'top_rhs', 'norm_blend', 'batched']} == {
        'rank': 16, 'strength': 64., 'calibration': 'lr', 'top_rhs': 'gradient', 'norm_blend': 0., 'batched': False}
    assert topfree.param_groups[0]['weight_decay'] == .05
    assert train.shadow_config()['weight_decay'] == 0. and train.shadow_config()['lr'] == 3e-3
    assert Registry(cifar.make_cifar_model('M1d6', 0, 'cpu')).size == 308042
