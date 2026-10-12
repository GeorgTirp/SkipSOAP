"""SN curvature study tests (PROTOCOL.md)."""
import torch
from torch.func import jacrev

import diagnostics
import run
import sn
from residual_curvature.curvature.information import information_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.curvature.solves import relative_error
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits


def batch(mnist, n=16):
    c, x, y, manifest = mnist
    ids = manifest['training'][:n]
    return c, x[ids], y[ids]


def test_inactive_sn_is_bitwise_identical(mnist):
    c, xb, yb = batch(mnist)
    plain, wrapped = make_model(c), sn.make_sn_model(c, 1e9)
    rp, rw = Registry(plain), Registry(wrapped)
    assert torch.equal(functional_logits(plain, rp, xb)(rp.flatten(dict(plain.named_parameters())).detach()),
                       functional_logits(wrapped, rw, xb)(rw.flatten(dict(wrapped.named_parameters())).detach()))
    assert len(sn.sn_module_names(wrapped)) == 12 and all(not r['active'] for r in sn.spectral_report(wrapped, 1e9).values())


def test_sn_map_and_effective_twin(mnist):
    c, xb, yb = batch(mnist)
    model = sn.make_sn_model(c, .5)
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    report = sn.spectral_report(model, .5)
    for row in report.values():
        assert abs(row['sigma_effective']-min(row['sigma_raw'], .5)) <= 1e-12
    twin, twin_registry = sn.effective_twin(model, c)
    twin_flat = twin_registry.flatten(dict(twin.named_parameters())).detach()
    assert relative_error(functional_logits(twin, twin_registry, xb)(twin_flat), functional_logits(model, registry, xb)(flat)) <= 1e-14


def test_raw_ggn_is_chain_rule_of_effective_ggn_and_has_scale_null(mnist):
    c, xb, yb = batch(mnist)
    model = sn.make_sn_model(c, .5)
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    blocks = information_blocks(model, registry, flat, xb, yb, ranks=(1,), repeats=1, budget_mib=3072)[0]['ggn']
    twin, twin_registry = sn.effective_twin(model, c)
    twin_flat = twin_registry.flatten(dict(twin.named_parameters())).detach()
    twin_blocks = information_blocks(twin, twin_registry, twin_flat, xb, yb, ranks=(1,), repeats=1, budget_mib=3072)[0]['ggn']
    twin_index = {s.name: i for i, s in enumerate(twin_registry.specs)}
    bound = sn.SpectralNorm(.5)
    checked = 0
    for spec, block in zip(registry.specs, blocks):
        if '.parametrizations.' not in spec.name:
            continue
        w = flat[spec.slice].reshape(spec.shape)
        p = jacrev(lambda v: bound(v.reshape(spec.shape)).reshape(-1))(w.reshape(-1))
        effective = twin_blocks[twin_index[spec.name.replace('parametrizations.weight.original', 'weight')]]
        assert relative_error(block, p.T@effective@p) <= 1e-8
        assert float((block@w.reshape(-1)).norm()/(torch.linalg.eigvalsh(block)[-1]*w.norm())) <= 1e-10
        checked += 1
    assert checked == 12


def test_full_ggn_and_distance_preservation(mnist):
    c, x, y, manifest = mnist
    model = sn.make_sn_model(c, 2.)
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    xb, yb = x[manifest['training'][:16]], y[manifest['training'][:16]]
    full = diagnostics.full_ggn(model, registry, flat, xb)
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    v = torch.randn(registry.size, generator=torch.Generator().manual_seed(1), dtype=flat.dtype)
    assert relative_error(full@v, op.action(v)) <= 1e-8
    out = diagnostics.distance_preservation(model, x, manifest)
    for value in out.values():
        q = value['quantiles_1_5_50_95_99']
        assert all(a <= b for a, b in zip(q, q[1:])) and value['distortion_q95_q5'] >= 1


def test_unit_grid():
    units = run.units()
    assert len(units) == 16 and {u['bound'] for u in units.values()} == {None, .95, 2., 6.}
