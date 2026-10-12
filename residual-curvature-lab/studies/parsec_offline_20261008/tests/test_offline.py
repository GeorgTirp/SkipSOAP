"""PARSEC offline study tests (PROTOCOL.md)."""
import torch
from torch.func import functional_call, grad, jacrev, vmap

import c_sn_split
import probes
from residual_curvature.curvature.information import information_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import capacity


def tiny(mnist, n=16):
    c, x, y, manifest = mnist
    model = make_model(c)
    registry = Registry(model)
    ids = manifest['training'][:n]
    return model, registry, registry.flatten(dict(model.named_parameters())).detach(), x[ids], y[ids]


def test_probe_rows_are_plain_backprop_with_sampled_labels(mnist):
    """The layer-parallel route: s ordinary backward passes with sampled labels give every block's probe columns."""
    model, registry, flat, xb, _ = tiny(mnist)
    s, seed = 3, 11
    u = probes.sampled_factor(model, registry, flat, xb, s, seed)
    gen = torch.Generator().manual_seed(seed)
    uniforms = torch.rand(len(xb), s, generator=gen, dtype=flat.dtype)
    p = functional_logits(model, registry, xb)(flat).detach().softmax(-1)
    labels = torch.searchsorted(p.cumsum(-1).contiguous(), uniforms.contiguous()).clamp_max(9)
    def per_example(w, xi, yi):
        return torch.nn.functional.cross_entropy(functional_call(model, registry.unflatten(w), (xi[None],)), yi[None])
    for j in range(s):
        g = vmap(grad(per_example), in_dims=(None, 0, 0))(flat, xb, labels[:, j])
        rows = u.reshape(len(xb), s, -1)[:, j]*(len(xb)*s)**.5
        torch.testing.assert_close(rows, g, rtol=1e-10, atol=1e-12)


def test_probe_expectation_is_the_ggn(mnist):
    model, registry, flat, xb, yb = tiny(mnist, 8)
    fn = functional_logits(model, registry, xb)
    p = fn(flat).detach().softmax(-1)
    jac = jacrev(fn)(flat).detach()
    mean_jac = (p[..., None]*jac).sum(1)
    rows = ((mean_jac[:, None]-jac)*(p/len(xb)).sqrt()[..., None]).reshape(-1, registry.size)
    blocks = information_blocks(model, registry, flat, xb, yb, ranks=(1,), repeats=1, budget_mib=3072)[0]['ggn']
    for spec, block in zip(registry.specs, blocks):
        torch.testing.assert_close(rows[:, spec.slice].T@rows[:, spec.slice], block, rtol=1e-10, atol=1e-13)


def test_randomized_top_and_cores_match_exact():
    gen = torch.Generator().manual_seed(4)
    q, _ = torch.linalg.qr(torch.randn(30, 30, generator=gen, dtype=torch.float64))
    eig = torch.cat([torch.tensor([50., 20., 10., 5.], dtype=torch.float64), 1e-3*torch.rand(26, generator=gen, dtype=torch.float64)])
    kc = (q*eig)@q.T
    z = q*eig.sqrt()
    w, kappa = probes.randomized_top(z, 4, 1)
    exact_w, exact_kappa = probes.exact_top(torch.eye(30, dtype=torch.float64)+kc, 4)
    assert probes.sin2(w, exact_w) < 1e-10
    torch.testing.assert_close(kappa, exact_kappa, rtol=1e-9, atol=1e-12)
    k = torch.eye(30, dtype=torch.float64)+kc
    ours = capacity.dense_core(probes.subspace_core(exact_w, exact_kappa), 30, k)
    theirs = capacity.dense_core(capacity.fit_core(k, 'teacher', 4), 30, k)
    torch.testing.assert_close(ours, theirs, rtol=1e-10, atol=1e-12)


def test_transport_is_identity_for_unchanged_factor_and_consistent():
    gen = torch.Generator().manual_seed(5)
    f = torch.eye(6, dtype=torch.float64)+.1*torch.randn(6, 6, generator=gen, dtype=torch.float64)
    w, _ = torch.linalg.qr(torch.randn(6, 2, generator=gen, dtype=torch.float64))
    assert probes.sin2(probes.transport(w, f, f), w) < 1e-14
    g = torch.eye(6, dtype=torch.float64)+.1*torch.randn(6, 6, generator=gen, dtype=torch.float64)
    there = probes.transport(w, f, g)
    back = probes.transport(there, g, f)
    assert probes.sin2(back, w) < 1e-12


def test_roles():
    assert c_sn_split.role('blocks.1.attention.output.parametrizations.weight.original') == 'stream_writer'
    assert c_sn_split.role('blocks.0.w2.weight') == 'stream_writer'
    assert c_sn_split.role('blocks.0.attention.key.weight') == 'first_in_branch'
    assert c_sn_split.role('blocks.2.w1.parametrizations.weight.original') == 'first_in_branch'
    assert c_sn_split.role('input.projection.weight') == 'input' and c_sn_split.role('input.weight') == 'input'
    assert c_sn_split.role('head.weight') == 'head' and c_sn_split.role('blocks.0.w1.bias') is None
