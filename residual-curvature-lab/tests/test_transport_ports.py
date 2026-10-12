import copy

import torch
from torch.func import jacrev

from residual_curvature.curvature.wasserstein import graph_laplacian, probability_metric_solve, streamed_wasserstein_blocks, action
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import parsec, ni_soap


def test_wasserstein_mass_gauge_permutation_and_uniform_scaling():
    p = torch.tensor([[.2, .3, .5], [1/3, 1/3, 1/3]])
    lap = graph_laplacian(p)
    torch.testing.assert_close(lap.sum(-1), torch.zeros_like(p), rtol=0, atol=1e-15)
    assert torch.linalg.eigvalsh(lap).min() > -1e-14
    tangent = torch.tensor([[.4, -.7, .3], [-.2, .1, .1]])
    potential = probability_metric_solve(p, tangent[..., None]).squeeze(-1)
    torch.testing.assert_close((lap@potential[..., None]).squeeze(-1), tangent, rtol=1e-12, atol=1e-14)
    permutation = torch.tensor([2, 0, 1])
    permuted = probability_metric_solve(p[:, permutation], tangent[:, permutation, None]).squeeze(-1)
    torch.testing.assert_close(permuted, potential[:, permutation], rtol=1e-12, atol=1e-14)
    h = torch.diag(p[1])-p[1, :, None]*p[1, None, :]
    w = h@probability_metric_solve(p[1], h)
    torch.testing.assert_close(w, h/3, rtol=1e-12, atol=1e-14)


def test_wasserstein_streamed_and_independent_action(tiny):
    model, registry, flat, x, _, _, _, _ = tiny
    fn = lambda v: functional_logits(model, registry, x)(v).softmax(-1)
    p = fn(flat)
    jp = jacrev(fn)(flat)
    dense = torch.einsum('ncd,nce->de', jp, probability_metric_solve(p, jp))/len(x)
    blocks, info = streamed_wasserstein_blocks(model, registry, flat, x, microbatch=3)
    for spec, block in zip(registry.specs, blocks):
        torch.testing.assert_close(block, dense[spec.slice, spec.slice], rtol=1e-9, atol=1e-12)
    v = torch.linspace(-1, 1, registry.size)
    torch.testing.assert_close(action(model, registry, flat, x, v), dense@v, rtol=1e-9, atol=1e-12)
    assert abs(info['trace']-float(dense.trace())) < 1e-10


def test_parsec_eigenspace_transform_refresh_and_replay():
    h = torch.diag(torch.tensor([.2, 1., 4., 9.]))
    gradient = torch.tensor([1., -2., 3., -.5])
    rng = torch.get_rng_state().clone()
    eigen, vectors, info = parsec.lanczos_ritz(lambda v: h@v, gradient, rank=2, iterations=4)
    torch.testing.assert_close(eigen, torch.tensor([9., 4.]), rtol=1e-10, atol=1e-12)
    assert max(info['relative_ritz_residuals']) < 1e-10
    transformed = parsec.transform_gradient(gradient, eigen, vectors, .1)
    expected = gradient.clone()
    expected[2:] /= torch.tensor([4.1, 9.1]).sqrt()
    torch.testing.assert_close(transformed, expected, rtol=1e-10, atol=1e-12)
    settings = {'rank': 2, 'lanczos_iters': 4, 'refresh_every': 3, 'seed': 0,
                'damping': .1, 'betas': (.9, .999), 'lr': .003, 'eps': 1e-8}
    class Op:
        def action(self, v):
            return h@v
    state = base = None
    for step in range(1, 5):
        g = gradient*step + .1*step**2
        before = copy.deepcopy(state)
        delta, state, diagnostics = parsec.propose(g, gradient, state, settings, Op())
        repeated, replay, _ = parsec.propose(g, gradient, before, settings, Op())
        torch.testing.assert_close(repeated, delta, rtol=0, atol=0)
        torch.testing.assert_close(state['m'], replay['m'], rtol=0, atol=0)
        native, base, _ = parsec.propose(g, gradient, base, dict(settings, refresh_every=0))
        if step < 3:
            torch.testing.assert_close(native, delta, rtol=0, atol=0)
        assert (diagnostics['refresh'] is not None) == (step == 3)
    assert torch.equal(rng, torch.get_rng_state())


def test_kron_noise_energy_exact_basis_sum():
    ql = torch.linalg.qr(torch.tensor([[1., 2.], [-2., 1.]]))[0]
    qr = torch.linalg.qr(torch.tensor([[2., 1., 3.], [1., -2., 1.], [1., 0., 2.]]))[0]
    el, er = torch.tensor([.3, 2.]), torch.tensor([1., 4., .2])
    energy = 0.
    for z in torch.eye(6).reshape(6, 2, 3):
        energy += float(ni_soap.kron_noise(z, ql, qr, el, er).square().sum())
    assert abs(energy-float(el.sum()*er.sum())) < 1e-12


def test_ni_soap_zero_noise_state_refresh_and_replay(tiny):
    model, registry, flat, _, _, op, _, _ = tiny
    settings = {'lr': .003, 'betas': (.95, .95), 'eps': 1e-8, 'shampoo_beta': .95,
                'refresh_every': 2, 'batch_size': 512, 'seed': 0, 'alpha': 1., 'weight_decay': 0.}
    routes = ni_soap.matrix_routes(model)
    state = None
    rng = torch.get_rng_state().clone()
    for step in range(1, 5):
        g = op.gradient + .0001*step
        before = copy.deepcopy(state)
        noisy, state, info = ni_soap.propose(g, flat, registry, routes, state, settings)
        replay, replay_state, _ = ni_soap.propose(g, flat, registry, routes, before, settings)
        plain, zero_state, zero_info = ni_soap.propose(g, flat, registry, routes, before, dict(settings, alpha=0.))
        torch.testing.assert_close(noisy, replay, rtol=0, atol=0)
        assert zero_info['noise_energy'] == 0 and zero_info['expected_noise_energy'] == 0
        assert abs(info['update_energy']-info['drift_energy']-info['noise_energy']-info['drift_noise_cross_term']) < 1e-12
        for spec in registry.specs:
            for key, value in state['per_param'][spec.name].items():
                other = zero_state['per_param'][spec.name][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
                else:
                    assert value == other
            if step == 1 and routes[spec.name]:
                assert float(noisy[spec.slice].norm()) == 0
            elif step == 1:
                expected = -settings['lr']*g[spec.slice]/(g[spec.slice].abs()+settings['eps'])
                torch.testing.assert_close(plain[spec.slice], expected, rtol=1e-12, atol=1e-14)
    assert torch.equal(rng, torch.get_rng_state())
