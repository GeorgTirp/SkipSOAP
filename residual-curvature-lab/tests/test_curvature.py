import torch
from torch.func import grad, hessian, jacrev

from residual_curvature.curvature.operators import loss, metric_action, output_factor, residual
from residual_curvature.curvature.solves import IdentityMetric, cg, diagonal_solve, factor_solve, sketch_sweeps
from residual_curvature.curvature.transport import residual_stages, transported_metrics, reverse_message_sweep, stagewise_factors
from residual_curvature.curvature.sketch import categorical_probes, factors_from_probes, microbatch_factors


def test_registry_gradient_and_normalization(tiny):
    model, registry, flat, x, y, op, j, u = tiny
    assert torch.equal(flat, registry.flatten(registry.unflatten(flat)))
    assert registry.size == sum(p.numel() for p in model.parameters())
    assert len({s.name for s in registry.specs}) == len(list(model.parameters()))
    direct = grad(lambda p: loss(op.fn(p), y))(flat)
    torch.testing.assert_close(direct, j.T @ residual(op.logits, y).flatten(), rtol=1e-9, atol=1e-11)
    z = op.logits.detach()
    r = residual(z, y)
    torch.testing.assert_close(r, (z.softmax(-1)-torch.nn.functional.one_hot(y, 3))/4)
    torch.testing.assert_close(metric_action(z.repeat(2, 1), torch.ones_like(z).repeat(2, 1)), torch.zeros(8, 3), atol=1e-14, rtol=0)


def test_dense_matrixfree_factor_linearized_hessian(tiny):
    _, _, flat, _, y, op, j, u = tiny
    root, _ = output_factor(op.logits)
    dense = j.T @ root @ root.T @ j
    torch.testing.assert_close(dense, u @ u.T, rtol=1e-9, atol=1e-11)
    gen = torch.Generator().manual_seed(333)
    for _ in range(3):
        v = torch.randn(flat.shape, generator=gen)
        torch.testing.assert_close(op.action(v), dense @ v, rtol=1e-9, atol=1e-11)
    linearized = lambda d: loss(op.logits.detach() + (j @ d).reshape_as(op.logits), y)
    torch.testing.assert_close(hessian(linearized)(torch.zeros_like(flat)), dense, rtol=1e-9, atol=1e-11)
    assert torch.linalg.eigvalsh(dense).min() >= -1e-10 * max(1, dense.norm())
    assert torch.linalg.matrix_rank(dense) <= len(y) * 2
    offdiag = dense - torch.diag(dense.diag())
    assert (offdiag < -1e-8).any() and (offdiag > 1e-8).any()


def test_output_weights_and_analytic_categorical_covariance():
    z = torch.tensor([[1.2, -.3, .1], [-.4, .9, .5]])
    weights = torch.tensor([2., 0.])
    p = z.softmax(-1)
    xi = torch.eye(3) - p[0]
    analytic = xi.T @ torch.diag(p[0]) @ xi
    root, _ = output_factor(z, weights)
    torch.testing.assert_close(root @ root.T, torch.block_diag(analytic, torch.zeros(3, 3)), rtol=1e-9, atol=1e-12)
    targets = torch.tensor([1, 0])
    torch.testing.assert_close(grad(lambda v: loss(v, targets, weights))(z), residual(z, targets, weights))


def test_transport_cross_blocks_messages_and_stagewise_probes(tiny):
    model, registry, flat, x, _, op, j, u = tiny
    stages = residual_stages(model, registry, flat, x)
    root, _ = output_factor(op.logits)
    hz = root @ root.T
    metrics = transported_metrics(stages, hz)
    # Independent downstream differentiation, not only products of local Jacobians.
    for i, h in enumerate(stages['h']):
        def downstream(state):
            for block in model.blocks[i:]:
                state = block(state)
            return model.head(state)
        direct_t = jacrev(downstream)(h).reshape(op.logits.numel(), -1).detach()
        torch.testing.assert_close(metrics[i], direct_t.T @ hz @ direct_t, rtol=1e-9, atol=1e-11)
    dense = u @ u.T
    for i, si in enumerate(stages['slices']):
        torch.testing.assert_close(dense[si, si], stages['B'][i].T @ metrics[i+1] @ stages['B'][i], rtol=1e-9, atol=1e-11)
        for k in range(i+1, len(stages['slices'])):
            transport = torch.eye(stages['R'][i].shape[0])
            for l in range(i+1, k+1):
                transport = stages['R'][l] @ transport
            sk = stages['slices'][k]
            predicted = stages['B'][i].T @ transport.T @ metrics[k+1] @ stages['B'][k]
            torch.testing.assert_close(dense[si, sk], predicted, rtol=1e-9, atol=1e-11)
    # Nonzero terminal head proposal seeds the initial message.
    terminal = torch.linspace(-.05, .05, op.logits.numel())
    mu = .1
    message = reverse_message_sweep(stages, hz, op.gradient, mu, torch.eye(len(flat)), terminal)
    explicit = torch.zeros_like(flat)
    for sl in reversed(stages['slices']):
        rhs = op.gradient[sl] + dense[sl] @ explicit + j[:, sl].T @ hz @ terminal
        explicit[sl] = torch.linalg.solve(dense[sl, sl] + mu * torch.eye(sl.stop-sl.start), -rhs)
    torch.testing.assert_close(message, explicit, rtol=1e-9, atol=1e-11)
    xi = categorical_probes(op.logits, 8, 88)
    whole = factors_from_probes(op.fn, flat, xi)
    torch.testing.assert_close(stagewise_factors(model, registry, flat, x, xi, stages), whole, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(microbatch_factors(model, registry, flat, x, xi, 2), whole, rtol=1e-9, atol=1e-11)


def test_solves_diagonals_and_nonzero_sweeps(tiny):
    _, registry, flat, _, _, op, _, u = tiny
    mu = .2
    dense = u @ u.T + mu * torch.eye(len(flat))
    reference = torch.linalg.solve(dense, -op.gradient)
    dual = factor_solve(op.gradient, u, mu, lambda x: x)
    iterative = cg(lambda v: op.action(v) + mu*v, -op.gradient, tolerance=1e-10)
    assert iterative.converged
    torch.testing.assert_close(dual, reference, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(iterative.delta, reference, rtol=1e-9, atol=1e-11)
    diagonal = diagonal_solve(op.gradient, u, mu, lambda x: x, registry.groups())
    for sl in registry.groups():
        torch.testing.assert_close(diagonal[sl], torch.linalg.solve(dense[sl, sl], -op.gradient[sl]), rtol=1e-9, atol=1e-11)
    initial = torch.linspace(-.1, .1, len(flat))
    one = sketch_sweeps(op.gradient, u, mu, IdentityMetric(), registry.groups(), initial=initial)
    explicit = initial.clone()
    for sl in reversed(registry.groups()):
        explicit[sl] = 0
        explicit[sl] = torch.linalg.solve(dense[sl, sl], -(op.gradient[sl] + dense[sl] @ explicit))
    torch.testing.assert_close(one, explicit, rtol=1e-9, atol=1e-11)
    converged = sketch_sweeps(op.gradient, u, mu, IdentityMetric(), registry.groups(), sweeps=400)
    torch.testing.assert_close(converged, reference, rtol=1e-8, atol=1e-10)


def test_rank_deficient_and_poorly_conditioned_solve():
    gen = torch.Generator().manual_seed(17)
    u = torch.randn(20, 3, generator=gen)
    u[:, 2] = u[:, 0]
    g = torch.randn(20, generator=gen)
    mu = 1e-5
    delta = factor_solve(g, u, mu, lambda v: v)
    assert (u @ (u.T @ delta) + mu * delta + g).norm() / g.norm() < 1e-8
    stopped = cg(lambda v: u @ (u.T @ v) + mu*v, -g, max_iterations=1)
    assert not stopped.converged


def test_nested_probes_and_matched_diagonal(tiny):
    _, registry, flat, _, _, op, _, _ = tiny
    xi4, xi8 = categorical_probes(op.logits, 4, 13), categorical_probes(op.logits, 8, 13)
    torch.testing.assert_close(xi4, xi8[..., :4] * 2**.5)
    y = factors_from_probes(op.fn, flat, xi8)
    coupled = y @ y.T
    diagonal = torch.zeros_like(coupled)
    for sl in registry.groups():
        diagonal[sl, sl] = y[sl] @ y[sl].T
        torch.testing.assert_close(diagonal[sl, sl], coupled[sl, sl], rtol=1e-9, atol=1e-11)
