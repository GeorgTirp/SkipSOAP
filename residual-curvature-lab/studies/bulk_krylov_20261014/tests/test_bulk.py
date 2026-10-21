"""Spectra, power-law fits, Krylov and Galerkin steps, deflated CG, Chebyshev iteration, the matrix-free whitened GGN
operator and state loading against references (CPU; PROTOCOL.md, section 5, G0)."""
import copy

import torch

import common  # first: it puts the earlier studies' modules on the path
import algebra  # noqa: E402
import bulk  # noqa: E402
import ggn  # noqa: E402
import run  # noqa: E402
import spectral  # noqa: E402
import states  # noqa: E402

D = torch.float64


def spd(n, gen):
    a = torch.randn(n, n+3, generator=gen, dtype=D)
    return a@a.T+.1*torch.eye(n, dtype=D)


def with_spectrum(values, gen):
    q, _ = torch.linalg.qr(torch.randn(len(values), len(values), generator=gen, dtype=D))
    return (q*values)@q.T, q


def test_top_eigenpairs_exact_and_by_subspace_iteration():
    gen = torch.Generator().manual_seed(1)
    small = spd(40, gen)
    values, vectors = spectral.top_eigenpairs(small, 10, gen)
    assert torch.allclose(values, torch.linalg.eigvalsh(small).flip(0)[:10], rtol=1e-10)
    assert torch.allclose(small@vectors, vectors*values, atol=1e-9)
    truth = torch.arange(1, 601, dtype=D)**-1.
    large, _ = with_spectrum(truth, gen)
    values, vectors = spectral.top_eigenpairs(large, 50, gen, oversample=50)
    assert torch.allclose(values, truth[:50], rtol=1e-8)
    assert float((large@vectors[:, -1]-values[-1]*vectors[:, -1]).norm()/values[-1]) <= 1e-4


def test_loglog_fit_recovers_exact_power_laws():
    k = torch.arange(1, 201, dtype=D)
    alpha, r2 = spectral.loglog_fit(3*k**-1.3, 1, 200)
    assert abs(alpha-1.3) <= 1e-10 and abs(r2-1) <= 1e-10
    broken = torch.where(k <= 16, 3*k**-.5, 3*16**.7*k**-1.2)
    assert abs(spectral.loglog_fit(broken, 1, 16)[0]-.5) <= 1e-10
    assert abs(spectral.loglog_fit(broken, 17, 200)[0]-1.2) <= 1e-10
    assert spectral.loglog_fit(broken, 1, 200)[1] < 1-1e-3
    assert spectral.loglog_fit(torch.tensor([1., 0., 0., 0.]), 1, 4) == (None, None)
    assert spectral.loglog_fit(3*k[:10]**-1.3, 17, 200) == (None, None)  # head.bias, 10 values (DEVIATIONS.md D1)


def test_alpha_from_trace_inverts_the_power_law_sum():
    for alpha, dimension in [(.7, 4096), (1.4, 64), (2.5, 16384)]:
        trace = float((2.*torch.arange(1, dimension+1, dtype=D)**-alpha).sum())
        assert abs(spectral.alpha_from_trace(2., trace, dimension)-alpha) <= 1e-8
    assert spectral.alpha_from_trace(2., 2.*4096+1, 4096) is None  # trace above the flat spectrum's
    assert spectral.alpha_from_trace(2., 1., 4096) is None  # trace below the top eigenvalue


def test_krylov_basis_spans_the_krylov_space_and_galerkin_there_is_cg():
    gen = torch.Generator().manual_seed(2)
    a, b = spd(30, gen), torch.randn(30, generator=gen, dtype=D)
    for m in [1, 2, 4, 8]:
        q = spectral.krylov_basis(lambda v: a@v, b, m)
        assert q.shape == (30, m) and torch.allclose(q.T@q, torch.eye(m, dtype=D), atol=1e-12)
        powers = torch.stack([torch.linalg.matrix_power(a, j)@b for j in range(m)], 1)
        assert float((powers-q@(q.T@powers)).norm()/powers.norm()) <= 1e-10
        y, reduction = algebra.galerkin(a, b, q)
        cg = spectral.conjugate_gradient(lambda v: a@v, -b, m)
        assert torch.allclose(y, cg, rtol=1e-8, atol=1e-10)
        assert abs(reduction-algebra.ray(a, b, cg)) <= 1e-8*reduction
    exact = spectral.conjugate_gradient(lambda v: a@v, -b, 60)
    assert torch.allclose(exact, -torch.linalg.solve(a, b), rtol=1e-8, atol=1e-10)
    assert spectral.krylov_basis(lambda v: a@v, torch.zeros(30, dtype=D), 3).shape == (30, 0)


def test_deflated_krylov_with_invariant_top_is_top_solve_plus_bulk_cg():
    gen = torch.Generator().manual_seed(3)
    values = torch.arange(1, 41, dtype=D)**-1.
    k, q = with_spectrum(values, gen)
    a, b, u = k+1e-3*torch.eye(40, dtype=D), torch.randn(40, generator=gen, dtype=D), q[:, :16]
    projector = lambda v: v-u@(u.T@v)
    for m in [2, 4, 8]:
        bulk_basis = spectral.krylov_basis(lambda v: projector(a@projector(v)), projector(b), m)
        y, reduction = algebra.galerkin(a, b, algebra.orth(torch.cat([u, bulk_basis], 1)))
        y_top, r_top = algebra.galerkin(a, b, u)
        y_bulk = spectral.conjugate_gradient(lambda v: projector(a@projector(v)), -projector(b), m)
        assert torch.allclose(y, y_top+y_bulk, rtol=1e-8, atol=1e-10)
        assert reduction >= algebra.galerkin(a, b, spectral.krylov_basis(lambda v: a@v, b, m))[1]-1e-12
    full = spectral.krylov_basis(lambda v: projector(a@projector(v)), projector(b), 24)
    y, _ = algebra.galerkin(a, b, algebra.orth(torch.cat([u, full], 1)))
    assert torch.allclose(y, -torch.linalg.solve(a, b), rtol=1e-8, atol=1e-10)


def test_chebyshev_iteration_has_the_chebyshev_residual_polynomial():
    gen = torch.Generator().manual_seed(4)
    low, high = .05, 2.
    lam = low+(high-low)*torch.rand(25, generator=gen, dtype=D)
    lam[0], lam[1] = low, high
    b = torch.randn(25, generator=gen, dtype=D)
    theta, delta = (high+low)/2, (high-low)/2

    def chebyshev(m, t):
        t = torch.as_tensor(t, dtype=D)  # inside [-1, 1] up to rounding, or a scalar above 1
        return torch.cos(m*torch.arccos(t.clamp(-1, 1))) if bool((t.abs() <= 1+1e-12).all()) else torch.cosh(m*torch.arccosh(t))

    for m in [1, 2, 4, 8]:
        x = spectral.chebyshev_solve(lambda v: lam*v, b, low, high, m)
        residual = chebyshev(m, (theta-lam)/delta)/chebyshev(m, torch.tensor(theta/delta, dtype=D))
        assert torch.allclose(x, (1-residual)*b/lam, rtol=1e-9, atol=1e-11)
    a, _ = with_spectrum(lam, gen)
    x = spectral.chebyshev_solve(lambda v: a@v, b, low, high, 120)
    assert torch.allclose(x, torch.linalg.solve(a, b), rtol=1e-8, atol=1e-10)


def test_powerlaw_step_assigns_by_probe_diagonal_rank():
    zg = torch.tensor([1., 2., 3., 4.], dtype=D)
    diagonal = torch.tensor([.1, .9, .5, .2], dtype=D)  # ranks 4, 1, 2, 3
    step = bulk.powerlaw_step(zg, 2., 1., diagonal, .5)
    assumed = 2.*torch.tensor([4., 1., 2., 3.], dtype=D)**-1.
    assert torch.allclose(step, -zg/(assumed+.5), rtol=1e-14)


def replay_after_two_steps():
    data = states.GGN_TRAIN.Data('cpu')
    gen = torch.Generator().manual_seed(8)
    x, y = torch.randn(6, 3, 32, 32, generator=gen), torch.randint(0, 10, (6,), generator=gen)
    replay = states.Replay('topfree', 1, 'cpu', data)
    for _ in range(2):
        replay.step(x, y)
    return replay, x, y


def test_whitened_operator_matches_the_dense_whitened_blocks():
    replay, x, _ = replay_after_two_steps()
    model, registry = replay.arm.model, replay.arm.registry
    model.double()
    flat = replay.flat().double()
    contexts = {name: {'basis': [q.double() if torch.is_tensor(q) else q for q in c['basis']], 'denominator': c['denominator'].double()}
                for name, c in states.contexts(replay.arm.optimizer, registry).items()}
    x = x[:3].double()
    damping = .37
    apply = bulk.whitened_operator(model, registry, flat, x, contexts, damping)
    gen = torch.Generator().manual_seed(9)
    names = ['blocks.0.norm1.weight', 'input.cls_token', 'blocks.0.attention.query.weight']
    chosen = [s for s in registry.specs if s.name in names]
    blocks, trace = ggn.accumulate(model, registry, flat, x, chosen, 2, all_parameters=True, contexts=contexts)
    for spec in chosen:
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=D)
        out = apply(v)
        dense = blocks[spec.name]@v[spec.slice]+damping*v[spec.slice]
        assert float((out[spec.slice]-dense).norm()/dense.norm()) <= 1e-8
    u, w = torch.randn(len(flat), generator=gen, dtype=D), torch.randn(len(flat), generator=gen, dtype=D)
    assert abs(float(u@apply(w))-float(w@apply(u))) <= 1e-8*abs(float(u@apply(w)))
    assert trace > 0


def test_proposal_leaves_the_state_unchanged():
    replay, x, y = replay_after_two_steps()
    flat, saved = replay.flat().clone(), copy.deepcopy(replay.arm.optimizer.state_dict())
    generator = replay.arm.probe_generator.get_state()
    step = bulk.proposal(replay, x, y)
    assert float(step.norm()) > 0 and torch.equal(replay.flat(), flat)
    assert torch.equal(replay.arm.probe_generator.get_state(), generator)
    after = replay.arm.optimizer.state_dict()
    for key, state in saved['state'].items():
        for name, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(after['state'][key][name], value)
    assert [g['weight_decay'] for g in after['param_groups']] == [g['weight_decay'] for g in saved['param_groups']]


def test_ladder_snapshot_loads_with_its_parameters_and_step():
    data = states.GGN_TRAIN.Data('cpu')
    snapshot = torch.load(run.LADDER_STATES/'topfree_s1_t0010.pt', weights_only=False)
    replay = run.restored(1, 10, 'cpu', data)
    assert replay.t == 10 and torch.equal(replay.flat(), snapshot['parameters'])
    xs, _ = replay.following(0)
    x, _ = replay.next_batch()
    assert x.shape == xs.shape == (states.BATCH, 3, 32, 32)
