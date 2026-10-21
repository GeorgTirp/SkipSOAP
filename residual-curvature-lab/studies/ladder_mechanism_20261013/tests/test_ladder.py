"""Galerkin steps, subspace iteration, the held-brake reconstruction, whitening round trips and snapshot round trips
against references (CPU)."""
import pytest
import torch

import common  # first: it puts the earlier studies' modules on the path
import algebra  # noqa: E402
import ggn  # noqa: E402
import ladder  # noqa: E402
import micro  # noqa: E402
import states  # noqa: E402
from gpu_parsec import PARSECL  # noqa: E402

D = torch.float64


def spd(n, gen):
    a = torch.randn(n, n+3, generator=gen, dtype=D)
    return a@a.T+.1*torch.eye(n, dtype=D)


def test_galerkin_is_the_restricted_optimum_and_full_basis_is_the_solve():
    gen = torch.Generator().manual_seed(1)
    a, g = spd(12, gen), torch.randn(12, generator=gen, dtype=D)
    full, reduction = algebra.galerkin(a, g, torch.eye(12, dtype=D))
    assert torch.allclose(full, -torch.linalg.solve(a, g)) and abs(reduction-.5*float(g@torch.linalg.solve(a, g))) <= 1e-12
    b = algebra.orth(torch.randn(12, 4, generator=gen, dtype=D))
    y, r = algebra.galerkin(a, g, b)
    q = lambda v: float(g@v+.5*v@(a@v))
    assert abs(q(y)+r) <= 1e-12
    for _ in range(20):
        other = b@torch.randn(4, generator=gen, dtype=D)
        assert q(other) >= q(y)-1e-12
    assert algebra.ray(a, g, y) <= r+1e-12 and abs(algebra.ray(a, g, full)-reduction) <= 1e-10


def test_orth_drops_dependent_columns():
    gen = torch.Generator().manual_seed(2)
    u = torch.randn(10, 3, generator=gen, dtype=D)
    b = algebra.orth(torch.cat([u, u[:, :1]+u[:, 1:2], torch.zeros(10, 1, dtype=D)], 1))
    assert b.shape[1] == 3 and torch.allclose(b.T@b, torch.eye(3, dtype=D), atol=1e-12)


def test_top_eigenvectors_by_subspace_iteration():
    gen = torch.Generator().manual_seed(3)
    q = torch.linalg.qr(torch.randn(600, 600, generator=gen, dtype=D))[0]
    values = torch.cat([torch.linspace(100, 50, 16, dtype=D), torch.linspace(5, .01, 584, dtype=D)])
    m = q@torch.diag(values)@q.T
    v = algebra.top_eigenvectors(m, 16, torch.Generator().manual_seed(4))
    projector = q[:, :16]@q[:, :16].T
    assert (projector@v-v).norm() <= 1e-6*v.norm()


def test_held_brake_reconstruction_matches_parsec_l_at_the_probe_step():
    model = states.GGN_TRAIN.build_model({'arm': 'topfree', 'seed': 1}, 'cpu').double()
    gen = torch.Generator().manual_seed(5)
    x, y = torch.randn(4, 3, 32, 32, generator=gen, dtype=D), torch.randint(0, 10, (4,), generator=gen)
    optimizer = PARSECL(model.parameters(), lr=3e-2, betas=(.95, .95), weight_decay=.05, precondition_frequency=10,
                        rank=16, norm_blend=0., batched=False, signals=False, calibration='lr', strength=64.,
                        top_rhs='gradient', seed=1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        probes = micro.lean_probes(model, x, samples=1, generator=torch.Generator().manual_seed(6)) if optimizer.wants_probes() else None
        optimizer.step(probes=probes)
    params = list(model.parameters())
    for index, row in optimizer.last_info['probe']['tensors'].items():
        p = params[index]
        u, kappa, _ = algebra.brake_directions(optimizer, p, 3e-2, optimizer.state[p]['step'])
        assert abs(float(kappa.max())-row['kappa_max']) <= 1e-8*max(row['kappa_max'], 1e-30)


def test_whitening_round_trips():
    gen = torch.Generator().manual_seed(7)
    q = [torch.linalg.qr(torch.randn(5, 5, generator=gen, dtype=D))[0], torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=D))[0]]
    context = {'basis': q, 'denominator': torch.rand(5, 3, generator=gen, dtype=D)+.1}
    xi = torch.randn(5, 3, generator=gen, dtype=D)
    step = ggn.unwhiten(xi, context)
    assert torch.allclose(ladder.coordinates(step, context).reshape(5, 3), xi, atol=1e-12)
    g = torch.randn(5, 3, generator=gen, dtype=D)
    assert abs(float(ladder.whitened(g, context)@ladder.coordinates(step, context))-float((g*step).sum())) <= 1e-12


def test_snapshot_round_trip_reproduces_the_next_steps_bitwise():
    data = states.GGN_TRAIN.Data('cpu')
    gen = torch.Generator().manual_seed(8)
    x, y = torch.randn(6, 3, 32, 32, generator=gen), torch.randint(0, 10, (6,), generator=gen)
    a = states.Replay('topfree', 1, 'cpu', data)
    for t in range(1, 3):
        a.t += 1
        a.arm.step(a.t, x, y, None)
    snap = a.snapshot()
    b = states.Replay('topfree', 1, 'cpu', data)
    b.restore(snap)
    for _ in range(2):
        for r in (a, b):
            r.t += 1
            r.arm.step(r.t, x, y, None)
    assert torch.equal(a.flat(), b.flat())
