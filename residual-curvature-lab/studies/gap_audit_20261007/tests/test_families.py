"""Check 2c family tests: Kronecker convention, nearest-Kronecker fitting, PCG."""
from types import SimpleNamespace

import torch

import families as fam
from residual_curvature.config import DEFAULT
from residual_curvature.models import Registry, make_model


def spd(n, gen, shift=.1):
    a = torch.randn(n, n, generator=gen, dtype=torch.float64)
    return a@a.T+shift*torch.eye(n, dtype=torch.float64)


def test_registry_row_major_kronecker_convention():
    gen = torch.Generator().manual_seed(0)
    m, n = 3, 5
    a, b = torch.randn(m, m, generator=gen, dtype=torch.float64), torch.randn(n, n, generator=gen, dtype=torch.float64)
    w = torch.randn(m, n, generator=gen, dtype=torch.float64)
    torch.testing.assert_close(torch.kron(a, b)@w.reshape(-1), (a@w@b.T).reshape(-1), rtol=1e-13, atol=1e-13)
    model = make_model(DEFAULT)
    registry = Registry(model)
    params = dict(model.named_parameters())
    flat = registry.flatten(params)
    for spec in registry.specs:
        if len(spec.shape) == 2:
            assert spec.orientation == 'out,in'
            assert torch.equal(flat[spec.slice], params[spec.name].detach().reshape(-1))
            assert torch.equal(flat[spec.slice].reshape(spec.shape)[1, 0], params[spec.name].detach()[1, 0])


def test_rearrangement_and_nearest_kronecker():
    gen = torch.Generator().manual_seed(1)
    m, n = 4, 3
    a, b = spd(m, gen), spd(n, gen)
    c = torch.kron(a, b)
    torch.testing.assert_close(fam.rearrange(c, m, n), torch.outer(a.reshape(-1), b.reshape(-1)), rtol=1e-13, atol=1e-13)
    spec = SimpleNamespace(shape=(m, n), count=m*n)
    block, params, info = fam.kron_sum_block(c, spec, 1)
    torch.testing.assert_close(block, c, rtol=1e-11, atol=1e-11)
    a1, b1 = params['terms'][0]
    assert float(torch.linalg.eigvalsh(a1).min()) > 0 and float(torch.linalg.eigvalsh(b1).min()) > 0
    assert not info['clipped']
    diag, _, info = fam.kron_basis_diag_block(c, spec)
    torch.testing.assert_close(diag, c, rtol=1e-11, atol=1e-11)
    assert info['storage'] == m*m+n*n+m*n


def test_kronecker_sum_rank_two_and_basis_diagonal_is_psd():
    gen = torch.Generator().manual_seed(2)
    m, n = 3, 4
    c = torch.kron(spd(m, gen), spd(n, gen))+torch.kron(spd(m, gen), spd(n, gen))
    spec = SimpleNamespace(shape=(m, n), count=m*n)
    block, _, info = fam.kron_sum_block(c, spec, 2)
    torch.testing.assert_close(block, c, rtol=1e-10, atol=1e-10)
    general = spd(m*n, gen)
    diag, params, _ = fam.kron_basis_diag_block(general, spec)
    assert float(torch.linalg.eigvalsh(diag).min()) >= -1e-10
    ua = torch.linalg.eigh(params['A1'])[1]
    ub = torch.linalg.eigh(params['B1'])[1]
    u = torch.kron(ua, ub)
    torch.testing.assert_close(torch.diag(u.T@diag@u), torch.diag(u.T@general@u), rtol=1e-10, atol=1e-10)


def test_clip_psd_reports_negative_mass():
    matrix = torch.diag(torch.tensor([2., 1., -.5], dtype=torch.float64))
    clipped, info = fam.clip_psd(matrix)
    assert info['clipped'] and info['min_eigenvalue'] == -.5
    assert abs(info['clipped_trace_fraction']-.5/3.5) < 1e-15
    assert float(torch.linalg.eigvalsh(clipped).min()) >= 0


def test_truncated_pcg_counts_and_converges():
    gen = torch.Generator().manual_seed(3)
    a = spd(6, gen)
    precondition = torch.diag(torch.rand(6, generator=gen, dtype=torch.float64)+.5)
    rhs = torch.randn(6, generator=gen, dtype=torch.float64)
    calls = []
    def action(v):
        calls.append(1)
        return a@v
    x, matvecs = fam.pcg(action, lambda v: precondition@v, rhs, 6)
    assert matvecs == len(calls) == 6
    torch.testing.assert_close(x, torch.linalg.solve(a, rhs), rtol=1e-8, atol=1e-8)
    x1, _ = fam.pcg(action, lambda v: precondition@v, rhs, 1)
    z = precondition@rhs
    torch.testing.assert_close(x1, z*(rhs@z)/(z@a@z), rtol=1e-13, atol=1e-13)
