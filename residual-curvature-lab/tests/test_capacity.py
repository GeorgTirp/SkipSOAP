from types import SimpleNamespace

import pytest
import torch

from residual_curvature.optim import capacity
from residual_curvature.optim.soap import frozen_apply


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_nonorthogonal_factor_inverse_and_apply(seed):
    gen = torch.Generator().manual_seed(seed)
    q1 = torch.eye(3, dtype=torch.float64)+.1*torch.randn(3, 3, generator=gen, dtype=torch.float64)
    q2 = torch.eye(2, dtype=torch.float64)+.1*torch.randn(2, 2, generator=gen, dtype=torch.float64)
    d = torch.rand(3, 2, generator=gen, dtype=torch.float64)+.3
    context = {'basis': [q1, q2], 'denominator': d}
    f = torch.kron(q1, q2)@torch.diag(d.flatten().rsqrt())
    v = torch.randn(6, 4, generator=gen, dtype=torch.float64)
    for transpose in [False, True]:
        matrix = f.T if transpose else f
        for inverse in [False, True]:
            result = capacity.factor(v.reshape(3, 2, 4), context, transpose, inverse).reshape(6, 4)
            target = torch.linalg.solve(matrix, v) if inverse else matrix@v
            torch.testing.assert_close(result, target, rtol=1e-12, atol=1e-12)
    registry = SimpleNamespace(specs=[SimpleNamespace(name='a', count=6, shape=(3,2), slice=slice(0,6))])
    k = torch.diag(torch.linspace(.1, 3, 6, dtype=torch.float64))
    for family in ['identity', 'scalar', 'diagonal', 'teacher', 'coordinate', 'state', 'exact']:
        core = capacity.fit_core(k, family, 4, v[:, 0], v[:, 1], d)
        got = capacity.apply([core], registry, {'a': context}, .7, v)
        m = capacity.dense_core(core, 6, k)
        torch.testing.assert_close(got, .7*f@m@f.T@v, rtol=1e-12, atol=1e-12)
        assert float(torch.linalg.eigvalsh(m).min()) > 0
    baseline = capacity.apply([{'kind': 'identity'}], registry, {'a': context}, 1., v)
    torch.testing.assert_close(baseline, frozen_apply(v.reshape(3,2,4), context).reshape(6,4), rtol=1e-12, atol=1e-12)


def test_population_fit_and_teacher_ranking():
    k = torch.diag(torch.tensor([200., .01, 1., 2.], dtype=torch.float64))
    core = capacity.fit_core(k, 'teacher', 2)
    m = capacity.dense_core(core, 4, k)
    torch.testing.assert_close(m, torch.diag(torch.tensor([.005, 100., 1., 1.], dtype=torch.float64)))
    full = capacity.dense_core(capacity.fit_core(k, 'teacher', 4), 4, k)
    torch.testing.assert_close(full, torch.linalg.inv(k))
    # Perturb the fitted correction in its allowed span; exact population loss must increase.
    kinv = torch.linalg.inv(k)
    objective = lambda value: torch.trace((value-kinv)@k@(value-kinv))
    for perturbation in [-.001, .001]:
        assert objective(m+perturbation*core['U']@core['U'].T) >= objective(m)-1e-12
    diagonal = capacity.fit_core(k, 'diagonal')
    torch.testing.assert_close(capacity.dense_core(diagonal, 4, k), kinv)
    clipped = capacity.fit_core(torch.diag(torch.tensor([1e-10,1e10],dtype=torch.float64)), 'diagonal')
    assert clipped['clipped'] == 2
    assert clipped['diagonal'].tolist() == [1e6,1e-6]


def test_global_ray_error_does_not_rescale_individual_tensors():
    registry = SimpleNamespace(specs=[SimpleNamespace(slice=slice(0,2)),SimpleNamespace(slice=slice(2,4))])
    systems = [torch.diag(torch.tensor([2.,3.],dtype=torch.float64)),torch.eye(2,dtype=torch.float64)]
    rhs = torch.eye(4,dtype=torch.float64)+torch.ones(4,4,dtype=torch.float64)
    exact = torch.cat([torch.linalg.solve(a,rhs[s.slice]) for a,s in zip(systems,registry.specs)])
    good = capacity.energy_errors(rhs, 3*exact, exact, systems, registry)
    assert max(good['ray']) < 1e-25
    assert max(abs(v-4) for v in good['raw']) < 1e-12
    bad = capacity.energy_errors(rhs, torch.cat([2*exact[:2],4*exact[2:]]), exact, systems, registry)
    assert min(bad['ray']) > .01


def test_energy_error_distinguishes_weak_and_strong_curvature():
    registry=SimpleNamespace(specs=[SimpleNamespace(slice=slice(0,2))])
    a=torch.diag(torch.tensor([1e-12,1.],dtype=torch.float64))
    rhs=torch.tensor([0.,1.],dtype=torch.float64)
    weak=rhs+torch.tensor([1e-5,0.],dtype=torch.float64)
    strong=rhs+torch.tensor([0.,1e-5],dtype=torch.float64)
    assert capacity.energy_errors(rhs,weak,rhs,[a],registry)['raw'][0]**.5 < 1e-8
    assert capacity.energy_errors(rhs,strong,rhs,[a],registry)['raw'][0]**.5 > 1e-8


def test_damping_calibration_recovers_exact_zero_curvature_inverse():
    f=torch.tensor([[2.,.3],[.2,1.]],dtype=torch.float64)
    mu,scale=.03,.7
    inv=torch.linalg.inv(f)
    omega=scale*inv.T@inv
    beta=1/(mu*scale)
    k=beta*f.T@(mu*omega)@f
    torch.testing.assert_close(k,torch.eye(2,dtype=torch.float64),atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(beta*f@f.T,torch.linalg.inv(mu*omega),atol=1e-12,rtol=1e-12)
