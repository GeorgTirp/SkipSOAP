import pytest
import torch

from residual_curvature.optim import capacity
from residual_curvature.curvature.wasserstein import graph_laplacian


@pytest.mark.parametrize('family',['combined_teacher','combined_state'])
def test_combined_spd_application_storage_and_calibration(family):
    gen=torch.Generator().manual_seed(811)
    dtype=torch.float64
    v=torch.randn(6,6,generator=gen,dtype=dtype)
    k=v.T@v+.5*torch.eye(6,dtype=dtype)
    zg,zm=torch.randn(2,6,generator=gen,dtype=dtype)
    denominator=torch.rand(2,3,generator=gen,dtype=dtype)+.3
    core=capacity.fit_core(k,family,3,zg,zm,denominator)
    root=core['root']
    inner=core['inner']
    u=inner['U']
    dense=root[:,None]*(torch.eye(6,dtype=dtype)+u@inner['correction']@u.T)*root[None,:]
    assert torch.linalg.eigvalsh(dense).min()>0
    assert core['storage']==6+u.numel()+inner['correction'].numel()
    rhs=torch.randn(6,4,generator=gen,dtype=dtype)
    torch.testing.assert_close(capacity.core_apply(core,rhs),dense@rhs,rtol=1e-12,atol=1e-12)
    scaled=capacity.fit_core(7*k,family,3,7**.5*zg,7**.5*zm,denominator)
    torch.testing.assert_close(7*capacity.dense_core(scaled,6,k),dense,rtol=1e-10,atol=1e-11)
    if family=='combined_teacher':
        full=capacity.fit_core(k,family,6,zg,zm,denominator)
        torch.testing.assert_close(capacity.dense_core(full,6,k),torch.linalg.inv(k),rtol=1e-11,atol=1e-11)


def test_fisher_output_factor_is_also_a_graph_laplacian():
    p=torch.tensor([.03,.17,.3,.5],dtype=torch.float64)
    conductance=2*p[:,None]*p[None,:]/(p[:,None]+p[None,:])
    conductance.fill_diagonal_(0)
    torch.testing.assert_close(graph_laplacian(p,conductance),torch.diag(p)-p[:,None]*p[None,:],rtol=1e-14,atol=1e-14)
