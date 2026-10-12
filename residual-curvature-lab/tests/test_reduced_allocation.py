import torch

from residual_curvature.curvature.reduced import reduced_ggn, trace_estimate
from residual_curvature.curvature.solves import IdentityMetric
from residual_curvature.optim.allocation import direction_matrix, nonnegative_qp
from residual_curvature.optim.reduced_allocation import reduced_allocation


def test_reduced_jvp_matches_dense_and_allocation_kkt(tiny):
    _, registry, flat, _, _, operator, _, factor = tiny
    native = -operator.gradient
    e, _, _ = direction_matrix(native, registry)
    k, cost = reduced_ggn(operator, e)
    reference = (e.T @ factor) @ (e.T @ factor).T
    torch.testing.assert_close(k, reference, rtol=1e-9, atol=1e-11)
    candidates, info, arrays = reduced_allocation(operator, native, registry, IdentityMetric(), .01)
    assert info['full_quadratic_gain'] >= info['native_quadratic_ray_gain'] - 1e-10
    assert max(info['kkt'].values()) < 1e-8
    torch.testing.assert_close(candidates['allocation_full'], arrays['E'] @ arrays['amplitudes_full'])
    assert cost['jvps'] == e.shape[1]
    h = arrays['H']
    general_diagonal, _ = nonnegative_qp(torch.diag(h.diag()), arrays['phi'])
    torch.testing.assert_close(general_diagonal, arrays['amplitudes_diagonal'], rtol=1e-9, atol=1e-12)


def test_trace_probe_normalization_and_local_rng(tiny):
    _, _, flat, _, _, op, _, factor = tiny
    before = torch.get_rng_state().clone()
    estimated, info = trace_estimate(op, 8, 901)
    gen = torch.Generator().manual_seed(901)
    values = []
    for _ in range(8):
        v = torch.randint(2, flat.shape, generator=gen).to(flat)*2-1
        values.append(float((factor.T @ v).square().sum()))
    torch.testing.assert_close(torch.tensor(info['samples']), torch.tensor(values), rtol=1e-9, atol=1e-11)
    assert abs(estimated-sum(values)/8) < 1e-10
    assert torch.equal(before, torch.get_rng_state())


def test_negative_alignment_can_be_active_through_coupling():
    h = torch.tensor([[1., -.9], [-.9, 1.]])
    phi = torch.tensor([1., -.1])
    amplitudes, kkt = nonnegative_qp(h, phi)
    assert (amplitudes > 0).all()
    assert max(kkt.values()) < 1e-8
    assert (phi/h.diag()).clamp_min(0)[1] == 0
