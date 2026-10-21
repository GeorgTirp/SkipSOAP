"""Quad machinery, SOAP's refresh applied now, and the reconstruction of PARSEC-L's brake, against independent
references (float64, CPU)."""
import copy

import pytest
import torch

import common  # first: it puts the earlier studies' modules on the path
import ggn  # noqa: E402
import quad  # noqa: E402
import refresh  # noqa: E402
import snr  # noqa: E402
import states  # noqa: E402
from gpu_parsec import PARSECL  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402
from residual_curvature.optim.soap import frozen_apply  # noqa: E402

D = torch.float64


def orthonormal(n, gen):
    return torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=D))[0]


def kron_dense(bases, shape):
    mats = [b if len(b) else torch.eye(k, dtype=D) for b, k in zip(bases, shape)]
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


@pytest.mark.parametrize('shape', [(5, 3), (1, 4), (6,)])
def test_exact_diagonal_damping_diagonal_and_operator(shape):
    gen = torch.Generator().manual_seed(1)
    count = int(torch.tensor(shape).prod())
    a = torch.randn(count, count+3, generator=gen, dtype=D)
    block = a@a.T
    bases = [orthonormal(k, gen) for k in shape] if len(shape) == 2 else [[]]
    dense = kron_dense(bases, shape)
    expected = (dense.T@block@dense).diagonal().reshape(shape)
    assert torch.allclose(quad.exact_diagonal(block, shape, bases, budget=7), expected, rtol=1e-12, atol=1e-12)
    soap = [orthonormal(k, gen)+.01*torch.randn(k, k, generator=gen, dtype=D) for k in shape] if len(shape) == 2 else [[]]
    den = torch.rand(shape, generator=gen, dtype=D)+.1
    context = {'basis': soap, 'denominator': den}
    damping = torch.stack([(dense[:, i].reshape(shape)*frozen_apply(dense[:, i].reshape(shape), context, inverse=True)).sum()
                           for i in range(count)]).reshape(shape)
    assert torch.allclose(quad.damping_diagonal(den, soap, bases), damping, rtol=1e-10, atol=1e-12)
    g = torch.randn(shape, generator=gen, dtype=D)
    diagonal = expected+1.
    d = quad.apply_diagonal(g, bases, diagonal)
    reference = -(dense@((dense.T@g.reshape(-1))/diagonal.reshape(-1)))
    assert torch.allclose(d.reshape(-1), reference, rtol=1e-12, atol=1e-12)


def test_nearest_kronecker_recovers_a_kronecker_product():
    gen = torch.Generator().manual_seed(2)
    a, b = torch.randn(4, 4, generator=gen, dtype=D), torch.randn(3, 3, generator=gen, dtype=D)
    a, b = a@a.T+torch.eye(4, dtype=D), b@b.T+torch.eye(3, dtype=D)
    left, right = quad.nearest_kronecker(torch.kron(a, b), (4, 3))
    scale = float((torch.kron(left, right)*torch.kron(a, b)).sum()/torch.kron(left, right).square().sum())
    assert torch.allclose(scale*torch.kron(left, right), torch.kron(a, b), rtol=1e-10, atol=1e-10)
    for f, ref in zip(quad.factor_bases([left, right]), quad.factor_bases([a, b])):
        assert torch.allclose((f.T@ref).abs(), torch.eye(len(f), dtype=D), atol=1e-8)


def small_cifar(seed=1):
    model = states.SNS_TRAIN.build_model({'arm': 'soap', 'seed': seed}, 'cpu').double()
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 3, 32, 32, generator=gen, dtype=D)
    y = torch.randint(0, 10, (4,), generator=gen)
    return model, Registry(model), x, y


def test_tensor_quadratics_equal_dense_blocks():
    model, registry, x, y = small_cifar()
    flat = registry.flatten(dict(model.named_parameters())).detach()
    gen = torch.Generator().manual_seed(3)
    directions = [torch.randn(registry.size, generator=gen, dtype=D) for _ in range(2)]
    out, trace = quad.tensor_quadratics(model, registry, flat, x, directions, chunk=3)
    specs = [s for s in registry.specs if s.name.startswith('blocks.0.') or s.name.startswith('head')]
    blocks, total = ggn.accumulate(model, registry, flat, x, specs, 3, all_parameters=True)
    assert abs(trace-total) <= 1e-10*total
    for spec in specs:
        for k, d in enumerate(directions):
            ref = float(d[spec.slice]@(blocks[spec.name]@d[spec.slice]))
            assert abs(out[spec.name][k]-ref) <= 1e-10*abs(ref)+1e-14


def soap_steps(model, x, y, steps, **kwargs):
    optimizer = PARSECL(model.parameters(), lr=1e-2, betas=(.95, .95), weight_decay=.05, precondition_frequency=10,
                        norm_blend=0., batched=False, signals=False, **kwargs)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    return optimizer


def test_refresh_now_is_soaps_refresh_with_the_momentum_carried_over():
    """refresh_now equals upstream's update_preconditioner sequence without the factor update, bitwise; the momentum
    in original coordinates survives to float32 precision (SOAP's bases are float32 even for float64 parameters)."""
    model, registry, x, y = small_cifar()
    optimizer = soap_steps(model, x, y, 4, rank=0)
    momentum, expected = {}, {}
    for g in optimizer.param_groups:
        for p in g['params']:
            state = copy.deepcopy(optimizer.state[p])
            momentum[id(p)] = optimizer.project_back(state['exp_avg'], state).clone()
            state['exp_avg'] = optimizer.project_back(state['exp_avg'], state)
            state['Q'] = optimizer.get_orthogonal_matrix_QR(state, g['max_precond_dim'], g['merge_dims'])
            state['exp_avg'] = optimizer.project(state['exp_avg'], state)
            expected[id(p)] = state
    states.refresh_now(optimizer)
    for g in optimizer.param_groups:
        for p in g['params']:
            state, e = optimizer.state[p], expected[id(p)]
            assert all((len(q) == 0 and len(r) == 0) or torch.equal(q, r) for q, r in zip(state['Q'], e['Q']))
            assert torch.equal(state['exp_avg'], e['exp_avg']) and torch.equal(state['exp_avg_sq'], e['exp_avg_sq'])
            back = optimizer.project_back(state['exp_avg'], state)
            assert (back-momentum[id(p)]).norm() <= 1e-5*momentum[id(p)].norm()


def test_refresh_signals_on_known_matrices():
    gen = torch.Generator().manual_seed(4)
    q = orthonormal(5, gen)
    diag = torch.diag(torch.arange(5., 0., -1., dtype=D))
    f = q@diag@q.T
    assert refresh.rotation([f], [q]) <= 1e-12 and refresh.spread([f], [q], top=3) <= 1e-12
    other = orthonormal(5, gen)
    assert refresh.rotation([f], [other]) > .1 and refresh.spread([f], [other], top=3) > .1


def test_brake_reconstruction_matches_parsec_l_at_the_probe_step():
    model, registry, x, y = small_cifar()
    optimizer = PARSECL(model.parameters(), lr=3e-2, betas=(.95, .95), weight_decay=.05, precondition_frequency=10,
                        rank=16, norm_blend=0., batched=False, signals=False, calibration='lr', strength=64.,
                        top_rhs='gradient', seed=1)
    import micro
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        probes = micro.lean_probes(model, x, samples=1, generator=torch.Generator().manual_seed(5)) if optimizer.wants_probes() else None
        optimizer.step(probes=probes)
    logged = optimizer.last_info['probe']['tensors']
    params = list(model.parameters())
    checked = 0
    for index, row in logged.items():
        p = params[index]
        u, kappa, beta = snr.brake_directions(optimizer, p, 3e-2, optimizer.state[p]['step'])
        assert abs(float(kappa.max())-row['kappa_max']) <= 1e-8*max(row['kappa_max'], 1e-30)
        assert torch.allclose(u.T@u, torch.eye(u.shape[1], dtype=D), atol=1e-10)
        checked += 1
    assert checked == len(params)


@pytest.mark.skipif(not torch.cuda.is_available() or not (states.GGN_EVAL/'topfree__lr0.03__s1').exists(),
                    reason='needs the GPU and the stored top-free checkpoints')
def test_topfree_checkpoint_loads_on_the_gpu():
    """D1: the PARSEC-L generator state stays on the CPU; the held factors are present."""
    model, registry, optimizer, _ = states.topfree(1, 50, torch.device('cuda'))
    assert optimizer._generator is not None and optimizer._generator.device.type == 'cpu'
    held = [p for p in model.parameters() if optimizer.state[p].get('parsec', {}).get('wt') is not None]
    assert len(held) == len(registry.specs)
