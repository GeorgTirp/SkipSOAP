"""The basis rebuild against dense references, the arms against cifar_ggn's (bitwise on the CPU), and the rebuild
arms' first steps."""
import copy
import importlib.util
import sys

import pytest
import torch

import common  # first: it puts the earlier studies' modules on the path
import micro  # noqa: E402
import n9  # noqa: E402
import train  # noqa: E402
from gpu_parsec import PARSECL  # noqa: E402
from gpu_parsec.optimizer import NEVER  # noqa: E402


def ggn_train():
    if 'ggn_train' not in sys.modules:
        spec = importlib.util.spec_from_file_location('ggn_train', common.LIB/'train.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules['ggn_train'] = module
        spec.loader.exec_module(module)
    return sys.modules['ggn_train']


def batch(n=6, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 32, 32, generator=gen, dtype=dtype), torch.randint(0, 10, (n,), generator=gen)


def soap_state(steps=3):
    model = train.build_model({'arm': 'soap', 'seed': 1}, 'cpu').double()
    x, y = batch(dtype=torch.float64)
    optimizer = PARSECL(model.parameters(), lr=1e-2, betas=(.95, .95), weight_decay=.05, precondition_frequency=NEVER,
                        rank=0, norm_blend=0., batched=False, signals=False)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    return model, optimizer, x, y


def test_probe_factors_equal_explicit_sums():
    gen = torch.Generator().manual_seed(1)
    rows = torch.randn(7, 4, 3, generator=gen, dtype=torch.float64)
    left, right = n9.probe_factors(rows, {0, 1})
    assert torch.allclose(left, sum(r@r.T for r in rows)) and torch.allclose(right, sum(r.T@r for r in rows))
    assert n9.probe_factors(rows, {1})[0] == []


def test_rebase_carries_momentum_and_second_moments():
    model, optimizer, _, _ = soap_state()
    p = model.head.weight
    state = optimizer.state[p]
    gen = torch.Generator().manual_seed(2)
    factors = [(lambda a: a@a.T)(torch.randn(k, k+2, generator=gen, dtype=torch.float64)) for k in p.shape]
    old_q = [q.clone() for q in state['Q']]
    v = state['exp_avg_sq'].clone()
    momentum = optimizer.project_back(state['exp_avg'], state).clone()
    checks = n9.rebase(optimizer, p, factors)
    for q, f in zip(state['Q'], factors):
        reference = torch.linalg.eigh(f)[1].flip(1).to(q.dtype)
        assert torch.allclose((q.T@reference).abs(), torch.eye(len(q), dtype=q.dtype), atol=1e-5)
    dense_old = torch.kron(old_q[0].double(), old_q[1].double())
    dense_new = torch.kron(state['Q'][0].double(), state['Q'][1].double())
    sigma = dense_old@torch.diag(v.reshape(-1).double())@dense_old.T
    expected = (dense_new.T@sigma@dense_new).diagonal().reshape(p.shape)
    assert torch.allclose(state['exp_avg_sq'].double(), expected, rtol=1e-5, atol=1e-12*float(v.max()))
    back = optimizer.project_back(state['exp_avg'], state)
    assert (back-momentum).norm() <= 1e-5*momentum.norm() and checks['momentum'] <= 1e-5


def test_rebuild_from_own_factors_and_g2_failure():
    model, optimizer, _, _ = soap_state()
    worst = n9.rebuild(optimizer, 'own')
    assert worst['parameters'] == sum(1 for p in model.parameters() if p.ndim == 2)
    assert worst['momentum'] <= 1e-5 and worst['second_moment_sum'] <= 1e-5 and worst['min_second_moment'] >= 0
    p = model.head.weight
    optimizer.state[p]['exp_avg_sq'][0, 0] = -1.
    with pytest.raises(RuntimeError, match='G2'):
        n9.rebase(optimizer, p, optimizer.state[p]['GG'])


def test_rebuild_schedule_is_parsec_l_probe_schedule():
    model = train.build_model({'arm': 'topfree', 'seed': 1}, 'cpu')
    optimizer = train.make_optimizer({'arm': 'topfree', 'lr': 3e-2, 'seed': 1}, model)
    for t in range(1, 40):
        assert optimizer.wants_probes() == train.rebuild_step(t)
        optimizer._t += 1


@pytest.mark.parametrize('arm', ['soap', 'topfree'])
def test_baseline_arms_equal_cifar_ggn_arms_bitwise(arm):
    lr = train.FIXED[arm][0]
    spec = {'arm': arm, 'lr': lr, 'seed': 1, 'measure': False}
    ours, theirs = train.Arm(spec, 'cpu'), ggn_train().Arm(spec, 'cpu')
    x, y = batch()
    for t in range(1, 4):
        a, b = ours.step(t, x, y), theirs.step(t, x, y, None)
        assert a == b
    assert torch.equal(ours.flat(), theirs.flat())


@pytest.mark.parametrize('arm', ['soap_eigh', 'soap_n9', 'topfree_n9'])
def test_rebuild_arms_run_and_rebuild_at_step_2(arm):
    model_arm = train.Arm({'arm': arm, 'lr': 1e-2, 'seed': 1, 'measure': False}, 'cpu')
    x, y = batch()
    bases = None
    for t in range(1, 4):
        info = model_arm.step(t, x, y)
        assert ('rebuild' in info) == (t == 2)
        if t == 1:
            bases = {id(p): [q.clone() if len(q) else q for q in model_arm.optimizer.state[p]['Q']]
                     for p in model_arm.model.parameters()}
        if t == 2:
            assert info['rebuild']['parameters'] == sum(1 for p in model_arm.model.parameters() if p.ndim == 2)
            changed = [p for p in model_arm.model.parameters() if p.ndim == 2 and p.shape[0] > 1
                       and not torch.equal(model_arm.optimizer.state[p]['Q'][0], bases[id(p)][0])]
            assert changed
    assert torch.isfinite(model_arm.flat()).all()
    assert all(s.get('precondition_frequency') == NEVER for s in model_arm.optimizer.state.values() if 'Q' in s)
