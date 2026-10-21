"""Unit plan, schedule, and the arms against cifar_ggn's (bitwise on the CPU under the same schedule factor)."""
import importlib.util
import sys

import pytest
import torch

import common  # first: it puts the earlier studies' modules on the path
import train  # noqa: E402


def ggn_train():
    if 'ggn_train' not in sys.modules:
        spec = importlib.util.spec_from_file_location('ggn_train', common.LIB/'train.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules['ggn_train'] = module
        spec.loader.exec_module(module)
    return sys.modules['ggn_train']


def test_schedule_and_unit_plan():
    assert train.STEPS == 2000 and train.lr_factor(1) == 1/100 and train.lr_factor(100) == 1. and train.lr_factor(1600) == 1.
    assert abs(train.lr_factor(2000)-.01) <= 1e-15 and abs(train.lr_factor(1601)-1.) <= 1e-15
    assert train.VALIDATION[:5] == [0, 10, 25, 50, 100] and train.VALIDATION[-1] == 2000
    units = train.tune_units()
    assert sorted((s['arm'], s['lr']) for s in units.values()) == sorted((a, lr) for a in train.ARMS for lr in train.LR_GRID)
    evaluation = train.eval_units({'soap': {'lr': 1e-2}, 'topfree': {'lr': 3e-2}, 'topfree_n9': {'lr': 3e-2}})
    assert len(evaluation) == 15 and {s['seed'] for s in evaluation.values()} == {1, 2, 3, 4, 5}


@pytest.mark.parametrize('arm,lr', [('soap', 1e-2), ('topfree', 3e-2)])
def test_arms_equal_cifar_ggn_under_the_same_schedule(arm, lr, monkeypatch):
    gt = ggn_train()
    monkeypatch.setattr(train, 'lr_factor', gt.lr_factor)
    spec = {'arm': arm, 'lr': lr, 'seed': 1, 'measure': False}
    ours, theirs = train.Arm(spec, 'cpu'), gt.Arm(spec, 'cpu')
    gen = torch.Generator().manual_seed(0)
    x, y = torch.randn(6, 3, 32, 32, generator=gen), torch.randint(0, 10, (6,), generator=gen)
    for t in range(1, 4):
        assert ours.step(t, x, y) == theirs.step(t, x, y, None)
    assert torch.equal(ours.flat(), theirs.flat())


def test_topfree_n9_rebuilds_at_step_2_with_refresh_off():
    from gpu_parsec.optimizer import NEVER
    arm = train.Arm({'arm': 'topfree_n9', 'lr': 3e-2, 'seed': 1, 'measure': False}, 'cpu')
    gen = torch.Generator().manual_seed(0)
    x, y = torch.randn(6, 3, 32, 32, generator=gen), torch.randint(0, 10, (6,), generator=gen)
    for t in range(1, 4):
        info = arm.step(t, x, y)
        assert ('rebuild' in info) == (t == 2) and info['probe'] == (t == 2)
    assert torch.isfinite(arm.flat()).all()
    assert all(s.get('precondition_frequency') == NEVER for s in arm.optimizer.state.values() if 'Q' in s)
