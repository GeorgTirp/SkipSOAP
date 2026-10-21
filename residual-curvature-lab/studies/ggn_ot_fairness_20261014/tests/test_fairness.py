"""Unit plan and extension rule; topfree (GGN probes) at c = 64 equal to cifar_ggn's topfree, and topfree_ot equal to
ot_tuned_1000's arm (bitwise on the CPU under the same schedule factor)."""
import importlib.util
import json
import sys

import pytest
import torch

import common  # first: it puts the earlier studies' modules on the path
import train  # noqa: E402


def load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def batch():
    gen = torch.Generator().manual_seed(0)
    return torch.randn(6, 3, 32, 32, generator=gen), torch.randint(0, 10, (6,), generator=gen)


def test_unit_plan_and_extensions():
    assert sorted(train.tune_units()) == sorted(f'topfree__lr{lr:g}__c{c:g}__s0' for lr in (.03, .09) for c in (64, 256, 1024))
    ext = train.extension_units({'topfree': {'lr': .09, 'strength': 1024., 'edge': 'high', 'c_edge': 'high',
                                             'lr_grid': [.03, .09], 'c_grid': [64., 256., 1024.]}})
    assert sorted(ext) == sorted(['topfree__lr0.27__c1024__s0', 'topfree__lr0.09__c4096__s0'])
    evaluation = train.eval_units({'topfree': {'lr': .09, 'strength': 256.}})
    assert sorted(evaluation) == sorted([f'topfree__lr0.09__c256__s{s}' for s in (1, 2, 3)]+['topfree_ot__lr0.09__c4096__s1'])


def test_selection_flags_lr_and_c_edges(tmp_path):
    units = {f'topfree__lr{lr:g}__c{c:g}__s0': {'arm': 'topfree', 'lr': lr, 'strength': c} for lr in (.03, .09) for c in (64., 256.)}
    (tmp_path/'protocol.json').write_text(json.dumps({'units': units}))
    for i, name in enumerate(sorted(units)):
        (tmp_path/name).mkdir()
        value = .5 if name != 'topfree__lr0.09__c256__s0' else .4
        (tmp_path/name/'validation.json').write_text(json.dumps([{'step': train.STEPS, **{f'ema{d:g}': {'validation_loss': value} for d in train.EMA_DECAYS}}]))
    row = train.selection([tmp_path])['topfree']
    assert (row['lr'], row['strength'], row['edge'], row['c_edge']) == (.09, 256., 'high', 'high')


@pytest.mark.parametrize('arm', ['topfree', 'topfree_ot'])
def test_arms_equal_references(arm, monkeypatch):
    if arm == 'topfree':
        ref = load('ggn_train', common.LIB/'train.py')
        spec, refspec = {'arm': 'topfree', 'lr': 3e-2, 'seed': 1, 'measure': False, 'strength': 64.}, {'arm': 'topfree', 'lr': 3e-2, 'seed': 1, 'measure': False}
        call = lambda r, t, x, y: r.step(t, x, y, None)
    else:
        ref = load('ott_train', common.LAB/'studies'/'ot_tuned_1000_20261014'/'scripts'/'train.py')
        spec = refspec = {'arm': 'topfree_ot', 'lr': 9e-2, 'seed': 1, 'measure': False, 'strength': 4096.}
        call = lambda r, t, x, y: r.step(t, x, y)
    monkeypatch.setattr(train, 'lr_factor', ref.lr_factor)
    ours, theirs = train.Arm(spec, 'cpu'), ref.Arm(refspec, 'cpu')
    x, y = batch()
    for t in range(1, 4):
        a, b = ours.step(t, x, y), call(theirs, t, x, y)
        assert a == {k: v for k, v in b.items() if k != 'eta_lambda_max'} or a == b
    assert torch.equal(ours.flat(), theirs.flat())
