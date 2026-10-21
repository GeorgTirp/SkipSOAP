"""Schedule, unit plan, selection rule, and the arms against cifar_ggn's and probe_variants' (bitwise on the CPU under
the same schedule factor)."""
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


def test_schedule_and_unit_plan():
    assert train.STEPS == 1000 and train.lr_factor(50) == 1. and abs(train.lr_factor(1000)-.01) <= 1e-15
    units = train.tune_units()
    assert sorted(units) == sorted(['soap__lr0.03__s0', 'soap__lr0.09__s0', 'soap__lr0.27__s0',
                                    'topfree_ot__lr0.03__c1024__s0', 'topfree_ot__lr0.03__c4096__s0',
                                    'topfree_ot__lr0.09__c1024__s0', 'topfree_ot__lr0.09__c4096__s0'])
    chosen = {'soap': {'lr': .09, 'strength': None}, 'topfree_ot': {'lr': .09, 'strength': 4096.}}
    evaluation = train.eval_units(chosen)
    assert sorted(evaluation) == sorted([f'soap__lr0.09__s{s}' for s in (1, 2, 3)]+[f'topfree_ot__lr0.09__c4096__s{s}' for s in (1, 2, 3)])
    ext = train.extension_units({'soap': {'edge': 'high', 'grid': [.03, .09, .27], 'strength': None},
                                 'topfree_ot': {'edge': 'low', 'grid': [.03, .09], 'strength': 1024.}})
    assert sorted((s['arm'], round(s['lr'], 6), s.get('strength')) for s in ext.values()) == [('soap', .81, None), ('topfree_ot', .01, 1024.)]


def test_selection_picks_lowest_ema_and_flags_edges(tmp_path):
    units = {'topfree_ot__lr0.03__c1024__s0': {'arm': 'topfree_ot', 'lr': .03, 'strength': 1024.},
             'topfree_ot__lr0.09__c4096__s0': {'arm': 'topfree_ot', 'lr': .09, 'strength': 4096.},
             'soap__lr0.09__s0': {'arm': 'soap', 'lr': .09}, 'soap__lr0.27__s0': {'arm': 'soap', 'lr': .27}}
    (tmp_path/'protocol.json').write_text(json.dumps({'units': units}))
    values = {'topfree_ot__lr0.03__c1024__s0': .50, 'topfree_ot__lr0.09__c4096__s0': .45, 'soap__lr0.09__s0': .44, 'soap__lr0.27__s0': None}
    for name, value in values.items():
        (tmp_path/name).mkdir()
        rows = [{'step': 0}] if value is None else [{'step': train.STEPS, **{f'ema{d:g}': {'validation_loss': value+d/10} for d in train.EMA_DECAYS}}]
        (tmp_path/name/'validation.json').write_text(json.dumps(rows))
    chosen = train.selection([tmp_path])
    assert chosen['topfree_ot']['lr'] == .09 and chosen['topfree_ot']['strength'] == 4096. and chosen['topfree_ot']['edge'] == 'high'
    assert chosen['soap']['lr'] == .09 and chosen['soap']['edge'] == 'low' and chosen['soap']['ema_decay'] == min(train.EMA_DECAYS)


@pytest.mark.parametrize('arm', ['soap', 'topfree_ot'])
def test_arms_equal_earlier_implementations(arm, monkeypatch):
    if arm == 'soap':
        ref = load('ggn_train', common.LIB/'train.py')
        spec = {'arm': 'soap', 'lr': 3e-2, 'seed': 1, 'measure': False}
    else:
        ref = load('pv_train', common.LAB/'studies'/'probe_variants_20261013'/'scripts'/'train.py')
        spec = {'arm': 'topfree_ot', 'lr': 3e-2, 'seed': 1, 'measure': False, 'strength': 4096.}
    monkeypatch.setattr(train, 'lr_factor', ref.lr_factor)
    ours, theirs = train.Arm(spec, 'cpu'), ref.Arm(spec, 'cpu')
    x, y = batch()
    for t in range(1, 4):
        a, b = ours.step(t, x, y), theirs.step(t, x, y, None)
        assert {k: v for k, v in b.items() if k != 'eta_lambda_max'} == a
    assert torch.equal(ours.flat(), theirs.flat())
