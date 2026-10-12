import copy
import json

import pytest
import torch
from torch.func import grad

from residual_curvature import checkpoint
from residual_curvature.cli import estimate, main
from residual_curvature.config import load_config, validate
from residual_curvature.curvature.operators import GGN, guard, loss
from residual_curvature.curvature.sketch import categorical_probes, factors_from_probes
from residual_curvature.data import DataOrder, synthetic
from residual_curvature.experiments.search import local_direction_grid, training_backtracking
from residual_curvature.experiments.snapshot import snapshot
from residual_curvature.experiments.training import run, training_step
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import soap_propose
from test_soap_allocation import assert_tree


def test_configuration_and_guards(config, tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('soap:\n  learning_rate: 0.2\n')
    with pytest.raises(ValueError, match='Unknown'):
        load_config(path)
    for section, key, value in [('resources', 'budget_mib', -1), ('resources', 'dual_cap', 0), ('damping', 'floor', 0), ('solver', 'max_iterations', 0)]:
        c = copy.deepcopy(config)
        c[section][key] = value
        with pytest.raises(ValueError):
            validate(c)
    with pytest.raises(MemoryError):
        guard(2049, 12, 4, config['resources'], torch.float64, dense=True)
    with pytest.raises(MemoryError):
        guard(10, 2049, 4, config['resources'], torch.float64)
    with pytest.raises(MemoryError):
        guard(200000, 100, 4, config['resources'], torch.float64)
    deferred = copy.deepcopy(config)
    deferred['model']['kind'] = 'transformer'
    with pytest.raises(NotImplementedError, match='deferred'):
        estimate(deferred)


def test_dry_runs_have_no_dataset_or_output_side_effects(tmp_path, monkeypatch):
    c = load_config('configs/mnist_snapshots.yaml')
    c['data']['root'] = str(tmp_path / 'missing-data')
    c['output'] = str(tmp_path / 'missing-runs')
    plan = estimate(c)
    assert plan['parameters'] == 6634 and plan['outputs'] == 640
    assert not (tmp_path / 'missing-data').exists() and not (tmp_path / 'missing-runs').exists()


def test_nonmutation_real_snapshot(config, tmp_path):
    config['damping']['gammas'] = [.1]
    config['training']['gamma'] = .1
    config['sketch'].update(ranks=[4], repeats=1)
    config['training']['rank'] = 4
    model, registry = make_model(config), None
    registry = Registry(model)
    x, y, _ = synthetic(config)
    state, _, _ = training_step(model, registry, None, x[:4], y[:4], config, 1)
    before_params = copy.deepcopy(model.state_dict())
    before_state = copy.deepcopy(state)
    rng = checkpoint.capture_rng()
    diagnostic = snapshot(model, registry, state, x, y, list(range(4)),
                           [list(range(4+i*4, 8+i*4)) for i in range(4)], config, 1, tmp_path / 'audit')
    assert_tree(before_params, model.state_dict())
    assert_tree(before_state, state)
    assert torch.equal(rng['torch'], torch.get_rng_state())
    assert rng['python'] == checkpoint.capture_rng()['python']
    import numpy as np
    assert np.array_equal(rng['numpy'][1], checkpoint.capture_rng()['numpy'][1])
    assert diagnostic['checks']['ggn_action'] < 1e-9
    assert diagnostic['checks']['microbatch_0'] < 1e-9
    assert diagnostic['checks']['dense_dual'] < diagnostic['dense']['dual_forward_error_bound']
    assert diagnostic['dense']['dense_solve_residual'] < 1e-8
    rows = json.loads((tmp_path / 'audit' / 'candidates.json').read_text())
    assert all(len(r['search']['trials']) == 13 for r in rows if r['search']['available'])
    assert all(len(r['selected']['independent_decreases']) == 4 for r in rows)


def test_resume_and_monitored_training_equality(config, tmp_path):
    config['output'] = str(tmp_path / 'runs')
    config['training'].update(steps=6, snapshots=[], checkpoint_every=1)
    config['soap']['precondition_frequency'] = 2
    full_dir, full_model, full_state = run(config, analyze=False)
    interrupted_dir, _, _ = run(config, analyze=False, stop_after=3)
    resumed_dir, resumed_model, resumed_state = run(config, analyze=False, resume=interrupted_dir / 'checkpoint.pt')
    assert_tree(full_model.state_dict(), resumed_model.state_dict())
    assert_tree(full_state, resumed_state)
    full = torch.load(full_dir / 'checkpoint.pt', weights_only=False)
    resumed = torch.load(resumed_dir / 'checkpoint.pt', weights_only=False)
    for key in ['data_order', 'used_context', 'step', 'accepted', 'probe_stream']:
        assert_tree(full[key], resumed[key])
    def monitoring(model, registry, state, x, y, c, step):
        flat = registry.flatten(dict(model.named_parameters())).detach()
        fn = functional_logits(model, registry, x[:4])
        op = GGN(fn, flat, y[:4])
        delta, _, _ = soap_propose(registry.unflatten(flat), registry.unflatten(op.gradient), state, c['soap'])
        xi = categorical_probes(op.logits, 4, c['analysis_seed'] + step)
        factors_from_probes(fn, flat, xi)
        direction = registry.flatten(delta)
        local_direction_grid(flat, direction, float(direction.norm()), lambda p: loss(fn(p), y[:4]), c['search']['grid_exponents'])
    _, monitored_model, monitored_state = run(config, analyze=False, monitor=monitoring)
    assert_tree(full_model.state_dict(), monitored_model.state_dict())
    assert_tree(full_state, monitored_state)


@pytest.mark.parametrize('method', ['sketch_diagonal', 'sketch_coupled'])
def test_training_statistics_advance_once_when_rejected(config, method):
    config['training'].update(method=method, policy='training_backtracking')
    config['search'].update(initial=1e8, max_evaluations=1)
    model = make_model(config)
    registry = Registry(model)
    x, y, _ = synthetic(config)
    state, _, first = training_step(model, registry, None, x[:4], y[:4], config, 1)
    assert not first['accepted']
    flat = registry.flatten(dict(model.named_parameters())).detach()
    fn = functional_logits(model, registry, x[4:8])
    gradient = grad(lambda v: loss(fn(v), y[4:8]))(flat)
    _, expected_state, _ = soap_propose(registry.unflatten(flat), registry.unflatten(gradient), state, config['soap'])
    next_state, _, row = training_step(model, registry, state, x[4:8], y[4:8], config, 2)
    assert not row['accepted']
    assert torch.equal(flat, registry.flatten(dict(model.named_parameters())))
    assert_tree(expected_state, next_state)


def test_search_policies(config):
    flat, g = torch.tensor([1.]), torch.tensor([1.])
    objective = lambda p: .5 * p.square().sum()
    delta, detail = training_backtracking(flat, g, g, objective, config['search'])
    assert not detail['accepted'] and torch.equal(delta, g*0)
    delta, detail = training_backtracking(flat, -g, g, objective, config['search'])
    assert detail['accepted'] and objective(flat+delta) < objective(flat)
    _, unavailable = local_direction_grid(flat, -g, 0, objective, config['search']['grid_exponents'])
    assert not unavailable['available']


def test_failed_snapshot_preserves_consumed_step(config, tmp_path, monkeypatch):
    import residual_curvature.experiments.training as training
    config['output'] = str(tmp_path / 'runs')
    config['training'].update(steps=3, snapshots=[2], checkpoint_every=10)
    def fail_snapshot(*args, **kwargs):
        raise RuntimeError('injected numerical gate failure')
    monkeypatch.setattr(training, 'snapshot', fail_snapshot)
    with pytest.raises(RuntimeError, match='injected numerical gate'):
        training.run(config, analyze=True)
    directory = next((tmp_path / 'runs').iterdir())
    saved = torch.load(directory / 'checkpoint.pt', weights_only=False)
    assert saved['step'] == 2
    assert json.loads((directory / 'training.json').read_text())[-1]['step'] == 2
    assert json.loads((directory / 'failure.json').read_text())['step'] == 2


def test_legacy_mlp_checkpoint_accepts_new_inactive_vit_defaults(config, tmp_path):
    legacy = copy.deepcopy(config)
    legacy.pop('vit')
    model = make_model(legacy)
    order = DataOrder(list(range(16)), 123)
    order.take(4)
    path = tmp_path / 'legacy.pt'
    checkpoint.save(path, model, None, None, 1, 0, order, legacy)
    restored_model = make_model(config)
    restored_order = DataOrder(list(range(16)), 123)
    restored = checkpoint.load(path, restored_model, restored_order, config)
    assert restored['step'] == 1
    assert_tree(model.state_dict(), restored_model.state_dict())
    assert_tree(order.state_dict(), restored_order.state_dict())
    changed = copy.deepcopy(config)
    changed['soap']['lr'] *= 2
    with pytest.raises(ValueError, match='configuration mismatch'):
        checkpoint.load(path, restored_model, restored_order, changed)
    changed = copy.deepcopy(config)
    changed['vit']['heads'] += 1
    with pytest.raises(ValueError, match='configuration mismatch'):
        checkpoint.load(path, restored_model, restored_order, changed)
