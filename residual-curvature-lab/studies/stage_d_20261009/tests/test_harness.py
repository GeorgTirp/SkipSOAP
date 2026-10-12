"""Stage D harness tests (PROTOCOL.md)."""
import torch

import arms
import common
import ggn_selected
import train
from residual_curvature.curvature.operators import GGN
from residual_curvature.data import DataOrder
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from run_exact_training_followup import evaluate


def test_unit_sets_and_explicit_settings():
    tune, fixed = train.tune_units(), train.eval_fixed_units()
    assert len(tune) == 19 and len(fixed) == 12 and len(set(tune) | set(fixed)) == 31
    selection = {'brake': {'calibration': 'lr', 'strength': 4., 'lr': .02}, 'early_newton_rhs': 'momentum'}
    tuned = train.eval_tuned_units(selection)
    assert len(tuned) == 9 and {s['seed'] for s in tuned.values()} == {1, 2, 3}
    c = train.unit_config(1)
    assert c['model']['branch_scale'] == 1. and c['model']['blocks'] == 2 and c['soap']['weight_decay'] == 0
    model = make_model(c)
    brake = train.make_optimizer({'arm': 'brake', 'calibration': 'lr', 'strength': 30., 'lr': .03, 'seed': 1}, model, c)
    assert brake.parsec['norm_blend'] == 0. and brake.parsec['calibration'] == 'lr' and brake.parsec['strength'] == 30.
    assert brake.param_groups[0]['lr'] == .03 and brake.parsec['batched'] is False and brake.parsec['rank'] == 16
    early = train.make_optimizer({'arm': 'early_newton', 'rhs': 'gradient', 'handover': True, 'seed': 1}, model, c)
    assert early.parsec['calibration'] == 'damping' and early.parsec['select_handover'] == 10 and early.parsec['rhs'] == 'gradient'
    assert early.param_groups[0]['lr'] == .01 and early.parsec['norm_blend'] == 0.
    soap = train.make_optimizer({'arm': 'soap', 'lr': .02, 'seed': 1}, model, c)
    assert soap.parsec['rank'] == 0 and soap.param_groups[0]['lr'] == .02


def test_soap_arm_reproduces_gap_audit_native_soap(mnist):
    c0, x, y, manifest = mnist
    c = train.unit_config(1)
    model = make_model(c)
    registry = Registry(model)
    optimizer = train.make_optimizer({'arm': 'soap', 'lr': .01, 'seed': 1}, model, c)
    order = DataOrder(manifest['training'], common.data_order_seed(1))
    reference = {r['step']: r['validation_loss'] for r in common.read(common.GAP_EVAL/'native_soap__lr0.01__s1'/'validation.json')}
    for step in range(1, 26):
        ids = order.take(512)
        flat = registry.flatten(dict(model.named_parameters())).detach()
        gradient = registry.unflatten(GGN(functional_logits(model, registry, x[ids]), flat, y[ids]).gradient)
        for name, p in model.named_parameters():
            p.grad = gradient[name].detach().clone()
        optimizer.step()
        if step in [10, 25]:
            assert abs(evaluate(model, x, y, manifest['validation'])['validation_loss']-reference[step]) <= 1e-9


def test_ggn_selected_on_the_training_batch_equals_the_gap_audit_step(mnist):
    c0, x, y, manifest = mnist
    c = train.unit_config(1)
    model, twin = make_model(c), make_model(c)
    registry = Registry(model)
    state = twin_state = None
    for step in range(1, 4):
        ids = manifest['training'][64*step:64*(step+1)]
        flat = registry.flatten(dict(model.named_parameters())).detach()
        delta, state, _, row, _ = ggn_selected.propose(.001, model, registry, flat, state, x[ids], y[ids], x[ids], y[ids], c, step)
        twin_flat = registry.flatten(dict(twin.named_parameters())).detach()
        reference, twin_state, _, _ = arms.study_ggn_propose(.001, twin, registry, twin_flat, twin_state, x[ids], y[ids], c, step)
        assert torch.equal(delta, reference)
        commit(model, registry, flat+delta)
        commit(twin, registry, twin_flat+reference)


def test_json_safe_records_non_finite_values_as_null(tmp_path):
    row = {'optimism': {'r_soap': float('inf'), 'r_parsec_l': 0.5}, 'values': [1., float('nan')], 'n': 3}
    safe = train.json_safe(row)
    assert safe == {'optimism': {'r_soap': None, 'r_parsec_l': 0.5}, 'values': [1., None], 'n': 3}
    common.write_compact(tmp_path/'row.json', [safe])
