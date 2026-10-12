"""Top-free confirmation harness tests (PROTOCOL.md)."""
import train
from residual_curvature.models import make_model


def test_units_gates_and_settings():
    tune, fixed = train.tune_units(), train.eval_fixed_units()
    assert len(tune) == 5 and len(fixed) == 9
    assert {s['seed'] for s in fixed.values()} == {4, 5, 6}
    tuned = train.eval_tuned_units({'topfree_strength': 64.})
    assert len(tuned) == 6 and all(s['strength'] == 64. for s in tuned.values())
    assert sorted(n for n, s in {**fixed, **tuned}.items() if s['checkpoints']) == [
        'brake__constant__lr0.01__s4', 'soap__constant__lr0.01__s4', 'topfree__constant__lr0.01_c64__s4']
    gated = {n: train.reference_unit(s) for n, s in tune.items()}
    assert {n for n, r in gated.items() if r is not None} == {
        'soap__constant__lr0.01__s0', 'soap__wsd__lr0.01__s0', 'brake__constant__lr0.01__s0', 'topfree__constant__lr0.01_c16__s0'}
    assert all(r.name == n for n, r in gated.items() if r is not None)
    c = train.unit_config(4)
    model = make_model(c)
    topfree = train.make_optimizer(train.spec_of('topfree', 'wsd', .01, 4, strength=64.), model, c)
    assert topfree.parsec['top_rhs'] == 'gradient' and topfree.parsec['strength'] == 64. and topfree.parsec['norm_blend'] == 0.
    assert train.lr_at('wsd', .01, 500) == .0001 and train.lr_at('wsd', .01, 400) == .01
