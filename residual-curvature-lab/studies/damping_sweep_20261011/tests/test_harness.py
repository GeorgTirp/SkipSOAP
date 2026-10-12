"""Damping-sweep harness tests (PROTOCOL.md part 1)."""
import train


def test_sweep_units_and_gate():
    units = train.sweep_units()
    assert len(units) == 13 and sorted({s['strength'] for s in units.values()}) == [64., 256., 1024., 4096., 1e9]
    gated = {n: train.reference_unit(s) for n, s in units.items()}
    assert [n for n, r in gated.items() if r is not None] == ['topfree__constant__lr0.01_c64__s4']
    assert (gated['topfree__constant__lr0.01_c64__s4']/'complete.json').exists()
    assert all(s['arm'] == 'topfree' and s['schedule'] == 'constant' and s['lr'] == .01 for s in units.values())
