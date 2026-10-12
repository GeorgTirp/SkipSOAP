"""Part 1 readings S1-S2 (PROTOCOL.md), the c = 64 gate and tables. Usage: summarize.py SWEEP_RUN"""
import json
import statistics
import sys
from pathlib import Path

import common  # noqa: E402
from common import read  # noqa: E402
import train  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

SEEDS, DECAY, GRID = train.EVAL_SEEDS, .98, [64., *train.SWEEP_STRENGTHS]
CONFIRM_FIXED = train.CONFIRM_TUNED.parent/'20261011T135638-221c0b6a'
REPORT_STEPS = [100, 200, 300, 400, 500]


def rows(unit):
    return {r['step']: r for r in read(unit/'validation.json')}


def loss(unit, step=train.STEPS, decay=DECAY):
    r = rows(unit)[step]
    return r['validation_loss'] if decay is None else r[f'ema{decay:g}']['validation_loss']


def reading(a, b):
    if any(v is None for v in a+b):
        return 'undefined (divergence)'
    return 'better' if all(x < y for x, y in zip(a, b)) else 'worse' if all(x > y for x, y in zip(a, b)) else 'mixed'


def main():
    common.guard()
    run = Path(sys.argv[1])
    name = lambda c, s: train.name_of(train.spec_of('topfree', 'constant', .01, s, strength=c))
    # c = 64 comes from topfree_confirm (seeds 4-6); its seed-4 rerun here is the bitwise gate.
    units = {c: [(train.CONFIRM_TUNED if c == 64. else run)/name(c, s) for s in SEEDS] for c in GRID}
    gate = read(run/name(64., 4)/'reproduction_gate.json')
    gate_passed = gate['max_difference'] is not None and gate['max_difference'] <= 1e-12
    ema = {c: [loss(u) for u in us] for c, us in units.items()}
    s1 = {f'c{c:g}_vs_c64': reading(ema[c], ema[64.]) for c in train.SWEEP_STRENGTHS}
    frozen, middle = ema[1e9], [ema[c] for c in train.SWEEP_STRENGTHS[:-1]]
    interior = [c for c in train.SWEEP_STRENGTHS[:-1] if reading(ema[c], ema[64.]) == 'better' and reading(ema[c], frozen) == 'better']
    monotone = reading(frozen, ema[64.]) == 'better' and all(reading(frozen, m) == 'better' for m in middle)
    s2 = 'interior optimum' if interior else 'monotone to freezing' if monotone else 'flat or undetermined'
    reference = {arm: [loss(CONFIRM_FIXED/f'{arm}__constant__lr0.01__s{s}') for s in SEEDS] for arm in ['soap', 'brake']}
    reported = {f'c{c:g}_vs_soap_ema': reading(ema[c], reference['soap']) for c in GRID}
    out = {'readings': {'S1': s1, 'S2': s2, 'interior_candidates': interior},
           'gate': {'unit': name(64., 4), 'max_difference': gate['max_difference'], 'passed': gate_passed},
           'ema098_500': {f'{c:g}': v for c, v in ema.items()},
           'ema098_500_mean': {f'{c:g}': statistics.mean(v) for c, v in ema.items()},
           'raw_500': {f'{c:g}': [loss(u, decay=None) for u in us] for c, us in units.items()},
           'ema_other_500': {f'{c:g}': {f'{d:g}': [loss(u, decay=d) for u in us] for d in [.9, .995]} for c, us in units.items()},
           'ema098_over_time_mean': {f'{c:g}': {t: statistics.mean(loss(u, t) for u in us) for t in REPORT_STEPS} for c, us in units.items()},
           'sharpness': {f'{c:g}': {s: [rows(u)[t].get('sharpness') for t in train.SHARPNESS_STEPS] for s, u in zip(SEEDS, us)}
                         for c, us in units.items()},
           'reference_ema098_500': reference, 'reported': reported,
           'units': {f'{c:g}': [str(u) for u in us] for c, us in units.items()}}
    write_json(run/'damping_sweep_summary.json', json.loads(json.dumps(out, default=str)))
    (common.STUDY/'results').mkdir(exist_ok=True)
    (common.STUDY/'results'/'sweep_summary.json').write_text(json.dumps(out, indent=1, default=str)+'\n')
    print(json.dumps({k: out[k] for k in ['readings', 'gate', 'reported']}, indent=1))
    for c in GRID:
        k = f'{c:g}'
        print(f"c={k:>6s} ema0.98 {out['ema098_500_mean'][k]:.4f} {[round(v, 4) for v in ema[c]]} | raw "
              f"{statistics.mean(out['raw_500'][k]):.4f} | d=0.9 {statistics.mean(out['ema_other_500'][k]['0.9']):.4f} "
              f"d=0.995 {statistics.mean(out['ema_other_500'][k]['0.995']):.4f} | over time "
              f"{[round(v, 4) for v in out['ema098_over_time_mean'][k].values()]}")
        print('   sharpness', {s: [None if v is None else round(v, 1) for v in vs] for s, vs in out['sharpness'][k].items()})
    for arm, v in reference.items():
        print(f'{arm} + EMA {statistics.mean(v):.4f} {[round(x, 4) for x in v]}')


if __name__ == '__main__':
    main()
