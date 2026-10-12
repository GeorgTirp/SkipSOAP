"""Readings P1-P2 (PROTOCOL.md section 5), gates and tables. Usage: summarize.py TUNE_RUN FIXED_RUN TUNED_RUN"""
import json
import statistics
import sys
from pathlib import Path

import common  # noqa: E402
from common import read  # noqa: E402
import train  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

SEEDS = train.EVAL_SEEDS


def final(unit):
    return {r['step']: r for r in read(unit/'validation.json')}[train.STEPS]


def reading(a, b):
    if any(v is None for v in a+b):
        return 'undefined (divergence)'
    return 'better' if all(x < y for x, y in zip(a, b)) else 'worse' if all(x > y for x, y in zip(a, b)) else 'mixed'


def main():
    common.guard()
    tune, fixed, tuned = map(Path, sys.argv[1:4])
    selection = read(tune/'selection.json')
    c, decay = selection['topfree_strength'], selection['ema_decay']
    name = lambda arm, schedule, s, strength=None: train.name_of(train.spec_of(arm, schedule, .01, s, strength))
    arms = {'soap|constant': [fixed/name('soap', 'constant', s) for s in SEEDS],
            'soap|wsd': [fixed/name('soap', 'wsd', s) for s in SEEDS],
            'brake|constant': [fixed/name('brake', 'constant', s) for s in SEEDS],
            'topfree|constant': [tuned/name('topfree', 'constant', s, c) for s in SEEDS],
            'topfree|wsd': [tuned/name('topfree', 'wsd', s, c) for s in SEEDS]}
    decays = dict(decay, **{'topfree|wsd': decay['topfree|constant']})
    raw = {k: [final(u)['validation_loss'] for u in us] for k, us in arms.items()}
    ema = {k: [final(u)[f'ema{decays[k]:g}']['validation_loss'] for u in us] for k, us in arms.items()}
    readings = {'P1_topfree_ema_vs_soap_ema': reading(ema['topfree|constant'], ema['soap|constant']),
                'P2_topfree_ema_vs_soap_wsd_ema': reading(ema['topfree|constant'], ema['soap|wsd'])}
    readings['verdict'] = ('in-loop damping adds beyond averaging' if readings['P1_topfree_ema_vs_soap_ema'] == 'better'
                           and readings['P2_topfree_ema_vs_soap_wsd_ema'] == 'better' else 'not confirmed')
    readings['reported'] = {'topfree_ema_vs_brake_ema': reading(ema['topfree|constant'], ema['brake|constant']),
                            'topfree_raw_vs_soap_raw': reading(raw['topfree|constant'], raw['soap|constant']),
                            'topfree_wsd_ema_vs_soap_wsd_ema': reading(ema['topfree|wsd'], ema['soap|wsd'])}
    gates = {u.name: read(u/'reproduction_gate.json')['max_difference'] for u in sorted(tune.iterdir())
             if u.is_dir() and (u/'reproduction_gate.json').exists()}
    sharp = {k: {s: {st: {r['step']: r for r in read(u/'validation.json')}[st].get('sharpness') for st in train.SHARPNESS_STEPS}
                 for s, u in zip(SEEDS, arms[k])} for k in ['soap|constant', 'brake|constant', 'topfree|constant']}
    out = {'readings': readings, 'gates': gates, 'selection': selection, 'raw_500': raw, 'ema_500': ema, 'ema_decays': decays,
           'sharpness': sharp, 'runs': {'tune': str(tune), 'fixed': str(fixed), 'tuned': str(tuned)}}
    write_json(fixed/'topfree_confirm_summary.json', json.loads(json.dumps(out, default=str)))
    print(json.dumps({k: v for k, v in out.items() if k not in ['sharpness']}, indent=1, default=str))
    for k in arms:
        print(f"{k:18s} raw {statistics.mean(raw[k]):.4f} {[round(v, 4) for v in raw[k]]} | ema(d={decays[k]:g}) "
              f"{statistics.mean(ema[k]):.4f} {[round(v, 4) for v in ema[k]]}")
    for k, series in sharp.items():
        print(k, {s: [round(v, 1) for v in d.values()] for s, d in series.items()})


if __name__ == '__main__':
    main()
