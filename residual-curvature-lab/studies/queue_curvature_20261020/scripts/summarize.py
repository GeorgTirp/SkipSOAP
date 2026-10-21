"""Readings V1-V3, S1-S7 and F1 (PROTOCOL.md, sections 3-5) from results/queue/*.json and spectral_rotation's summary.

Usage (from residual-curvature-lab/): summarize.py
Writes studies/queue_curvature_20261020/results/summary.json and summary.md. Capture statistics are medians over
the matrix tensors of one case (seed x state).
"""
import json
import statistics
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY/'results'
ROTATION = STUDY.parent/'spectral_rotation_20261020'/'results'/'summary.json'
SEEDS, STATES = [1, 2, 3], [50, 100, 250, 400]
SOURCES = ['cur', 'next8', 'last8', 'last8own', 'rand8', 'rand8_rad']
STEPS = SOURCES+['identity']
NEED = 10
HIGH, LOW, MIN_PREDICTED = .8, .3, 4


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def capture(case, target, source, k):
    return median(r[str(k)] for r in case['capture'][target][source].values())


def main():
    outs = {s: json.loads((RESULTS/'queue'/f'topfree_s{s}.json').read_text()) for s in SEEDS
            if (RESULTS/'queue'/f'topfree_s{s}.json').exists()}
    cases = [(s, t, outs[s]['states'][str(t)]) for s in SEEDS if s in outs for t in STATES if str(t) in outs[s]['states']]
    complete = len(cases) == len(SEEDS)*len(STATES)
    summary = {'complete': complete, 'gates': {s: o['gate'] for s, o in outs.items()}, 'cases': {}, 'readings': {}}
    for s, t, case in cases:
        v = case['visit']
        summary['cases'][f's{s}_t{t}'] = {
            'visit': {k: v[k] for k in ['D_loss', 'D_trace', 'D_loss_placebo', 'D_trace_placebo', 'next_batch', 'last_batch',
                                        'train_mean', 'validation_mean']},
            'capture': {target: {source: {k: capture(case, target, source, k) for k in [16, 64]} for source in SOURCES}
                        for target in ['queue', 'held']},
            'steps': case['steps'], 'delta': case['delta'], 'lr_factor': case['lr_factor']}
    c = summary['cases']
    if complete:
        d_loss = [x['visit']['D_loss'] for x in c.values()]
        d_trace = [x['visit']['D_trace'] for x in c.values()]
        p_loss = [abs(x['visit']['D_loss_placebo']) for x in c.values()]
        p_trace = [abs(x['visit']['D_trace_placebo']) for x in c.values()]
        r = summary['readings']
        r['V1'] = all(d > 0 for d in d_loss) and min(d_loss) > max(p_loss)
        r['V2'] = all(d > 0 for d in d_trace) and min(d_trace) > max(p_trace)
        count = lambda f: sum(bool(f(x)) for x in c.values())
        r['S1_count'] = count(lambda x: x['capture']['queue']['next8'][64] > x['capture']['queue']['rand8'][64])
        r['S2_count'] = count(lambda x: x['capture']['held']['rand8'][64] > x['capture']['held']['last8'][64])
        r['S3_count'] = count(lambda x: x['capture']['queue']['last8'][64] > x['capture']['queue']['last8own'][64])
        r['S4_count'] = count(lambda x: x['capture']['held']['rand8'][64] > x['capture']['held']['cur'][64])
        r['S5_count'] = count(lambda x: x['steps']['next8']['decrease']['queue'] > x['steps']['rand8']['decrease']['queue'])
        r['S6_count'] = count(lambda x: x['steps']['next8']['decrease']['held'] > x['steps']['rand8']['decrease']['held'])
        for code in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']:
            r[code] = r[f'{code}_count'] >= NEED
        if ROTATION.exists():
            rotation = json.loads(ROTATION.read_text())
            predictions = []
            for s, t, _ in cases:
                tr = rotation['trajectories'].get(f'topfree_s{s}')
                if tr is None or f'{t}->{t+10}' not in tr['pairs']:
                    continue
                floor_loss = 1-tr['floor'][str(t)]['capture']['16']
                drift_loss = 1-tr['pairs'][f'{t}->{t+10}']['capture']['16']
                ratio = floor_loss/drift_loss if drift_loss > 0 else None
                observed = c[f's{s}_t{t}']['capture']['held']['last8own'][16]-c[f's{s}_t{t}']['capture']['held']['cur'][16]
                predicted = None if ratio is None else ('trailing' if ratio >= HIGH else 'single' if ratio <= LOW else None)
                agree = None if predicted is None else (observed >= 0 if predicted == 'trailing' else observed <= 0)
                predictions.append({'case': f's{s}_t{t}', 'R': ratio, 'observed_last8own_minus_cur': observed,
                                    'predicted': predicted, 'agree': agree})
            made = [p for p in predictions if p['predicted'] is not None]
            r['F1'] = ('undetermined' if len(made) < MIN_PREDICTED else
                       'holds' if all(p['agree'] for p in made) else 'not supported')
            r['F1_cases'] = predictions
        else:
            r['F1'] = 'spectral_rotation summary missing'
    (RESULTS/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    f = lambda v, d=3: '' if v is None else f'{v:.{d}f}'
    lines = ['# queue_curvature_20261020 summary (generated by scripts/summarize.py)', '', f'Complete: {complete}', '']
    for code, value in summary['readings'].items():
        if code != 'F1_cases':
            lines.append(f'- {code}: {value}')
    lines += ['', '| case | D_loss | placebo | D_trace | placebo | next-batch loss | last-batch loss | train loss | validation loss |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, x in c.items():
        v = x['visit']
        lines.append(f"| {key} | {f(v['D_loss'], 4)} | {f(v['D_loss_placebo'], 4)} | {f(v['D_trace'], 4)} | {f(v['D_trace_placebo'], 4)} | "
                     f"{f(v['next_batch']['loss'], 4)} | {f(v['last_batch']['loss'], 4)} | {f(v['train_mean']['loss'], 4)} | "
                     f"{f(v['validation_mean']['loss'], 4)} |")
    for target in ['queue', 'held']:
        lines += ['', f'Capture fraction on T_{target} (k = 64; k = 16 in brackets)', '',
                  '| case | ' + ' | '.join(SOURCES) + ' |', '| --- | ' + ' | '.join('---:' for _ in SOURCES) + ' |']
        for key, x in c.items():
            lines.append(f'| {key} | ' + ' | '.join(f"{f(x['capture'][target][s][64])} ({f(x['capture'][target][s][16])})"
                                                    for s in SOURCES) + ' |')
    lines += ['', 'Step decreases at the selected multiplier: queue set / held-out', '',
              '| case | ' + ' | '.join(STEPS) + ' |', '| --- | ' + ' | '.join('---:' for _ in STEPS) + ' |']
    for key, x in c.items():
        lines.append(f'| {key} | ' + ' | '.join(f"{f(x['steps'][s]['decrease']['queue'], 4)} / {f(x['steps'][s]['decrease']['held'], 4)}"
                                                + ('' if x['steps'][s]['interior'] else ' (edge)') for s in STEPS) + ' |')
    (RESULTS/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:16]))


if __name__ == '__main__':
    sys.exit(main())
