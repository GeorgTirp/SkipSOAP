"""Readings R1-R7 (PROTOCOL.md section 6), gates, tables and figure. Usage: summarize.py TUNE_RUN FIXED_RUN TUNED_RUN"""
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import read  # noqa: E402
import train  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

SEEDS = [1, 2, 3]
LIMIT = 2*(1+.95)/(1-.95)


def curve(unit):
    return {r['step']: r for r in read(unit/'validation.json')}


def raw(unit, step):
    row = curve(unit).get(step)
    return None if row is None else row['validation_loss']


def ema(unit, step, decay):
    row = curve(unit).get(step)
    return None if row is None else row[f'ema{decay:g}']['validation_loss']


def reading(a, b):
    if any(v is None for v in a+b):
        return 'undefined (divergence)'
    return 'better' if all(x < y for x, y in zip(a, b)) else 'worse' if all(x > y for x, y in zip(a, b)) else 'mixed'


def mean_sd(values):
    values = [v for v in values if v is not None]
    return (statistics.mean(values), statistics.stdev(values)) if len(values) > 1 else (values[0] if values else None, 0.)


def matched_ratio(brake_unit, soap_unit):
    """Brake's median effective step over SOAP's interpolated at the same training loss (steps >= 100)."""
    def points(unit):
        rows = [r for s, r in curve(unit).items() if s >= 100 and r.get('effective_step')]
        return sorted((r['train_loss_last25'], math.log(r['effective_step']['median'])) for r in rows)
    soap, brake = points(soap_unit), points(brake_unit)
    lo, hi = max(soap[0][0], brake[0][0]), min(soap[-1][0], brake[-1][0])
    ratios = []
    for loss_value, log_e in brake:
        if not lo <= loss_value <= hi:
            continue
        for (l0, e0), (l1, e1) in zip(soap, soap[1:]):
            if l0 <= loss_value <= l1:
                w = 0. if l1 == l0 else (loss_value-l0)/(l1-l0)
                ratios.append(math.exp(log_e-(e0+w*(e1-e0))))
                break
    return ratios


def main():
    common.guard()
    tune, fixed, tuned = map(Path, sys.argv[1:4])
    selection = read(tune/'selection.json')
    decay = selection['ema_decay']
    name = lambda arm, schedule, lr, s, strength=None: train.name_of(train.spec_of(arm, schedule, lr, s, strength))
    soap_c = [fixed/name('soap', 'constant', .01, s) for s in SEEDS]
    brake_c = [fixed/name('brake', 'constant', .01, s) for s in SEEDS]
    sched = {key: [tuned/name(*key.split('|')[:1], key.split('|')[1], choice['lr'], s) for s in SEEDS]
             for key, choice in selection['schedules'].items()}
    topfree = [tuned/name('topfree', 'constant', .01, s, selection['topfree']['strength']) for s in SEEDS]
    R = lambda units, step: [raw(u, step) for u in units]
    E = lambda units, step: [ema(u, step, decay) for u in units]
    readings = {}
    for schedule in ['cosine', 'wsd']:
        soap_s, brake_s = sched[f'soap|{schedule}'], sched[f'brake|{schedule}']
        readings[f'R1_soap_{schedule}_vs_brake_constant_500'] = reading(R(soap_s, 500), R(brake_c, 500))
        readings[f'R2_brake_{schedule}_vs_soap_{schedule}_500'] = reading(R(brake_s, 500), R(soap_s, 500))
        readings[f'R3_brake_constant_vs_soap_{schedule}'] = {s: reading(R(brake_c, s), R(soap_s, s)) for s in [200, 300]}
    readings['R4a_soap_ema_vs_brake_raw_500'] = reading(E(soap_c, 500), R(brake_c, 500))
    readings['R4b_brake_ema_vs_soap_ema_500'] = reading(E(brake_c, 500), E(soap_c, 500))
    readings['R5_topfree_vs_brake'] = {s: reading(R(topfree, s), R(brake_c, s)) for s in [50, 100, 500]}
    ratios = {s: matched_ratio(b, a) for s, b, a in zip(SEEDS, brake_c, soap_c)}
    readings['R6_matched_loss_ratio'] = {s: {'min': min(r) if r else None, 'median': statistics.median(r) if r else None, 'points': len(r)}
                                        for s, r in ratios.items()}
    readings['R6_statistics_effect'] = 'present' if all(r and min(r) > 1.05 for r in ratios.values()) else 'not detected'
    sharp = {s: {step: curve(u)[step].get('sharpness') for step in train.SHARPNESS_STEPS} for s, u in zip(SEEDS, soap_c)}
    sharp_brake = {s: {step: curve(u)[step].get('sharpness') for step in train.SHARPNESS_STEPS} for s, u in zip(SEEDS, brake_c)}

    def signature(series):
        values = [series[s] for s in train.SHARPNESS_STEPS]
        inside = [v is not None and .75*LIMIT <= v <= 1.25*LIMIT for v in values]
        return any(a and b for a, b in zip(inside, inside[1:]))
    readings['R7_eos_signature'] = 'present' if all(signature(sharp[s]) for s in SEEDS) else 'absent'
    readings['R7_any_seed'] = {s: signature(sharp[s]) for s in SEEDS}
    gates = {}
    for s, u in [(s, u) for s, u in zip(SEEDS, soap_c+brake_c)]:
        pass
    for unit in soap_c+brake_c+[tune/name('soap', 'constant', .01, 0), tune/name('brake', 'constant', .01, 0)]:
        gates[unit.name] = read(unit/'reproduction_gate.json')['max_difference']
    for arm in ['soap', 'brake']:
        wsd = tune/name(arm, 'wsd', .01, 0)
        const = tune/name(arm, 'constant', .01, 0)
        gates[f'{arm}_wsd_vs_constant_to_400_s0'] = max(abs(raw(wsd, st)-raw(const, st)) for st in curve(wsd) if st <= 400)
    readings['gates'] = gates
    arms = {'SOAP constant': soap_c, 'Brake constant': brake_c,
            **{f"{k.split('|')[0].upper() if k.startswith('soap') else 'Brake'} + {k.split('|')[1]} (peak {v['lr']:g})".replace('SOAP', 'SOAP'): sched[k]
               for k, v in selection['schedules'].items()},
            f"Top-free brake (c = {selection['topfree']['strength']:g})": topfree}
    table = {}
    for label, units in arms.items():
        table[label] = {**{f'raw_{s}': mean_sd(R(units, s)) for s in [50, 100, 200, 300, 400, 500]},
                        **{f'ema_{s}': mean_sd(E(units, s)) for s in [100, 300, 500]},
                        'per_seed_raw_500': R(units, 500), 'per_seed_ema_500': E(units, 500),
                        'accuracy_500': mean_sd([curve(u)[500]['validation_accuracy'] for u in units])}
    out = {'readings': readings, 'selection': selection, 'table': table, 'sharpness_soap': sharp, 'sharpness_brake': sharp_brake,
           'matched_ratios': ratios, 'runs': {'tune': str(tune), 'fixed': str(fixed), 'tuned': str(tuned)}}
    write_json(fixed/'noise_controls_summary.json', json.loads(json.dumps(out, default=str)))
    print(json.dumps(readings, indent=1, default=str))
    print(json.dumps(selection, indent=1, default=str))
    for label, row in table.items():
        cells = ' | '.join(f"{k} {v[0]:.4f}±{v[1]:.4f}" for k, v in row.items() if isinstance(v, tuple) and v[0] is not None)
        print(f'{label:34s} {cells} | raw500 {[round(v, 4) for v in row["per_seed_raw_500"]]} ema500 {[round(v, 4) for v in row["per_seed_ema_500"]]}')
    for s in SEEDS:
        print(f'sharpness s{s} SOAP {[round(v, 1) for v in sharp[s].values()]} brake {[round(v, 1) for v in sharp_brake[s].values()]}')
    colors = {'SOAP constant': '#2a78d6', 'Brake constant': '#eb6834'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor='#fcfcfb')
    ax = axes[0]
    palette = ['#2a78d6', '#eb6834', '#1baf7a', '#7d4fd6', '#d6455f', '#52514e', '#b8860b']
    for (label, units), color in zip(arms.items(), palette):
        steps = [s for s in sorted(curve(units[0])) if s >= 100]
        ax.plot(steps, [mean_sd(R(units, s))[0] for s in steps], color=color, linewidth=2, label=label)
    ax.set_title('Validation loss, raw weights (mean of seeds 1–3)', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    labels = list(arms)
    for i, (label, color) in enumerate(zip(labels, palette)):
        r, e = mean_sd(R(arms[label], 500))[0], mean_sd(E(arms[label], 500))[0]
        ax.plot([i, i], [e, r], color=color, linewidth=2)
        ax.scatter([i], [r], color=color, s=40, zorder=3)
        ax.scatter([i], [e], color='#fcfcfb', edgecolor=color, s=40, linewidth=2, zorder=3)
    short = ['SOAP\nconst', 'Brake\nconst', 'SOAP\ncosine', 'SOAP\nWSD', 'Brake\ncosine', 'Brake\nWSD', 'Top-free\nc=16']
    ax.set_xticks(range(len(labels)), short[:len(labels)], fontsize=7)
    ax.set_title(f'Step 500: raw (filled) vs EMA d={decay:g} (open)', fontsize=10, loc='left')
    ax = axes[2]
    for s, style in zip(SEEDS, ['-', (0, (4, 2)), (0, (1, 2))]):
        ax.plot(train.SHARPNESS_STEPS, [sharp[s][t] for t in train.SHARPNESS_STEPS], color=colors['SOAP constant'], linestyle=style, linewidth=2,
                label=f'SOAP s{s}')
        ax.plot(train.SHARPNESS_STEPS, [sharp_brake[s][t] for t in train.SHARPNESS_STEPS], color=colors['Brake constant'], linestyle=style,
                linewidth=2, label=f'brake s{s}')
    ax.axhline(LIMIT, color='#52514e', linewidth=.8, linestyle=(0, (3, 3)))
    ax.text(300, LIMIT+2, 'EMA-momentum limit 78', fontsize=7, color='#52514e')
    ax.set_xscale('log')
    ax.set_title('lr·λ_max(P̂H), full network', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes:
        ax.set_facecolor('#fcfcfb')
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.grid(True, color='#e4e3df', linewidth=.6)
        ax.tick_params(colors='#52514e', labelsize=8)
    fig.tight_layout()
    fig.savefig(common.STUDY/'figures'/'noise_controls.png', dpi=150, facecolor='#fcfcfb')


if __name__ == '__main__':
    main()
