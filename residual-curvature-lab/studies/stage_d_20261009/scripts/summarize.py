"""Stage D summary: pre-declared readings (PROTOCOL.md section 7), comparison with all earlier optimizers on the
same seeds and data order, seed-0 grid, diagnostics and figure. Usage: summarize.py TUNE_RUN FIXED_RUN TUNED_RUN"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import read  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

SEEDS, STEPS = [1, 2, 3], [50, 100, 200, 500]
TARGETS = {'T50': 0.355037, 'T100': 0.265401}
GEOMETRY = Path.home()/'skipsoap-study-data'/'geometry_followup_20261008'/'runs'
EARLIER = {  # label: (run directory, unit pattern)
    'Native SOAP, lr 1e-2 (gap audit)': (common.GAP_EVAL, 'native_soap__lr0.01__s{s}'),
    'Adam, lr 1e-2 (gap audit)': (common.GAP_EVAL, 'adam_native__lr0.01__s{s}'),
    'Searched SOAP, tuned (gap audit)': (common.GAP_EVAL, 'searched_soap_tuned__b0.9-0.99_pf10__s{s}'),
    'Exact tensor GGN, same-batch search (gap audit)': (common.GAP_EVAL, 'ggn_exact_trace__gamma0.001__s{s}'),
    'Dense empirical Fisher (gap audit)': (common.GAP_EVAL, 'dense_ef__gamma0.001__s{s}'),
    'Original PARSEC, default (geometry)': (GEOMETRY/'20261008T103332-f9ee97b7', 'parsec_default__*__s{s}'),
    'Original PARSEC, split (geometry)': (GEOMETRY/'20261008T103332-f9ee97b7', 'parsec_split__*__s{s}'),
    'Tensor OT, gamma 0.1 (geometry)': (GEOMETRY/'20261008T103332-f9ee97b7', 'ot_tensor__*__s{s}'),
    'Full GGN (geometry)': (GEOMETRY/'20261008T092440-b7620df1', 'ggn_full__*__s{s}'),
}


def unit_dir(directory, pattern, seed):
    matches = sorted(Path(directory).glob(pattern.format(s=seed)))
    if len(matches) != 1:
        raise RuntimeError(f'{directory} {pattern} seed {seed}: {matches}')
    return matches[0]


def curve(unit):
    return {r['step']: r for r in read(unit/'validation.json')}


def loss(unit, step):
    row = curve(unit).get(step)
    return None if row is None else row['validation_loss']


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.)


def reading(a, b):
    if any(v is None for v in a+b):
        return 'undefined (divergence)'
    return 'better' if all(x < y for x, y in zip(a, b)) else 'worse' if all(x > y for x, y in zip(a, b)) else 'mixed'


def target_hit(unit, target):
    for step, row in sorted(curve(unit).items()):
        if step > 0 and row['validation_loss'] is not None and row['validation_loss'] <= target:
            return step, row.get('training_seconds')
    return None


def transfer(unit, lo, hi):
    rows = [r for r in read(unit/'training.json') if lo <= r['step'] <= hi and r.get('own_before') is not None and r.get('own_after') is not None]
    own = sum(r['own_before']-r['own_after'] for r in rows)
    nxt = sum(r['next_before']-r['next_after'] for r in rows)
    return nxt/own if own > 0 else None


def main():
    common.guard()
    tune, fixed, tuned = map(Path, sys.argv[1:4])
    selection = read(tune/'selection.json')
    b = selection['brake']
    arms = {
        'SOAP, lr 1e-2 (this study)': [fixed/f'soap__lr0.01__s{s}' for s in SEEDS],
        'SOAP, lr 2e-2': [fixed/f'soap__lr0.02__s{s}' for s in SEEDS],
        'SOAP, lr 3e-2': [fixed/f'soap__lr0.03__s{s}' for s in SEEDS],
        f"PARSEC-L brake ({b['calibration']}-calibrated, c = {b['strength']:g}, lr {b['lr']:g})":
            [tuned/f"brake__{b['calibration']}{b['strength']:g}_lr{b['lr']:g}__s{s}" for s in SEEDS],
        f"PARSEC-L Early-Newton ({selection['early_newton_rhs']}, no hand-over)":
            [tuned/f"early_newton__{selection['early_newton_rhs']}_handover0__s{s}" for s in SEEDS],
        f"PARSEC-L Early-Newton ({selection['early_newton_rhs']}, hand-over)":
            [tuned/f"early_newton__{selection['early_newton_rhs']}_handover1__s{s}" for s in SEEDS],
        'Exact tensor GGN, selected step': [fixed/f'ggn_selected__gamma0.001__s{s}' for s in SEEDS],
    }
    for label, (directory, pattern) in EARLIER.items():
        arms[label] = [unit_dir(directory, pattern, s) for s in SEEDS]
    keys = list(arms)
    # Readings use the gap audit's native SOAP (lr 1e-2, same seeds and data order): this study's lr 1e-2 seed-1
    # unit failed the pre-declared 1e-6 reproduction gate at step 300 (DEVIATIONS.md #2).
    own_soap, brake_key, en_off, en_on, ggn_sel = keys[0], keys[3], keys[4], keys[5], keys[6]
    soap = 'Native SOAP, lr 1e-2 (gap audit)'
    ggn_same = 'Exact tensor GGN, same-batch search (gap audit)'
    L = lambda key, step: [loss(u, step) for u in arms[key]]
    readings = {
        'R1_brake_vs_soap_step500': reading(L(brake_key, 500), L(soap, 500)),
        'R2_early_newton_no_handover_vs_soap': {s: reading(L(en_off, s), L(soap, s)) for s in [50, 100]},
        'R2_early_newton_handover_vs_soap': {s: reading(L(en_on, s), L(soap, s)) for s in [50, 100]},
        'R3_ggn_selected_vs_same_batch': {s: reading(L(ggn_sel, s), L(ggn_same, s)) for s in [200, 500]},
        'gate_soap_reproduction_per_seed': {s: max(abs(r['validation_loss']-curve(g)[r['step']]['validation_loss'])
                                                  for r in read(u/'validation.json') if r['validation_loss'] is not None)
                                            for s, u, g in zip(SEEDS, arms[own_soap], arms[soap])},
        'own_soap_completed_seeds': [s for s, u in zip(SEEDS, arms[own_soap]) if (u/'complete.json').exists()],
    }
    table = {}
    for key, units in arms.items():
        row = {f'loss_{s}': mean_sd(L(key, s)) for s in STEPS}
        row['accuracy_500'] = mean_sd([curve(u).get(500, {}).get('validation_accuracy') for u in units])
        seconds, steps_run, examples = [], [], []
        for u in units:
            rows = read(u/'training.json')
            last = curve(u)[max(curve(u))]  # partial for a failed unit
            seconds.append(last.get('training_seconds'))
            steps_run.append(len(rows))
            examples.append((rows[-1].get('examples_gradient', 512*len(rows)), rows[-1].get('examples_selection', 0)))
        row['seconds_per_step'] = statistics.mean(s/n for s, n in zip(seconds, steps_run) if s and n)
        row['examples_gradient_selection'] = (statistics.mean(e[0] for e in examples), statistics.mean(e[1] for e in examples))
        for name, target in TARGETS.items():
            hits = [target_hit(u, target) for u in units]
            row[name] = {'reached': sum(h is not None for h in hits),
                         'steps': statistics.mean(h[0] for h in hits) if all(hits) else None,
                         'seconds': statistics.mean(h[1] for h in hits) if all(hits) and all(h[1] is not None for h in hits) else None}
        if key in keys[:7]:
            row['transfer'] = {w: mean_sd([transfer(u, *map(int, w.split('-'))) for u in units]) for w in ['1-50', '51-200', '201-500']}
        table[key] = row
    grid = {}
    for name, spec in read(tune/'protocol.json')['units'].items():
        grid[name] = {'spec': spec, 'loss_100': loss(tune/name, 100), 'loss_500': loss(tune/name, 500)}
    diagnostics = {}
    for key in keys[:7]:
        d = {'eta_lambda_max': [], 'r_soap': [], 'r_own': [], 'alpha': [], 'handover_steps': [], 'winner_parsec_share': []}
        for u in arms[key]:
            rows = read(u/'training.json')
            d['eta_lambda_max'] += [(r['step'], r['probe']['eta_lambda_max']) for r in rows if r.get('probe')]
            d['r_soap'] += [(r['step'], r['optimism']['r_soap']) for r in rows if r.get('optimism')]
            own = 'r_ggn' if key == ggn_sel else 'r_parsec_l'
            d['r_own'] += [(r['step'], r['optimism'].get(own)) for r in rows if r.get('optimism')]
            sel = [r['select'] for r in rows if r.get('select')]
            d['alpha'] += [s['alpha'] for s in sel]
            if sel:
                d['winner_parsec_share'].append(statistics.mean(s['winner'] == 'parsec' for s in sel))
            hand = [r['step'] for r in rows if r.get('handover')]
            d['handover_steps'].append(hand[0] if hand else None)
        diagnostics[key] = d
    out = {'readings': readings, 'selection': selection, 'table': table, 'seed0_grid': grid, 'diagnostics': diagnostics,
           'runs': {'tune': str(tune), 'eval_fixed': str(fixed), 'eval_tuned': str(tuned)}}
    write_json(fixed/'stage_d_summary.json', json.loads(json.dumps(out, default=str)))
    print(json.dumps(readings, indent=1))
    print(json.dumps(selection, indent=1, default=str))
    for key, row in table.items():
        cells = ' | '.join(f"{row[f'loss_{s}'][0]:.4f}±{row[f'loss_{s}'][1]:.4f}" if row[f'loss_{s}'][0] is not None else 'div'
                           for s in STEPS)
        print(f"{key:58s} | {cells} | s/step {row['seconds_per_step']:.3f} | T50 {row['T50']} | T100 {row['T100']} | "
              f"examples {row['examples_gradient_selection']} | transfer {row.get('transfer')}")
    for name, g in sorted(grid.items()):
        print(f"seed0 {name:45s} loss100 {g['loss_100']} loss500 {g['loss_500']}")
    for key in keys[:7]:
        d = diagnostics[key]
        def window(series, lo, hi):
            values = [v for s, v in series if lo <= s <= hi and v is not None]
            return round(statistics.median(values), 3) if values else None
        print(f"{key:58s} etaλmax med 1-50/51-200/201-500 {[window(d['eta_lambda_max'], *w) for w in [(1, 50), (51, 200), (201, 500)]]} "
              f"r_soap {[window(d['r_soap'], *w) for w in [(1, 50), (51, 200), (201, 500)]]} r_own {[window(d['r_own'], *w) for w in [(1, 50), (51, 200), (201, 500)]]} "
              f"alpha med {statistics.median(d['alpha']) if d['alpha'] else None} handover {d['handover_steps']} parsec-wins {d['winner_parsec_share']}")
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#7d4fd6', '#d6455f', '#52514e']
    # soap = the gap audit's native SOAP lr 1e-2
    show = [(soap, colors[0], '-'), (brake_key, colors[1], '-'), (en_off, colors[2], '-'), (en_on, colors[2], (0, (4, 2))),
            (ggn_sel, colors[3], '-'), (ggn_same, colors[3], (0, (4, 2))), ('Adam, lr 1e-2 (gap audit)', colors[5], (0, (1, 2)))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor='#fcfcfb')
    for ax, (lo, hi, log) in zip(axes, [(10, 500, True), (100, 500, False)]):
        for key, color, style in show:
            steps = sorted({s for u in arms[key] for s in curve(u) if lo <= s <= hi})
            means = [mean_sd([loss(u, s) for u in arms[key]])[0] for s in steps]
            ax.plot(steps, means, color=color, linestyle=style, linewidth=2, label=key)
        if log:
            ax.set_yscale('log')
        ax.set_xlabel('Consumed step', fontsize=9)
        ax.set_ylabel('Validation loss (mean of seeds 1–3)', fontsize=9)
        ax.set_facecolor('#fcfcfb')
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.grid(True, color='#e4e3df', linewidth=.6)
        ax.tick_params(colors='#52514e', labelsize=8)
    axes[0].set_title('Validation loss, steps 10–500 (log scale)', fontsize=10, loc='left')
    axes[1].set_title('Steps 100–500 (linear)', fontsize=10, loc='left')
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(common.STUDY/'figures'/'stage_d.png', dpi=150, facecolor='#fcfcfb')


if __name__ == '__main__':
    main()
