"""Summarize check 1 (with 2a/2b) from stored JSON: tables, targets, verdicts, figures.

Usage: summarize_check1.py --gate DIR --tune DIR --eval DIR --dryrun DIR
Writes check1_summary.json and check1_tables.md into the evaluation run and PNGs into figures/.
"""
import argparse
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import read  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

ORDER = ['ggn_exact_trace', 'searched_soap_tuned', 'dense_ef', 'soap_gradient_map', 'searched_soap', 'native_soap', 'adam_native']
COLORS = dict(zip(ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']))
LABELS = {'ggn_exact_trace': 'Exact tensor GGN + search', 'searched_soap_tuned': 'SOAP + search, tuned',
          'dense_ef': 'Dense empirical Fisher + search', 'soap_gradient_map': 'SOAP map of gradient + search',
          'searched_soap': 'SOAP + search, default', 'native_soap': 'Native SOAP, tuned lr', 'adam_native': 'Adam, tuned lr'}
BASELINES = ['native_soap', 'adam_native', 'searched_soap', 'searched_soap_tuned', 'soap_gradient_map']
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SEEDS = [1, 2, 3]


def load_eval(directory):
    units = read(directory/'protocol.json')['units']
    data = {}
    for name, spec in units.items():
        rows = read(directory/name/'validation.json')
        data.setdefault(spec['arm'], {})[spec['seed']] = {
            'validation': {r['step']: r for r in rows}, 'outcome': read(directory/name/'outcome.json'),
            'settings': spec['settings'], 'runtime': read(directory/name/'runtime.json')}
    return data


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.


def loss(data, arm, seed, step):
    row = data[arm][seed]['validation'].get(step)
    return None if row is None else row['validation_loss']


def first_reach(data, arm, seed, target, dry):
    validation = data[arm][seed]['validation']
    for step in sorted(validation):
        value = validation[step]['validation_loss']
        if value is not None and value <= target:
            key = arm
            settings = data[arm][seed]['settings']
            if arm == 'searched_soap_tuned' and settings.get('precondition_frequency') == 10:
                key = 'searched_soap_pf10_b9-99' if settings.get('betas') == [.9, .99] else 'searched_soap'
            per_step = dry['arms'][key]
            derived = per_step['step_seconds'][0]+(step-1)*per_step['per_step_after_init'] if step else 0.
            return {'step': step, 'seconds': validation[step]['training_seconds'], 'sequential_estimate_seconds': derived,
                    'cpu_seconds': validation[step].get('training_cpu_seconds')}
    return None


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=.6)
    ax.set_axisbelow(True)


def figures(data, steps, ratio, figure_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE)
    for ax, mode in zip(axes, ['step', 'seconds']):
        style(ax)
        for arm in ORDER:
            if arm not in data:
                continue
            means_x, means_y = [], []
            for step in steps:
                ys = [loss(data, arm, s, step) for s in SEEDS]
                xs = [data[arm][s]['validation'][step]['training_seconds'] if step in data[arm][s]['validation'] else None for s in SEEDS]
                if all(v is not None for v in ys) and (mode == 'step' or step > 0):
                    means_y.append(statistics.mean(ys))
                    means_x.append(step if mode == 'step' else statistics.mean(xs))
            for s in SEEDS:
                rows = [(k, v) for k, v in sorted(data[arm][s]['validation'].items()) if v['validation_loss'] is not None and (mode == 'step' or k > 0)]
                ax.plot([k if mode == 'step' else v['training_seconds'] for k, v in rows], [v['validation_loss'] for k, v in rows],
                        color=COLORS[arm], linewidth=.7, alpha=.35)
            ax.plot(means_x, means_y, color=COLORS[arm], linewidth=2, label=LABELS[arm])
        ax.set_yscale('log')
        if mode == 'seconds':
            ax.set_xscale('log')
        ax.set_xlabel('Consumed step' if mode == 'step' else 'Cumulative training seconds (excl. validation/I/O)', color=INK, fontsize=9)
        ax.set_ylabel('Validation loss (5,000 examples)', color=INK, fontsize=9)
    axes[0].set_title('Validation loss by step: seed mean (thick), seeds 1–3 (thin)', color=INK, fontsize=10, loc='left')
    axes[1].set_title('Validation loss by measured training time', color=INK, fontsize=10, loc='left')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, loc='lower center', ncol=4)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(figure_dir/'check1_validation.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=SURFACE)
    style(ax)
    for s in SEEDS:
        ax.plot(ratio['steps'], ratio['per_seed'][str(s)], color=COLORS['searched_soap_tuned'], linewidth=.8, alpha=.4)
    ax.plot(ratio['steps'], ratio['mean_of_ratios'], color=COLORS['searched_soap_tuned'], linewidth=2, label='Mean of per-seed ratios')
    for level, text in [(1.5, '1.5 (step-50 test for B)'), (1.25, '1.25 (A)'), (1.10, '1.10 (B)'), (1., 'parity')]:
        ax.axhline(level, color=MUTED, linewidth=.8, linestyle=(0, (3, 3)))
        ax.text(ratio['steps'][-1], level, text, color=MUTED, fontsize=7, ha='right', va='bottom')
    ax.set_xlabel('Consumed step', color=INK, fontsize=9)
    ax.set_ylabel('Loss ratio: tuned searched SOAP / exact GGN', color=INK, fontsize=9)
    ax.set_title('Validation-loss ratio by step (seeds 1–3, thin)', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(figure_dir/'check1_ratio.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['gate', 'tune', 'eval', 'dryrun']:
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    common.guard()
    data = load_eval(args.eval)
    dry = read(args.dryrun/'dryrun.json')
    horizon = read(args.eval/'protocol.json')['selection'].get('evaluation_steps', 500)
    steps = sorted(next(iter(next(iter(data.values())).values()))['validation'])
    report = [s for s in [50, 100, 200, 500] if s <= horizon]
    summary = {'horizon': horizon, 'report_steps': report, 'table': {}, 'per_seed': {}}
    for arm in ORDER:
        summary['table'][arm] = {}
        for step in report:
            losses = [loss(data, arm, s, step) for s in SEEDS]
            accuracy = [data[arm][s]['validation'][step]['validation_accuracy'] if step in data[arm][s]['validation'] else None for s in SEEDS]
            m, sd = mean_sd(losses)
            am, asd = mean_sd(accuracy)
            summary['table'][arm][step] = {'loss_mean': m, 'loss_sd': sd, 'accuracy_mean': am, 'accuracy_sd': asd}
            summary['per_seed'].setdefault(arm, {})[step] = {'loss': losses, 'accuracy': accuracy}
        summary['table'][arm]['outcomes'] = [data[arm][s]['outcome']['status'] for s in SEEDS]
        summary['table'][arm]['training_seconds'] = [data[arm][s]['runtime']['training_seconds'] for s in SEEDS]
        summary['table'][arm]['settings'] = data[arm][SEEDS[0]]['settings']
    per_seed_ratio = {str(s): [loss(data, 'searched_soap_tuned', s, k)/loss(data, 'ggn_exact_trace', s, k) for k in steps] for s in SEEDS}
    ratio = {'steps': steps, 'per_seed': per_seed_ratio,
             'mean_of_ratios': [statistics.mean(per_seed_ratio[str(s)][i] for s in SEEDS) for i in range(len(steps))],
             'ratio_of_means': [statistics.mean(loss(data, 'searched_soap_tuned', s, k) for s in SEEDS) /
                                statistics.mean(loss(data, 'ggn_exact_trace', s, k) for s in SEEDS) for k in steps]}
    summary['ratio'] = ratio
    at = lambda step: [loss(data, 'searched_soap_tuned', s, step)/loss(data, 'ggn_exact_trace', s, step) for s in SEEDS]
    final = at(horizon)
    early = at(50)
    if all(r >= 1.25 for r in final):
        verdict = 'A'
    elif sum(r <= 1.10 for r in final) >= 2 and sum(r >= 1.5 for r in early) >= 2:
        verdict = 'B'
    else:
        verdict = 'C'
    summary['verdict'] = {'category': verdict, f'ratio_step{horizon}': final, 'ratio_step50': early,
                          'note': 'Pre-declared descriptive categories; not a significance test.'}
    targets = {f'T{k}': statistics.mean(loss(data, 'ggn_exact_trace', s, k) for s in SEEDS) for k in [50, 100]}
    reach = {}
    for name, target in targets.items():
        reach[name] = {'target': target}
        for arm in ORDER:
            hits = [first_reach(data, arm, s, target, dry) for s in SEEDS]
            reach[name][arm] = {'per_seed': hits, 'reached_seeds': sum(h is not None for h in hits),
                                'mean_steps': statistics.mean(h['step'] for h in hits) if all(hits) else None,
                                'mean_seconds': statistics.mean(h['seconds'] for h in hits) if all(hits) else None,
                                'mean_sequential_estimate_seconds': statistics.mean(h['sequential_estimate_seconds'] for h in hits) if all(hits) else None,
                                'mean_cpu_seconds': statistics.mean(h['cpu_seconds'] for h in hits) if all(hits) and all(h['cpu_seconds'] is not None for h in hits) else None}
        baselines = {a: reach[name][a]['mean_seconds'] for a in BASELINES if reach[name][a]['mean_seconds'] is not None}
        sequential = {a: reach[name][a]['mean_sequential_estimate_seconds'] for a in BASELINES if reach[name][a]['mean_sequential_estimate_seconds'] is not None}
        ggn = reach[name]['ggn_exact_trace']
        reach[name]['wall_clock'] = {
            'fastest_baseline_measured': min(baselines, key=baselines.get) if baselines else None,
            'ggn_faster_measured': (ggn['mean_seconds'] is not None and (not baselines or ggn['mean_seconds'] < min(baselines.values()))),
            'fastest_baseline_sequential_estimate': min(sequential, key=sequential.get) if sequential else None,
            'ggn_faster_sequential_estimate': (ggn['mean_sequential_estimate_seconds'] is not None and
                                               (not sequential or ggn['mean_sequential_estimate_seconds'] < min(sequential.values())))}
    summary['targets'] = reach
    closure = {}
    for step in [50, 200]:
        if step > horizon:
            continue
        m = {arm: statistics.mean(loss(data, arm, s, step) for s in SEEDS) for arm in ['searched_soap_tuned', 'ggn_exact_trace', 'dense_ef']}
        gap = m['searched_soap_tuned']-m['ggn_exact_trace']
        closure[step] = {'seed_means': m, 'closure': (m['searched_soap_tuned']-m['dense_ef'])/gap if gap != 0 else None}
    worst = min(v['closure'] for v in closure.values())
    summary['dense_ef_reading'] = {'closures': closure, 'minimum': worst,
                                   'category': 'suffices' if worst >= .75 else 'partial' if worst >= .25 else 'no',
                                   'rule': 'minimum of the step-50 and step-200 closures (DEVIATIONS.md #7)'}
    summary['soap_gradient_map_vs_searched_soap'] = {
        step: {arm: summary['table'][arm][step]['loss_mean'] for arm in ['soap_gradient_map', 'searched_soap', 'searched_soap_tuned']}
        for step in report}
    tuning = read(args.tune/'tuning_results.json')
    summary['tuning'] = tuning
    pf = {}
    for name, row in tuning['results'].items():
        if row['arm'] == 'searched_soap_tuned':
            pf[name] = row['final_validation_loss']
    summary['tuning_precondition_frequency'] = pf
    summary['gate'] = read(args.gate/'gate.json')
    write_json(args.eval/'check1_summary.json', summary)
    figure_dir = common.STUDY/'figures'
    figure_dir.mkdir(exist_ok=True)
    figures(data, steps, ratio, figure_dir)
    lines = ['| Arm | Settings | '+' | '.join(f'Loss {s}' for s in report)+' | '+' | '.join(f'Acc. {s}' for s in report)+' |',
             '| --- | --- | '+' | '.join('---:' for _ in report*2)+' |']
    for arm in ORDER:
        t = summary['table'][arm]
        cells = [f"{fmt(t[s]['loss_mean'])} ± {fmt(t[s]['loss_sd'])}" for s in report]
        cells += [f"{100*t[s]['accuracy_mean']:.2f} ± {100*t[s]['accuracy_sd']:.2f}" for s in report]
        settings = ', '.join(f'{k}={v}' for k, v in t['settings'].items()) or 'default'
        lines.append(f'| {arm} | {settings} | '+' | '.join(cells)+' |')
    lines += ['', '| Arm | Seed | '+' | '.join(f'Loss {s}' for s in report)+' | '+' | '.join(f'Acc. {s}' for s in report)+' |',
              '| --- | ---: | '+' | '.join('---:' for _ in report*2)+' |']
    for arm in ORDER:
        for i, s in enumerate(SEEDS):
            row = summary['per_seed'][arm]
            lines.append(f'| {arm} | {s} | '+' | '.join(fmt(row[k]['loss'][i], 6) for k in report)+' | '
                         + ' | '.join('—' if row[k]['accuracy'][i] is None else f"{100*row[k]['accuracy'][i]:.2f}" for k in report)+' |')
    for name in ['T50', 'T100']:
        lines += ['', f"Target {name} = {reach[name]['target']:.6f}", '',
                  '| Arm | Seeds reached | Mean steps | Mean measured seconds | Mean CPU seconds | Sequential-estimate seconds |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        for arm in ORDER:
            r = reach[name][arm]
            lines.append(f"| {arm} | {r['reached_seeds']}/3 | {fmt(r['mean_steps'], 1)} | {fmt(r['mean_seconds'], 1)} | {fmt(r['mean_cpu_seconds'], 1)} | {fmt(r['mean_sequential_estimate_seconds'], 1)} |")
    (args.eval/'check1_tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print({k: summary[k] for k in ['verdict', 'dense_ef_reading']})
    print({k: v['wall_clock'] for k, v in reach.items()})


if __name__ == '__main__':
    main()
