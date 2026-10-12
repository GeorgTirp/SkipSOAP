"""Summary of the geometry follow-up from stored JSON (PROTOCOL.md, pre-declared readings).

Usage: summarize_followup.py --eval DIR [DIR ...] --tune DIR --dryrun DIR
Writes followup_summary.json and followup_tables.md into the first evaluation run, PNGs into figures/.
"""
import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import read  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

SEEDS = [1, 2, 3]
NEW = ['parsec_default', 'parsec_split', 'ot_tensor', 'ggn_full', 'ot_full']
BASE = ['ggn_exact_trace', 'searched_soap_tuned', 'native_soap', 'adam_native']
ORDER = ['ggn_exact_trace', 'ggn_full', 'ot_tensor', 'ot_full', 'searched_soap_tuned', 'native_soap', 'adam_native',
         'parsec_default', 'parsec_split']
COLORS = dict(zip(ORDER, ['#2a78d6', '#2a78d6', '#eb6834', '#eb6834', '#1baf7a', '#008300', '#4a3aa7', '#e87ba4', '#eda100']))
STYLES = {'ggn_full': (0, (4, 2)), 'ot_full': (0, (4, 2))}
COMPARISONS = [('parsec_default', 'adam_native'), ('parsec_split', 'adam_native'),
               ('parsec_default', 'searched_soap_tuned'), ('parsec_split', 'searched_soap_tuned'),
               ('parsec_default', 'ggn_exact_trace'), ('parsec_split', 'ggn_exact_trace'),
               ('ot_tensor', 'ggn_exact_trace'), ('ggn_full', 'ggn_exact_trace'), ('ot_full', 'ot_tensor')]
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
GAP_DRY = common.GAP_AUDIT_RUNS/'20261007T121137-28c507f9'/'dryrun.json'


def load(directories, arms):
    data = {}
    for directory in directories:
        for name, spec in read(directory/'protocol.json')['units'].items():
            if spec['arm'] not in arms:
                continue
            data.setdefault(spec['arm'], {})[spec['seed']] = {
                'unit': name, 'run': directory.name, 'settings': spec['settings'],
                'validation': {r['step']: r for r in read(directory/name/'validation.json')},
                'training': read(directory/name/'training.json'), 'runtime': read(directory/name/'runtime.json'),
                'outcome': read(directory/name/'outcome.json')}
    return data


def loss(data, arm, seed, step):
    row = data[arm][seed]['validation'].get(step)
    return None if row is None else row['validation_loss']


def verdict(data, a, b, step):
    pairs = [(loss(data, a, s, step), loss(data, b, s, step)) for s in SEEDS]
    if any(x is None or y is None for x, y in pairs):
        return 'n/a'
    if all(x < y for x, y in pairs):
        return 'better'
    if all(x > y for x, y in pairs):
        return 'worse'
    return 'mixed'


def per_step_cost(arm, settings, dry, gap_dry):
    if arm in dry['arms'] and not arm.startswith('parsec'):
        return dry['arms'][arm]['step_seconds'][0], dry['arms'][arm]['per_step_after_init']
    if arm.startswith('parsec'):
        base = dry['arms'][arm]['per_step_after_init']
        refresh = dry['lanczos'][str(settings['rank'])]
        return base, base+refresh/settings['refresh_every']
    key = {'searched_soap_tuned': 'searched_soap_pf10_b9-99'}.get(arm, arm)
    return gap_dry['arms'][key]['step_seconds'][0], gap_dry['arms'][key]['per_step_after_init']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', type=Path, nargs='+', required=True)
    parser.add_argument('--tune', type=Path, required=True)
    parser.add_argument('--dryrun', type=Path, required=True)
    args = parser.parse_args()
    common.guard()
    data = {**load(args.eval, NEW), **load([common.GAP_EVAL], BASE)}
    dry, gap_dry = read(args.dryrun/'dryrun.json'), read(GAP_DRY)
    steps = sorted(next(iter(data['ggn_exact_trace'].values()))['validation'])
    report = [50, 100, 200, 500]
    out = {'table': {}, 'per_seed': {}, 'readings': {}, 'targets': {}, 'diagnostics': {}}
    for arm in ORDER:
        if arm not in data:
            continue
        out['table'][arm] = {'settings': data[arm][SEEDS[0]]['settings'], 'run': data[arm][SEEDS[0]]['run'],
                             'outcomes': [data[arm][s]['outcome']['status'] for s in SEEDS],
                             'seconds_per_step_measured': statistics.mean(data[arm][s]['runtime']['training_seconds']/len(data[arm][s]['training']) for s in SEEDS)}
        for step in report:
            losses = [loss(data, arm, s, step) for s in SEEDS]
            acc = [data[arm][s]['validation'][step]['validation_accuracy'] for s in SEEDS]
            out['table'][arm][step] = {'loss_mean': statistics.mean(losses), 'loss_sd': statistics.stdev(losses),
                                       'acc_mean': statistics.mean(acc), 'acc_sd': statistics.stdev(acc)}
            out['per_seed'].setdefault(arm, {})[step] = losses
    for a, b in COMPARISONS:
        out['readings'][f'{a} vs {b}'] = {step: verdict(data, a, b, step) for step in [50, 200, 500]}
    gap_targets = read(common.GAP_EVAL/'check1_summary.json')['targets']
    targets = {name: gap_targets[name]['target'] for name in ['T50', 'T100']}
    for name, target in targets.items():
        out['targets'][name] = {'target': target}
        for arm in ORDER:
            hits = []
            for s in SEEDS:
                d = data[arm][s]
                reached = next((k for k in sorted(d['validation']) if d['validation'][k]['validation_loss'] is not None
                                and d['validation'][k]['validation_loss'] <= target), None)
                if reached is None:
                    hits.append(None)
                    continue
                first, per = per_step_cost(arm, d['settings'], dry, gap_dry)
                hits.append({'step': reached, 'seconds': d['validation'][reached]['training_seconds'],
                             'cpu_seconds': d['validation'][reached].get('training_cpu_seconds'),
                             'sequential_estimate_seconds': first+(reached-1)*per if reached else 0.})
            ok = all(hits)
            out['targets'][name][arm] = {'reached': sum(h is not None for h in hits),
                                         **({k: statistics.mean(h[k] for h in hits) for k in ['step', 'seconds', 'sequential_estimate_seconds']} if ok else {})}
    for arm in NEW:
        rows = [r for s in SEEDS for r in data[arm][s]['training']]
        diag = {'max_solve_residual': max((r['solve_residual'] for r in rows if r.get('solve_residual') is not None), default=None),
                'max_audit_check': max((r['max_audit_check'] for r in rows if r.get('max_audit_check') is not None), default=None),
                'searches': sum(1 for r in rows if r.get('search')), 'rejected': sum(1 for r in rows if r.get('search') and r['search']['rejected'])}
        if arm.startswith('parsec'):
            ritz = [r['max_ritz_residual'] for r in rows if r.get('max_ritz_residual') is not None]
            diag.update(refreshes=len(ritz), max_ritz_residual=max(ritz, default=None),
                        median_ritz_residual=statistics.median(ritz) if ritz else None,
                        top_eigenvalue_range=[min(r['ritz_eigenvalues'][0] for r in rows if r.get('ritz_eigenvalues')),
                                              max(r['ritz_eigenvalues'][0] for r in rows if r.get('ritz_eigenvalues'))] if ritz else None)
            if arm == 'parsec_split':
                active = [r for r in rows if r.get('active_rank')]
                diag.update(mean_top_to_perp_norm=statistics.mean(r['top_step_norm']/max(r['perp_step_norm'], 1e-300) for r in active),
                            max_perp_top_space_fraction=max(r['perp_step_top_space_fraction'] for r in active))
        if arm.endswith('_full'):
            energy = [r['off_diagonal_energy_fraction'] for r in rows if r.get('off_diagonal_energy_fraction') is not None]
            diag.update(off_diagonal_energy_fraction=energy)
        out['diagnostics'][arm] = diag
    out['tuning'] = {'selection': read(args.tune/'selection.json'), 'results': read(args.tune/'tuning_results.json')}
    write_json(args.eval[0]/'followup_summary.json', out)
    lines = ['| Arm | Settings | '+' | '.join(f'Loss {s}' for s in report)+' | Acc. 500 | s/step |', '| --- | --- | '+' | '.join('---:' for _ in report)+' | ---: | ---: |']
    for arm in ORDER:
        t = out['table'][arm]
        settings = ', '.join(f'{k}={v:g}' if isinstance(v, (int, float)) else f'{k}={v}' for k, v in t['settings'].items())
        lines.append(f'| {arm} | {settings} | '+' | '.join(f"{t[s]['loss_mean']:.4f} ± {t[s]['loss_sd']:.4f}" for s in report)
                     + f" | {100*t[500]['acc_mean']:.2f} ± {100*t[500]['acc_sd']:.2f}% | {t['seconds_per_step_measured']:.3f} |")
    lines += ['', '| Comparison | Step 50 | Step 200 | Step 500 |', '| --- | --- | --- | --- |']
    for key, v in out['readings'].items():
        lines.append(f'| {key} | {v[50]} | {v[200]} | {v[500]} |')
    for name in ['T50', 'T100']:
        lines += ['', f"{name} = {out['targets'][name]['target']:.6f}", '', '| Arm | Reached | Mean steps | Measured s | Sequential-estimate s |', '| --- | ---: | ---: | ---: | ---: |']
        for arm in ORDER:
            r = out['targets'][name][arm]
            lines.append(f"| {arm} | {r['reached']}/3 | {r.get('step', float('nan')):.1f} | {r.get('seconds', float('nan')):.1f} | {r.get('sequential_estimate_seconds', float('nan')):.1f} |")
    lines += ['', '| Arm | Seed | '+' | '.join(f'Loss {s}' for s in report)+' |', '| --- | ---: | '+' | '.join('---:' for _ in report)+' |']
    for arm in ORDER:
        for i, s in enumerate(SEEDS):
            lines.append(f'| {arm} | {s} | '+' | '.join(f"{out['per_seed'][arm][k][i]:.6f}" for k in report)+' |')
    (args.eval[0]/'followup_tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:30]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE)
    for ax, mode in zip(axes, ['step', 'seconds']):
        ax.set_facecolor(SURFACE)
        for arm in ORDER:
            xs, ys = [], []
            for step in steps:
                vals = [loss(data, arm, s, step) for s in SEEDS]
                if mode == 'seconds' and step == 0:
                    continue
                xs.append(step if mode == 'step' else statistics.mean(data[arm][s]['validation'][step]['training_seconds'] for s in SEEDS))
                ys.append(statistics.mean(vals))
            ax.plot(xs, ys, color=COLORS[arm], linestyle=STYLES.get(arm, '-'), linewidth=2, label=arm)
        ax.set_yscale('log')
        if mode == 'seconds':
            ax.set_xscale('log')
        ax.set_xlabel('Consumed step' if mode == 'step' else 'Cumulative training seconds (measured)', color=INK, fontsize=9)
        ax.set_ylabel('Validation loss, mean of seeds 1–3', color=INK, fontsize=9)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_title('Validation loss by step (dashed: full, non-block)', color=INK, fontsize=10, loc='left')
    axes[1].set_title('Validation loss by measured training time', color=INK, fontsize=10, loc='left')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, loc='lower center', ncol=5)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(common.STUDY/'figures'/'followup_validation.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main()
