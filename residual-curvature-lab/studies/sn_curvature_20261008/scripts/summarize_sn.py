"""SN curvature study summary (PROTOCOL.md pre-declared readings). Usage: summarize_sn.py --run DIR"""
import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import read  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

STEPS, SEEDS = [10, 50, 200], [0, 1]
CONDITIONS = ['nosn', 'sn0.95', 'sn2', 'sn6']
COLORS = {'nosn': '#52514e', 'sn0.95': '#2a78d6', 'sn2': '#eb6834', 'sn6': '#1baf7a'}
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
METRICS = {
    'distortion_global_res': lambda d: d['distance']['global_phi_res']['distortion_q95_q5'],
    'distortion_global_head': lambda d: d['distance']['global_phi_head']['distortion_q95_q5'],
    'distortion_local_res': lambda d: d['distance']['local_phi_res']['distortion_q95_q5'],
    'distortion_local_head': lambda d: d['distance']['local_phi_head']['distortion_q95_q5'],
    'raw_top_eigenvalue': lambda d: d['raw']['spectrum']['top'],
    'raw_effective_rank': lambda d: d['raw']['spectrum']['effective_rank_entropy'],
    'raw_cross_tensor_energy': lambda d: d['raw']['cross_tensor_energy'],
    'raw_nearest_kronecker_error': lambda d: d['raw']['nearest_kronecker_error'],
    'raw_diagonal_error': lambda d: d['raw']['diagonal_error'],
    'raw_empirical_fisher_error': lambda d: d['raw']['estimation_frobenius']['empirical_fisher'],
    'raw_mc1_fisher_error': lambda d: d['raw']['estimation_frobenius']['mc_fisher_1'],
    'raw_hutchinson8_error': lambda d: d['raw']['hutchinson8_relative_rms'],
    'raw_rank16_truncation_error': lambda d: d['raw']['spectrum']['rank16_truncation_error'],
    'energy_ray_diagonal': lambda d: d['estimation']['energy_errors']['diagonal']['gradient']['pooled_ray'],
    'energy_ray_kron_basis_diag': lambda d: d['estimation']['energy_errors']['kron_basis_diag']['gradient']['pooled_ray'],
    'energy_ray_empirical_fisher': lambda d: d['estimation']['energy_errors']['empirical_fisher']['gradient']['pooled_ray'],
    'exact_ggn_headroom': lambda d: d['estimation']['headroom'],
    'kron_basis_diag_recovery': lambda d: d['estimation']['recovery']['kron_basis_diag'],
    'effective_top_eigenvalue': lambda d: d['effective']['spectrum']['top'],
    'effective_effective_rank': lambda d: d['effective']['spectrum']['effective_rank_entropy'],
    'effective_cross_tensor_energy': lambda d: d['effective']['cross_tensor_energy'],
    'effective_nearest_kronecker_error': lambda d: d['effective']['nearest_kronecker_error'],
    'effective_diagonal_error': lambda d: d['effective']['diagonal_error'],
    'effective_hutchinson8_error': lambda d: d['effective']['hutchinson8_relative_rms'],
    'effective_rank16_truncation_error': lambda d: d['effective']['spectrum']['rank16_truncation_error'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    common.guard()
    values, validation = {}, {}
    for name in read(args.run/'protocol.json')['units']:
        model, condition, seed = name.split('__')
        validation[(model, condition, int(seed[1:]))] = {r['step']: r['validation_loss'] for r in read(args.run/name/'validation.json')}
        for step in STEPS:
            d = read(args.run/name/f'state-{step:04d}'/'diagnostics.json')
            values[(model, condition, int(seed[1:]), step)] = {k: f(d) for k, f in METRICS.items()}
            values[(model, condition, int(seed[1:]), step)]['active_sn_layers'] = sum(r['active'] for r in d['spectral'].values())
    readings, means = {}, {}
    for model in ['vit', 'mlp']:
        for condition in CONDITIONS:
            for metric in list(METRICS)+['active_sn_layers']:
                for step in STEPS:
                    xs = [values[(model, condition, s, step)][metric] for s in SEEDS]
                    if all(x is not None for x in xs):
                        means.setdefault(model, {}).setdefault(condition, {}).setdefault(metric, {})[step] = statistics.mean(xs)
            if condition == 'nosn':
                continue
            for metric in METRICS:
                pairs = [(values[(model, condition, s, t)][metric], values[(model, 'nosn', s, t)][metric]) for s in SEEDS for t in STEPS]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                label = ('consistently lower' if pairs and all(a < b for a, b in pairs) else
                         'consistently higher' if pairs and all(a > b for a, b in pairs) else 'mixed')
                readings.setdefault(model, {}).setdefault(condition, {})[metric] = {'reading': label, 'pairs': len(pairs)}
    write_json(args.run/'sn_summary.json', {'means': means, 'readings': readings,
                                            'values': {'|'.join(map(str, k)): v for k, v in values.items()},
                                            'validation': {'|'.join(map(str, k)): v for k, v in validation.items()}})
    lines = []
    for model in ['vit', 'mlp']:
        lines += [f'### {model.upper()}', '', '| Metric | '+' | '.join(f'{c} (10/50/200)' for c in CONDITIONS)+' | '+' | '.join(f'{c} vs no SN' for c in CONDITIONS[1:])+' |',
                  '| --- | '+' | '.join('---' for _ in CONDITIONS)+' | '+' | '.join('---' for _ in CONDITIONS[1:])+' |']
        for metric in list(METRICS)+['active_sn_layers']:
            cells = ['/'.join(f'{means[model][c][metric][t]:.3g}' if t in means[model][c].get(metric, {}) else '—' for t in STEPS) for c in CONDITIONS]
            reads = [readings[model][c][metric]['reading'] if metric in readings[model][c] else '' for c in CONDITIONS[1:]]
            lines.append(f'| {metric} | '+' | '.join(cells)+' | '+' | '.join(reads)+' |')
        lines += ['', '| Condition | Validation loss at 10/50/100/200 (seed mean) |', '| --- | --- |']
        for c in CONDITIONS:
            lines.append(f'| {c} | '+'/'.join(f'{statistics.mean(validation[(model, c, s)][t] for s in SEEDS):.4f}' for t in [10, 50, 100, 200])+' |')
        lines.append('')
    (args.run/'sn_tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    panels = [('distortion_global_res', 'Global distortion q95/q5, residual stream'),
              ('raw_top_eigenvalue', 'Top GGN eigenvalue (raw parameters)'),
              ('raw_nearest_kronecker_error', 'Nearest-Kronecker error (raw)'),
              ('raw_empirical_fisher_error', 'Empirical-Fisher error vs GGN (raw)'),
              ('exact_ggn_headroom', 'Exact-GGN headroom over SOAP map'),
              ('energy_ray_empirical_fisher', 'Empirical-Fisher inverse A-energy error')]
    fig, axes = plt.subplots(2, len(panels), figsize=(18, 6.4), facecolor=SURFACE)
    for row, model in enumerate(['vit', 'mlp']):
        for col, (metric, title) in enumerate(panels):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            for c in CONDITIONS:
                series = means[model][c].get(metric, {})
                ax.plot([t for t in STEPS if t in series], [series[t] for t in STEPS if t in series], color=COLORS[c],
                        linewidth=2, marker='o', markersize=5, markeredgecolor=SURFACE, label=c)
            ax.set_xscale('log')
            ax.set_xticks(STEPS, [str(t) for t in STEPS])
            ax.set_title(f'{model.upper()}: {title}', color=INK, fontsize=8.5, loc='left')
            for side in ['top', 'right']:
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7)
            ax.grid(True, color=GRID, linewidth=.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ['no SN', 'SN c=0.95', 'SN c=2', 'SN c=6'], frameon=False, fontsize=8, labelcolor=INK, loc='lower center', ncol=4)
    fig.text(.01, .01, 'Lines coincide where SN is inactive (sigma(W) below c): ViT c=6 at all steps and c=2 before step 200 equal no SN. '
             'Seed means of 2 seeds; headroom ratios omitted (undefined near zero headroom).', color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(common.STUDY/'figures'/'sn_curvature.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main()
