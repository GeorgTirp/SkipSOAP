"""Check 2c tables and figure from the run's summary.json and per-state JSON.

Usage: summarize_check2c.py --run DIR
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import read  # noqa: E402

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
RHS_COLORS = {'gradient': '#2a78d6', 'momentum': '#eb6834'}


def pct(v):
    return '—' if v is None else f'{100*v:.1f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    common.guard()
    summary = read(args.run/'summary.json')
    families = list(summary['families'])
    states = summary['states']
    lines = ['| Family | Recovery g | Recovery m | Mean decrease g | Mean decrease m | Ray error g | Ray error m | Ray error factor | '
             'Storage numbers | Model vectors | x SOAP factors | Matvecs/RHS | Tier |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name in families:
        f = summary['families'][name]
        e = f['pooled_ray_errors']
        lines.append(f"| {name} | {pct(f['gradient_recovery'])} | {pct(f['momentum_recovery'])} | {f['gradient_mean_decrease']:.6f} | "
                     f"{f['momentum_mean_decrease']:.6f} | {e['gradient']:.3e} | {e['momentum']:.3e} | {e['factor']:.3e} | "
                     f"{f['storage_numbers']:,} | {f['storage_model_vectors']:.2f} | {f['storage_relative_to_soap_factors']:.2f} | "
                     f"{f['matvecs_per_rhs']} | {f.get('tier', 'reference')} |")
    lines += ['', '| Family | '+' | '.join(states)+' |', '| --- | '+' | '.join('---:' for _ in states)+' |']
    for name in families:
        per = summary['families'][name]['gradient_per_state']
        lines.append(f'| {name} | '+' | '.join(f"{per[s]['decrease']:.6f}" for s in states)+' |')
    lines += ['', '| Family (momentum RHS) | '+' | '.join(states)+' |', '| --- | '+' | '.join('---:' for _ in states)+' |']
    for name in families:
        per = summary['families'][name]['momentum_per_state']
        lines.append(f'| {name} | '+' | '.join(f"{per[s]['decrease']:.6f}" for s in states)+' |')
    lines += ['', '| Sanity row | State | Reproduced | Stored | Absolute difference |', '| --- | --- | ---: | ---: | ---: |']
    for state, values in summary['reproduction'].items():
        for name, v in values.items():
            lines.append(f"| {name} | {state} | {v['reproduced']:.9f} | {v['stored']:.9f} | {v['absolute_difference']:.2e} |")
    (args.run/'check2c_tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.4), facecolor=SURFACE, sharey=True)
    names = [n for n in families if n not in ['identity', 'exact']]
    rows = range(len(names))
    left.barh(rows, [100*summary['families'][n]['gradient_recovery'] for n in names], height=.6,
              color=RHS_COLORS['gradient'], edgecolor=SURFACE, linewidth=1)
    for level in [60, 80]:
        left.axvline(level, color=MUTED, linewidth=.8, linestyle=(0, (3, 3)))
    left.text(59, -.75, '60% promising', color=MUTED, fontsize=7, ha='right', va='bottom')
    left.text(81, -.75, '80% bar', color=MUTED, fontsize=7, ha='left', va='bottom')
    left.set_xlabel('Current-gradient local recovery (%), six-state aggregate', color=INK, fontsize=9)
    left.set_title('Recovery of exact tensor-GGN headroom', color=INK, fontsize=10, loc='left')
    for i, rhs in enumerate(['gradient', 'momentum']):
        right.barh([k+(i-.5)*.38 for k in rows], [summary['families'][n][f'{rhs}_mean_decrease'] for n in names], height=.36,
                   color=RHS_COLORS[rhs], edgecolor=SURFACE, linewidth=1,
                   label='Current gradient' if rhs == 'gradient' else 'Effective SOAP momentum')
    for name, style in [('identity', (0, (1, 2))), ('exact', (0, (4, 2)))]:
        right.axvline(summary['families'][name]['gradient_mean_decrease'], color=MUTED, linewidth=.9, linestyle=style)
        right.text(summary['families'][name]['gradient_mean_decrease'], -.75,
                   'SOAP map, g' if name == 'identity' else 'exact GGN, g', color=MUTED, fontsize=7, ha='center', va='bottom')
    right.set_xlabel('Mean independent decrease over six states', color=INK, fontsize=9)
    right.set_title('Absolute decreases, both RHS', color=INK, fontsize=10, loc='left')
    right.legend(frameon=False, fontsize=8, labelcolor=INK, loc='lower right', bbox_to_anchor=(1, -.2), ncol=2)
    for ax in [left, right]:
        ax.set_facecolor(SURFACE)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, axis='x', color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
        ax.axvline(0, color=MUTED, linewidth=.8)
    left.set_yticks(list(rows), names, fontsize=8, color=INK)
    left.invert_yaxis()
    fig.tight_layout()
    (common.STUDY/'figures').mkdir(exist_ok=True)
    fig.savefig(common.STUDY/'figures'/'check2c_recovery.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main()
