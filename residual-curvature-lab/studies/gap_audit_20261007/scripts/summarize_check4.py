"""Check 4a/4b tables and figure from stored JSON.

Usage: summarize_check4.py --family DIR --headroom DIR
"""
import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import read  # noqa: E402
from check4b_headroom import STEPS, candidate_names, recovery  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
DEPTH_COLORS = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}


def aggregate(directory):
    """Per-state, per-architecture and per-split recovery from every completed 4b unit."""
    protocol = read(directory/'protocol.json')
    best = protocol['best_cheap_family']
    families = candidate_names(best)[2:]
    records, missing = [], []
    for name, spec in protocol['units'].items():
        if not common.unit_complete(directory/name):
            missing.append(name)
            continue
        for step in STEPS:
            for row in read(directory/name/f'candidates-{step:04d}.json'):
                records.append({**row, 'unit': name, 'step': step, 'blocks': spec['blocks'], 'width': spec['width'],
                                'seed': spec['seed'], 'split': spec['split']})
    summary = {'best_cheap_family': best, 'missing_units': missing, 'per_state': {}, 'per_architecture': {}, 'per_split': {}}
    for name in sorted({r['unit'] for r in records}):
        for step in STEPS:
            subset = [r for r in records if r['unit'] == name and r['step'] == step]
            summary['per_state'][f'{name}-{step}'] = {
                rhs: {'headroom': recovery(subset, 'exact', rhs)[1], 'identity': recovery(subset, 'exact', rhs)[2]['identity'],
                      'recovery': {f: recovery(subset, f, rhs)[0] for f in families}} for rhs in ['gradient', 'momentum']}
    for key, group in [('per_architecture', lambda r: f"b{r['blocks']}_w{r['width']}"), ('per_split', lambda r: r['split'])]:
        for label in sorted({group(r) for r in records}):
            subset = [r for r in records if group(r) == label]
            summary[key][label] = {rhs: {'aggregate_headroom': recovery(subset, 'exact', rhs)[1],
                                         'recovery': {f: recovery(subset, f, rhs)[0] for f in families},
                                         'states': len(subset)//(2*len(candidate_names(best)))} for rhs in ['gradient', 'momentum']}
    summary['train_split_best_cheap_recovery'] = summary['per_split']['train']['gradient']['recovery'][best]
    summary['searches_bracketed_or_rejected'] = all(r['search']['bracketed'] or r['search']['rejected'] for r in records)
    return summary


def pct(v):
    return '—' if v is None else f'{100*v:.1f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', type=Path, required=True)
    parser.add_argument('--headroom', type=Path, required=True)
    args = parser.parse_args()
    common.guard()
    family = read(args.family/'protocol.json')['units']
    units = read(args.family/'units_summary.json')
    summary = aggregate(args.headroom)
    write_json(args.headroom/'summary_partial.json' if summary['missing_units'] else args.headroom/'summary.json', summary)
    best = summary['best_cheap_family']
    families = ['soap_basis_diag', best, 'pcg_full_2', 'pcg_full_5']
    lines = ['| Architecture | Split | Params | Val. loss step 50 (seeds) | Headroom g | Headroom m | '+' | '.join(f'{f} g' for f in families)+' |',
             '| --- | --- | ---: | --- | ---: | ---: | '+' | '.join('---:' for _ in families)+' |']
    for arch, entry in summary['per_architecture'].items():
        names = [n for n in family if n.startswith(arch+'__')]
        losses = ', '.join(f"{units[n]['final_validation_loss']:.3f}" for n in sorted(names))
        splits = sorted({units[n]['split'] for n in names})
        g, m = entry['gradient'], entry['momentum']
        lines.append(f"| {arch} | {'/'.join(splits)} | {units[names[0]]['parameters']:,} | {losses} | {g['aggregate_headroom']:.6f} | "
                     f"{m['aggregate_headroom']:.6f} | "+' | '.join(pct(g['recovery'][f]) for f in families)+' |')
    lines += ['', '| Split | States | Headroom g | '+' | '.join(f'{f} g' for f in families)+' | Headroom m | '+' | '.join(f'{f} m' for f in families)+' |',
              '| --- | ---: | ---: | '+' | '.join('---:' for _ in families)+' | ---: | '+' | '.join('---:' for _ in families)+' |']
    for split, entry in summary['per_split'].items():
        g, m = entry['gradient'], entry['momentum']
        lines.append(f"| {split} | {g['states']} | {g['aggregate_headroom']:.6f} | "+' | '.join(pct(g['recovery'][f]) for f in families)
                     + f" | {m['aggregate_headroom']:.6f} | "+' | '.join(pct(m['recovery'][f]) for f in families)+' |')
    lines += ['', '| State | SOAP map g | Headroom g | '+' | '.join(f'{f} g' for f in families)+' |',
              '| --- | ---: | ---: | '+' | '.join('---:' for _ in families)+' |']
    for key, entry in summary['per_state'].items():
        g = entry['gradient']
        lines.append(f"| {key} | {g['identity']:.6f} | {g['headroom']:.6f} | "+' | '.join(pct(g['recovery'][f]) for f in families)+' |')
    if summary['missing_units']:
        lines = [f"PARTIAL: check 4b stopped; missing units: {', '.join(summary['missing_units'])}", ''] + lines
    (args.headroom/'check4_tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:30]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for ax, (title, getter) in zip(axes, [
            ('Exact tensor-GGN local headroom (current gradient)', lambda e: e['gradient']['aggregate_headroom']),
            (f'{best} recovery of that headroom', lambda e: 100*(e['gradient']['recovery'][best] or 0))]):
        ax.set_facecolor(SURFACE)
        for blocks in [1, 2, 3]:
            widths = [8, 16, 24]
            values = [getter(summary['per_architecture'][f'b{blocks}_w{w}']) for w in widths]
            ax.plot(widths, values, color=DEPTH_COLORS[blocks], linewidth=2, marker='o', markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f'{blocks} block{"s" if blocks > 1 else ""}')
        main = getter(summary['per_architecture']['b2_w16'])
        ax.annotate('main ViT', (16, main), xytext=(8, 8), textcoords='offset points', color=MUTED, fontsize=7)
        ax.set_xticks([8, 16, 24])
        ax.set_xlabel('Width (expansion = 2 x width)', color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc='left')
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean independent decrease, exact − SOAP map', color=INK, fontsize=9)
    axes[1].set_ylabel('Recovery (%)', color=INK, fontsize=9)
    axes[1].axhline(60, color=MUTED, linewidth=.8, linestyle=(0, (3, 3)))
    axes[1].text(24, 60, '60%', color=MUTED, fontsize=7, ha='right', va='bottom')
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    if summary['missing_units']:
        fig.text(.01, .01, f"Partial (check 4b stopped): missing {', '.join(summary['missing_units'])}; seed-mean over available seeds",
                 color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(common.STUDY/'figures'/'check4_headroom.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main()
