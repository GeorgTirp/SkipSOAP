"""Summary of the PARSEC offline study (PROTOCOL.md readings). Usage: summarize_offline.py B1_RUN B2_RUN C_RUN"""
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import read  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
RANK_COLORS = {4: '#2a78d6', 8: '#eb6834', 16: '#1baf7a'}


def b1(run):
    units = read(run/'protocol.json')['units']
    decrease, states = {}, {}
    for name, spec in units.items():
        rows = read(run/name/'candidates.json')
        states[name] = read(run/name/'state.json')
        for r in rows:
            decrease.setdefault((r['name'], r['rhs']), {})[name] = r['selected']['independent_mean_decrease']
    subsets = {'all': list(units), 'main': [n for n in units if n.startswith('main')], 'family': [n for n in units if n.startswith('fam')]}
    out = {'means': {}, 'R': {}, 'angles': {}, 'kappa1_relative_error': {}, 'probe_trace_relative_error': {}}
    for subset, names in subsets.items():
        mean = lambda core, rhs: statistics.mean(decrease[(core, rhs)][n] for n in names)
        out['means'][subset] = {f'{core}|{rhs}': mean(core, rhs) for core, rhs in decrease}
        for rhs in ['gradient', 'momentum']:
            for r in [4, 8, 16]:
                denominator = mean(f'exact_r{r}', rhs)-mean('identity', rhs)
                for s in [1, 4]:
                    value = (mean(f'probe_s{s}_r{r}', rhs)-mean('identity', rhs))/denominator if abs(denominator) > 1e-4 else None
                    out['R'].setdefault(subset, {})[f's{s}_r{r}_{rhs}'] = value
        out['angles'][subset] = {k: statistics.mean(states[n]['angles'][k] for n in names) for k in states[names[0]]['angles']}
        out['kappa1_relative_error'][subset] = {k: statistics.mean(states[n]['kappa1_relative_error'][k] for n in names) for k in states[names[0]]['kappa1_relative_error']}
        out['probe_trace_relative_error'][subset] = {k: statistics.mean(states[n]['probe_trace_relative_error'][k] for n in names) for k in ['s1', 's4']}
    primary = out['R']['all']['s4_r16_momentum']
    out['kill_I2'] = None if primary is None else primary < .70
    out['primary_R'] = primary
    out['gates'] = {n: s['gate'] for n, s in states.items() if 'gate' in s}
    return out


def c(run):
    sn_values = read(common.SN_RUN/'sn_summary.json')['values']
    rows = {n: read(run/n/'groups.json') for n in read(run/'protocol.json')['units']}
    out = {'values': rows, 'readings': {}}
    for model in ['vit', 'mlp']:
        for cond in ['sn0.95', 'sn2', 'sn6']:
            for param in ['effective', 'raw']:
                for group in ['stream_writer', 'first_in_branch']:
                    for metric in ['ks1_error', 'kron_basis_diag_error']:
                        pairs = []
                        for seed in [0, 1]:
                            for step in ['10', '50', '200']:
                                a = rows[f'{model}__{cond}__s{seed}'][step]
                                if a['active_sn_layers'] == 0:
                                    continue
                                b = rows[f'{model}__nosn__s{seed}'][step]
                                pairs.append((a[param][group][metric], b[param][group][metric]))
                        label = ('consistently lower' if pairs and all(x < y for x, y in pairs) else
                                 'consistently higher' if pairs and all(x > y for x, y in pairs) else 'mixed' if pairs else 'no active pairs')
                        out['readings'][f'{model}|{cond}|{param}|{group}|{metric}'] = {'reading': label, 'pairs': len(pairs)}
        writer = out['readings'][f'{model}|sn0.95|effective|stream_writer|ks1_error']['reading']
        first = out['readings'][f'{model}|sn0.95|effective|first_in_branch|ks1_error']['reading']
        distortion_lower = all(sn_values[f'{model}|sn0.95|{s}|10']['distortion_local_res'] < sn_values[f'{model}|nosn|{s}|10']['distortion_local_res'] for s in [0, 1])
        out[f'S2_{model}'] = ('supported' if writer == 'consistently lower' and first != 'consistently lower' else
                              'absent' if writer != 'consistently lower' and distortion_lower else 'undecided')
        out[f'S2_{model}_inputs'] = {'stream_writer': writer, 'first_in_branch': first, 'local_distortion_lower_at_step10': distortion_lower}
    return out


def main():
    common.guard()
    b1_run, b2_run, c_run = map(Path, sys.argv[1:4])
    result = {'B1': b1(b1_run), 'B2': read(b2_run/'trajectory'/'summary.json'), 'C': c(c_run)}
    write_json(b1_run/'offline_summary.json', result)
    tracking = read(b2_run/'trajectory'/'tracking.json')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE)
    ax = axes[0]
    labels = [f'r{r}' for r in [4, 8, 16]]
    for i, s in enumerate([1, 4]):
        for j, rhs in enumerate(['gradient', 'momentum']):
            values = [100*(result['B1']['R']['all'][f's{s}_r{r}_{rhs}'] or 0) for r in [4, 8, 16]]
            ax.bar([k+(2*i+j-1.5)*.2 for k in range(3)], values, width=.18, color=['#2a78d6', '#eb6834'][j],
                   alpha=.55 if s == 1 else 1., edgecolor=SURFACE, label=f's={s}, {rhs}')
    ax.axhline(70, color=MUTED, linewidth=.8, linestyle=(0, (3, 3)))
    ax.text(-.45, 64, '70% kill bar', color=MUTED, fontsize=7)
    ax.set_ylim(0, 125)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel('Probe / exact PARSEC-L local gain (%)', color=INK, fontsize=9)
    ax.set_title('B1: probe-estimated vs exact (72 states)', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK, ncol=2, loc='upper center')
    ax = axes[1]
    keys = [('drift', 'drift per step', '#52514e'), ('exact_tracking', 'one-step tracking, exact K', '#2a78d6'),
            ('probe_tracking', 'one-step tracking, probes', '#eb6834'), ('fresh_extraction', 'fresh probe extraction', '#1baf7a')]
    for j, (key, label, color) in enumerate(keys):
        ax.bar([i+(j-1.5)*.2 for i in range(3)], [result['B2']['summary'][f'r{r}'][key]['median'] for r in [4, 8, 16]],
               width=.18, color=color, edgecolor=SURFACE, label=label)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel('Median sin²Θ, steps 21–50 (energy-weighted)', color=INK, fontsize=9)
    ax.set_title('B2: tracking vs fresh extraction (kill if orange > green)', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    ax = axes[2]
    for model, marker in [('vit', 'o'), ('mlp', 's')]:
        for group, color in [('stream_writer', '#2a78d6'), ('first_in_branch', '#eb6834')]:
            xs, ys = [], []
            for cond, xpos in [('nosn', 0), ('sn6', 1), ('sn2', 2), ('sn0.95', 3)]:
                vals = [result['C']['values'][f'{model}__{cond}__s{s}'][t]['effective'][group]['ks1_error'] for s in [0, 1] for t in ['10', '50']]
                xs.append(xpos); ys.append(statistics.mean(vals))
            ax.plot(xs, ys, color=color, marker=marker, linewidth=1.8, markersize=6, markeredgecolor=SURFACE,
                    label=f'{model.upper()} {group.replace("_", " ")}', linestyle='-' if model == 'vit' else (0, (4, 2)))
    ax.set_xticks(range(4), ['no SN', 'c=6', 'c=2', 'c=0.95'])
    ax.set_ylabel('Nearest-Kronecker error (effective weights)', color=INK, fontsize=9)
    ax.set_title('C: stream writers vs first-in-branch (steps 10, 50)', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.text(.67, .01, 'Step 200 omitted: MLP branches dead without SN and at c=6 (zero curvature).', color=MUTED, fontsize=7)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(common.STUDY/'figures'/'parsec_offline.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)
    import json
    print(json.dumps({'B1_primary_R': result['B1']['primary_R'], 'kill_I2': result['B1']['kill_I2'], 'R_all': result['B1']['R']['all'],
                      'angles_all': result['B1']['angles']['all'], 'B2': result['B2']['summary'],
                      'S2': {m: (result['C'][f'S2_{m}'], result['C'][f'S2_{m}_inputs']) for m in ['vit', 'mlp']}}, indent=1))


if __name__ == '__main__':
    main()
