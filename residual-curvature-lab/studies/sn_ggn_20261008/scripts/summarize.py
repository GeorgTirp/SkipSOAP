"""Readings of PROTOCOL.md section 4, tables and figure. Usage: summarize.py RUN_DIR"""
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

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COLORS = {'ggn': '#2a78d6', 'soap': '#eb6834'}
LABELS = {'ggn': 'per-tensor GGN', 'soap': 'native SOAP', 'nosn': 'no SN', 'sn0.95': 'SN c=0.95'}
CONDITIONS, SEEDS, STEPS = ['nosn', 'sn0.95'], [1, 2, 3], [50, 100, 250]
WINDOWS = [(1, 50), (51, 150), (151, 250)]
TARGETS = {'T50': 0.355037, 'T100': 0.265401}


def reading(a, b):
    """'lower'/'higher' if a < b / a > b in every seed, else 'mixed'."""
    return 'lower' if all(x < y for x, y in zip(a, b)) else 'higher' if all(x > y for x, y in zip(a, b)) else 'mixed'


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def main():
    common.guard()
    run = Path(sys.argv[1])
    val, train, spectra, runtime = {}, {}, {}, {}
    for opt in ['ggn', 'soap']:
        for cond in CONDITIONS:
            for s in SEEDS:
                unit = run/f'{opt}__{cond}__s{s}'
                val[opt, cond, s] = {r['step']: r for r in read(unit/'validation.json')}
                train[opt, cond, s] = read(unit/'training.json')
                spectra[opt, cond, s] = read(unit/'spectra.json')
                runtime[opt, cond, s] = read(unit/'runtime.json')
    loss = lambda opt, cond, step: [val[opt, cond, s][step]['validation_loss'] for s in SEEDS]
    out = {'readings': {}, 'tables': {}, 'transfer': {}, 'targets': {}, 'exploratory': {}}
    for opt in ['ggn', 'soap']:
        out['readings'][f'sn_effect_{opt}'] = {step: reading(loss(opt, 'sn0.95', step), loss(opt, 'nosn', step)) for step in STEPS}
    ratio = {cond: {step: [a/b for a, b in zip(loss('soap', cond, step), loss('ggn', cond, step))] for step in STEPS} for cond in CONDITIONS}
    out['ratio_soap_over_ggn'] = ratio
    out['readings']['ggn_advantage_under_sn'] = {
        step: {'lower': 'shrinks', 'higher': 'grows', 'mixed': 'mixed'}[reading(ratio['sn0.95'][step], ratio['nosn'][step])] for step in STEPS}
    schedule = sorted(val['ggn', 'nosn', 1])
    for opt in ['ggn', 'soap']:
        for cond in CONDITIONS:
            key = f'{opt}|{cond}'
            out['tables'][key] = {step: {'loss': mean_sd(loss(opt, cond, step)),
                                         'accuracy': mean_sd([val[opt, cond, s][step]['validation_accuracy'] for s in SEEDS])}
                                  for step in [10, 25, *STEPS]}
            out['tables'][key]['per_seed'] = {step: loss(opt, cond, step) for step in schedule}
            seconds = [runtime[opt, cond, s]['training_seconds']/len(train[opt, cond, s]) for s in SEEDS]
            out['exploratory'].setdefault('seconds_per_step', {})[key] = statistics.mean(seconds)
            transfer, logbased = {}, {}
            for lo, hi in WINDOWS:
                rows = [r for s in SEEDS for r in train[opt, cond, s] if lo <= r['step'] <= hi]
                own = statistics.mean(r['loss_before']-r['loss_after'] for r in rows)
                nxt = statistics.mean(r['next_loss_before']-r['next_loss_after'] for r in rows)
                transfer[f'{lo}-{hi}'] = {'own_drop': own, 'next_drop': nxt, 'transfer': nxt/own}
                pairs = [(t[i], t[i+1]) for s in SEEDS for t in [train[opt, cond, s]] for i in range(len(t)-1) if lo <= t[i]['step'] <= min(hi, 249)]
                net = statistics.mean(a['loss_before']-b['loss_before'] for a, b in pairs)
                own_lb = statistics.mean(a['loss_before']-a['loss_after'] for a, _ in pairs)
                logbased[f'{lo}-{min(hi, 249)}'] = net/own_lb
            out['transfer'][key] = {'paired': transfer, 'log_based': logbased}
            for name, target in TARGETS.items():
                hits = []
                for s in SEEDS:
                    hit = next((st for st in schedule if st > 0 and val[opt, cond, s][st]['validation_loss'] <= target), None)
                    hits.append(None if hit is None else (hit, val[opt, cond, s][hit]['training_seconds']))
                reached = [h for h in hits if h]
                out['targets'][f'{key}|{name}'] = {'reached': len(reached), 'per_seed': hits,
                                                   'mean_steps': statistics.mean(h[0] for h in reached) if len(reached) == 3 else None,
                                                   'mean_seconds': statistics.mean(h[1] for h in reached) if len(reached) == 3 else None}
            sigma = {}
            for step in schedule:
                values = [r['sigma_raw'] for s in SEEDS for r in spectra[opt, cond, s][str(step)].values()]
                sigma[step] = {'mean': statistics.mean(values), 'max': max(values),
                               'active': statistics.mean(sum(r['active'] for r in spectra[opt, cond, s][str(step)].values()) for s in SEEDS)}
            out['exploratory'].setdefault('sigma_raw', {})[key] = sigma
            if opt == 'ggn':
                rows = [r for s in SEEDS for r in train[opt, cond, s]]
                out['exploratory'].setdefault('audits', {})[key] = {
                    'max_action_check': max(r['max_action_check'] for r in rows if r['max_action_check'] is not None),
                    'max_solve_residual': max(r['solve_residual'] for r in rows if r['solve_residual'] is not None),
                    'searches': len([r for r in rows if r['search']]),
                    'bracketed': sum(bool(r['search'] and r['search']['bracketed']) for r in rows),
                    'rejected': sum(bool(r['search'] and r['search']['rejected']) for r in rows),
                    'zero_selected': sum(bool(r['search'] and r['search']['selected_alpha'] == 0) for r in rows)}
                for lo, hi in WINDOWS:
                    w = [r for r in rows if lo <= r['step'] <= hi and r['mu'] is not None]
                    out['exploratory'].setdefault('mu_trace_norm', {}).setdefault(key, {})[f'{lo}-{hi}'] = {
                        'mu': statistics.mean(r['mu'] for r in w), 'trace': statistics.mean(r['trace'] for r in w),
                        'step_norm': statistics.mean(r['norm'] for r in w)}
    out['reproduction_gate'] = {f'{opt}__nosn__s{s}': read(run/f'{opt}__nosn__s{s}'/'reproduction_gate.json')['max_difference']
                                for opt in ['ggn', 'soap'] for s in SEEDS}
    write_json(run/'sn_ggn_summary.json', out)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE)
    ax = axes[0]
    for opt in ['ggn', 'soap']:
        for cond, style in [('nosn', '-'), ('sn0.95', (0, (4, 2)))]:
            steps = [st for st in schedule if st >= 10]
            ax.plot(steps, [statistics.mean(loss(opt, cond, st)) for st in steps], color=COLORS[opt], linestyle=style, linewidth=2,
                    label=f'{LABELS[opt]}, {LABELS[cond]}')
    ax.set_yscale('log')
    ax.set_xlabel('Consumed step', color=INK, fontsize=9)
    ax.set_ylabel('Validation loss (mean of seeds 1–3, log scale)', color=INK, fontsize=9)
    ax.set_title('Validation loss', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    ax = axes[1]
    keys = [(o, c) for o in ['ggn', 'soap'] for c in CONDITIONS]
    for j, (opt, cond) in enumerate(keys):
        values = [100*out['transfer'][f'{opt}|{cond}']['paired'][f'{lo}-{hi}']['transfer'] for lo, hi in WINDOWS]
        ax.bar([i+(j-1.5)*.2 for i in range(3)], values, width=.18, color=COLORS[opt], alpha=1. if cond == 'nosn' else .5,
               hatch=None if cond == 'nosn' else '//', edgecolor=SURFACE, label=f'{LABELS[opt]}, {LABELS[cond]}')
    ax.set_xticks(range(3), [f'steps {lo}–{hi}' for lo, hi in WINDOWS])
    ax.set_ylabel('Next-batch drop / own-batch drop (%)', color=INK, fontsize=9)
    ax.set_title('Transfer of each update to the next batch', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    ax = axes[2]
    for opt in ['ggn', 'soap']:
        for cond, style in [('nosn', '-'), ('sn0.95', (0, (4, 2)))]:
            sig = out['exploratory']['sigma_raw'][f'{opt}|{cond}']
            ax.plot(schedule, [sig[st]['mean'] for st in schedule], color=COLORS[opt], linestyle=style, linewidth=2,
                    label=f'{LABELS[opt]}, {LABELS[cond]}')
    ax.axhline(.95, color=MUTED, linewidth=.8, linestyle=(0, (3, 3)))
    ax.text(205, 1.0, 'c = 0.95', color=MUTED, fontsize=7)
    ax.set_xlabel('Consumed step', color=INK, fontsize=9)
    ax.set_ylabel('Mean σ of the 12 raw branch weights', color=INK, fontsize=9)
    ax.set_title('Raw spectral norms (SN acts on W·c/σ)', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(common.STUDY/'figures'/'sn_ggn.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(json.dumps({k: out[k] for k in ['readings', 'ratio_soap_over_ggn', 'reproduction_gate']}, indent=1))
    print(json.dumps({k: {kk: {w: round(v['transfer'], 4) for w, v in vv['paired'].items()} | {'log': {w: round(v, 4) for w, v in vv['log_based'].items()}}
                          for kk, vv in out[k].items()} for k in ['transfer']}, indent=1))
    print(json.dumps(out['targets'], indent=1))
    print(json.dumps({k: out['exploratory'][k] for k in ['seconds_per_step', 'audits', 'mu_trace_norm']}, indent=1))
    for key, table in out['tables'].items():
        print(key, {st: (round(table[st]['loss'][0], 4), round(table[st]['loss'][1], 4), round(100*table[st]['accuracy'][0], 2)) for st in [10, 25, *STEPS]})
    for key, sig in out['exploratory']['sigma_raw'].items():
        print(key, {st: (round(sig[st]['mean'], 3), round(sig[st]['max'], 3), sig[st]['active']) for st in [0, 10, 50, 100, 250]})


if __name__ == '__main__':
    main()
