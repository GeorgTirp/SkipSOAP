"""Check 3 analysis (docs/ot_ggn_capacity_protocol_20261007.md and section 7).

All families, both RHS and both targets; per-state and aggregate own-teacher recovery;
absolute independent decreases; exact-OT versus exact-GGN teacher decrease at every
state; the pre-declared reading. Usage: summarize_check3.py --run DIR
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

STATES = [(t, s) for t in ['searched_soap', 'exact_tensor_ggn'] for s in [10, 25, 50]]
READING_FAMILIES = ['diagonal', 'state_4', 'combined_state_3']
TARGET_COLORS = {'ggn': '#2a78d6', 'ot': '#eb6834'}
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    common.guard()
    rows = read(args.run/'all_candidates.json')
    operators = read(args.run/'all_operators.json')
    states = read(args.run/'all_states.json')
    families = [f[0] for f in read(args.run/'protocol.json')['families']]
    def value(target, name, rhs, trajectory, step):
        return next(r for r in rows if r['target'] == target and r['name'] == name and r['rhs'] == rhs
                    and r['trajectory'] == trajectory and r['step'] == step)['selected']['independent_mean_decrease']
    def mean(target, name, rhs):
        return statistics.mean(value(target, name, rhs, t, s) for t, s in STATES)
    result = {'families': {}, 'teachers': {}, 'native_soap': {}}
    for target in ['ggn', 'ot']:
        result['native_soap'][target] = {f'{t}-{s}': value(target, 'native_soap', 'momentum', t, s) for t, s in STATES}
        for rhs in ['gradient', 'momentum']:
            reference = mean(target, 'exact', rhs)-mean(target, 'identity', rhs)
            result['teachers'][f'{target}_{rhs}_aggregate_reference_advantage'] = reference
            for name in families:
                entry = result['families'].setdefault(name, {}).setdefault(target, {})
                entry[f'{rhs}_mean_decrease'] = mean(target, name, rhs)
                entry[f'{rhs}_recovery'] = (mean(target, name, rhs)-mean(target, 'identity', rhs))/reference if reference > 1e-4 else None
                per = {}
                for t, s in STATES:
                    local = value(target, 'exact', rhs, t, s)-value(target, 'identity', rhs, t, s)
                    per[f'{t}-{s}'] = {'decrease': value(target, name, rhs, t, s),
                                       'recovery': (value(target, name, rhs, t, s)-value(target, 'identity', rhs, t, s))/local if abs(local) > 1e-4 else None}
                entry[f'{rhs}_per_state'] = per
        for name in families:
            selected = [o for o in operators if o['target'] == target and o['name'] == name]
            entry = result['families'][name][target]
            entry['pooled_ray_errors'] = {k: sum(o['errors'][k]['ray_numerator_sum'] for o in selected)/sum(o['errors'][k]['denominator_sum'] for o in selected)
                                          for k in ['gradient', 'momentum', 'physical', 'factor']}
            entry['equal_state_mean_ray_errors'] = {k: statistics.mean(sum(o['errors'][k]['ray'])/len(o['errors'][k]['ray']) for o in selected)
                                                    for k in ['gradient', 'momentum', 'physical', 'factor']}
            entry['storage_model_vectors'] = max(o['storage_model_vectors'] for o in selected)
            ratios = [o['apply_seconds_median']/next(v['apply_seconds_median'] for v in operators if v['target'] == target and v['name'] == 'identity'
                                                     and v['trajectory'] == o['trajectory'] and v['step'] == o['step']) for o in selected]
            entry['median_application_ratio'] = statistics.median(ratios)
            deployable = not name.startswith(('teacher', 'combined_teacher')) and name != 'exact'
            entry['shortlist'] = bool(deployable and entry['gradient_recovery'] is not None and entry['gradient_recovery'] >= .8
                                      and entry['storage_model_vectors'] <= 4.25 and entry['median_application_ratio'] <= 2.)
    result['teachers']['per_state'] = {f'{t}-{s}': {f'{target}_{rhs}': value(target, 'exact', rhs, t, s)
                                                    for target in ['ggn', 'ot'] for rhs in ['gradient', 'momentum']} for t, s in STATES}
    result['teachers']['mu'] = {f"{s['target']}-{s['trajectory']}-{s['step']}": s['mu'] for s in states}
    result['audits'] = {k: max(s[k] for s in states) for k in ['max_factor_error', 'max_action_error', 'cached_teacher_energy_error',
                                                               'teacher_gradient_residual', 'teacher_max_backward_error']}
    result['audits']['max_dense_application_error'] = max(o['max_dense_application_error'] for o in operators)
    result['audits']['searches_bracketed_or_rejected'] = all(r['search']['bracketed'] or r['search']['rejected'] for r in rows)
    result['counts'] = {'candidate_rows': len(rows), 'operator_states': len(operators), 'states': len(states)}
    reading = {}
    for name in READING_FAMILIES:
        g, o = result['families'][name]['ggn'], result['families'][name]['ot']
        recovery_gain = None if g['gradient_recovery'] is None or o['gradient_recovery'] is None else o['gradient_recovery']-g['gradient_recovery']
        reading[name] = {'ggn_recovery': g['gradient_recovery'], 'ot_recovery': o['gradient_recovery'],
                         'recovery_difference_pp': None if recovery_gain is None else 100*recovery_gain,
                         'ggn_mean_decrease': g['gradient_mean_decrease'], 'ot_mean_decrease': o['gradient_mean_decrease'],
                         'both_conditions': bool(recovery_gain is not None and recovery_gain >= .10
                                                 and o['gradient_mean_decrease'] >= g['gradient_mean_decrease'])}
    passed = sum(r['both_conditions'] for r in reading.values())
    result['reading'] = {'families': reading, 'qualifying_count': passed,
                         'verdict': 'OT is easier to compress' if passed >= 2 else 'no evidence that OT is easier to compress'}
    write_json(args.run/'check3_summary.json', result)
    lines = ['| Family | GGN recovery g | OT recovery g | GGN recovery m | OT recovery m | GGN mean dec. g | OT mean dec. g | Storage (model vectors) | Shortlist GGN/OT |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    pct = lambda v: '—' if v is None else f'{100*v:.1f}%'
    for name in families:
        g, o = result['families'][name]['ggn'], result['families'][name]['ot']
        lines.append(f"| {name} | {pct(g['gradient_recovery'])} | {pct(o['gradient_recovery'])} | {pct(g['momentum_recovery'])} | "
                     f"{pct(o['momentum_recovery'])} | {g['gradient_mean_decrease']:.6f} | {o['gradient_mean_decrease']:.6f} | "
                     f"{g['storage_model_vectors']:.3f} | {g['shortlist']}/{o['shortlist']} |")
    lines += ['', '| State | Exact GGN g | Exact OT g | Exact GGN m | Exact OT m | Native SOAP (GGN run) |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, v in result['teachers']['per_state'].items():
        lines.append(f"| {key} | {v['ggn_gradient']:.6f} | {v['ot_gradient']:.6f} | {v['ggn_momentum']:.6f} | {v['ot_momentum']:.6f} | {result['native_soap']['ggn'][key]:.6f} |")
    (args.run/'check3_tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print(result['reading'])
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    names = [n for n in families if n not in ['identity', 'exact']]
    for i, target in enumerate(['ggn', 'ot']):
        values = [100*(result['families'][n][target]['gradient_recovery'] or 0) for n in names]
        ax.barh([k+(i-.5)*.38 for k in range(len(names))], values, height=.36, color=TARGET_COLORS[target],
                label='GGN teacher' if target == 'ggn' else 'OT teacher', edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(len(names)), names, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=.8)
    ax.axvline(80, color=MUTED, linewidth=.8, linestyle=(0, (3, 3)))
    ax.text(80, -.9, '80% shortlist bar', color=MUTED, fontsize=7)
    ax.set_xlabel('Own-teacher current-gradient local recovery (%), six states', color=INK, fontsize=9)
    ax.set_title('OT vs GGN teacher compressibility (damping calibration)', color=INK, fontsize=10, loc='left')
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis='x', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc='lower right')
    fig.tight_layout()
    (common.STUDY/'figures').mkdir(exist_ok=True)
    fig.savefig(common.STUDY/'figures'/'check3_recovery.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main()
