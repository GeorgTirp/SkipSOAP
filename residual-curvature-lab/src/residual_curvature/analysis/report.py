import csv
import json
import os
from pathlib import Path


def summarize(run_dir):
    run_dir = Path(run_dir)
    # A writable, run-local cache also works in restricted environments.
    os.environ.setdefault('MPLCONFIGDIR', str((run_dir / '.matplotlib').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((run_dir / '.cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows = []
    interpretations = []
    for snapshot in sorted(run_dir.glob('snapshot-*')):
        candidates = json.loads((snapshot / 'candidates.json').read_text())
        diag = json.loads((snapshot / 'diagnostics.json').read_text())
        interpretations.append(f"## Snapshot {diag['step']}\n\n"
                               f"Scope: all parameters; partition: {diag['partition']}; common native-radius direction grid.\n\n"
                               "| gamma | method | raw damped model decrease | selected construction decrease | selected panel mean | boundary |\n"
                               "| --- | --- | --- | --- | --- | --- |")
        for r in candidates:
            rows.append({'step': r['step'], 'name': r['name'], 'gamma': r['gamma'], 'rank': r.get('rank'),
                         'repeat': r.get('repeat'), 'raw_predicted': r['raw']['undamped_decrease'],
                         'raw_actual': r['raw']['actual_decrease'], 'selected_predicted': r['selected']['undamped_decrease'],
                         'selected_actual': r['selected']['actual_decrease'],
                         'independent_mean': np.mean(r['selected']['independent_decreases']),
                         'sketch_error': r.get('sketch_frobenius_error'),
                         'wall_seconds': diag['wall_seconds'], 'peak_rss_bytes': diag['process_peak_rss_bytes']})
            if r['name'] in ['native_soap', 'exact_diagonal', 'exact_full', 'allocation_cone', 'allocation_span']:
                interpretations.append(f"| {r['gamma']:g} | {r['name']} | {r['raw']['damped_decrease']:.6g} | "
                                       f"{r['selected']['actual_decrease']:.6g} | {np.mean(r['selected']['independent_decreases']):.6g} | "
                                       f"{r['search'].get('boundary', False)} |")
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
        ax = axes.flat
        ax[0].scatter([r['selected']['undamped_decrease'] for r in candidates], [r['selected']['actual_decrease'] for r in candidates], s=12)
        ax[0].set(xlabel='Selected update: predicted GGN decrease', ylabel='Same update: actual decrease', title='Construction batch')
        gamma = candidates[0]['gamma']
        local = [r for r in candidates if r['gamma'] == gamma and
                 (r['name'] in ['native_soap', 'exact_diagonal', 'exact_full', 'allocation_cone'] or
                  (r['name'] in ['sketch_diagonal', 'sketch_coupled'] and r['repeat'] == 0 and r['rank'] == max(s.get('rank', 0) for s in candidates)))]
        labels = [r['name'].replace('_', '\n') for r in local]
        ax[1].bar(labels, [r['selected']['actual_decrease'] for r in local])
        ax[1].tick_params(axis='x', labelsize=8)
        ax[1].set(title=f'Common direction grid; gamma={gamma}', ylabel='Actual construction decrease')
        sketches = [r for r in candidates if r['gamma'] == gamma and r['name'] == 'sketch_coupled']
        ax[2].scatter([r['rank'] for r in sketches], [r['sketch_frobenius_error'] for r in sketches], label='Relative factor-model error')
        quality = ax[2].twinx()
        quality.scatter([r['rank'] for r in sketches], [r['selected']['actual_decrease'] for r in sketches], c='orange', marker='x')
        ax[2].set(xlabel='Rank (dots are sketch repeats)', ylabel='Relative Frobenius error', title='Sketch accuracy and candidate quality')
        quality.set_ylabel('Selected actual decrease', color='orange')
        names = list(local[0]['raw']['shares'])
        ax[3].bar(range(len(names)), list(local[0]['raw']['shares'].values()))
        ax[3].set(xlabel='Registry tensor index', ylabel='Squared-update share', title='Raw native SOAP shares')
        coupling = diag['coupling'][0]
        ax[4].bar(['diagonal', 'positive pairs', 'negative pairs'], [coupling['diagonal'], coupling['positive_pairs'], coupling['negative_pairs']])
        ax[4].set(title='Native exposure numerator; unordered pairs ×2')
        ax[5].axis('off')
        ax[5].text(.02, .9, f"Snapshot {diag['step']} / {diag['partition']} partition\nTotal analysis: {diag['wall_seconds']:.3f} s\nProcess peak RSS: {diag['process_peak_rss_bytes']/2**20:.1f} MiB\nExact dual residual: {local[0]['full_solve_residual']:.2e}\nMetric: {local[0]['metric']}\n\nCPU toy result; no efficiency claim.\nRSS is process lifetime, not incremental.\nNo pooling across training seeds.", va='top')
        fig.savefig(snapshot / 'comparison.png', dpi=160)
        plt.close(fig)
    if rows:
        with (run_dir / 'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    training = json.loads((run_dir / 'training.json').read_text()) if (run_dir / 'training.json').exists() else []
    if training:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axs[0].plot([r['accepted_updates'] for r in training], [r['loss_after'] for r in training], marker='.')
        axs[0].set(xlabel='Accepted updates', ylabel='Training minibatch loss')
        axs[1].plot([r['wall_seconds'] for r in training], [r['loss_after'] for r in training], marker='.')
        axs[1].set(xlabel='Elapsed seconds (includes prior monitoring)', ylabel='Training minibatch loss')
        fig.savefig(run_dir / 'training.png', dpi=160)
        plt.close(fig)
    (run_dir / 'REPORT.md').write_text(
        '# Synthetic/selected-run report\n\n'
        'See summary.csv for per-state, per-damping, per-rank and per-repeat measurements. '
        'Plots are regenerated exclusively from stored results. Raw and selected predictions '
        'are compared to the actual loss of the same displacement. Independent-panel values '
        'use the construction-selected update without retuning. Sketch repeats are not training seeds.\n\n'
        'Coupling opportunity: compare exact_diagonal with exact_full. Fixed-direction opportunity: '
        'compare native optimized-ray, allocation_cone, allocation_span and exact_full damped values '
        'in candidates.json. Raw directions can have poor nonlinear loss despite a positive quadratic value. '
        'No percentage gain is pooled or claimed. Algorithm B and language-model efficiency remain untested.\n\n'
        + '\n'.join(interpretations) + '\n\n'
        'Interpretation limits: quadratic optimality applies to the stated damped model. '
        'The common finite direction window can truncate useful lengths; inspect boundary selections. '
        'Raw diagonal candidates are scored under the full GGN and can overshoot. '
        'Any loss decrease that fails on the fixed independent panel is retained. '
        'Sketch error, exact-oracle headroom, nonlinear selection and transfer are separate measurements; '
        'these data do not identify a single universal failure mechanism or establish training benefit.\n')
    return run_dir / 'REPORT.md'
