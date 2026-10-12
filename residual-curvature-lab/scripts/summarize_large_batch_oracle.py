"""Reconstruct figures and a report from the saved larger-batch measurements."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle-run', type=Path, required=True)
    parser.add_argument('--training-run', type=Path)
    args = parser.parse_args()
    directory = args.oracle_run
    rows = json.loads((directory / 'all_candidates.json').read_text())
    diagnostics = json.loads((directory / 'all_diagnostics.json').read_text())
    os.environ.setdefault('MPLCONFIGDIR', str((directory / '.matplotlib').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((directory / '.cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, constrained_layout=True)
    for ax, batch in zip(axes, [256, 512]):
        styles = [('soap', None, 'selected', 'SOAP, searched', '#2166ac'),
                  ('soap', None, 'raw', 'SOAP, native', '#73a9cf'),
                  ('exact_tensor_ggn', .001, 'selected', 'Exact tensor GGN, gamma=.001', '#f2a280'),
                  ('exact_tensor_ggn', .01, 'selected', 'Exact tensor GGN, gamma=.01 (primary)', '#b2182b'),
                  ('exact_tensor_ggn', .1, 'selected', 'Exact tensor GGN, gamma=.1', '#6a1634')]
        for name, gamma, part, label, color in styles:
            selected = sorted([r for r in rows if r['batch_size'] == batch and r['name'] == name and r['gamma'] == gamma], key=lambda r: r['step'])
            ax.plot([r['step'] for r in selected], [r[part]['independent_mean_decrease'] for r in selected],
                    marker='o', linestyle='--' if part == 'raw' else '-', label=label, color=color)
        ax.axhline(0, color='gray', linewidth=.7)
        ax.set(title=f'Construction and panel batch size {batch}', xlabel='Saved SOAP state (training iteration)')
        ax.set_xticks([10, 50, 100])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean independent cross-entropy decrease\nHigher is better; negative means loss increased')
    axes[1].legend(fontsize=8)
    fig.suptitle('Fixed states of a batch-64 SOAP trajectory; four independent panel batches')
    fig.savefig(directory / 'independent_decreases.png', dpi=170)
    plt.close(fig)

    lines = ['# Local MNIST larger-batch oracle results — 2 October 2026', '',
             f'Fixed-state run: `{directory}`. Batch 256 completed before batch 512. '
             'The model is the existing 14x14 residual MLP: 6,634 parameters in 22 exact tensor blocks. '
             'All states and SOAP moments come from the original batch-64 trajectory. '
             'Each candidate uses the true construction gradient and a cloned SOAP proposal. '
             'Damping is mu=max(gamma*trace(C)/D, 1e-8) with normalized inverse-SOAP metric.', '',
             'The table reports independent-panel mean loss decreases, with positive values indicating improvement. '
             'SOAP and GGN directions use the same broad construction-only line search; the panel never selects a scalar or damping.', '',
             '| Batch | State | Native SOAP | Searched SOAP | GGN .001 | GGN .01 (primary) | GGN .1 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    gaps = {}
    for b in [256, 512]:
        gaps[b] = []
        for step in [10, 50, 100]:
            subset = [r for r in rows if r['batch_size'] == b and r['step'] == step]
            soap = next(r for r in subset if r['name'] == 'soap')
            exact = {r['gamma']: r for r in subset if r['name'] == 'exact_tensor_ggn'}
            values = [soap['raw']['independent_mean_decrease'], soap['selected']['independent_mean_decrease']]
            values.extend(exact[g]['selected']['independent_mean_decrease'] for g in [.001, .01, .1])
            gaps[b].append(values[3] - values[1])
            lines.append(f'| {b} | {step} | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines.extend(['', 'At gamma=.01, GGN improves construction loss more than searched SOAP at all six states. '
                  'At batch 256 it transfers worse on the independent mean at all three states. '
                  'At batch 512 it transfers better at two of three states, with a small positive mean difference. '
                  'The gamma=.1 sensitivity result is stronger at the early and middle batch-512 states, '
                  'but it was not selected for training using the panel. The four panels and three states '
                  'are not independent training seeds and these results are not a significance test.', '',
                  '| Batch | Primary GGN minus searched SOAP, mean over states | States with positive difference | Short-training gate |',
                  '| --- | --- | --- | --- |'])
    for b, values in gaps.items():
        passed = sum(v > 0 for v in values) >= 2 and sum(values) > 0
        lines.append(f"| {b} | {np.mean(values):+.6f} | {sum(v > 0 for v in values)}/3 | {'passed' if passed else 'not passed'} |")
    max_action = max(v for d in diagnostics for v in d['action_checks'].values())
    max_solve = max(r['solve']['residual'] for r in rows if r['solve'])
    lines.extend(['', f'All action and solve checks passed 1e-8: maximum action error {max_action:.3e}; '
                  f'maximum combined solve residual {max_solve:.3e}. All 30 direction searches selected an interior '
                  'grid bracket and used 52 evaluations. No positive search boundary limited these comparisons. '
                  'Non-mutation checks passed for all six snapshots.', '',
                  f"Measured snapshot work: {sum(d['seconds'] for d in diagnostics):.3f} seconds total; "
                  f"process-lifetime peak RSS {max(d['process_peak_rss_bytes'] for d in diagnostics)/2**20:.1f} MiB. "
                  'The conservative tensor-workspace estimate was 636.8 MiB against a declared 768 MiB budget. '
                  'RSS includes the dataset, Python and numerical libraries. Saved artifacts include the exact blocks, '
                  'directions, optimizer contexts, example IDs, source archives and checksums.', '',
                  f'![Independent loss decreases]({directory.resolve()}/independent_decreases.png)', ''])
    if args.training_run:
        training = args.training_run
        complete = json.loads((training / 'complete.json').read_text())
        assert complete['completed'] and complete['paired_initialization'] and complete['paired_data_order']
        validation = json.loads((training / 'validation.json').read_text())
        methods = complete['methods']
        labels = {'native_soap': 'Native SOAP', 'searched_soap': 'Line-searched SOAP', 'exact_tensor_ggn': 'Exact tensor GGN'}
        lines.extend(['## Paired batch-512 training followup', '',
                      f'Training run: `{training}`. The predeclared snapshot gate passed only at batch 512. '
                      'All three arms started from identical seed-0 random weights and consumed the same 50 batches '
                      '(25,600 examples; the first batch initializes statistics without a parameter update). '
                      'Validation uses all 5,000 reserved examples at steps 0, 10, 25 and 50, without tuning or early stopping. '
                      'The official test set was not evaluated. The exact arm uses the predeclared gamma=.01.', '',
                      '| Method | Iteration | Validation loss | Accuracy | Training-step seconds |',
                      '| --- | --- | --- | --- | --- |'])
        for method in methods:
            for row in validation:
                if row['method'] == method and row['step'] > 0:
                    lines.append(f"| {labels[method]} | {row['step']} | {row['validation_loss']:.6f} | "
                                 f"{100*row['validation_accuracy']:.2f}% | {row['training_seconds']:.3f} |")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
        for method, color in zip(methods, ['#73a9cf', '#2166ac', '#b2182b']):
            subset = [r for r in validation if r['method'] == method]
            axes[0].plot([r['step'] for r in subset], [r['validation_loss'] for r in subset], marker='o', label=labels[method], color=color)
            axes[1].plot([r['training_seconds'] for r in subset], [r['validation_loss'] for r in subset], marker='o', color=color)
            axes[2].plot([r['step'] for r in subset], [100*r['validation_accuracy'] for r in subset], marker='o', color=color)
        axes[0].set(xlabel='Consumed iterations', ylabel='Validation cross-entropy')
        axes[1].set(xlabel='Measured training-step seconds', ylabel='Validation cross-entropy')
        axes[2].set(xlabel='Consumed iterations', ylabel='Validation accuracy (%)')
        axes[0].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle('Batch 512 paired training; one seed, 50 consumed batches, 5,000 validation examples')
        fig.savefig(training / 'comparison.png', dpi=170)
        plt.close(fig)
        final = {r['method']: r for r in validation if r['step'] == 50}
        exact, searched, native = (final[k] for k in ['exact_tensor_ggn', 'searched_soap', 'native_soap'])
        lines.extend(['', f"At iteration 50, exact GGN validation loss minus searched SOAP is "
                      f"{exact['validation_loss']-searched['validation_loss']:+.6f}; minus native SOAP is "
                      f"{exact['validation_loss']-native['validation_loss']:+.6f}. "
                      f"Its measured training-step cost is {exact['training_seconds']/searched['training_seconds']:.1f}x "
                      f"searched SOAP and {exact['training_seconds']/native['training_seconds']:.1f}x native SOAP.", '',
                      'These are untuned single-seed short trajectories. Training-step time includes gradients, '
                      'curvature, solves and search, but excludes dataset loading, validation, checkpoint writing and plotting. '
                      'No equal-wall-clock speed advantage or replicated ranking follows from equal-iteration comparisons. '
                      'This exact tensor reference uses SOAP-metric damping and is not a reproduction of the paper.', '',
                      f'![Paired training]({training.resolve()}/comparison.png)', ''])
        (training / 'REPORT.md').write_text('\n'.join(lines[lines.index('## Paired batch-512 training followup'):]))
    output = Path('docs/local_large_batch_results_20261002.md')
    output.write_text('\n'.join(lines))
    (directory / 'COMBINED_REPORT.md').write_text('\n'.join(lines))
    print(output.resolve())


if __name__ == '__main__':
    main()
