"""Report the saved output-information intervention, without further model calls."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from residual_curvature.provenance import checksum, write_json


METHODS = [
    ('soap', 'SOAP momentum + frozen map'),
    ('soap_current_gradient', 'SOAP frozen map + current gradient'),
    ('ggn_gradient', 'Dense GGN + current gradient'),
    ('ggn_momentum', 'Dense GGN + effective SOAP momentum'),
    ('ggn_diagonal_gradient', 'GGN diagonal in SOAP basis + gradient'),
    ('ggn_diagonal_momentum', 'GGN diagonal in SOAP basis + momentum'),
    ('empirical_fisher_raw', 'Dense empirical Fisher, raw'),
    ('empirical_fisher_matched', 'Dense empirical Fisher, trace matched'),
    ('empirical_fisher_diagonal_matched', 'Empirical Fisher diagonal in SOAP basis, trace matched'),
    ('ggn_directions_soap_norms', 'GGN directions / SOAP tensor norms'),
    ('soap_directions_ggn_norms', 'SOAP directions / GGN tensor norms'),
    ('sampled_1', 'Model-sampled labels, 1 per example'),
    ('sampled_4', 'Model-sampled labels, 4 per example'),
    ('sampled_16', 'Model-sampled labels, 16 per example'),
]


def read(directory, name):
    return json.loads((directory/name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    directory = args.run
    assert read(directory, 'complete.json')['completed'] and not (directory/'failure.json').exists()
    rows, diagnostics = read(directory, 'all_candidates.json'), read(directory, 'all_diagnostics.json')
    for name, digest in read(directory, 'source_archive_checksums.json').items():
        assert checksum(directory/'source'/name) == digest, name
    for name, digest in read(directory, 'parent_checkpoint_checksums.json').items():
        assert checksum(name) == digest, name
    for d in diagnostics:
        path = directory/f"{d['trajectory']}-{d['step']:04d}"
        for name, digest in read(path, 'artifacts.json').items():
            assert checksum(path/name) == digest, name
        assert d['numerical_gate_passed'] and d['search_gate_passed'] and d['nonmutation_passed']
    assert len(rows) == 102 and len(diagnostics) == 6
    assert all(r['search']['bracketed'] or r['search']['rejected'] for r in rows)
    def select(trajectory, step, name):
        return [r for r in rows if r['trajectory']==trajectory and r['step']==step and
                (r['name']==name or r['name'].startswith(name+'_repeat_'))]
    def values(trajectory, step, name):
        return [r['selected']['independent_mean_decrease'] for r in select(trajectory, step, name)]
    contrasts = []
    for trajectory in ['searched_soap', 'exact_tensor_ggn']:
        for step in [10, 25, 50]:
            for name, _ in METHODS:
                selected = select(trajectory, step, name)
                panel = np.mean([r['selected']['independent_decreases'] for r in selected], axis=0)
                ggn = np.array(select(trajectory, step, 'ggn_gradient')[0]['selected']['independent_decreases'])
                difference = panel-ggn
                contrasts.append({'trajectory': trajectory, 'step': step, 'name': name,
                                  'independent_mean_decrease': float(panel.mean()),
                                  'repeat_means': values(trajectory, step, name),
                                  'panel_differences_vs_ggn': difference.tolist(),
                                  'mean_difference_vs_ggn': float(difference.mean()),
                                  'panel_sample_standard_error': float(difference.std(ddof=1)/2)})
    action = max(v for d in diagnostics for v in d['exact_action_checks'].values())
    gradient = max(d['gradient_error'] for d in diagnostics)
    momentum = max(d['momentum_ray_error'] for d in diagnostics)
    solve = max(v for r in rows if r['solve'] for v in r['solve']['tensor_residuals'].values())
    assert max(action, gradient, momentum, solve) < 1e-8
    verification = {'source_parent_and_array_checksums_passed': True, 'nonmutation_passed': True,
                    'max_action_error': action, 'max_gradient_error': gradient,
                    'max_momentum_ray_error': momentum, 'max_solve_residual': solve,
                    'searches': len(rows), 'rejected_searches': sum(r['search']['rejected'] for r in rows)}
    write_json(directory/'verification.json', verification)
    write_json(directory/'contrasts.json', contrasts)

    lines = ['# What information separates SOAP from tensor GGN?', '',
             f'Run: `runs/{directory.name}`. The [prospective protocol](jacobian_information_protocol_20261005.md) '
             'selected six saved ViT states, batch 512, gamma .01, identical normalized inverse-SOAP damping '
             'and global scalar searches. The forward network and ordinary training gradient are unchanged. '
             'The four evaluation panels (2,048 examples) were unused in earlier experiments and are disjoint '
             'from construction/training/validation. They are reused across these six states; they do not '
             'select settings. This is a local intervention, not another training comparison.', '',
             '## Measurements', '',
             'Each entry is the independent mean loss decrease; positive is better. Model-sampled-label '
             'entries average two prespecified repeats, retaining both repeats and every panel in raw data. '
             'Every row gets the same global scalar-search budget, selected only on construction loss.']
    for trajectory, title in [('searched_soap', 'States from searched-SOAP training'),
                              ('exact_tensor_ggn', 'States from exact tensor-GGN training')]:
        lines.extend(['', f'### {title}', '', '| Candidate | Step 10 | Step 25 | Step 50 |',
                      '| --- | ---: | ---: | ---: |'])
        for name, label in METHODS:
            lines.append('| '+label+' | '+' | '.join(f'{np.mean(values(trajectory, step, name)):.6f}' for step in [10, 25, 50])+' |')
    def mean(trajectory, step, name):
        return float(np.mean(values(trajectory, step, name)))
    lines.extend(['', '## Observed interpretation', '',
                  'The strongest simple explanation is not supported uniformly: access to every output-Jacobian '
                  'direction is not necessary to recover most of the early local gain. At the searched-SOAP '
                  f"step-10 state, dense trace-matched empirical Fisher gives {mean('searched_soap',10,'empirical_fisher_matched'):.6f} "
                  f"independent decrease versus GGN {mean('searched_soap',10,'ggn_gradient'):.6f}; at step 25 these are "
                  f"{mean('searched_soap',25,'empirical_fisher_matched'):.6f} and {mean('searched_soap',25,'ggn_gradient'):.6f}. "
                  'The raw empirical-Fisher values are very similar, so oracle trace matching does not account '
                  'for these early improvements. This control uses per-example ordinary gradients and a dense '
                  'tensor covariance, not SOAP\'s usual aggregate-gradient statistic.', '',
                  'At those same two states, compressing either curvature source to a diagonal in SOAP\'s '
                  'basis removes much of the gain. Conversely, GGN tensor directions keep the early advantage '
                  'when assigned SOAP tensor norms. These interventions implicate within-tensor geometry '
                  'more strongly than simply copying a different allocation or global update length.', '',
                  'This is state dependent. On searched-SOAP step 50, current-gradient GGN and empirical Fisher '
                  'both increase independent loss. GGN with effective SOAP momentum improves it. On the GGN '
                  'trajectory at steps 25/50, current-gradient GGN beats dense empirical Fisher, but the frozen '
                  'SOAP map with the current gradient and the GGN diagonal are also strong. Changing momentum '
                  'alone is therefore not a universal explanation either; its interaction with the preconditioner '
                  'and the trajectory matters.', '',
                  'Model-sampled output directions approach the exact-GGN local measurements at early/middle '
                  'states as the number of draws rises. At the late GGN state, even 16 draws retains a noticeable '
                  'gap. This shows potential for extra output probes, without establishing a cheap representation '
                  'or a successful training algorithm. All sampled controls still use dense tensor matrices.', '',
                  'A direct next test of the training-gap explanation would compare paired dense empirical-Fisher '
                  'training with GGN under the same damping/search/RHS protocol, alongside a current-gradient '
                  'SOAP-map control. It has not been launched here. These controls also do not separately identify '
                  'EMA lag, basis refresh, inverse versus inverse-square-root scaling, or a new factorization.'])
    lines.extend(['', '## What each intervention identifies', '',
                  'Write C_bb = mean J_ib^T H_i J_ib, with H_i=diag(p_i)-p_i p_i^T. '
                  'The dense empirical Fisher is F_bb=mean g_ib g_ib^T, where '
                  'g_ib=J_ib^T(p_i-onehot(y_i)). It can be obtained from per-example ordinary backprop; '
                  'it does not require observing the Jacobian in all output directions. It nevertheless '
                  'retains all within-tensor parameter correlations. It is richer than SOAP\'s batch-gradient '
                  'statistics and is not an implementation of SOAP.', '',
                  'The dense C versus dense F intervention changes the output-information target while '
                  'holding the true gradient, dense representation, damping, partition and scalar budget fixed. '
                  'Raw F and per-tensor trace-matched F are both retained. The trace-matched control adds '
                  'oracle trace information; do not count it as a practical estimator. F is not an unbiased '
                  'GGN estimate for fixed observed labels. This intervention changes output weighting/rank '
                  'together, and cannot distinguish those finer mechanisms by itself.', '',
                  'The dense-versus-diagonal intervention keeps the curvature source and actual SOAP basis '
                  'fixed. The diagonal model is Q^-T diag(diag(Q^T C_bb Q)) Q^-1, or its F counterpart, '
                  'with physical tensor traces matched to C_bb. Thus a dense advantage at fixed source '
                  'demonstrates the value of correlations discarded by this representation. It does not '
                  'prove that a different factorization or basis could not retain those correlations cheaply.', '',
                  'The RHS intervention reconstructs an effective momentum vector from -P^-1 delta_SOAP, '
                  'normalized to the gradient norm. Applying -P returns the native SOAP ray, verified using '
                  'the actual, slightly nonorthogonal saved bases. Comparing GGN on this RHS with GGN on '
                  'the current gradient measures sensitivity to that change. SOAP without momentum remains '
                  'a separate control. The interaction means one cannot assign independent percentages of '
                  'the training gap to momentum and curvature.', '',
                  'For labels sampled from the current model, '
                  'E[(p-onehot(label))(p-onehot(label))^T]=diag(p)-pp^T. Their per-example gradient '
                  'outer products therefore estimate C. Nested counts 1/4/16, with two fixed repeats, '
                  'test extra output directions without changing the target or dense tensor representation. '
                  'The diagnostic reuses the full Jacobian to construct these matrices; measured time is '
                  'not the cost of a VJP-only estimator or a practical optimizer.', '',
                  'Norm transfers preserve each source tensor direction and assign the other method\'s '
                  'complete tensor norms before global search. They separate the local value of within-tensor '
                  'directions from tensor allocation. They do not prescribe universal Muon or GGN shares.', '',
                  '## Numerical checks and reproducibility', '',
                  f'All 75 tests passed before execution. Across six states, maximum exact tensor-action '
                  f'error was {action:.3e}, mean-gradient error {gradient:.3e}, native momentum-ray '
                  f'reconstruction error {momentum:.3e}, and tensor-solve residual {solve:.3e}. '
                  'Model, optimizer and Torch RNG were preserved. Source archives, all six parent '
                  'checkpoints and saved arrays passed checksum verification.', '',
                  f"All {len(rows)} searches were bracketed or explicitly rejected; "
                  f"{verification['rejected_searches']} selected rejection. Snapshot work took "
                  f"{sum(d['seconds'] for d in diagnostics):.2f} seconds. Maximum estimated tensor workspace was "
                  f"{max(d['construction']['estimated_workspace_bytes'] for d in diagnostics)/2**20:.2f} MiB; "
                  f"cumulative process peak RSS was {max(d['process_peak_rss_bytes'] for d in diagnostics)/2**20:.2f} MiB. "
                  'Exact/sample information is recomputed on the same construction batch; no damping '
                  'or rank was selected on panels.', '',
                  'Raw candidates, scalar trials and panels are in `all_candidates.json`; discrepancies '
                  'and trace scales are in `all_diagnostics.json`. Each state saves parameters, optimizer '
                  'context, exact C/F blocks, sampled labels and candidate directions in `arrays.pt`. '
                  '`contrasts.json` reports paired panel differences and descriptive standard errors, '
                  'not independent training-replicate uncertainty. Rebuild this report with:', '', '```sh',
                  f'.venv/bin/python scripts/summarize_jacobian_information.py --run runs/{directory.name}', '```', '',
                  '## Scope of a conclusion', '',
                  'A dense empirical-Fisher win would show that observing every output-Jacobian direction '
                  'is not necessary for that local improvement. A dense C/F advantage over both SOAP-coordinate '
                  'diagonals would support preserving within-tensor correlations. Momentum and direction/norm '
                  'controls further restrict the explanation. These are falsifiable local mechanisms, '
                  'not a unique causal decomposition of the previous 50-step training gap. A dense empirical '
                  'Fisher training arm was not run. Six related states and two probe repeats are not optimizer '
                  'replication; there is no conclusion about skip-specific information.', '',
                  'Primary context: [SOAP](https://arxiv.org/abs/2409.11321) and '
                  '[the full/layerwise GGN study](https://arxiv.org/abs/2510.09378).', ''])

    os.environ.setdefault('MPLCONFIGDIR', str((directory/'.matplotlib').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((directory/'.cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotted = [('soap', 'SOAP'), ('soap_current_gradient', 'SOAP map / gradient'),
               ('ggn_momentum', 'GGN / momentum'), ('ggn_diagonal_gradient', 'GGN diagonal / gradient'),
               ('empirical_fisher_matched', 'Dense empirical Fisher'),
               ('empirical_fisher_diagonal_matched', 'EF diagonal'),
               ('sampled_1', 'Sampled labels: 1'), ('sampled_4', 'Sampled labels: 4'),
               ('sampled_16', 'Sampled labels: 16'), ('ggn_gradient', 'Exact GGN / gradient')]
    colors = ['#2166ac', '#76a5c8', '#9467bd', '#b39dcf', '#26845b', '#98c6a9', '#e8baba', '#d98888', '#c65555', '#a51c30']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True, constrained_layout=True)
    for i, trajectory in enumerate(['searched_soap', 'exact_tensor_ggn']):
        for j, step in enumerate([10, 25, 50]):
            ax = axes[i, j]
            vals = [values(trajectory, step, name) for name, _ in plotted]
            ax.barh(range(len(plotted)), [np.mean(v) for v in vals], color=colors,
                    xerr=[(max(v)-min(v))/2 for v in vals], capsize=2)
            ax.axvline(0, color='black', linewidth=.7)
            ax.set(title=f'{"SOAP" if i==0 else "GGN"} trajectory: step {step}', xlabel='Independent loss decrease')
            ax.grid(axis='x', alpha=.2)
    axes[0, 0].set_yticks(range(len(plotted)), [label for _, label in plotted])
    axes[0, 0].invert_yaxis()
    fig.suptitle('Output information versus within-tensor representation\nHigher is better; sampled-label bars show mean and half-range of two repeats; all use global search')
    fig.savefig(directory/'comparison.png', dpi=170)
    fig.savefig(directory/'comparison.pdf')
    plt.close(fig)
    report = '\n'.join(lines)
    (directory/'REPORT.md').write_text(report.replace('(jacobian_information_protocol_20261005.md)',
         '(../../docs/jacobian_information_protocol_20261005.md)')+'\n![Information comparison](comparison.png)\n')
    Path('docs/jacobian_information_results_20261005.md').write_text(report+f'\n![Information comparison](../runs/{directory.name}/comparison.png)\n')
    shutil.copyfile(__file__, directory/'analysis_runner.py')
    write_json(directory/'analysis_artifacts.json', {name: checksum(directory/name) for name in
        ['REPORT.md', 'comparison.png', 'comparison.pdf', 'contrasts.json', 'verification.json', 'analysis_runner.py']})
    print(json.dumps(verification, indent=2))
    for trajectory in ['searched_soap', 'exact_tensor_ggn']:
        print(trajectory)
        for name, label in METHODS:
            print(label, [round(float(np.mean(values(trajectory, s, name))), 6) for s in [10, 25, 50]])


if __name__ == '__main__':
    main()
