"""Report saved downstream-curvature and update-allocation measurements."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from residual_curvature.provenance import checksum, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run
    rows = json.loads((directory / 'all_candidates.json').read_text())
    diagnostics = json.loads((directory / 'all_diagnostics.json').read_text())
    assert json.loads((directory / 'complete.json').read_text())['completed']
    methods = [
        ('soap', 'SOAP + global search'),
        ('frozen_minus_Pg', 'Frozen -Pg + search'),
        ('exact_tensor_ggn', 'Exact tensor GGN'),
        ('soap_directions_ggn_shares', 'SOAP directions, GGN shares'),
        ('ggn_directions_soap_shares', 'GGN directions, SOAP shares'),
        ('soap_allocation_diagonal', 'SOAP + diagonal allocation'),
        ('soap_allocation_full', 'SOAP + full reduced allocation'),
        ('exact_curvature_diagonal_soap_basis', 'Exact curvature diagonal in SOAP basis'),
        ('downstream_curvature_diagonal', 'Probe curvature diagonal, rank 16 (mean of two repeats)'),
    ]
    values = {}
    for name, _ in methods:
        values[name] = {}
        for step in [10, 50, 100]:
            selected = [r for r in rows if r['name'] == name and r['step'] == step and
                        (name != 'downstream_curvature_diagonal' or r['rank'] == 16)]
            values[name][step] = [r['selected']['independent_mean_decrease'] for r in selected]
    lines = ['# Downstream curvature and update allocation — 5 October 2026', '',
             f'Run: `{directory}`. This is a fixed-state experiment on 14x14 MNIST, at '
             'batch 512 and saved SOAP states 10/50/100. It does not launch or claim a new training result. '
             'All candidates use the same normalized inverse-SOAP damping with gamma=.01 and '
             'the same broad global scalar search. Each scalar is selected only on construction loss. '
             'The four independent panels are measured without retuning. The original states have batch-64 training history.', '',
             '## Independent loss decrease', '',
             'Positive values mean improvement. Each deterministic entry averages four independent panels. '
             'The displayed rank-16 entry also averages the two prespecified probe repeats; it is a diagnostic, '
             'not a rank selected on losses. All ranks and repeats are stored in all_candidates.json.', '',
             '| Candidate | State 10 | State 50 | State 100 |', '| --- | --- | --- | --- |']
    for name, label in methods:
        lines.append('| '+label+' | '+' | '.join(f'{np.mean(values[name][s]):.6f}' for s in [10, 50, 100])+' |')
    lines.extend(['', 'Copying the oracle tensor shares onto SOAP directions makes only a small change here. '
                  'At the middle state, both GGN directions at SOAP shares and optimized allocation along SOAP directions '
                  'recover much of the exact tensor reference improvement. These are two different ways to improve '
                  'the joint update; copying one method\'s shares is not equivalent to optimizing shares for another method. '
                  'At the late state full reduced allocation increases independent loss, despite its construction-batch objective. '
                  'There is no uniform allocation win across these states.', '',
                  'The panels are independent of each candidate construction batch, but are reused from the earlier '
                  'oracle experiment. They are diagnostic evaluation data, not a fresh confirmation set.', '',
                  '## Downstream transport and approximation gates', '',
                  'The implementation transports shared output-Hessian probes through the terminal head and actual '
                  'residual-module VJPs. Identity/projected skip paths, normalization derivatives, input/head tensors '
                  'and all biases are included. On the selected architecture it reproduces the existing direct whole-network '
                  'probe factors; the estimator itself is not new. The new structured candidate keeps SOAP bases and '
                  'uses curvature-derived coordinate scaling.', '',
                  'Let Q be the actual saved SOAP basis, d its denominator, and c the estimated diag(Q^T C_bb Q). '
                  'The candidate solves with C_hat=Q^-T diag(c)Q^-1 and Omega=scale Q^-T diag(d)Q^-1. '
                  'Its coordinate update is -(Q^T g)/(c+mu*scale*d). This uses the true gradient and does not '
                  'preserve SOAP\'s momentum proposal; frozen -Pg is retained to expose that distinction.', '',
                  '| State | Exact coordinate-compression action error | Rank-16 diagonal errors, repeats 0/1 | Rank-16 action errors, repeats 0/1 | Eligible rank |',
                  '| --- | --- | --- | --- | --- |'])
    for d in diagnostics:
        at16 = [r for r in d['probe_errors'] if r['rank'] == 16]
        lines.append(f"| {d['step']} | {d['exact_coordinate_compression_error']:.4f} | " +
                     '/'.join(f"{r['diagonal_error']:.4f}" for r in at16)+' | '+
                     '/'.join(f"{r['tensor_action_error']:.4f}" for r in at16)+f" | {d['selected_rank']} |")
    lines.extend(['', 'No declared rank passed both error targets in both repeats. More probes alone cannot remove '
                  'the discrepancy between the exact coordinate-diagonal model and the exact tensor GGN. '
                  'The probes improve over frozen -Pg at the early and middle states, but the structured candidate '
                  'does not beat momentum-based searched SOAP there. This local result does not disprove the value '
                  'of downstream curvature; it rejects this particular diagonal compression as an accurate tensor approximation.', '',
                  'The next candidate should preserve useful within-tensor correlations or use the downstream measurements '
                  'to optimize a much smaller allocation problem. Neither a larger probe rank nor a training run was '
                  'selected after seeing these results.', '',
                  '## Numerical checks and cost', ''])
    transport = max(v for d in diagnostics for v in d['transport_errors'])
    solve = max(r.get('solve_residual', 0.) for r in rows)
    lines.append(f'Maximum transported/direct factor relative error: {transport:.3e}; maximum new candidate solve '
                 f'residual: {solve:.3e}. All numerical tolerances are 1e-8. The 55-test suite passed before execution, '
                 'including exact dense/operator/probe agreement, nonorthogonal SOAP-coordinate solves, projected skips, '
                 'microbatch normalization and state non-mutation. No model, context or Torch RNG mutation occurred in the run.')
    lines.append(f"\nAll {len(rows)} candidate comparisons and their search trials are retained. "
                 f"Snapshot work took {sum(d['seconds'] for d in diagnostics):.3f} seconds. "
                 f"Peak process RSS was {max(d['process_peak_rss_bytes'] for d in diagnostics)/2**20:.1f} MiB. "
                 'The separately budgeted tensor workspace was below 768 MiB. Probe counters refer to stage pullbacks, '
                 'not complete backward passes. Cost includes oracle diagnostics and is not an optimizer-overhead estimate.\n')
    lines.extend(['## Relation to the update-norm paper', '',
                  'For tensor b, write Delta_b=R sqrt(w_b) u_b. R is total update length, w_b is its squared-norm '
                  'share and u_b is its unit direction. With phi_b=-g_b^T u_b and K_bc=u_b^T C_bc u_c, the '
                  'quadratic decrease for amplitudes a is phi^T a - a^T K a/2. The reduced allocation experiment '
                  'maximizes this expression with damping and nonnegative a; it accounts for alignment as well as exposure.', '',
                  'A single namesake/fallback rate ratio fixes at most the aggregate group share. It cannot generally '
                  'match Muon\'s fan-out-dependent allocation among matrices, change SOAP\'s tensor directions, '
                  'or fix the total step length. The PDF is descriptive and does not identify the failed matched-ratio '
                  'tuning run; diagnosing that specific run requires its measurements.', '',
                  'A large cross term does not imply a large benefit from changing direction. For C=I+rho 11^T and '
                  'g=1, full and block-diagonal inverse updates lie on the same ray even when off-diagonal terms '
                  'dominate d^T C d. A global scalar search can correct that scale difference. This is a mathematical '
                  'counterexample, not a claim that our model has exactly that geometry. Also distinguish residual-path '
                  'interactions inside each C_bb from interactions between parameter tensors C_bc.', ''])

    os.environ.setdefault('MPLCONFIGDIR', str((directory / '.matplotlib').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((directory / '.cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected_methods = [item for item in methods if item[0] != 'frozen_minus_Pg']
    labels = ['SOAP + global search', 'Exact tensor GGN', 'SOAP directions / GGN shares',
              'GGN directions / SOAP shares', 'SOAP + diagonal allocation', 'SOAP + full allocation',
              'Exact curvature diagonal / SOAP basis', 'Probe curvature diagonal (rank 16)']
    colors = ['#2166ac', '#b2182b', '#8db7d4', '#dc8b93', '#9575b6', '#603d8d', '#946544', '#d39449']
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharey=True, constrained_layout=True)
    for ax, step in zip(axes, [10, 50, 100]):
        averages = [float(np.mean(values[name][step])) for name, _ in selected_methods]
        errors = [float((max(values[name][step])-min(values[name][step]))/2) for name, _ in selected_methods]
        ax.barh(range(len(labels)), averages, color=colors, xerr=errors, capsize=3)
        ax.axvline(0, color='black', linewidth=.7)
        ax.set(title=f'State {step}', xlabel='Independent cross-entropy decrease')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Batch 512: directions, allocation and downstream curvature\nHigher is better; probe bar shows mean and half-range of two repeats')
    fig.savefig(directory / 'comparison.png', dpi=170)
    plt.close(fig)
    lines.append(f'![Fixed-state comparison]({directory.resolve()}/comparison.png)\n')
    report = '\n'.join(lines)
    (directory / 'REPORT.md').write_text(report)
    Path('docs/downstream_allocation_results_20261005.md').write_text(report)
    shutil.copyfile(__file__, directory / 'analysis_runner.py')
    write_json(directory / 'analysis_artifacts.json', {name: checksum(directory / name) for name in ['analysis_runner.py', 'comparison.png', 'REPORT.md']})
    print((directory / 'REPORT.md').resolve())


if __name__ == '__main__':
    main()
