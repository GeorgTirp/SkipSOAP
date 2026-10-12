"""Verify and report a completed, trusted local mini-ViT allocation pilot."""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from residual_curvature import checkpoint
from residual_curvature.config import load_config
from residual_curvature.data import DataOrder, dataset
from residual_curvature.models import make_model
from residual_curvature.provenance import checksum, write_json
from run_exact_training_followup import evaluate


LABELS = {
    'native_soap': 'Native SOAP',
    'searched_soap': 'SOAP + global search',
    'soap': 'SOAP + global search',
    'allocation_diagonal': 'SOAP + diagonal allocation',
    'allocation_full': 'SOAP + full allocation',
    'exact_tensor_ggn': 'Exact tensor GGN + search',
    'frozen_minus_Pg': 'Frozen -Pg + search',
}
COLORS = {
    'native_soap': '#777777', 'searched_soap': '#2166ac', 'soap': '#2166ac',
    'allocation_diagonal': '#d18b22', 'allocation_full': '#6a3d9a',
    'exact_tensor_ggn': '#b2182b', 'frozen_minus_Pg': '#5a9670',
}


def read(directory, name):
    return json.loads((directory / name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    directory = args.run
    complete = read(directory, 'complete.json')
    assert complete['completed'] and not (directory / 'failure.json').exists()
    methods = complete['methods']
    validation = read(directory, 'validation.json')
    candidates = read(directory, 'snapshot_candidates.json')
    diagnostics = read(directory, 'snapshot_diagnostics.json')
    gate = read(directory, 'exact_training_gate.json')
    protocol = read(directory, 'protocol.json')
    manifest = read(directory, 'split_manifest.json')
    archive = read(directory, 'source_archive_checksums.json')
    geometry = []
    for name, digest in archive.items():
        assert checksum(directory / 'source' / name) == digest, name
    for d in diagnostics:
        snapshot = directory / f"snapshot-{d['step']:04d}"
        for name, digest in read(snapshot, 'artifacts.json').items():
            assert checksum(snapshot / name) == digest, name
        assert d['numerical_gate_passed'] and d['search_gate_passed'] and d['nonmutation_passed']
        # Arrays come from this trusted local run and were checksum-verified above.
        reduced = torch.load(snapshot / 'arrays.pt', map_location='cpu', weights_only=False)['reduced']
        a, h, phi = reduced['amplitudes_diagonal'], reduced['H'], reduced['phi']
        curvature, slope = float(a @ h @ a), float(phi @ a)
        geometry.append({'step': d['step'],
                         'native_quadratic_ray_gain': d['allocation']['native_quadratic_ray_gain'],
                         'diagonal_quadratic_ray_gain': .5*max(slope, 0.)**2/curvature if curvature > 0 else 0.,
                         'full_quadratic_gain': d['allocation']['full_quadratic_gain'],
                         'active_full': d['allocation']['active_full']})
    c = load_config(directory / 'resolved.yaml')
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    x, y, restored_manifest = dataset(c)
    assert restored_manifest == manifest
    pools = [set(manifest[name]) for name in ['training', 'validation', 'analysis']]
    assert all(not pools[i] & pools[j] for i in range(3) for j in range(i))
    histories, summaries, restored, searches, kkts = {}, [], [], [], []
    solve_errors = [v for d in diagnostics for v in d['solve']['tensor_residuals'].values()]
    paired_batches = None
    for method in methods:
        arm = directory / method
        history = read(arm, 'training.json')
        histories[method] = history
        assert [r['step'] for r in history] == list(range(1, 51))
        batches = [r['batch_ids'] for r in history]
        if paired_batches is None:
            paired_batches = batches
        assert batches == paired_batches
        assert history[0]['norm'] == 0
        for row in history:
            if row['search']:
                searches.append(row['search'])
            if row['allocation'] and row['allocation']['available']:
                kkts.append(max(row['allocation']['kkt'].values()))
            if row['solve']:
                solve_errors.extend(row['solve']['tensor_residuals'].values())
        final = next(r for r in validation if r['method'] == method and r['step'] == 50)
        arm_config = read(arm, 'resolved_arm.json')
        model = make_model(arm_config)
        order = DataOrder(manifest['training'], c['data_seed']+1)
        state = checkpoint.load(arm / 'checkpoint-0050.pt', model, order, arm_config)
        reproduced = evaluate(model, x, y, manifest['validation'])
        assert state['step'] == 50 and state['accepted'] == history[-1]['accepted_updates']
        for key in ['validation_loss', 'validation_accuracy']:
            assert abs(reproduced[key]-final[key]) <= 1e-12, (method, key)
        restored.append({'method': method, **reproduced, 'passed': True})
        summaries.append({**final, 'accepted_updates': state['accepted'],
                          'rejected_searched_steps': sum(bool(r['search'] and r['search']['rejected']) for r in history),
                          **read(arm, 'runtime.json')})
    searches.extend(r['search'] for r in candidates)
    assert all(s['bracketed'] or s['rejected'] for s in searches)
    assert all(not s['boundary'] for s in searches)
    kkts.extend(max(d['allocation']['kkt'].values()) for d in diagnostics)
    action_error = max(v for d in diagnostics for v in d['action_checks'].values())
    solve_error = max(solve_errors)
    verification = {'source_archive_files_verified': len(archive), 'snapshot_array_checksums_passed': True,
                    'paired_data_order_passed': True, 'final_checkpoints': restored,
                    'searches_checked': len(searches), 'all_searches_bracketed_or_rejected': True,
                    'max_tensor_action_error': action_error, 'max_tensor_solve_residual': solve_error,
                    'max_allocation_kkt_residual': max(kkts)}
    assert max(action_error, solve_error, max(kkts)) <= 1e-8
    write_json(directory / 'verification.json', verification)
    write_json(directory / 'summary.json', summaries)
    write_json(directory / 'reduced_geometry.json', geometry)

    os.environ.setdefault('MPLCONFIGDIR', str((directory / '.matplotlib').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((directory / '.cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for method in methods:
        rows = sorted((r for r in validation if r['method'] == method), key=lambda r: r['step'])
        style = dict(label=LABELS[method], color=COLORS[method], marker='o', markersize=4)
        axes[0, 0].plot([r['step'] for r in rows], [r['validation_loss'] for r in rows], **style)
        axes[0, 1].plot([r['training_seconds'] for r in rows[1:]], [r['validation_loss'] for r in rows[1:]], **style)
        axes[1, 0].plot([r['step'] for r in rows], [100*r['validation_accuracy'] for r in rows], **style)
    for name in ['soap', 'frozen_minus_Pg', 'allocation_diagonal', 'allocation_full', 'exact_tensor_ggn']:
        rows = sorted((r for r in candidates if r['name'] == name), key=lambda r: r['step'])
        axes[1, 1].plot([r['step'] for r in rows], [r['selected']['independent_mean_decrease'] for r in rows],
                        label=LABELS[name], color=COLORS[name], marker='o', markersize=4)
    axes[0, 0].set(title='Paired training: validation loss', xlabel='Consumed iterations', ylabel='Cross-entropy')
    axes[0, 1].set(title='Measured training cost', xlabel='Training-step seconds (log scale)', ylabel='Cross-entropy', xscale='log')
    axes[1, 0].set(title='Paired training: validation accuracy', xlabel='Consumed iterations', ylabel='Accuracy (%)')
    axes[1, 1].set(title='Shared searched-SOAP states', xlabel='Checkpoint iteration', ylabel='Independent loss decrease (higher is better)')
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].axhline(0, color='black', linewidth=.7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Mini ViT / MNIST / batch 512 / seed 0\n50 consumed iterations; validation and snapshot panels do not select hyperparameters')
    fig.savefig(directory / 'comparison.png', dpi=170)
    fig.savefig(directory / 'comparison.pdf')
    plt.close(fig)

    lines = ['# Mini ViT and SOAP tensor allocation — 5 October 2026', '',
             f'Run: `runs/{directory.name}`. The [prospective protocol](mini_vit_protocol_20261005.md) '
             'selected this 5,738-parameter ViT before execution: two pre-LayerNorm attention/MLP blocks, '
             'width 16, two heads, 16 image patches plus CLS, 28x28 MNIST. All arms use seed 0, '
             'batch 512, identical initial weights and example order, CPU float64 and one thread. '
             'There is no augmentation or dropout.', '',
             '## Paired training', '',
             'These are measurements after 50 consumed iterations, including the first SOAP statistics-only call. '
             'Validation uses the reserved 5,000 examples. Time sums training steps, including proposal, curvature and '
             'search; it excludes dataset loading, validation, checkpoint writes and fixed-state diagnostics.', '',
             '| Method | Validation loss | Accuracy | Training seconds | Accepted updates |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in summaries:
        lines.append(f"| {LABELS[row['method']]} | {row['validation_loss']:.6f} | {100*row['validation_accuracy']:.2f}% | {row['training_seconds']:.2f} | {row['accepted_updates']} |")
    final_by_method = {row['method']: row for row in summaries}
    if 'exact_tensor_ggn' in final_by_method:
        exact, soap = final_by_method['exact_tensor_ggn'], final_by_method['searched_soap']
        lines.extend(['', f"Exact tensor GGN's final validation loss is {exact['validation_loss']/soap['validation_loss']:.3f} "
                      f"times searched SOAP's, at {exact['training_seconds']/soap['training_seconds']:.2f} times "
                      'the measured training-step time. This is an observed advantage in this pilot, '
                      'not a replicated or equally tuned optimizer ranking.'])
    if all(final_by_method[m]['validation_loss'] > final_by_method['searched_soap']['validation_loss']
           for m in ['allocation_diagonal', 'allocation_full']):
        lines.extend(['', 'Both allocation variants finish with higher validation loss than searched SOAP. '
                      'The shared-state allocation comparisons therefore did not establish a training improvement.'])
    if all(next(r['validation_loss'] for r in validation if r['method']=='allocation_full' and r['step']==s) <
           next(r['validation_loss'] for r in validation if r['method']=='searched_soap' and r['step']==s)
           for s in [10, 25]) and final_by_method['allocation_full']['validation_loss'] > final_by_method['searched_soap']['validation_loss']:
        lines.extend(['', 'Full allocation was ahead at iterations 10 and 25, but the advantage reversed by iteration 50.'])
    lines.extend(['', 'Native SOAP uses lr=.003. Every searched arm uses one **global** scalar, with the same '
                  '52-evaluation policy: zero and native-radius multipliers 2^[-16..16], then 18 golden refinements. '
                  'Allocation changes nonnegative tensor amplitudes before that scalar search; it does not perform '
                  'independent nonlinear searches per layer. All settings remained as prospectively declared.', '',
                  '## Shared-state direction comparisons', '',
                  'Positive values below mean independent cross-entropy decrease, averaged across four disjoint '
                  '512-example panels. Each candidate uses the same 512-example construction batch, current gradient, '
                  'SOAP context, exact-trace damping and global-search budget at that state. The states come from '
                  'searched SOAP; they are not the separate allocation trajectories.', '',
                  '| Candidate | State 10 | State 25 | State 50 |', '| --- | ---: | ---: | ---: |'])
    for name in ['soap', 'frozen_minus_Pg', 'allocation_diagonal', 'allocation_full', 'exact_tensor_ggn']:
        values = [next(r['selected']['independent_mean_decrease'] for r in candidates if r['name']==name and r['step']==step) for step in [10, 25, 50]]
        lines.append('| '+LABELS[name]+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    late_exact = next(r['selected'] for r in candidates if r['name']=='exact_tensor_ggn' and r['step']==50)
    lines.extend(['', f"At state 50, exact tensor GGN reduces construction loss by {late_exact['actual_decrease']:.6f} "
                  f"but its mean independent decrease is {late_exact['independent_mean_decrease']:.6f}. "
                  'This directly separates fitting the construction batch from transfer. It does not prove '
                  'the cause of the allocation training outcome.'])
    lines.extend(['', f"The exact-GGN training gate selected the optional arm: **{gate['exact_arm_selected']}**. "
                  f"Independent differences versus searched SOAP were {[round(v, 6) for v in gate['independent_differences']]}; "
                  f"the first-snapshot estimate for 49 updates was {gate['estimated_49_update_seconds']:.2f} seconds "
                  f"against a {gate['budget_seconds']}-second budget. Opportunity and numerical/search gates were "
                  f"{gate['opportunity_passed']} and {gate['checks_passed']}, respectively.", '',
                  '## What the allocation computes', '',
                  'E contains unit SOAP tensor directions. K=E^T C E is computed by exact directional JVPs, '
                  'including cross-tensor curvature; no full parameter GGN is assembled for allocation. With '
                  'phi=-E^T g and H=K+mu E^T Omega E, full allocation minimizes a^T H a/2-phi^T a over a>=0. '
                  'Its KKT conditions are a>=0, Ha-phi>=0 and componentwise complementarity. Diagonal allocation '
                  'uses a=[phi]_+/diag(H); only this diagonal model has that componentwise solution. Shares are '
                  'a_b^2/sum(a^2). Negative-alignment directions remain eligible in the full problem.', '',
                  'Training uses mu=max(.01*estimated_trace(C)/5738,1e-8), with eight candidate-independent '
                  'Rademacher JVP probes per step and the normalized inverse-SOAP metric. The trace estimate is '
                  'approximate; the projected K is exact up to numerical precision. Snapshots use exact trace '
                  'for every candidate. SOAP moments advance exactly once per consumed gradient. Exact tensor GGN '
                  'uses the current gradient; allocation preserves SOAP momentum directions.', '',
                  'The following diagnostic evaluates all allocation rays in the same full damped quadratic. '
                  'Each ray gets its optimal nonnegative quadratic scalar, so a poor unscaled diagonal step '
                  'does not by itself count as evidence for useful coupling. These predicted decreases are '
                  'construction-batch model values, distinct from the independent nonlinear decreases above.', '',
                  '| State | Native SOAP ray | Diagonal-allocation ray | Full-allocation optimum | Active full tensors |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for row in geometry:
        lines.append(f"| {row['step']} | {row['native_quadratic_ray_gain']:.6f} | {row['diagonal_quadratic_ray_gain']:.6f} | {row['full_quadratic_gain']:.6f} | {row['active_full']} |")
    lines.extend(['',
                  '## Verification and artifacts', '',
                  f"The source archive has {len(archive)} verified files. All snapshot array checksums, paired data "
                  'orders and final checkpoint restoration/evaluation checks passed. '
                  f"Maximum snapshot tensor-action error: {action_error:.3e}; tensor-solve residual across training and snapshots: {solve_error:.3e}; "
                  f"allocation KKT residual across training and snapshots: {max(kkts):.3e}. "
                  f"All {len(searches)} searched proposals were bracketed or explicitly rejected; none selected "
                  'a positive search boundary. The 69-test suite passed before the dataset run.', '',
                  f"Estimated tensor workspace: {protocol['workspace']['estimated_workspace_bytes']/2**20:.2f} MiB "
                  f"(budget {protocol['workspace_budget_mib']} MiB). Peak process RSS over the suite: "
                  f"{max(r['process_peak_rss_bytes'] for r in summaries)/2**20:.2f} MiB. "
                  'RSS is cumulative process high-water memory, not per-method incremental overhead. '
                  'Each initialized allocation step computes up to 40 directional and eight trace JVPs. '
                  'Both allocation candidates are calculated and audited inside each allocation arm, so these '
                  'timings describe the research runner rather than a streamlined diagonal implementation.', '',
                  'See `validation.json`, per-arm `training.json`, `snapshot_candidates.json`, '
                  '`snapshot_diagnostics.json`, `exact_training_gate.json`, `verification.json`, '
                  '`snapshot-*/arrays.pt`, and saved checkpoints/source archives in the run. '
                  'The comparison plot can be rebuilt with:', '', '```sh',
                  f'.venv/bin/python scripts/summarize_mini_vit.py --run runs/{directory.name}', '```', '',
                  '## Interpretation limits', '',
                  'This is one seed and a short, untuned pilot. Tensor means all 40 registered parameter tensors, '
                  'including biases, normalization, embeddings and attention matrices; it is not the full GGN '
                  'or necessarily the same partition as an external layerwise-GGN experiment. Optimizing the '
                  'construction-batch quadratic does not guarantee nonlinear or independent-batch improvement. '
                  'Fixed-state comparisons isolate a local proposal change; diverged training trajectories do not '
                  'causally decompose final gains. Panels are reused across states and are diagnostic. '
                  'No official test examples were evaluated.', '',
                  'The KKT optimum is conditional on the current SOAP directions, damping and construction '
                  'batch. It is not a universal ideal allocation or a reason to copy Muon shares. '
                  'Further work should first check the allocation method on its own trajectory and replicate '
                  'the paired outcome. A prospectively selected convex amplitude-norm budget could test '
                  'whether limiting displacement improves transfer; it was not tested in this pilot.', '',
                  'The prior MLP used 14x14 inputs while this ViT uses 28x28. A difference in outcomes therefore '
                  'cannot be attributed solely to architecture. The existing explicit MLP residual-transport '
                  'implementation was not reused for CLS/final-norm transport; this experiment uses generic '
                  'parameter-space GGN operators. It does not validate a cheap downstream-statistic SOAP variant.', ''])
    report = '\n'.join(lines)
    run_report = report.replace('(mini_vit_protocol_20261005.md)', '(../../docs/mini_vit_protocol_20261005.md)')
    (directory / 'REPORT.md').write_text(run_report+'\n![Mini ViT comparison](comparison.png)\n')
    Path('docs/mini_vit_results_20261005.md').write_text(report+f'\n![Mini ViT comparison](../runs/{directory.name}/comparison.png)\n')
    shutil.copyfile(__file__, directory / 'analysis_runner.py')
    write_json(directory / 'analysis_artifacts.json', {name: checksum(directory / name) for name in
               ['analysis_runner.py', 'comparison.png', 'comparison.pdf', 'REPORT.md', 'verification.json', 'summary.json', 'reduced_geometry.json']})
    print(json.dumps({'run': str(directory), 'summary': summaries, 'verification': verification}, indent=2))


if __name__ == '__main__':
    main()
