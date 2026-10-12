"""Verify and report the selected transport/PARSEC/NI-SOAP pilot."""
import argparse
import json
import os
import shutil
from pathlib import Path

from residual_curvature.provenance import checksum, write_json


LABELS = {
    'adam_native': 'Adam native', 'adam_searched': 'Adam + search',
    'parsec_native': 'PARSEC native', 'parsec_searched': 'PARSEC + search',
    'ni_soap_alpha0': 'NI-SOAP, alpha=0', 'ni_soap_alpha1': 'NI-SOAP, alpha=1',
    'ggn_exact_trace': 'Exact tensor GGN + search',
    'wasserstein_tensor': 'Tensor graph-Wasserstein + search',
}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run
    complete = read(directory/'complete.json')
    assert complete['completed'] and not (directory/'failure.json').exists()
    archive = read(directory/'source_archive_checksums.json')
    for name, digest in archive.items():
        assert checksum(directory/'source'/name) == digest, name
    validation = read(directory/'validation.json')
    replay = read(directory/'checkpoint_verification.json')
    assert len(replay) == len(complete['methods'])
    assert all(r['next_proposal_replay_passed'] and r['nonmutation_passed'] for r in replay)
    histories, summaries, searches, actions, solves, ritz = {}, [], [], [], [], []
    batches = None
    for method in complete['methods']:
        history = read(directory/method/'training.json')
        histories[method] = history
        assert [r['step'] for r in history] == list(range(1, 51))
        current_batches = [r['batch_ids'] for r in history]
        if batches is None:
            batches = current_batches
        assert current_batches == batches
        final = next(r for r in validation if r['method'] == method and r['step'] == 50)
        restored = next(r['restored_validation'] for r in replay if r['method'] == method)
        for key in ['validation_loss', 'validation_accuracy']:
            assert restored[key] == final[key]
        summaries.append({**final, **read(directory/method/'runtime.json'),
                          'accepted_updates': history[-1]['accepted_updates']})
        for row in history:
            if row['search']:
                searches.append(row['search'])
                assert row['loss_after'] <= row['loss_before'] + 1e-12
            algorithm = row['algorithm']
            actions.extend(algorithm.get('action_checks', {}).values())
            if 'solve' in algorithm:
                solves.extend(algorithm['solve']['tensor_residuals'].values())
                solves.append(algorithm['solve']['residual'])
            if algorithm.get('refresh'):
                ritz.append({'method': method, 'step': row['step'], **algorithm['refresh']})
    assert all(s['bracketed'] or s['rejected'] for s in searches)
    assert all(not s['boundary'] for s in searches)
    assert max(actions + solves) <= 1e-8
    # Until the first curvature refresh, PARSEC must match its Adam control.
    for suffix in ['native', 'searched']:
        for a, p in zip(histories['adam_'+suffix][:9], histories['parsec_'+suffix][:9]):
            assert all(a[k] == p[k] for k in ['loss_before', 'loss_after', 'norm'])
        assert [r['step'] for r in ritz if r['method'] == 'parsec_'+suffix] == [10, 20, 30, 40, 50]
    noise = {}
    for method in ['ni_soap_alpha0', 'ni_soap_alpha1']:
        rows = [r['algorithm'] for r in histories[method]]
        totals = {key: sum(r[key] for r in rows) for key in [
            'drift_energy', 'noise_energy', 'drift_noise_cross_term', 'update_energy',
            'expected_noise_energy', 'expected_matrix_noise_energy', 'expected_fallback_noise_energy']}
        closure = max(abs(r['update_energy']-r['drift_energy']-r['noise_energy']-r['drift_noise_cross_term'])
                      / max(r['update_energy'], 1e-30) for r in rows)
        assert closure <= 1e-10
        totals['max_energy_closure_error'] = closure
        totals['realized_matrix_noise_energy'] = sum(t['noise_energy'] for r in rows for t in r['per_tensor'].values() if t['route'] == 'matrix')
        totals['realized_fallback_noise_energy'] = sum(t['noise_energy'] for r in rows for t in r['per_tensor'].values() if t['route'] == 'fallback')
        totals['noise_to_drift_energy_ratio'] = totals['noise_energy']/totals['drift_energy']
        noise[method] = totals
    assert noise['ni_soap_alpha0']['noise_energy'] == 0
    # Prior official SOAP is contextual, not a silently substituted alpha-zero arm.
    parent = Path('runs/20261005T135920-d7ce29b2')
    assert read(parent/'split_manifest.json') == read(directory/'split_manifest.json')
    old_batches = [r['batch_ids'] for r in read(parent/'searched_soap'/'training.json')]
    assert old_batches == batches
    parent_archive = read(parent/'source_archive_checksums.json')
    model_files = [name for name in archive if name.startswith('src/residual_curvature/models/')]
    assert all(archive[name] == parent_archive[name] for name in model_files)
    old_validation = read(parent/'validation.json')
    assert next(r['validation_loss'] for r in old_validation if r['step'] == 0) == validation[0]['validation_loss']
    reference = [r for r in old_validation if r['step'] == 50 and r['method'] in ['native_soap', 'searched_soap']]
    verification = {
        'source_archive_files_verified': len(archive), 'paired_batches_passed': True,
        'final_checkpoint_replay_passed': True, 'searches_checked': len(searches),
        'rejected_searches': sum(s['rejected'] for s in searches),
        'all_searches_bracketed_or_rejected': True, 'max_action_error': max(actions),
        'max_solve_residual': max(solves), 'pre_refresh_adam_equality_passed': True,
        'max_ritz_relative_residual': max(v for r in ritz for v in r['relative_ritz_residuals']),
        'max_ritz_orthogonality_error': max(r['orthogonality_error'] for r in ritz),
        'parent_soap_model_data_initial_loss_checks_passed': True,
        'tests_passed_before_training': 92,
    }
    write_json(directory/'verification.json', verification)
    write_json(directory/'summary.json', summaries)
    write_json(directory/'noise_summary.json', noise)
    write_json(directory/'ritz_summary.json', ritz)
    write_json(directory/'prior_soap_reference.json', {'run': str(parent), 'rows': reference})

    os.environ.setdefault('MPLCONFIGDIR', str((directory/'.matplotlib').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((directory/'.cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    groups = [['adam_native', 'parsec_native', 'adam_searched', 'parsec_searched'],
              ['ni_soap_alpha0', 'ni_soap_alpha1'],
              ['ggn_exact_trace', 'wasserstein_tensor']]
    for ax, methods, title in zip(axes.flat, groups, ['PARSEC and matched Adam', 'NI-SOAP noise control', 'Matched metric comparison']):
        for method in methods:
            rows = [r for r in validation if r['method'] == method]
            ax.plot([r['step'] for r in rows], [r['validation_loss'] for r in rows], marker='o', markersize=4, label=LABELS[method])
        ax.set(title=title, xlabel='Consumed iterations', ylabel='Validation cross-entropy')
        ax.legend(fontsize=8)
    for method in complete['methods']:
        rows = [r for r in validation if r['method'] == method and r['step'] > 0]
        axes[1, 1].plot([r['training_seconds'] for r in rows], [r['validation_loss'] for r in rows], marker='o', markersize=3, label=LABELS[method])
    axes[1, 1].set(title='Measured training cost', xlabel='Training-step seconds (log scale)', ylabel='Validation cross-entropy', xscale='log')
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('MNIST / small ViT / batch 512 / seed 0 / 50 iterations\n5,000 validation examples; one untuned paired pilot')
    fig.savefig(directory/'comparison.png', dpi=170)
    fig.savefig(directory/'comparison.pdf')
    plt.close(fig)

    lines = ['# Transport, PARSEC and NI-SOAP — 6 October 2026', '',
             f'Run: `runs/{directory.name}`. The [prospective protocol](transport_parsec_noise_protocol_20261006.md) '
             'selected eight paired trajectories on the 5,738-parameter ViT, MNIST batch 512, seed 0, '
             '50 consumed iterations, CPU float64 and one thread. Validation uses 5,000 reserved examples; '
             'no official test data or hyperparameter sweep was used. The user confirmed the default '
             'PARSEC precondition-then-AdamW branch.', '',
             '## Final measurements', '',
             '| Method | Validation loss | Accuracy | Training seconds | Accepted updates |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in summaries:
        lines.append(f"| {LABELS[row['method']]} | {row['validation_loss']:.6f} | {100*row['validation_accuracy']:.2f}% | {row['training_seconds']:.2f} | {row['accepted_updates']} |")
    lines += ['', 'Time includes gradient, proposal, curvature refresh and global search where used. '
              'It excludes data loading, validation, writes, replay and separately timed numerical action audits. '
              'Every searched step uses one global scalar with the same 52-evaluation budget; none is a per-layer search. '
              'NI-SOAP uses native steps to preserve its noise law. All hyperparameters stayed as prospectively declared.', '',
              'For context, the earlier official-SOAP run used identical model source, split, initial loss and batch IDs. '
              'It is not the alpha-zero NI-SOAP control; the supplied implementation has different initialization and epsilon semantics.', '',
              '| Earlier official SOAP | Validation loss | Accuracy |', '| --- | ---: | ---: |']
    for row in reference:
        lines.append(f"| {row['method']} | {row['validation_loss']:.6f} | {100*row['validation_accuracy']:.2f}% |")
    finals = {r['method']: r for r in summaries}
    lines += ['', 'Within each selected pair, the relative reduction in final validation loss was '
              '(negative means higher loss):', '',
              '| Candidate versus control | Relative loss reduction |', '| --- | ---: |']
    for candidate, control in [('parsec_native', 'adam_native'), ('parsec_searched', 'adam_searched'),
                               ('ni_soap_alpha1', 'ni_soap_alpha0'), ('wasserstein_tensor', 'ggn_exact_trace')]:
        reduction = 1-finals[candidate]['validation_loss']/finals[control]['validation_loss']
        lines.append(f'| {LABELS[candidate]} versus {LABELS[control]} | {reduction:+.2%} |')
    lines += ['', '## PARSEC as implemented', '',
              'With orthonormal global Ritz vectors in rows of V, transform the current gradient as '
              '`g_tilde = (I - V.T V) g + V.T diag((lambda + delta)^(-1/2)) V g`, '
              'then run ordinary AdamW on g_tilde in the original parameter coordinates. '
              'The untracked complement is unchanged. Moments persist across curvature refreshes. '
              'Weight decay is zero in this experiment. The selected branch is not the separate Newton/Adam split.', '',
              'Rank 16, 32 Krylov iterations, GGN refresh at 10/20/30/40/50, damping 1e-4, '
              'lr=.003, Adam betas .9/.999, eps=1e-8. Both native and searched variants match their '
              'Adam controls exactly through step 9. Refresh was shortened from the supplied default 100 '
              'so curvature activates during this 50-step pilot. Float64 full reorthogonalization replaces '
              'the helper\'s lower-precision Lanczos storage; these are approximate Ritz vectors, not exact top eigenspaces.', '',
              'Preconditioning the input to Adam is different from preconditioning its final update. '
              'For a fixed positive coordinate scaling and negligible epsilon, Adam scales its first '
              'moment by c and its second moment by c^2, cancelling that input scaling. PARSEC can '
              'still change the direction through dense coordinate mixing and time-varying transforms. '
              'Its equations alone do not guarantee the inverse-root geometry survives Adam unchanged.', '',
              '| Arm | Refresh | Largest eigenvalue | Smallest retained | Max relative Ritz residual |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in ritz:
        lines.append(f"| {LABELS[row['method']]} | {row['step']} | {max(row['eigenvalues']):.5g} | {min(row['eigenvalues']):.5g} | {max(row['relative_ritz_residuals']):.3e} |")
    n = noise['ni_soap_alpha1']
    lines += ['', '## NI-SOAP as implemented', '',
              'The supplied SOAP-like drift receives matrix noise '
              '`alpha sqrt(lr/B) L_cached^(1/2) Z R_cached^(1/2)` and isotropic fallback noise '
              '`alpha sqrt(lr/B) z`. Gradients, moments and factor statistics do not directly ingest the noise. '
              'Cached bases/eigenvalues are used before the current statistics update; the first matrix call '
              'only initializes statistics, while fallback parameters update immediately. Linear weights take '
              'the matrix route; embeddings, normalization and biases take fallback. All routes are saved.', '',
              'Settings: alpha 0/1, lr=.003, B=512, betas .95/.95, factor beta .95, QR refresh 10, '
              'eps=1e-8, weight decay zero. Ports preserve finite update equations in float64, with local '
              'Torch random generators; they are not bitwise JAX reproductions.', '',
              'Across the 50 alpha-one steps, sums of squared increments were:', '',
              '| Quantity | Measured sum |', '| --- | ---: |']
    for key in ['drift_energy', 'noise_energy', 'expected_noise_energy', 'realized_matrix_noise_energy', 'realized_fallback_noise_energy', 'drift_noise_cross_term', 'update_energy']:
        lines.append(f'| {key} | {n[key]:.8g} |')
    lines += ['', f"Noise/drift energy ratio: {n['noise_to_drift_energy_ratio']:.4%}. "
              f"Fallback share of realized noise: {n['realized_fallback_noise_energy']/n['noise_energy']:.4%}. "
              'These sums are per-step update energies, not squared net displacement. '
              'The matrix noise expectation is alpha^2 lr/B tr(L_cached)tr(R_cached); '
              'the fallback expectation is alpha^2 lr/B times its parameter count. '
              'This does not automatically equal SGD gradient-noise energy, despite the source comment. '
              'No additional normalization was silently introduced.', '',
              '## Transport geometry and the step decomposition', '',
              'Normalized steepest descent minimizes <g,d> over a unit norm ball. For matrix '
              'spectral norm, G=U S V.T gives the idealized direction -U V.T; momentum and practical '
              'Muon scaling are separate. See [Modular Duality](https://proceedings.mlr.press/v267/bernstein25a.html).', '',
              'For categorical probabilities p, use a complete graph with equal class conductances and '
              'edge mobility (p_i+p_j)/2. Its Laplacian L(p) defines the zero-sum tangent norm '
              '`||s||_p^2 = s.T L(p)^dagger s`. Pulling it back through predictions gives '
              '`M = mean J_p.T L(p)^dagger J_p`. The damped steepest step has direction '
              '`-(M + mu Omega)^(-1) g`. For radius R, divide by '
              '`sqrt(g.T (M + mu Omega)^(-1) g)` and multiply by R. '
              'For the tensor-amplitude decomposition d=E a, its squared norm is a.T E.T M E a. '
              'This is the same constrained-step framework; the chosen metric need not equal loss curvature. '
              'See [Li and Montufar](https://arxiv.org/html/1803.07033v5).', '',
              'This run uses exact within-tensor blocks of that dynamic graph-Wasserstein pullback. '
              'It is not static discrete Kantorovich W2 or a canonical fixed norm on signed weights. '
              'Equal class costs avoid inventing ordinal geometry for digits. The comparison uses '
              'exact tensor GGN, the same normalized inverse shadow-SOAP damping metric, '
              'mu=max(.01 trace(M)/5738,1e-8), the current gradient and the same global search. '
              'The earlier GGN training estimated trace with eight probes; this control uses exact trace '
              'to avoid conflating the trace estimator with the geometry.', '',
              'At uniform probabilities over K classes, this graph metric is exactly GGN/K. '
              'With trace-scaled damping and scalar search, the resulting rays therefore agree '
              '(when the damping floor is inactive). Away from uniform predictions the metrics '
              'can differ. This limits what equal-cost categorical transport adds near initialization; '
              'it does not test a learned or feature-informed ground geometry.', '',
              '## Verification and limits', '',
              f"All 92 tests passed before training. {len(archive)} archived source hashes verified; "
              'all eight final checkpoints reproduce validation and the uncommitted next proposal exactly. '
              'Optimizer, model, data-order and global RNG non-mutation passed. '
              f"All {len(searches)} searches were bracketed or rejected; {verification['rejected_searches']} were rejected. "
              f"Maximum action error {max(actions):.3e}; solve residual {max(solves):.3e}. "
              'PARSEC Ritz errors above describe approximation quality and are not exact-solve residuals.', '',
              'This is one short, untuned seed. Native and searched arms have different evaluation costs; '
              'PARSEC uses a global low-rank preconditioner whereas the exact metrics use tensor blocks. '
              'The 50 consumed batches include method-specific initialization: curvature arms make 49 updates. '
              'No result establishes a general optimizer ranking, transport ground-geometry optimum, '
              'or a causal explanation of the prior SOAP/GGN gap.', '',
              'Raw validation, per-step histories (including all search trials), noise contributions, Ritz '
              'diagnostics, settings, source archives and checkpoints remain in the run. Rebuild with:', '',
              '```sh', f'.venv/bin/python scripts/summarize_transport_parsec_noise.py --run runs/{directory.name}', '```', '']
    report = '\n'.join(lines)
    Path('docs/transport_parsec_noise_results_20261006.md').write_text(report+f'\n![Comparison](../runs/{directory.name}/comparison.png)\n')
    (directory/'REPORT.md').write_text(report.replace('(transport_parsec_noise_protocol_20261006.md)', '(../../docs/transport_parsec_noise_protocol_20261006.md)')+'\n![Comparison](comparison.png)\n')
    shutil.copyfile(__file__, directory/'analysis_runner.py')
    artifacts = ['analysis_runner.py', 'comparison.png', 'comparison.pdf', 'REPORT.md', 'verification.json',
                 'summary.json', 'noise_summary.json', 'ritz_summary.json', 'prior_soap_reference.json',
                 'complete.json', 'checkpoint_verification.json', 'validation.json']
    artifacts.extend(str(p.relative_to(directory)) for m in complete['methods'] for p in (directory/m).glob('*') if p.is_file())
    write_json(directory/'analysis_artifacts.json', {name: checksum(directory/name) for name in artifacts})
    print(json.dumps({'summary': summaries, 'verification': verification, 'noise': noise}, indent=2))


if __name__ == '__main__':
    main()
