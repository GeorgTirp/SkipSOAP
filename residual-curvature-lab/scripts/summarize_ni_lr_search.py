"""Verify the NI-SOAP learning-rate-search extension and compare saved controls."""
import argparse
import json
import os
import shutil
import statistics
from pathlib import Path

from residual_curvature.provenance import checksum, write_json
from summarize_transport_parsec_noise import LABELS, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    directory = args.run
    parent = Path('runs/20261006T100557-f1efb932')
    complete = read(directory/'complete.json')
    assert complete['completed'] and not (directory/'failure.json').exists()
    archive = read(directory/'source_archive_checksums.json')
    for name, digest in archive.items():
        assert checksum(directory/'source'/name) == digest
    for name, digest in read(parent/'analysis_artifacts.json').items():
        assert checksum(parent/name) == digest
    assert read(parent/'split_manifest.json') == read(directory/'split_manifest.json')
    validation = read(directory/'validation.json')
    old_validation = read(parent/'validation.json')
    assert validation[0]['validation_loss'] == old_validation[0]['validation_loss']
    model_files = [n for n in archive if n.startswith('src/residual_curvature/models/')]
    parent_archive = read(parent/'source_archive_checksums.json')
    assert all(archive[n] == parent_archive[n] for n in model_files)
    batches = [r['batch_ids'] for r in read(parent/'adam_native'/'training.json')]
    replay = read(directory/'checkpoint_verification.json')
    assert len(replay) == 2 and all(r['next_proposal_replay_passed'] and r['nonmutation_passed'] for r in replay)
    summaries, searches, energy_errors = [], [], []
    for method in complete['methods']:
        h = read(directory/method/'training.json')
        assert [r['step'] for r in h] == list(range(1, 51))
        assert [r['batch_ids'] for r in h] == batches
        assert all(r['loss_after'] <= r['loss_before']+1e-12 for r in h)
        searches.extend(r['search'] for r in h)
        final = next(r for r in validation if r['method'] == method and r['step'] == 50)
        restored = next(r['restored_validation'] for r in replay if r['method'] == method)
        assert all(restored[k] == final[k] for k in ['validation_loss', 'validation_accuracy'])
        lrs = [r['search']['selected_lr'] for r in h]
        totals = {k: sum(r['algorithm'][k] for r in h) for k in ['drift_energy', 'noise_energy', 'update_energy', 'drift_noise_cross_term']}
        for row in h:
            a = row['algorithm']
            energy_errors.append(abs(a['update_energy']-a['drift_energy']-a['noise_energy']-a['drift_noise_cross_term'])/max(a['update_energy'],1e-30))
            assert a['selected_learning_rate'] == row['search']['selected_lr']
        fallback = sum(t['noise_energy'] for r in h for t in r['algorithm']['per_tensor'].values() if t['route'] == 'fallback')
        summaries.append({**final, **read(directory/method/'runtime.json'),
                          'accepted_updates': h[-1]['accepted_updates'],
                          'rejected_searches': sum(r['search']['rejected'] for r in h),
                          'lr_min': min(lrs), 'lr_median': statistics.median(lrs), 'lr_max': max(lrs),
                          'total_search_evaluations': sum(r['search']['evaluations'] for r in h),
                          'noise_to_drift_energy': totals['noise_energy']/totals['drift_energy'],
                          'fallback_noise_fraction': fallback/totals['noise_energy'] if totals['noise_energy'] else None,
                          **totals})
    assert all(s['bracketed'] or s['rejected'] for s in searches)
    assert all(not s['boundary'] and s['fixed_draw'] for s in searches)
    assert max(energy_errors) <= 1e-10
    verification = {'source_files_verified': len(archive), 'parent_artifact_hashes_passed': True,
                    'paired_data_and_initialization_checks_passed': True, 'checkpoint_replay_passed': True,
                    'searches_checked': len(searches), 'all_searches_bracketed_or_rejected': True,
                    'max_energy_closure_error': max(energy_errors), 'tests_passed_before_run': 96}
    write_json(directory/'summary.json', summaries)
    write_json(directory/'verification.json', verification)
    labels = {**LABELS, 'ni_soap_alpha0_lr_searched': 'NI-SOAP alpha=0 + LR search',
              'ni_soap_alpha1_lr_searched': 'NI-SOAP alpha=1 + LR search'}
    all_summaries = read(parent/'summary.json') + summaries
    all_validation = old_validation + validation
    os.environ.setdefault('MPLCONFIGDIR', str((directory/'.matplotlib').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((directory/'.cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for method in ['ni_soap_alpha0', 'ni_soap_alpha1']+complete['methods']:
        rows = [r for r in all_validation if r['method'] == method]
        axes[0].plot([r['step'] for r in rows], [r['validation_loss'] for r in rows], marker='o', markersize=4, label=labels[method])
    for method in ['adam_searched', 'parsec_searched']+complete['methods']+['ggn_exact_trace', 'wasserstein_tensor']:
        rows = [r for r in all_validation if r['method'] == method]
        axes[1].plot([r['step'] for r in rows], [r['validation_loss'] for r in rows], marker='o', markersize=4, label=labels[method])
    for ax, title in zip(axes, ['NI-SOAP native versus learning-rate search', 'All searched candidates']):
        ax.set(title=title, xlabel='Consumed iterations', ylabel='Validation cross-entropy')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('MNIST / small ViT / batch 512 / seed 0 / 50 iterations\nSame initialization and batches; construction-only global search')
    fig.savefig(directory/'comparison.png', dpi=170)
    fig.savefig(directory/'comparison.pdf')
    plt.close(fig)
    lines = ['# NI-SOAP learning-rate search — 6 October 2026', '',
             f'Extension run: `runs/{directory.name}`; parent: `runs/{parent.name}`. '
             'The user requested search for both supplied algorithms. PARSEC already had it; '
             'this extension adds NI-SOAP with and without noise. The '
             '[prospective protocol](ni_soap_lr_search_protocol_20261006.md) was saved before execution. '
             'All arms below use the same 5,738-parameter ViT, MNIST batch 512, seed 0, '
             '50 consumed iterations and 5,000 validation examples.', '',
             '| Method | Validation loss | Accuracy | Training seconds |', '| --- | ---: | ---: | ---: |']
    for row in all_summaries:
        lines.append(f"| {labels[row['method']]} | {row['validation_loss']:.6f} | {100*row['validation_accuracy']:.2f}% | {row['training_seconds']:.2f} |")
    lines += ['', 'The earlier official searched SOAP achieved loss .731623 and accuracy 77.40% '
              'on this paired workload. It differs from the supplied NI-SOAP alpha-zero drift, '
              'including first-call and epsilon semantics; use alpha zero as the matched noise control.', '',
              '## Search definition', '',
              'Decompose one native lr=.003 update as drift d plus noise n. For learning-rate '
              'multiplier a, search `Delta(a)=a*d+sqrt(a)*n`, so `eta=.003*a`. '
              'Each candidate holds the true gradient, moments and noise draw fixed. Only the '
              'chosen displacement is committed; optimizer statistics advance once, including '
              'when zero is selected. The grid is zero plus 2^[-16..16], with 18 golden-section '
              'refinements for an interior positive grid winner. This is a single global search, '
              'not per-layer search.', '',
              'The alpha-zero path is the ordinary update ray. The noisy path preserves the '
              'source formula at each fixed learning rate, with drift proportional to eta and '
              'noise proportional to sqrt(eta); rescaling the entire noisy step would not. '
              'However, selecting eta using the noisy candidate loss changes its conditional '
              'noise distribution. These are searched variants, not unchanged native stochastic '
              'dynamics. Alpha and covariance structure are still untuned.', '',
              '| Searched NI arm | Min eta | Median eta | Max eta | Rejections | Noise/drift energy |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        lines.append(f"| {labels[row['method']]} | {row['lr_min']:.6g} | {row['lr_median']:.6g} | {row['lr_max']:.6g} | {row['rejected_searches']} | {row['noise_to_drift_energy']:.3%} |")
    zero, noisy = summaries
    lines += ['', f"The noisy searched arm's final loss relative to the zero-noise searched control is "
              f"{noisy['validation_loss']/zero['validation_loss']:.6f}; values above one favor zero noise. "
              f"Fallback tensors received {noisy['fallback_noise_fraction']:.4%} of realized noise energy. "
              'Energy ratios sum squared per-step increments, not squared total displacement. '
              'Pre-selection expected-energy formulas saved per step are not unconditional '
              'expectations after loss-based selection.', '',
              '## Verification and interpretation', '',
              f"All 96 tests passed before the extension (and the 12 proposal/path tests passed after "
              f"the final metadata change). {len(archive)} source archive files and all parent analysis "
              'artifact hashes verified. Both final checkpoints reproduce validation and the next '
              'uncommitted proposal exactly. Model, optimizer, data-order and RNG non-mutation passed. '
              f"All {len(searches)} searches were bracketed or explicitly rejected. "
              f"Maximum noise/drift energy-closure error: {max(energy_errors):.3e}.", '',
              'Training seconds include proposals and searches, excluding validation, file writes '
              'and replay. Learning-rate search tests whether adaptive global step size changes '
              'these trajectories. It does not prove all remaining differences are curvature '
              'effects: momentum, noise geometry, per-tensor scaling and approximation still differ. '
              'This is one short untuned seed, not a replicated ranking. No official test data '
              'were evaluated.', '',
              'The parent [transport/PARSEC/NI-SOAP report](transport_parsec_noise_results_20261006.md) '
              'contains algorithm equations, native noise accounting, Ritz residuals and the '
              'transport derivation. Rebuild this extension report with:', '', '```sh',
              f'.venv/bin/python scripts/summarize_ni_lr_search.py --run runs/{directory.name}', '```', '']
    report = '\n'.join(lines)
    Path('docs/ni_soap_lr_search_results_20261006.md').write_text(report+f'\n![Comparison](../runs/{directory.name}/comparison.png)\n')
    (directory/'REPORT.md').write_text(report.replace('(ni_soap_lr_search_protocol_20261006.md)', '(../../docs/ni_soap_lr_search_protocol_20261006.md)').replace('(transport_parsec_noise_results_20261006.md)', '(../../docs/transport_parsec_noise_results_20261006.md)')+'\n![Comparison](comparison.png)\n')
    shutil.copyfile(__file__, directory/'analysis_runner.py')
    names = ['analysis_runner.py', 'comparison.png', 'comparison.pdf', 'REPORT.md', 'summary.json', 'verification.json', 'validation.json', 'complete.json', 'checkpoint_verification.json']
    names.extend(str(p.relative_to(directory)) for m in complete['methods'] for p in (directory/m).glob('*') if p.is_file())
    write_json(directory/'analysis_artifacts.json', {n: checksum(directory/n) for n in names})
    print(json.dumps({'summary': summaries, 'verification': verification}, indent=2))


if __name__ == '__main__':
    main()
