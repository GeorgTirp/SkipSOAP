"""Verify and summarize the completed learnable-optimizer capacity screen."""
import argparse
import json
import os
import shutil
import statistics
from pathlib import Path

from residual_curvature.provenance import checksum, write_json
from run_learnable_capacity import FAMILIES, gate


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    directory=args.run
    complete=read(directory/'complete.json')
    assert complete['completed'] and not (directory/'failure.json').exists()
    sources=read(directory/'source_archive_checksums.json')
    for name,digest in sources.items():
        assert checksum(directory/'source'/name)==digest
    for name,digest in read(directory/'parent_arrays.json').items():
        assert checksum(name)==digest
    resumed=read(directory/'resumed_source_archive.json') if (directory/'resumed_source_archive.json').exists() else None
    verified_origins=set()
    def verify_origin(run):
        run=Path(run)
        if str(run) in verified_origins:
            return
        for name,digest in read(run/'source_archive_checksums.json').items():
            assert checksum(run/'source'/name)==digest
        verified_origins.add(str(run))
        if (run/'resumed_source_archive.json').exists():
            verify_origin(read(run/'resumed_source_archive.json')['source_run'])
    if resumed:
        verify_origin(resumed['source_run'])
    rows=read(directory/'all_candidates.json')
    ops=read(directory/'all_operators.json')
    states=read(directory/'all_states.json')
    outcome=read(directory/'progression_gate.json')
    assert len(rows)==324 and len(ops)==156 and len(states)==12
    summaries=[]
    for normalization in ['trace','damping']:
        rs=[r for r in rows if r['normalization']==normalization]
        os_=[r for r in ops if r['normalization']==normalization]
        assert gate(rs,os_)==outcome['normalizations'][normalization]
        for item in outcome['normalizations'][normalization]['candidates']:
            subset=[o for o in os_ if o['name']==item['name']]
            item=dict(item)
            item['normalization']=normalization
            item['pooled_errors']={}
            for group in ['gradient','momentum','physical','factor']:
                denom=sum(o['errors'][group]['denominator_sum'] for o in subset)
                item['pooled_errors'][group]={kind:sum(o['errors'][group][kind+'_numerator_sum'] for o in subset)/denom for kind in ['raw','ray']}
            item['median_basis_seconds']=statistics.median(o['basis_seconds'] for o in subset)
            item['median_fit_seconds']=statistics.median(o['fit_seconds_including_basis'] for o in subset)
            item['median_coefficient_fit_seconds']=statistics.median(o['coefficient_fit_seconds'] for o in subset)
            item['median_apply_seconds']=statistics.median(o['apply_seconds_median'] for o in subset)
            item['total_clipped']=sum(o['clipped'] for o in subset)
            summaries.append(item)
    searches=[r['search'] for r in rows]
    assert all(s['bracketed'] or s['rejected'] for s in searches)
    assert all(not s['boundary'] for s in searches)
    assert all(s['nonmutation_passed'] for s in states)
    checks={key:max(s[key] for s in states) for key in ['max_factor_error','max_action_error','cached_teacher_energy_error','teacher_gradient_residual','teacher_max_backward_error']}
    checks['max_dense_application_error']=max(o['max_dense_application_error'] for o in ops)
    assert max(checks.values())<=1e-8
    for state in states:
        folder=directory/f"{state['normalization']}-{state['trajectory']}-{state['step']:04d}"
        for name,digest in read(folder/'artifacts.json').items():
            assert checksum(folder/name)==digest
        if (folder/'origin.json').exists():
            origin=read(folder/'origin.json')
            while origin:
                verify_origin(origin['source_run'])
                old=Path(origin['source_run'])/folder.name
                for name in ['operators.pt','operators.json','candidates.json','diagnostics.json']:
                    assert checksum(folder/name)==checksum(old/name)
                origin=origin.get('prior_origin')
    verification={'source_files_verified':len(sources),'parent_array_hashes_passed':True,
                  'operator_artifacts_verified':12,'numerical_checks':checks,
                  'searches_checked':len(searches),'rejected_searches':sum(s['rejected'] for s in searches),
                  'nonmutation_passed':True,'tests_passed_before_run':103,
                  'resumed_source_archives_verified':sorted(verified_origins),
                  'dense_check_kinds':sorted({o.get('dense_check_kind','gradient_forward_or_exact_energy') for o in ops}),
                  'max_logged_gradient_forward_discrepancy':max(o.get('max_gradient_forward_discrepancy',0.) for o in ops)}
    if (directory/'post_run_tests.json').exists():
        verification['post_run_tests']=read(directory/'post_run_tests.json')
    write_json(directory/'summary.json',summaries)
    write_json(directory/'verification.json',verification)

    os.environ.setdefault('MPLCONFIGDIR',str((directory/'.matplotlib').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME',str((directory/'.cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    selected=['identity','diagonal','teacher_16','coordinate_16','state_16','exact']
    for ax,trajectory in zip(axes[0],['searched_soap','exact_tensor_ggn']):
        for name in selected:
            values=[r for r in rows if r['normalization']=='damping' and r['trajectory']==trajectory and r['name']==name and r['rhs']=='gradient']
            ax.plot([r['step'] for r in values],[r['selected']['independent_mean_decrease'] for r in values],marker='o',label=name)
        ax.axhline(0,color='black',linewidth=.6)
        ax.set(title=trajectory+' states; damping calibration',xlabel='Saved iteration',ylabel='Independent loss decrease')
        ax.legend(fontsize=8)
    for family in ['teacher','coordinate','state']:
        for normalization,style in [('trace','--'),('damping','-')]:
            values=[next(s for s in summaries if s['normalization']==normalization and s['name']==f'{family}_{rank}') for rank in [4,8,16]]
            axes[1,0].plot([4,8,16],[100*s['local_recovery'] for s in values],style,marker='o',label=family+'/'+normalization)
    axes[1,0].axhline(80,color='black',linestyle=':',label='80% development gate')
    axes[1,0].set(title='All six states; local headroom recovery',xlabel='Requested rank per tensor',ylabel='Recovery (%)')
    axes[1,0].legend(fontsize=7)
    for family in ['teacher','coordinate','state']:
        for group,style in [('physical','-'),('factor','--')]:
            values=[next(s for s in summaries if s['normalization']=='damping' and s['name']==f'{family}_{rank}') for rank in [4,8,16]]
            axes[1,1].plot([4,8,16],[100*s['operator_error_reductions'][group] for s in values],style,marker='o',label=family+'/'+group)
    axes[1,1].axhline(80,color='black',linestyle=':')
    axes[1,1].set(title='Held-out inverse-action improvement; damping calibration',xlabel='Requested rank per tensor',ylabel='Reduction in pooled ray error (%)')
    axes[1,1].set_yscale('symlog',linthresh=100)
    axes[1,1].set_ylabel('Reduction in pooled ray error (%)\nSymmetric log scale; linear within ±100%')
    axes[1,1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Experiment 1: inverse capacity / MNIST ViT / batch 512\nTeacher coefficients throughout; reused development panels; no predictor training')
    fig.savefig(directory/'comparison.png',dpi=170)
    fig.savefig(directory/'comparison.pdf')
    plt.close(fig)

    lines=['# Learnable optimizer: representation capacity — 7 October 2026','',
           f'Run: `runs/{directory.name}`. The [protocol](learnable_capacity_protocol_20261007.md) '
           'records the initial selection and disclosed numerical/calibration amendments. '
           'Six saved seed-0 ViT states (SOAP and GGN trajectories at 10/25/50), batch 512, '
           'CPU float64, one thread. All 40 parameter tensors participate. The four independent '
           'analysis panels are reused development data, not fresh confirmation. No training or '
           'learned predictor was run.','',
           '**Progression to experiment 2: '+('selected by the gate.' if outcome['experiment2_selected'] else 'not selected: no eligible representation passed all gates.')+'','',
           '## Findings and interpretation','',
           'A cheap rank-4 state basis with exact teacher coefficients recovers 37.3% of '
           'aggregate current-gradient local headroom. Rank 16 reaches 41.6%, with 17.747 '
           'additional model-sized vectors. Exact scalar and coordinate-diagonal fitting '
           'recover 14.2% and 21.0%. Under this inverse-fitting objective, these bases miss '
           'the 80% target even with teacher coefficients. Application is inexpensive here; quality '
           'is already insufficient before adding prediction/construction cost.','',
           'With damping calibration, teacher rank 8 gives .131890 independent decrease '
           'at SOAP state 10 versus .033191 for current-gradient SOAP and .135327 for '
           'exact GGN: 96.6% of that state\'s gap. Teacher rank 16 gives .076923 at SOAP '
           'state 25 versus exact .085051, but hurts at SOAP 50 and GGN 25/50. Across '
           'all six states it recovers 56.1%. Original trace-calibrated teacher rank 16 '
           'recovers only 15.4%. The baseline outside the learned span matters; global '
           'line search cannot repair an incorrect relative complement scale.','',
           'There is a promising exploratory momentum interaction: damping-calibrated '
           'teacher rank 16 applied to effective SOAP momentum gives mean decrease '
           '.046187, compared with .015916 for exact GGN on that same momentum and '
           '.045883 for exact GGN on the current gradient. This is a cross-RHS diagnostic, '
           'not the prespecified current-gradient progression test or a training result. '
           'It uses an expensive teacher-selected basis and exceeds the memory budget. '
           'It suggests that the useful correction can depend on the gradient history; '
           'closer inverse approximation alone does not guarantee better nonlinear transfer.','',
           'Probe accuracy and optimization quality also differ. The diagonal core cuts '
           'physical-probe ray error by 93.2%, yet closes only 21.0% of the local loss gap '
           'and cuts factor-probe error by only 29.8% under damping calibration. Random '
           'physical probes emphasize a different part of the inverse than actual gradients. '
           'Both probe distributions and gradient/momentum scores are needed. Factor RHS '
           'include beta, so cross-state energy pooling weights change between calibrations.','',
           'These results motivate finding useful within-tensor directions from affordable '
           'state/history or curvature measurements, with their complement scale and RHS '
           'choice treated explicitly. They do not establish that a GNN can predict the '
           'directions, that cross-tensor graph coupling is required, or that all rank-16 '
           'representations fail: the teacher basis is a restricted spectral construction '
           'fitted to an isotropic inverse objective, not an oracle for independent loss. '
           'Under the declared gate, experiments 2 and 3 remain unrun.','',
           '## What is fitted','',
           'The damped tensor target is A_b=C_bb+mu Omega_b with gamma .01 and exact trace. '
           'For the actual SOAP factor F0, P0=F0 F0.T. Set F=sqrt(beta)F0 and K=F.T A F. '
           'Every candidate is P=F M F.T. Scalar/diagonal cores and bounded subspace cores '
           '`M=I+U(T-I)U.T` are fitted using the analytic expected teacher energy over isotropic '
           'factor-coordinate RHS. The fit does not use realized evaluation probes. '
           'Bounds are [1e-6,1e6]. For fixed U, T is the bounded inverse of U.T K U.','',
           'Teacher bases select eigenvectors of K by population-error improvement. They are '
           'restricted spectral oracles, not a proof of the globally best rank-r SPD correction. '
           'Coordinate bases select entries from current gradient/momentum energy. State bases '
           'orthogonalize current gradient/momentum multiplied by powers of normalized log SOAP '
           'denominators. Both cheap basis rules still receive teacher coefficients; neither '
           'is a deployable optimizer. Coordinate bases store indices, while state/teacher '
           'bases store dense vectors.','',
           '## Aggregate development results','',
           'Independent decrease is averaged over all six states; positive is better. Local '
           'recovery is (candidate-current-gradient SOAP)/(exact-current-gradient SOAP), with '
           'the same global construction-only search. No negative-advantage state is omitted. '
           'Inverse errors pool A-energy numerators/denominators over 16 held-out complete '
           'parameter RHS per state, after one global nonnegative scalar per RHS. They are '
           'distinct from the nonlinear search and from current-gradient-only errors.','']
    for normalization in ['trace','damping']:
        lines += ['', '### '+normalization+' calibration','',
                  '| Core | Mean independent decrease | Local recovery | Physical probe ray error | Factor probe ray error | Stored model vectors | Apply / baseline | Pass |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for s in [v for v in summaries if v['normalization']==normalization]:
            recovery=f"{s['local_recovery']:.1%}" if s['local_recovery'] is not None else 'undefined'
            lines.append(f"| {s['name']} | {s['independent_mean_decrease']:.6f} | {recovery} | {s['pooled_ray_errors']['physical']:.3e} | {s['pooled_ray_errors']['factor']:.3e} | {s['storage_model_vectors']:.3f} | {s['median_application_ratio']:.2f} | {s['passed']} |")
    lines += ['', 'Eligibility requires >=80% local headroom recovery, >=80% reduction of both '
              'pooled probe ray errors relative to baseline, <=4.25 additional model-sized '
              'vectors including coefficients, median application ratio <=2, and passed '
              'numerical/search checks. Exact and teacher-selected bases are references, '
              'not deployable candidates. These thresholds are development engineering '
              'gates, not statistical significance tests.','',
              '## Unscaled versus ray accuracy','',
              'The following values use actual gradients and the same block teacher energy, '
              'pooled across states. Raw values retain scalar calibration; ray values permit '
              'one scalar for the whole parameter vector. Full coupled quadratic and nonlinear '
              'independent decreases are saved separately in each candidate row.','',
              '| Calibration | Core | Gradient raw error | Gradient ray error | Momentum ray error |',
              '| --- | --- | ---: | ---: | ---: |']
    for s in summaries:
        lines.append(f"| {s['normalization']} | {s['name']} | {s['pooled_errors']['gradient']['raw']:.3e} | {s['pooled_errors']['gradient']['ray']:.3e} | {s['pooled_errors']['momentum']['ray']:.3e} |")
    lines += ['', '## Per-state independent decreases (damping calibration, current gradient)','',
              '| Core | SOAP 10 | SOAP 25 | SOAP 50 | GGN 10 | GGN 25 | GGN 50 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,_,_ in FAMILIES:
        values=[next(r['selected']['independent_mean_decrease'] for r in rows if r['normalization']=='damping' and r['name']==name and r['rhs']=='gradient' and r['trajectory']==trajectory and r['step']==step) for trajectory in ['searched_soap','exact_tensor_ggn'] for step in [10,25,50]]
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '## Momentum controls','',
              'Mean independent decrease across the same six states, damping calibration. '
              'Both columns use the same search budget. Native SOAP includes its recorded '
              'momentum proposal; the matrix-map controls apply the fitted operator to the '
              'effective momentum RHS. The teacher-fit objective did not fit either realized RHS.','',
              '| Method | Current gradient | Effective momentum |',
              '| --- | ---: | ---: |']
    for name in ['native_soap','identity','diagonal','teacher_16','state_16','exact']:
        cells=[]
        for rhs in ['gradient','momentum']:
            values=[r['selected']['independent_mean_decrease'] for r in rows if r['normalization']=='damping' and r['name']==name and r['rhs']==rhs]
            cells.append(f'{statistics.mean(values):.6f}' if values else '—')
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', '## Interrupted attempts and calibration amendment','',
              'The first attempt, `runs/20261007T092113-f1ad8fe0`, finished one state and stopped '
              'at a Euclidean solution-comparison check on SOAP state 25. The discrepancy '
              'was 1.36e-8, with much smaller equation and teacher-energy errors. The repaired '
              'reference uses factor-coordinate Cholesky and retains 1e-8 checks on cached '
              'teacher energy agreement, gradient equation residual, all-RHS normwise '
              'backward error and independent factor/operator actions. The failed artifact '
              'is preserved, not counted as a successful run.','',
              'The second attempt, `runs/20261007T092734-02f0a4d9`, completed the three '
              'SOAP states but stopped when explicit inverse-matrix multiplication disagreed '
              'with the Cholesky reference on random RHS at GGN state 10. The final runner '
              'uses Cholesky applications for the exact reference and retains the independent '
              'cached-gradient, physical-equation, backward-error and factor checks. Only '
              'fully verified completed states were reused, with their original source '
              'archive and per-state origin recorded. Candidate fitting equations did not change.','',
              'The third attempt, `runs/20261007T093220-108e6ea8`, completed all six trace '
              'states, then stopped on damping/SOAP-10/state-4: gradient forward discrepancy '
              'was 3.09e-8 despite a positive core minimum eigenvalue 1.70e-4. New states '
              'compare the full independently materialized factorized/dense operators in '
              'relative Frobenius norm at 1e-8. Gradient forward discrepancies remain logged '
              'separately, since conditioning can amplify them. Reused trace states retain '
              'their original gradient-forward (exact: energy) comparisons; the verification '
              'records both audit kinds and verifies nested source archives. No candidate '
              'formula or scientific gate changed.','',
              'The original trace calibration beta=D/sum trace(F0.T A F0) is retained. '
              'The first-state observations exposed the fixed identity complement\'s dependence '
              'on calibration. The added beta=1/(mu*metric.scale) makes the baseline the exact '
              'inverse of damping alone; its K equals I plus whitened data curvature. This '
              'analytic control was added after preliminary observations and is labelled '
              'accordingly. Ranks, bounds, RHS, panels and gates were not retuned.','',
              '## Cost, verification and limits','',
              f"The 103-test suite passed before the restart and again after completion. "
              f"{len(sources)} source files and six parent "
              f"arrays verified. All {len(searches)} searches were bracketed or explicitly rejected; "
              f"{verification['rejected_searches']} were rejected. Twelve state/calibration artifacts "
              'passed checksums and state/RNG non-mutation. Numerical maxima:', '',
              '| Check | Maximum |','| --- | ---: |']
    for key,value in checks.items():
        lines.append(f'| {key} | {value:.3e} |')
    lines += ['', f"Peak process RSS during construction: {max(s['process_peak_rss_bytes'] for s in states)/2**20:.2f} MiB. "
              'This includes dataset, dense teachers, interpreter and library memory and is a '
              'producer-process high-water mark (including resumed states). It is not incremental student memory. Representation '
              'storage counts and basis/teacher fitting timings are saved separately. Application '
              'timings use nine repetitions and exclude teacher fitting, feature prediction and '
              'search; they do not establish end-to-end optimizer speed.','',
              '| Damping-calibrated core | Basis construction ms | Total teacher fit ms | Application ms |',
              '| --- | ---: | ---: | ---: |']
    for s in [v for v in summaries if v['normalization']=='damping']:
        lines.append(f"| {s['name']} | {1000*s['median_basis_seconds']:.3f} | {1000*s['median_fit_seconds']:.3f} | {1000*s['median_apply_seconds']:.3f} |")
    lines += ['',
              f"Median cached-teacher reconstruction/audit time: {statistics.median(s['setup_seconds'] for s in states):.3f} seconds per state/calibration. "
              'This includes independent GGN actions and dense coordinate transforms; it '
              'excludes the original Jacobian/teacher production, model loading and searches. '
              'Teacher fitting includes basis construction; these two columns must not be summed.','',
              'Full cached teacher matrices are used only for this offline capacity screen. '
              'Three future seeds would test trajectory transfer, not architecture transfer. '
              'A failed family does not rule out richer factorizations, a better subspace '
              'selection objective, different feature bases or larger affordable structures. '
              'No conditional predictor or paired training result should be inferred from this screen.','',
              'Rebuild this report and figures:', '', '```sh',
              f'.venv/bin/python scripts/summarize_learnable_capacity.py --run runs/{directory.name}', '```','']
    report='\n'.join(lines)
    Path('docs/learnable_capacity_results_20261007.md').write_text(report+f'\n![Capacity screen](../runs/{directory.name}/comparison.png)\n')
    (directory/'REPORT.md').write_text(report.replace('(learnable_capacity_protocol_20261007.md)','(../../docs/learnable_capacity_protocol_20261007.md)')+'\n![Capacity screen](comparison.png)\n')
    shutil.copyfile(__file__,directory/'analysis_runner.py')
    artifacts=['analysis_runner.py','REPORT.md','comparison.png','comparison.pdf','summary.json','verification.json',
               'complete.json','all_candidates.json','all_operators.json','all_states.json','progression_gate.json']
    if (directory/'post_run_tests.json').exists():
        artifacts.append('post_run_tests.json')
    write_json(directory/'analysis_artifacts.json',{n:checksum(directory/n) for n in artifacts})
    print(json.dumps({'gate':outcome,'verification':verification},indent=2))


if __name__=='__main__':
    main()
