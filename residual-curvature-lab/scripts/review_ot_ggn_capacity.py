"""Read-only verification of the study's execution of the main OT/GGN protocol.

Writes a separate main-lab review, never modifies the study or external run.
Replays selected saved proposals without searching or fitting any candidate.
"""
import argparse
import json
import shutil
import statistics
from pathlib import Path

import torch

from residual_curvature.config import load_config
from residual_curvature.curvature.operators import loss
from residual_curvature.data import dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import capacity
from residual_curvature.provenance import checksum, write_json
from run_learnable_capacity import PARENT, TRAINING, read

STUDY=Path('studies/gap_audit_20261007')
DEFAULT_RUN=Path('/Users/georgtirpitz/skipsoap-study-data/gap_audit_20261007/runs/20261007T124647-1d8487a2')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--output',type=Path,default=Path('runs/20261008_ot_ggn_capacity_review'))
    args=parser.parse_args()
    run,out=args.run,args.output
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert read(run/'complete.json')['completed'] and not (run/'failure.json').exists()
    source_hashes=read(run/'source_archive_checksums.json')
    for name,digest in source_hashes.items():
        assert checksum(run/'source'/name)==digest
        # Replays use these modules, so do not accept a changed implementation.
        if name.startswith(('src/','scripts/','tests/','docs/')):
            assert checksum(name)==digest
    assert checksum(run/'supplied_critique.txt')==read(run/'protocol.json')['attachment_sha256']
    for name,digest in read(run/'parent_arrays.json').items():
        assert checksum(name)==digest
    rows,ops,states=[read(run/name) for name in ['all_candidates.json','all_operators.json','all_states.json']]
    assert (len(rows),len(ops),len(states))==(420,204,12)
    mirrors=STUDY/'runs'/run.name
    for name in ['all_candidates.json','all_operators.json','all_states.json','complete.json','protocol.json']:
        assert checksum(mirrors/name)==checksum(run/name)
    keys=[(r['target'],r['trajectory'],r['step'],r['name'],r['rhs']) for r in rows]
    assert len(set(keys))==420
    assert all((r['search']['bracketed'] or r['search']['rejected']) and not r['search']['boundary'] for r in rows)
    assert all(s['nonmutation_passed'] for s in states)
    numerical={k:max(s[k] for s in states) for k in ['max_factor_error','max_action_error',
               'cached_teacher_energy_error','teacher_gradient_residual','teacher_max_backward_error']}
    numerical['max_dense_application_error']=max(o['max_dense_application_error'] for o in ops)
    assert max(numerical.values())<=1e-8 and min(o['min_core_eigenvalue'] for o in ops)>0
    families=[f[0] for f in read(run/'protocol.json')['families']]
    mean=lambda target,name,rhs:statistics.mean(r['selected']['independent_mean_decrease'] for r in rows
        if r['target']==target and r['name']==name and r['rhs']==rhs)
    summary=[]
    for target in ['ggn','ot']:
        for name in families:
            selected=[o for o in ops if o['target']==target and o['name']==name]
            item={'target':target,'name':name}
            for rhs in ['gradient','momentum']:
                value=mean(target,name,rhs)
                gap=mean(target,'exact',rhs)-mean(target,'identity',rhs)
                item[rhs]={'mean_decrease':value,'reference_advantage':gap,
                           'recovery':(value-mean(target,'identity',rhs))/gap if gap>1e-4 else None}
            item['pooled_errors']={group:{kind:sum(o['errors'][group][kind+'_numerator_sum'] for o in selected)/
                                       sum(o['errors'][group]['denominator_sum'] for o in selected)
                                       for kind in ['raw','ray']} for group in ['gradient','momentum','physical','factor']}
            item['storage_model_vectors']=max(o['storage_model_vectors'] for o in selected)
            item['median_application_ratio']=statistics.median(o['apply_seconds_median']/next(b['apply_seconds_median']
                for b in ops if b['target']==target and b['name']=='identity' and b['trajectory']==o['trajectory'] and b['step']==o['step']) for o in selected)
            item['median_fit_seconds']=statistics.median(o['fit_seconds_including_basis'] for o in selected)
            item['median_basis_seconds']=statistics.median(o['basis_seconds'] for o in selected)
            eligible=name not in ['identity','exact'] and not name.startswith(('teacher','combined_teacher'))
            item['current_gradient_shortlist']=bool(eligible and item['gradient']['recovery'] is not None
                and item['gradient']['recovery']>=.8 and item['storage_model_vectors']<=4.25 and item['median_application_ratio']<=2.)
            prior=read(run/'check3_summary.json')['families'][name][target]
            assert abs(item['gradient']['mean_decrease']-prior['gradient_mean_decrease'])<1e-14
            assert item['current_gradient_shortlist']==prior['shortlist']
            summary.append(item)
    # Independently replay the stored operators and selected scalars, with no new search.
    c=load_config(TRAINING/'resolved.yaml')
    x,y,manifest=dataset(c)
    assert manifest==read(run/'split_manifest.json')
    model=make_model(c)
    registry=Registry(model)
    replay=[]
    count_artifacts=0
    for trajectory in ['searched_soap','exact_tensor_ggn']:
        for step in [10,25,50]:
            key=f'{trajectory}-{step:04d}'
            arrays=torch.load(PARENT/key/'arrays.pt',map_location='cpu',weights_only=False)
            info=read(PARENT/key/'diagnostics.json')
            flat,contexts=arrays['parameters'],arrays['contexts']
            commit(model,registry,flat)
            ids=[info['construction_ids']]+info['panel_ids']
            functions=[(lambda w,fn=functional_logits(model,registry,x[ii]),yy=y[ii]:loss(fn(w),yy)) for ii in ids]
            base=[float(fn(flat)) for fn in functions]
            paired_rhs=None
            for target in ['ggn','ot']:
                folder=run/f'{target}-{key}'
                for name,digest in read(folder/'artifacts.json').items():
                    assert checksum(folder/name)==digest
                    count_artifacts+=1
                saved=torch.load(folder/'operators.pt',map_location='cpu',weights_only=False)
                for name in ['candidates','operators']:
                    full=rows if name=='candidates' else ops
                    assert read(folder/(name+'.json'))==[r for r in full if r['target']==target and r['trajectory']==trajectory and r['step']==step]
                if paired_rhs is not None:
                    assert all(torch.equal(saved['rhs_groups'][name],paired_rhs[name]) for name in saved['rhs_groups'])
                paired_rhs=saved['rhs_groups']
                for name in ['identity','exact','state_4','combined_state_3','teacher_16']:
                    for rhs in ['gradient','momentum']:
                        row=next(r for r in rows if r['target']==target and r['trajectory']==trajectory and r['step']==step and r['name']==name and r['rhs']==rhs)
                        delta=-capacity.apply(saved['models'][name],registry,contexts,saved['beta'],saved['rhs_groups'][rhs])
                        scale=row['search']['radius']*row['search']['selected_alpha']/float(delta.norm())
                        delta=delta*scale
                        decreases=[b-float(fn(flat+delta)) for b,fn in zip(base,functions)]
                        expected=[row['selected']['actual_decrease']]+row['selected']['independent_decreases']
                        error=max(abs(a-b) for a,b in zip(decreases,expected))
                        assert error<=1e-8,(target,key,name,rhs,error)
                        replay.append({'state':key,'target':target,'name':name,'rhs':rhs,'max_loss_discrepancy':error})
            print(f'Verified source/paired RHS and replayed 20 selected proposals: {key}',flush=True)
    verification={'source_run':str(run),'new_experiment_launched':False,'archived_sources_verified':len(source_hashes),
                  'current_implementation_matches':True,'parent_arrays_verified':6,'operator_artifacts_verified':count_artifacts,
                  'study_mirror_matches':True,'identical_OT_GGN_rhs':True,'counts':[420,204,12],
                  'numerical_checks':numerical,'selected_proposals_replayed':len(replay),
                  'max_replayed_loss_discrepancy':max(v['max_loss_discrepancy'] for v in replay),
                  'current_gradient_shortlist':[s for s in summary if s['current_gradient_shortlist']]}
    write_json(out/'verification.json',verification)
    write_json(out/'summary.json',summary)
    write_json(out/'replayed_proposals.json',replay)
    study_report='../studies/gap_audit_20261007/RESULTS.md'
    lines=['# Matched OT/GGN capacity: verified results — 8 October 2026','',
        'The user-authorized comparison is complete. Claude Code executed the existing '
        'main-lab protocol in the isolated gap-audit study on 7 October, before this agent '
        'resumed. This review reuses that execution and independently verifies it; it does '
        'not launch duplicate experiments or change the study.','',
        f'External run: `{run}`. [Study report]({study_report}); '
        '[prospective main-lab protocol](ot_ggn_capacity_protocol_20261007.md). '
        'Six saved seed-0 ViT states (SOAP/GGN trajectories at 10/25/50), MNIST batch 512, '
        'all 40 tensors, float64, one thread, target-specific exact-trace gamma .01 damping '
        'and the same global 52-evaluation search. Four reused analysis panels are '
        'development data. No OT-trajectory states or new optimization trajectories are '
        'included in this capacity comparison.','',
        '## Findings','',
        'The strongest affordable current-gradient family is a teacher-fitted diagonal plus '
        'three state-derived directions. It recovers 57.0% of OT’s own local teacher advantage '
        'versus 46.4% for GGN. Storage is 4.063 model-sized vectors; neither target reaches '
        'the 80% current-gradient shortlist. These coefficients and the diagonal still require '
        'expensive teacher information; the family is not a deployable learned optimizer.','',
        'OT has a modest nonlinear-score advantage for the state and combined-state families. '
        'For combined rank 3, the absolute mean decrease is .032685 versus .031652 for GGN '
        '(difference .001033). OT’s exact teacher has less aggregate headroom: .023469 '
        'versus .026569. With the common GGN gap as denominator, the combined candidates '
        'recover 50.3% (OT) versus 46.4% (GGN). Thus the own-teacher 10.5-percentage-point '
        'difference partly reflects a smaller target gap.','',
        'The study’s prospectively recorded comparative reading is met: at least two of '
        '{diagonal, state_4, combined_state_3} gain >=10 percentage points of own-teacher '
        'recovery and do not decrease their absolute score. This is a thin development '
        'result, not broad evidence that OT is a better optimization target. Inverse '
        'accuracy does not improve: combined-rank-3 pooled gradient ray error is .601 '
        'under OT versus .568 under GGN, each measured in its own teacher energy.','',
        'The combined family addresses the attachment’s missing control: separate diagonal '
        'and identity-complement low-rank models were not the combined model. The new '
        'control improves over state rank 4 at a slightly smaller storage cost. It is a '
        'sequential fit in diagonally adjusted coordinates, so its isotropic fitting '
        'distribution also changes. No joint optimum or isolated representation effect '
        'is claimed. The separate gradient/history-weighted fitting experiment remains unrun.','',
        '## All current-gradient candidates','',
        '| Family | GGN mean decrease | OT mean decrease | GGN own recovery | OT own recovery | Stored vectors | OT apply / SOAP map |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    get=lambda target,name:next(s for s in summary if s['target']==target and s['name']==name)
    for name in families:
        g,o=get('ggn',name),get('ot',name)
        lines.append(f"| {name} | {g['gradient']['mean_decrease']:.6f} | {o['gradient']['mean_decrease']:.6f} | {g['gradient']['recovery']:.1%} | {o['gradient']['recovery']:.1%} | {o['storage_model_vectors']:.3f} | {o['median_application_ratio']:.2f} |")
    lines += ['', 'Application timing excludes teacher fitting, basis construction, prediction and '
        'line search; it is not end-to-end optimizer overhead. Original process timing was '
        'collected under the study’s concurrent workload. Fitting/basis times and errors '
        'are retained in the review JSON and original artifacts.','',
        '## Momentum and graph interpretation','',
        'Exact OT on momentum has mean independent decrease .012390, slightly below the '
        'SOAP-momentum map .012656. Its aggregate recovery denominator is negative, so '
        'OT momentum recovery is undefined. The rank-16 teacher correction on momentum '
        'gives .047773 (OT) and .046187 (GGN), but uses 17.747 model vectors and expensive '
        'teacher-selected directions. These absolute local scores do not establish a '
        'training win. Same-RHS GGN momentum ratios have a small .003260 denominator '
        'and should not be promoted as practical success.','',
        'The OT graph is over output classes. Its metric is the pullback '
        '`J_p.T L(p)^dagger J_p`, with zero-sum probability tangents. A graph over '
        'parameter tensors would be a different construction; the tested tensor teachers '
        'discard cross-tensor blocks. The output graph does not remove the Jacobian or '
        'the cost of the parameter-space inverse.','',
        'Categorical GGN also has an exact graph interpretation: `H=diag(p)-p p.T` is '
        'the Laplacian of a complete class graph with edge weights `p_i p_j`, and '
        '`C=J_z.T H J_z`. OT uses `J_z.T H L(p)^dagger H J_z`. This identity is '
        'covered by the new unit test. Graph structure is meaningful for both targets; '
        'OT’s explicit Laplacian alone does not establish a cheaper or better tensor GNN.','',
        '## Broader gap-audit context','',
        'The separate study reports three fresh-seed, 500-step comparisons: GGN loss '
        '.355 at step 50 versus .697 for tuned searched SOAP; at step 500, .182 '
        'versus .210. Tuned native SOAP ends at .161 and reaches the study’s targets '
        'much faster. GGN and SOAP were tuned at different horizons, and concurrent '
        'load affects timings. This strengthens evidence for an early per-step GGN '
        'advantage while weakening the case for a general runtime advantage.','',
        'The study also reports that dense empirical-Fisher training and removing SOAP '
        'momentum do not reproduce earlier local gains. Its check 4b remains partial '
        '(21/22 units); that interrupted study was not repaired or rerun here. Broader '
        'training outcomes are attributed to the study, not independently checkpoint-replayed '
        'by this review. See its report and deviations for full settings and limits.','',
        '## Independent verification','',
        f"Verified {len(source_hashes)} archived source hashes, current implementation agreement, "
        'six parent arrays, twelve operator artifacts, mirrored JSON, all 420 search '
        'records and positive cores. Identical physical/factor/gradient/momentum RHS '
        'were confirmed across OT and GGN. Independently applied stored operators and '
        'selected scalars for identity, exact, state_4, combined_state_3 and teacher_16, '
        f"both RHS, all states/targets: {len(replay)} selected proposals replayed. Maximum "
        f"loss discrepancy: {verification['max_replayed_loss_discrepancy']:.3e}. No search or fitting was repeated.", '',
        f'[Verification JSON](../{out}/verification.json); [all aggregate results](../{out}/summary.json). '
        'The main suite has 106 passing tests; the separate study suite has 12 passing tests.','',
        'The next decision remains how to obtain a more useful affordable representation '
        'and fitting signal. Neither target currently supplies an 80%-recovery cheap '
        'candidate on these states. Predictor training and new OT trajectories were not '
        'part of the selected steps 1 and 2 and were not launched.','']
    Path('docs/ot_ggn_capacity_results_20261008.md').write_text('\n'.join(lines))
    shutil.copyfile(__file__,out/'review_runner.py')
    artifacts=['verification.json','summary.json','replayed_proposals.json','review_runner.py']
    write_json(out/'artifacts.json',{name:checksum(out/name) for name in artifacts})
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':
    main()
