"""Experiment 1: saved-state inverse capacity screen, no learned predictors."""
import copy
import json
import resource
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch

from residual_curvature.config import load_config
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import capacity
from residual_curvature.optim.soap import SOAPMetric, frozen_apply
from residual_curvature.provenance import new_run, checksum, write_json
from run_large_batch_oracle import measurement, state_equal
from run_mini_vit_allocation import search_candidate


PARENT = Path('runs/20261005T143029-9b582ed5')
TRAINING = Path('runs/20261005T135920-d7ce29b2')
FAMILIES = [(name, name, None) for name in ['identity', 'scalar', 'diagonal', 'exact']]
FAMILIES += [(f'{name}_{rank}', name, rank) for name in ['teacher', 'coordinate', 'state'] for rank in [4,8,16]]


def read(path):
    return json.loads(path.read_text())


def benchmark(fn, value):
    fn(value)
    elapsed = []
    for _ in range(9):
        start = time.perf_counter()
        fn(value)
        elapsed.append(time.perf_counter()-start)
    return statistics.median(elapsed)


def audit(source, destination, c, x, y, registry, model, normalization,
          target='ggn', families=None, common_factor_rhs=False):
    families = FAMILIES if families is None else families
    destination.mkdir()
    for name, digest in read(source/'artifacts.json').items():
        assert checksum(source/name) == digest
    arrays = torch.load(source/'arrays.pt', map_location='cpu', weights_only=False)
    parent_info = read(source/'diagnostics.json')
    flat, g, momentum, contexts = [arrays[k] for k in ['parameters','gradient','momentum_rhs','contexts']]
    commit(model, registry, flat)
    model_before, contexts_before = copy.deepcopy(model.state_dict()), copy.deepcopy(contexts)
    rng = torch.get_rng_state().clone()
    construction, panel_ids = parent_info['construction_ids'], parent_info['panel_ids']
    fn = functional_logits(model, registry, x[construction])
    objective = lambda w: loss(fn(w), y[construction])
    op = GGN(fn, flat, y[construction])
    assert relative_error(op.gradient, g) < 1e-8
    metric = SOAPMetric(registry, contexts)
    if target == 'ggn':
        blocks = arrays['ggn_blocks']
        construction_info = {'source':'cached exact GGN blocks','construction_seconds':None}
        target_action = op.action
    elif target == 'ot':
        from residual_curvature.curvature import wasserstein
        blocks,construction_info = wasserstein.streamed_wasserstein_blocks(
            model,registry,flat,x[construction],8,c['resources']['budget_mib'])
        target_action = lambda v:wasserstein.action(model,registry,flat,x[construction],v)
    else:
        raise ValueError(target)
    mu = max(.01*sum(float(b.trace()) for b in blocks)/registry.size, c['damping']['floor'])
    start = time.perf_counter()
    systems, raw_ks, factors = [], [], []
    factor_errors, action_errors = [], []
    gen = torch.Generator().manual_seed(270010+parent_info['step'])
    for spec, block in zip(registry.specs, blocks):
        context = contexts[spec.name]
        eye = torch.eye(spec.count,dtype=flat.dtype)
        f = capacity.factor(eye.reshape(spec.shape+(spec.count,)), context).reshape(spec.count,spec.count)
        p = frozen_apply(eye.reshape(spec.shape+(spec.count,)),context).reshape(spec.count,spec.count)
        factor_errors.append(relative_error(f@f.T,p))
        omega = metric.scale*frozen_apply(eye.reshape(spec.shape+(spec.count,)),context,inverse=True).reshape(spec.count,spec.count)
        a = block+mu*omega
        systems.append(a)
        raw_ks.append(f.T@a@f)
        factors.append(f)
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count,dtype=flat.dtype,generator=gen)
        action_errors.append(relative_error(block@v[spec.slice],target_action(v)[spec.slice]))
    beta = registry.size/sum(float(k.trace()) for k in raw_ks) if normalization=='trace' else 1/(mu*metric.scale)
    ks = [beta*k for k in raw_ks]
    setup_seconds = time.perf_counter()-start
    assert max(factor_errors+action_errors) <= 1e-8
    # Evaluation draws are disjoint from operator fitting (which uses an analytic expectation).
    gen = torch.Generator().manual_seed(270100+parent_info['step']+(0 if parent_info['trajectory']=='searched_soap' else 1000))
    physical = torch.randn(registry.size,16,dtype=flat.dtype,generator=gen)
    zs = torch.randn(registry.size,16,dtype=flat.dtype,generator=gen)
    factor_rhs = torch.empty_like(zs)
    zg, zm = [], []
    for spec in registry.specs:
        context = contexts[spec.name]
        factor_rhs[spec.slice] = capacity.factor(zs[spec.slice].reshape(spec.shape+(16,)),context,transpose=True,inverse=True).reshape(spec.count,16)/(1. if common_factor_rhs else beta**.5)
        zg.append(beta**.5*capacity.factor(g[spec.slice].reshape(spec.shape),context,transpose=True).reshape(-1))
        zm.append(beta**.5*capacity.factor(momentum[spec.slice].reshape(spec.shape),context,transpose=True).reshape(-1))
    rhs_groups = {'gradient':g, 'momentum':momentum, 'physical':physical, 'factor':factor_rhs}
    chols = [torch.linalg.cholesky((k+k.T)/2) for k in ks]
    def reference(rhs):
        cores = [{'kind':'cholesky','cholesky':ch} for ch in chols]
        return capacity.apply(cores,registry,contexts,beta,rhs)
    exact_values = {name:reference(rhs) for name,rhs in rhs_groups.items()}
    # Ill-conditioned physical systems can amplify roundoff in Euclidean solutions.
    # Check teacher energy agreement AND equation residual, at unchanged tolerance.
    if target == 'ggn':
        independent_teacher = -arrays['raw_ggn_gradient']
    else:
        from residual_curvature.curvature.layerwise import solve_blocks
        teacher_steps,_ = solve_blocks(blocks,registry,g,metric,[mu],c['solver']['tolerance64'])
        independent_teacher = -teacher_steps[mu]
    cache_error = capacity.energy_errors(g,exact_values['gradient'],independent_teacher,systems,registry)['raw'][0]**.5
    gradient_residual = relative_error(torch.cat([a@exact_values['gradient'][s.slice] for a,s in zip(systems,registry.specs)]),g)
    assert max(cache_error,gradient_residual) <= 1e-8
    backward_errors = []
    for key,rhs in rhs_groups.items():
        for a,s in zip(systems,registry.specs):
            value = exact_values[key][s.slice]
            backward_errors.append(float((a@value-rhs[s.slice]).norm()/(a.norm()*value.norm()+rhs[s.slice].norm()).clamp_min(1e-300)))
    assert max(backward_errors) <= 1e-8
    panels = [lambda w,f=functional_logits(model,registry,x[ids]),targets=y[ids]:loss(f(w),targets) for ids in panel_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    native = arrays['raw_soap']
    rows, diagnostics, models = [], [], {}
    def measure(name, rhs_name, delta):
        selected, search = search_candidate(flat,delta,native,objective)
        values = measurement(selected,op,metric,mu,objective,base,panels,panel_base,registry)
        block_curvature = sum(float(selected[s.slice]@a@selected[s.slice]) for a,s in zip(systems,registry.specs))
        values['block_teacher_quadratic_decrease'] = -float(g@selected)-.5*block_curvature
        values['block_teacher_optimized_ray_decrease'] = .5*min(float(g@selected),0)**2/block_curvature if block_curvature else 0.
        target_curvature = float(selected@target_action(selected)) if target=='ot' else values['ggn_quadratic']
        values['full_teacher_quadratic_decrease'] = -float(g@selected)-.5*(target_curvature+mu*float(selected@metric.apply(selected)))
        row = {'target':target,'normalization':normalization,'trajectory':parent_info['trajectory'],'step':parent_info['step'],'name':name,'rhs':rhs_name,
               'selected':values,'raw_norm':float(delta.norm()),'search':search}
        rows.append(row)
        write_json(destination/'candidates.json',rows)
        return row
    measure('native_soap','momentum',native)
    for name,family,rank in families:
        started = time.perf_counter()
        cores = [capacity.fit_core(k,family,rank,a,b,contexts[s.name]['denominator'])
                 for k,a,b,s in zip(ks,zg,zm,registry.specs)]
        fit_seconds = time.perf_counter()-started
        models[name] = cores
        apply = lambda rhs:capacity.apply(cores,registry,contexts,beta,rhs)
        errors = {key:capacity.energy_errors(rhs,apply(rhs),exact_values[key],systems,registry) for key,rhs in rhs_groups.items()}
        if family == 'exact':
            assert max(errors['physical']['raw']) < 1e-16
            assert relative_error(apply(g),exact_values['gradient']) <= 1e-8
        dense_checks, minimum, gradient_checks = [], [], []
        applied_gradient = apply(g)
        for s,core,f,k in zip(registry.specs,cores,factors,ks):
            # Factorized deployment application independently checked against dense reference.
            dense = capacity.dense_core(core,s.count,k)
            minimum.append(float(torch.linalg.eigvalsh((dense+dense.T)/2).min()))
            expected_operator = beta*f@dense@f.T
            context = contexts[s.name]
            eye = torch.eye(s.count,dtype=g.dtype).reshape(s.shape+(s.count,))
            factored = capacity.factor(eye,context,transpose=True).reshape(s.count,s.count)
            factored = capacity.core_apply(core,factored).reshape(s.shape+(s.count,))
            got_operator = beta*capacity.factor(factored,context).reshape(s.count,s.count)
            dense_checks.append(relative_error(got_operator,expected_operator))
            # Forward relative error can be amplified by operator conditioning.
            gradient_checks.append(relative_error(applied_gradient[s.slice],expected_operator@g[s.slice]))
        assert min(minimum) > 0 and max(dense_checks) <= 1e-8
        apply_seconds = benchmark(apply,g)
        info = {'target':target,'normalization':normalization,'name':name,'trajectory':parent_info['trajectory'],'step':parent_info['step'],
                'fit_seconds_including_basis':fit_seconds,'apply_seconds_median':apply_seconds,
                'basis_seconds':sum(v.get('basis_seconds',0.) for v in cores),
                'coefficient_fit_seconds':sum(v.get('coefficient_fit_seconds',0.) for v in cores),
                'stored_numbers':sum(v['storage'] for v in cores),'storage_model_vectors':sum(v['storage'] for v in cores)/registry.size,
                'clipped':sum(v['clipped'] for v in cores),'min_core_eigenvalue':min(minimum),
                'max_dense_application_error':max(dense_checks),
                'dense_check_kind':'operator_frobenius',
                'max_gradient_forward_discrepancy':max(gradient_checks),'errors':errors,
                'effective_ranks':{s.name:v.get('rank',0) for s,v in zip(registry.specs,cores)}}
        diagnostics.append(info)
        write_json(destination/'operators.json',diagnostics)
        for rhs_name in ['gradient','momentum']:
            row = measure(name,rhs_name,-apply(rhs_groups[rhs_name]))
            if rhs_name=='gradient':
                print(f"{target} {normalization} {source.name} {name}: independent={row['selected']['independent_mean_decrease']:.6f}; physical ray error={sum(errors['physical']['ray'])/16:.4f}",flush=True)
    assert state_equal(model_before,model.state_dict()) and state_equal(contexts_before,contexts)
    assert torch.equal(rng,torch.get_rng_state())
    torch.save({'models':models,'beta':beta,'mu':mu,'rhs_groups':rhs_groups,
                'target':target,'teacher_blocks':blocks},destination/'operators.pt')
    write_json(destination/'artifacts.json',{'operators.pt':checksum(destination/'operators.pt')})
    info = {'target':target,'normalization':normalization,'trajectory':parent_info['trajectory'],'step':parent_info['step'],'beta':beta,'mu':mu,
            'teacher_construction':construction_info,'common_factor_rhs':common_factor_rhs,
            'setup_seconds':setup_seconds,'max_factor_error':max(factor_errors),'max_action_error':max(action_errors),
            'cached_teacher_energy_error':cache_error,'teacher_gradient_residual':gradient_residual,
            'teacher_max_backward_error':max(backward_errors),
            'nonmutation_passed':True,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    write_json(destination/'diagnostics.json',info)
    return rows,diagnostics,info


def gate(rows,operators):
    means = {name:statistics.mean(r['selected']['independent_mean_decrease'] for r in rows if r['name']==name and r['rhs']=='gradient') for name,_,_ in FAMILIES}
    gap = means['exact']-means['identity']
    pooled = {}
    for name,_,_ in FAMILIES:
        selected = [r for r in operators if r['name']==name]
        pooled[name] = {group:sum(r['errors'][group]['ray_numerator_sum'] for r in selected)/sum(r['errors'][group]['denominator_sum'] for r in selected) for group in ['physical','factor']}
    decisions = []
    for name,family,rank in FAMILIES:
        selected = [r for r in operators if r['name']==name]
        ratios = [r['apply_seconds_median']/next(v['apply_seconds_median'] for v in operators if v['name']=='identity' and v['trajectory']==r['trajectory'] and v['step']==r['step']) for r in selected]
        recovery = (means[name]-means['identity'])/gap if gap>1e-4 else None
        gains = {group:1-pooled[name][group]/pooled['identity'][group] for group in ['physical','factor']}
        storage = max(r['storage_model_vectors'] for r in selected)
        cost = statistics.median(ratios)
        eligible_family = family in ['scalar','diagonal','coordinate','state']
        passed = eligible_family and recovery is not None and recovery>=.8 and min(gains.values())>=.8 and storage<=4.25 and cost<=2.
        decisions.append({'name':name,'family':family,'independent_mean_decrease':means[name],
                          'local_recovery':recovery,'pooled_ray_errors':pooled[name],'operator_error_reductions':gains,
                          'storage_model_vectors':storage,'median_application_ratio':cost,'passed':passed})
    return {'aggregate_exact_headroom':gap,'candidates':decisions,'experiment2_selected':any(r['passed'] for r in decisions),
            'qualifying':[r['name'] for r in decisions if r['passed']]}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args()
    assert read(PARENT/'complete.json')['completed']
    c = load_config(TRAINING/'resolved.yaml')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    directory = new_run(c)
    print(f'Capacity experiment: {directory.resolve()}',flush=True)
    sources = read(directory/'provenance.json')['source_checksums']
    for name in ['scripts/run_learnable_capacity.py','scripts/run_large_batch_oracle.py','scripts/run_mini_vit_allocation.py',
                 'docs/learnable_capacity_protocol_20261007.md','tests/test_capacity.py']:
        sources[name] = checksum(name)
    for name,digest in sources.items():
        target = directory/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,target)
        assert checksum(target)==digest
    write_json(directory/'source_archive_checksums.json',sources)
    write_json(directory/'protocol.json',{'document':'docs/learnable_capacity_protocol_20261007.md','parent':str(PARENT),'families':FAMILIES,'training_launched':False})
    x,y,manifest = dataset(c)
    assert manifest==read(TRAINING/'split_manifest.json')
    write_json(directory/'split_manifest.json',manifest)
    model = make_model(c)
    registry = Registry(model)
    rows,operators,states,parents = [],[],[],{}
    completed = {}
    if args.resume_from:
        for saved in read(args.resume_from/'all_states.json'):
            key=f"{saved['normalization']}-{saved['trajectory']}-{saved['step']:04d}"
            completed[key]=saved
        resume_sources=read(args.resume_from/'source_archive_checksums.json')
        for name,digest in resume_sources.items():
            assert checksum(args.resume_from/'source'/name)==digest
        write_json(directory/'resumed_source_archive.json',{'source_run':str(args.resume_from),'source_checksums':resume_sources})
    for normalization,trajectory,step in [(n,t,s) for n in ['trace','damping'] for t in ['searched_soap','exact_tensor_ggn'] for s in [10,25,50]]:
            source = PARENT/f'{trajectory}-{step:04d}'
            key=f'{normalization}-{source.name}'
            parents[str(source/'arrays.pt')] = checksum(source/'arrays.pt')
            write_json(directory/'parent_arrays.json',parents)
            try:
                if key in completed:
                    old=args.resume_from/key
                    for name,digest in read(old/'artifacts.json').items():
                        assert checksum(old/name)==digest
                    s=completed[key]
                    assert s['nonmutation_passed']
                    r,o=read(old/'candidates.json'),read(old/'operators.json')
                    assert len(r)==27 and len(o)==13
                    shutil.copytree(old,directory/key)
                    prior_origin=read(old/'origin.json') if (old/'origin.json').exists() else None
                    write_json(directory/key/'origin.json',{'source_run':str(args.resume_from),
                               'verified_completed_state':True,'prior_origin':prior_origin})
                    print(f'Reused verified completed state: {key}',flush=True)
                else:
                    r,o,s = audit(source,directory/key,c,x,y,registry,model,normalization)
            except Exception as error:
                write_json(directory/'failure.json',{'state':source.name,'normalization':normalization,'error':repr(error)})
                raise
            rows.extend(r); operators.extend(o); states.append(s)
            write_json(directory/'all_candidates.json',rows)
            write_json(directory/'all_operators.json',operators)
            write_json(directory/'all_states.json',states)
    gates = {n:gate([r for r in rows if r['normalization']==n],[r for r in operators if r['normalization']==n]) for n in ['trace','damping']}
    outcome = {'normalizations':gates,'experiment2_selected':any(g['experiment2_selected'] for g in gates.values())}
    write_json(directory/'progression_gate.json',outcome)
    write_json(directory/'complete.json',{'completed':True,'states':6,'normalizations':2,'candidate_steps':len(rows),'operator_states':len(operators)})
    print(json.dumps(outcome,indent=2),flush=True)
    print(f'Completed: {directory.resolve()}',flush=True)


if __name__=='__main__':
    main()
