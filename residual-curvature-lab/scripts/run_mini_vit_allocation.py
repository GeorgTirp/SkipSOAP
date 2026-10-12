"""Selected paired small-ViT experiment with exact reduced SOAP allocation."""
import argparse
import copy
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import torch

from residual_curvature import checkpoint
from residual_curvature.config import load_config
from residual_curvature.curvature.layerwise import streamed_blocks, solve_blocks, workspace_estimate
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.reduced import trace_estimate
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.search import broad_direction_search
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.reduced_allocation import reduced_allocation
from residual_curvature.optim.soap import SOAPMetric, soap_propose
from residual_curvature.provenance import checksum, new_run, write_json
from run_exact_training_followup import evaluate
from run_large_batch_oracle import measurement, state_equal


def search_candidate(flat, delta, native, objective):
    if float(delta.norm()) == 0:
        return delta, {'policy': 'zero_direction', 'rejected': True, 'bracketed': False,
                       'boundary': False, 'trials': [], 'evaluations': 0, 'selected_alpha': 0.}
    selected, search = broad_direction_search(flat, delta, float(native.norm()), objective)
    if not search['bracketed'] and not search['rejected']:
        raise RuntimeError('Positive search selection reached the declared radius boundary')
    return selected, search


def propose(method, model, registry, flat, state, x, y, c, step):
    fn = functional_logits(model, registry, x)
    objective = lambda v: loss(fn(v), y)
    op = GGN(fn, flat, y)
    proposal, next_state, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(op.gradient), state, c['soap'])
    native = registry.flatten(proposal)
    delta = native
    info = {'method': method, 'native_norm': float(native.norm()), 'trace': None,
            'allocation': None, 'solve': None, 'search': None}
    if method in ['allocation_diagonal', 'allocation_full', 'exact_tensor_ggn'] and all(v is not None for v in contexts.values()):
        metric = SOAPMetric(registry, contexts)
        trace, trace_info = trace_estimate(op, 8, c['analysis_seed'] + step)
        mu = max(.01 * trace / registry.size, c['damping']['floor'])
        info.update(trace=trace_info, trace_estimate=trace, mu=mu)
        if method == 'exact_tensor_ggn':
            blocks, curvature = streamed_blocks(model, registry, flat, x, 8, c['resources']['budget_mib'])
            solutions, solve_info = solve_blocks(blocks, registry, op.gradient, metric, [mu], c['solver']['tolerance64'])
            delta = solutions[mu]
            info.update(curvature=curvature, solve=solve_info[mu])
        else:
            candidates, allocation, _ = reduced_allocation(op, native, registry, metric, mu)
            delta = candidates[method]
            info['allocation'] = allocation
    info['raw_norm'] = float(delta.norm())
    if method != 'native_soap' and float(native.norm()) > 0:
        delta, info['search'] = search_candidate(flat, delta, native, objective)
    info.update(loss_before=float(objective(flat)), loss_after=float(objective(flat+delta)),
                norm=float(delta.norm()), slope=float(op.gradient @ delta), counts=op.counts)
    return delta, next_state, contexts, info


def snapshot(model, registry, optimizer, x, y, ids, panels_ids, c, step, directory):
    directory.mkdir(parents=True)
    started = time.perf_counter()
    before_model, before_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer)
    rng = torch.get_rng_state().clone()
    flat = registry.flatten(dict(model.named_parameters())).detach()
    fn = functional_logits(model, registry, x[ids])
    op = GGN(fn, flat, y[ids])
    g = op.gradient
    soap_start = time.perf_counter()
    proposal, _, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), optimizer, c['soap'])
    native = registry.flatten(proposal)
    metric = SOAPMetric(registry, contexts)
    soap_seconds = time.perf_counter()-soap_start
    blocks, curvature = streamed_blocks(model, registry, flat, x[ids], 8, c['resources']['budget_mib'])
    mu = max(.01 * curvature['trace'] / registry.size, c['damping']['floor'])
    tolerance = c['solver']['tolerance64']
    checks = {}
    generator = torch.Generator().manual_seed(c['analysis_seed'] + step)
    for spec, block in zip(registry.specs, blocks):
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, generator=generator, dtype=flat.dtype)
        checks[spec.name] = relative_error(block @ v[spec.slice], op.action(v)[spec.slice])
    if max(checks.values()) > tolerance:
        raise RuntimeError(f'Exact tensor action check failed: {max(checks.values())}')
    exact, solve_info = solve_blocks(blocks, registry, g, metric, [mu], tolerance)
    allocated, allocation_info, reduced_arrays = reduced_allocation(op, native, registry, metric, mu)
    _, trace_timing = trace_estimate(op, 8, c['analysis_seed'] + step)
    objective = lambda v: loss(fn(v), y[ids])
    panels = [lambda v, f=functional_logits(model, registry, x[batch]), targets=y[batch]: loss(f(v), targets)
              for batch in panels_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    candidates = [('soap', native), ('frozen_minus_Pg', -metric.map(g)),
                  ('allocation_diagonal', allocated['allocation_diagonal']),
                  ('allocation_full', allocated['allocation_full']), ('exact_tensor_ggn', exact[mu])]
    rows, arrays = [], {'parameters': flat, 'gradient': g, 'contexts': contexts,
                       'optimizer': optimizer, 'exact_blocks': blocks, 'reduced': reduced_arrays}
    exact_search_seconds = 0.
    for name, delta in candidates:
        search_start = time.perf_counter()
        selected, search = search_candidate(flat, delta, native, objective)
        search['seconds'] = time.perf_counter()-search_start
        if name == 'exact_tensor_ggn':
            exact_search_seconds = search['seconds']
        rows.append({'name': name, 'step': step, 'gamma': .01, 'mu': mu, 'trace_source': 'exact GGN',
                     'raw': measurement(delta, op, metric, mu, objective, base, panels, panel_base, registry),
                     'selected': measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry),
                     'search': search})
        arrays[f'delta_{name}'], arrays[f'selected_{name}'] = delta, selected
        print(f"  snapshot {step} {name}: independent decrease={rows[-1]['selected']['independent_mean_decrease']:.6f}", flush=True)
    if not state_equal(before_model, model.state_dict()) or not state_equal(before_optimizer, optimizer) or not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError('Snapshot mutated training state')
    info = {'step': step, 'curvature': curvature, 'solve': solve_info[mu], 'action_checks': checks,
            'allocation': allocation_info, 'construction_ids': ids, 'panel_ids': panels_ids,
            'search_gate_passed': all(r['search']['bracketed'] or r['search']['rejected'] for r in rows),
            'numerical_gate_passed': True, 'nonmutation_passed': True, 'counts': op.counts,
            'exact_step_seconds_estimate': curvature['construction_seconds'] + solve_info[mu]['seconds'] + exact_search_seconds + soap_seconds + trace_timing['seconds'],
            'seconds': time.perf_counter()-started}
    torch.save(arrays, directory / 'arrays.pt')
    write_json(directory / 'diagnostics.json', info)
    write_json(directory / 'candidates.json', rows)
    write_json(directory / 'artifacts.json', {'arrays.pt': checksum(directory / 'arrays.pt')})
    return rows, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/mini_vit_allocation.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    c = load_config(args.config)
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    initial_model = make_model(c)
    registry = Registry(initial_model)
    estimate = workspace_estimate(registry, 10, 8)
    if estimate['estimated_workspace_bytes'] > c['resources']['budget_mib']*2**20:
        raise MemoryError('Exact reference exceeds declared tensor-workspace budget')
    methods = ['native_soap', 'searched_soap', 'allocation_diagonal', 'allocation_full']
    protocol = {'family': 'mini_vit_paired_SOAP_allocation', 'parameters': registry.size,
                'registered_tensors': len(registry.specs), 'batch_size': 512, 'steps': 50,
                'methods': methods.copy(), 'gamma': .01,
                'training_damping_trace': '8 candidate-independent Rademacher JVP probes',
                'snapshot_damping_trace': 'exact GGN', 'snapshot_trajectory': 'searched_soap',
                'snapshot_steps': [10, 25, 50], 'global_search': 'zero + 2^[-16..16], 18 golden refinements',
                'workspace': estimate, 'workspace_budget_mib': c['resources']['budget_mib'],
                'expected_allocation_jvps_per_initialized_step': len(registry.specs)+8,
                'expected_exact_jacobian_rows_per_snapshot': 5120,
                'optional_exact_arm': '>=2/3 positive independent differences vs searched SOAP, positive mean, passed checks, first-snapshot estimated 49-step time <=300 seconds',
                'protocol_document': 'docs/mini_vit_protocol_20261005.md'}
    print(json.dumps(protocol, indent=2), flush=True)
    if args.dry_run:
        return
    directory = new_run(c)
    print(f'Mini ViT comparison directory: {directory.resolve()}', flush=True)
    write_json(directory / 'protocol.json', protocol)
    write_json(directory / 'registry.json', registry.json())
    sources = json.loads((directory / 'provenance.json').read_text())['source_checksums']
    for name in ['scripts/run_mini_vit_allocation.py', 'scripts/run_exact_training_followup.py',
                 'scripts/run_large_batch_oracle.py', 'docs/mini_vit_protocol_20261005.md']:
        sources[name] = checksum(name)
    for name, expected in sources.items():
        if checksum(name) != expected:
            raise RuntimeError('Source changed before archiving')
        target = directory / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    write_json(directory / 'source_archive_checksums.json', sources)
    x, y, manifest = dataset(c)
    write_json(directory / 'split_manifest.json', manifest)
    initial = registry.flatten(dict(initial_model.named_parameters())).detach()
    validation, snapshot_rows, snapshot_info = [], [], []
    paired_batches = None
    for method in methods:
        arm = directory / method
        arm.mkdir()
        model = make_model(c)
        if not torch.equal(initial, registry.flatten(dict(model.named_parameters())).detach()):
            raise RuntimeError('Initialization pairing failed')
        arm_config = copy.deepcopy(c)
        arm_config['training'].update(method=method, policy='native' if method=='native_soap' else 'broad_direction_search')
        write_json(arm / 'resolved_arm.json', arm_config)
        order = DataOrder(manifest['training'], c['data_seed']+1)
        state, history, seconds, accepted = None, [], 0., 0
        validation.append(dict(method=method, step=0, training_seconds=0., **evaluate(model, x, y, manifest['validation'])))
        print(f'Starting {method}', flush=True)
        for step in range(1, 51):
            ids = order.take(512)
            flat = registry.flatten(dict(model.named_parameters())).detach()
            start = time.perf_counter()
            try:
                delta, state, contexts, row = propose(method, model, registry, flat, state, x[ids], y[ids], c, step)
            except Exception as error:
                write_json(directory / 'failure.json', {'method': method, 'step': step, 'phase': 'training', 'message': str(error)})
                raise
            commit(model, registry, flat+delta)
            elapsed = time.perf_counter()-start
            seconds += elapsed
            accepted += int(float(delta.norm()) > 0)
            row.update(step=step, batch_ids=ids, step_seconds=elapsed, accepted_updates=accepted)
            history.append(row)
            write_json(arm / 'training.json', history)
            if step in [10, 25, 50]:
                measured = dict(method=method, step=step, training_seconds=seconds, **evaluate(model, x, y, manifest['validation']))
                validation.append(measured)
                write_json(directory / 'validation.json', validation)
                checkpoint.save(arm / f'checkpoint-{step:04d}.pt', model, state, contexts, step, accepted, order, arm_config)
                print(f"{method} step {step}: validation={measured['validation_loss']:.6f}, accuracy={measured['validation_accuracy']:.2%}, step_seconds_total={seconds:.2f}", flush=True)
                if method == 'searched_soap':
                    construction = manifest['analysis'][:512]
                    panels = [manifest['analysis'][512*i:512*(i+1)] for i in range(1, 5)]
                    try:
                        rows, diagnostics = snapshot(model, registry, state, x, y, construction, panels, c, step, directory / f'snapshot-{step:04d}')
                    except Exception as error:
                        write_json(directory / 'failure.json', {'method': method, 'step': step, 'phase': 'snapshot', 'message': str(error)})
                        raise
                    snapshot_rows.extend(rows)
                    snapshot_info.append(diagnostics)
                    write_json(directory / 'snapshot_candidates.json', snapshot_rows)
                    write_json(directory / 'snapshot_diagnostics.json', snapshot_info)
        batches = [r['batch_ids'] for r in history]
        if paired_batches is None:
            paired_batches = batches
        elif batches != paired_batches:
            raise RuntimeError('Data-order pairing failed')
        write_json(arm / 'runtime.json', {'training_seconds': seconds,
                   'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)})
        if method == 'searched_soap':
            gaps = []
            for step in [10, 25, 50]:
                subset = {r['name']: r for r in snapshot_rows if r['step']==step}
                gaps.append(subset['exact_tensor_ggn']['selected']['independent_mean_decrease'] - subset['soap']['selected']['independent_mean_decrease'])
            estimated_seconds = 49*snapshot_info[0]['exact_step_seconds_estimate']
            opportunity = sum(g > 0 for g in gaps) >= 2 and sum(gaps) > 0
            checks = all(d['numerical_gate_passed'] and d['search_gate_passed'] for d in snapshot_info)
            selected = opportunity and checks and estimated_seconds <= 300
            gate = {'independent_differences': gaps, 'opportunity_passed': opportunity, 'checks_passed': checks,
                    'estimated_49_update_seconds': estimated_seconds, 'budget_seconds': 300, 'exact_arm_selected': selected}
            write_json(directory / 'exact_training_gate.json', gate)
            print('Exact training gate: '+json.dumps(gate), flush=True)
            if selected:
                methods.append('exact_tensor_ggn')
    write_json(directory / 'complete.json', {'completed': True, 'methods': methods,
               'paired_initialization': True, 'paired_data_order': True, 'steps': 50})
    print(f'Completed mini ViT experiment: {directory.resolve()}', flush=True)


if __name__ == '__main__':
    main()
