"""Selected MNIST oracle comparison: batch 256 first, then 512 at fixed SOAP states."""
import argparse
import copy
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from residual_curvature.config import load_config, validate
from residual_curvature.curvature.layerwise import streamed_blocks, solve_blocks, workspace_estimate
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import dataset
from residual_curvature.experiments.search import broad_direction_search
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import SOAPMetric, soap_propose
from residual_curvature.provenance import checksum, new_run, write_json


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def archive_sources(directory):
    provenance = json.loads((directory / 'provenance.json').read_text())
    for name, expected in provenance['source_checksums'].items():
        source, destination = Path(name), directory / 'source' / name
        if checksum(source) != expected:
            raise RuntimeError(f'Source changed before archive: {source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(__file__, directory / 'runner.py')
    write_json(directory / 'runner_checksum.json', {'runner.py': checksum(directory / 'runner.py')})


def measurement(delta, operator, metric, mu, objective, base, panels, panel_base, registry):
    slope = float(operator.gradient @ delta)
    curvature = float(delta @ operator.action(delta))
    penalty = mu * float(delta @ metric.apply(delta))
    norm2 = float(delta.square().sum())
    decreases = [initial - float(fn(operator.params + delta)) for fn, initial in zip(panels, panel_base)]
    exposure = curvature + penalty
    return {'norm': norm2**.5, 'slope': slope, 'ggn_quadratic': curvature,
            'predicted_decrease': -slope - .5 * exposure,
            'optimized_quadratic_ray_decrease': .5 * min(slope, 0)**2 / exposure if exposure > 0 else None,
            'actual_decrease': base - float(objective(operator.params + delta)),
            'independent_decreases': decreases, 'independent_mean_decrease': float(np.mean(decreases)),
            'tensor_update_shares': {s.name: float(delta[s.slice].square().sum()) / norm2 if norm2 else 0
                                     for s in registry.specs}}


def snapshot(model, registry, flat, optimizer, x, y, construction, panel_ids, c, step, directory):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer, before_rng = copy.deepcopy(optimizer), torch.get_rng_state().clone()
    xb, yb = x[construction], y[construction]
    fn = functional_logits(model, registry, xb)
    objective = lambda v: loss(fn(v), yb)
    operator = GGN(fn, flat, yb)
    g = operator.gradient
    soap_start = time.perf_counter()
    proposal, _, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), optimizer, c['soap'])
    native = registry.flatten(proposal)
    metric = SOAPMetric(registry, contexts)
    soap_seconds = time.perf_counter() - soap_start
    tolerance = c['solver']['tolerance64']
    blocks, diagnostics = streamed_blocks(model, registry, flat, xb, 8, c['resources']['budget_mib'])
    diagnostics['soap_proposal_seconds'] = soap_seconds
    print(f"  B={len(xb)} step={step}: exact blocks built in {diagnostics['construction_seconds']:.2f}s", flush=True)
    generator = torch.Generator().manual_seed(c['analysis_seed'] + step)
    checks = {}
    # Independent JVP-Hz-VJP route for every tensor, not reused to form blocks.
    for spec, block in zip(registry.specs, blocks):
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, generator=generator, dtype=flat.dtype)
        checks[spec.name] = relative_error(block @ v[spec.slice], operator.action(v)[spec.slice])
    if max(checks.values()) > tolerance:
        raise RuntimeError(f'Layerwise action check failed: {checks}')
    mus = {gamma: max(gamma * diagnostics['trace'] / registry.size, c['damping']['floor'])
           for gamma in c['damping']['gammas']}
    solutions, solve_info = solve_blocks(blocks, registry, g, metric, list(mus.values()), tolerance)
    torch.save({s.name: block for s, block in zip(registry.specs, blocks)}, directory / 'exact_blocks.pt')
    del blocks
    panels = []
    for ids in panel_ids:
        panel_fn = functional_logits(model, registry, x[ids])
        panels.append(lambda v, f=panel_fn, targets=y[ids]: loss(f(v), targets))
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    directions = [('soap', None, native), ('frozen_minus_Pg', None, -metric.map(g))]
    directions.extend(('exact_tensor_ggn', gamma, solutions[mu]) for gamma, mu in mus.items())
    rows, arrays = [], {'parameters': flat, 'gradient': g, 'native_soap': native}
    for index, (name, gamma, delta) in enumerate(directions):
        search_start = time.perf_counter()
        selected, search = broad_direction_search(flat, delta, float(native.norm()), objective)
        search['seconds'] = time.perf_counter() - search_start
        mu = mus[gamma] if gamma is not None else mus[.01]
        row = {'name': name, 'gamma': gamma, 'mu': mu, 'batch_size': len(xb), 'step': step,
               'scope': 'all registered parameters', 'partition': 'tensor',
               'metric': 'SOAP inverse normalized to trace / D = 1', 'metric_scale': metric.scale,
               'raw': measurement(delta, operator, metric, mu, objective, base, panels, panel_base, registry),
               'selected': measurement(selected, operator, metric, mu, objective, base, panels, panel_base, registry),
               'search': search, 'solve': solve_info[mu] if gamma is not None else None}
        rows.append(row)
        arrays[f'delta_{index}'], arrays[f'selected_{index}'] = delta, selected
        write_json(directory / 'candidates.json', rows)
        print(f"  B={len(xb)} step={step} {name} gamma={gamma}: construction decrease={row['selected']['actual_decrease']:.6f}; "
              f"independent={row['selected']['independent_mean_decrease']:.6f}; bracketed={search['bracketed']}", flush=True)
    if not state_equal(before_model, model.state_dict()) or not state_equal(before_optimizer, optimizer) or not torch.equal(before_rng, torch.get_rng_state()):
        raise RuntimeError('Snapshot mutated model, optimizer or global Torch RNG')
    torch.save({'arrays': arrays, 'pre_proposal_optimizer': optimizer, 'used_contexts': contexts}, directory / 'state_and_directions.pt')
    diagnostics.update(step=step, construction_ids=construction, panel_ids=panel_ids, base_loss=base,
                       independent_base_losses=panel_base, action_checks=checks, counts=operator.counts,
                       numerical_gate_passed=True, nonmutation_passed=True,
                       search_gate_passed=all(r['search']['bracketed'] or r['search']['rejected'] for r in rows),
                       seconds=time.perf_counter()-started,
                       process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    write_json(directory / 'diagnostics.json', diagnostics)
    write_json(directory / 'registry.json', registry.json())
    write_json(directory / 'artifacts.json', {name: checksum(directory / name) for name in ['exact_blocks.pt', 'state_and_directions.pt']})
    return rows, diagnostics


def report(directory, rows, diagnostics):
    lines = ['# MNIST exact tensor GGN versus SOAP at larger construction batches', '',
             'Fixed states from a batch-64 SOAP trajectory; construction batches run in order 256, then 512. '
             'All directions use a common broad radius grid and construction-only refinement. '
             'Independent batches do not select settings. Loss decreases are positive when loss improves.', '',
             'The primary damping is gamma=0.01 with normalized inverse-SOAP metric. '
             'Gamma=0.001 and 0.1 are prespecified sensitivity results, not held-out-selected settings.', '',
             '| Batch | State | SOAP construction decrease | GGN construction decrease | SOAP independent decrease | GGN independent decrease | Independent difference |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    differences = []
    for b in [256, 512]:
        for step in [10, 50, 100]:
            subset = [r for r in rows if r['batch_size'] == b and r['step'] == step]
            if not subset:
                continue
            soap = next(r for r in subset if r['name'] == 'soap')['selected']
            exact = next(r for r in subset if r['name'] == 'exact_tensor_ggn' and r['gamma'] == .01)['selected']
            gap = exact['independent_mean_decrease'] - soap['independent_mean_decrease']
            differences.append({'batch_size': b, 'step': step, 'independent_difference': gap})
            lines.append(f"| {b} | {step} | {soap['actual_decrease']:.6f} | {exact['actual_decrease']:.6f} | "
                         f"{soap['independent_mean_decrease']:.6f} | {exact['independent_mean_decrease']:.6f} | {gap:+.6f} |")
    lines.extend(['', 'These are one-step measurements, not exact-GGN training trajectories. '
                  'SOAP moments come from the original batch-64 training history. '
                  'The same four disjoint panel regions are used at both batch sizes; 256-example batches are nested in the 512-example batches. '
                  'Raw SOAP steps, all damping settings, all search trials, exact matrices and directions are retained.', '',
                  'Curvature blocks are accumulated in microbatches with complete-batch mean normalization. '
                  'Direct Cholesky solve residuals and independent JVP-Hz-VJP action checks must pass 1e-8. '
                  'The tensor-workspace estimate excludes dataset/interpreter/library RSS; measured RSS is process lifetime peak.', ''])
    (directory / 'REPORT.md').write_text('\n'.join(lines))
    write_json(directory / 'primary_comparison.json', differences)
    write_json(directory / 'all_candidates.json', rows)
    write_json(directory / 'all_diagnostics.json', diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-run', type=Path, default=Path('runs/20261001T080647-fd8e50e5'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    c = load_config(args.snapshot_run / 'resolved.yaml')
    c['resources']['budget_mib'] = 768
    c['data']['batch_size'] = 512
    validate(c)
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    model, registry = make_model(c), None
    registry = Registry(model)
    estimate = workspace_estimate(registry, c['model']['classes'], 8)
    protocol = {'family': 'exact_tensor_ggn_large_batch_fixed_state', 'batch_sizes_in_order': [256, 512],
                'states': [10, 50, 100], 'baseline_run': str(args.snapshot_run.resolve()),
                'state_training_batch_size': 64, 'partition': 'tensor', 'microbatch': 8,
                'damping_gammas': c['damping']['gammas'], 'primary_gamma': .01,
                'damping': 'max(gamma * exact trace(C) / D, 1e-8), normalized inverse SOAP metric',
                'search': {'grid_exponents': list(range(-16, 17)), 'golden_refinement_evaluations': 18,
                           'selection': 'construction loss only'},
                'panel_batches': 4, 'panel_region_stride': 512,
                'workspace_budget_mib': 768, 'workspace_estimate': estimate,
                'expected_jacobian_rows_per_state': {b: b * 10 for b in [256, 512]},
                'independent_action_checks_per_state': 22,
                'gate_for_short_training': 'At primary gamma, exact improves independent mean over searched SOAP at >=2/3 states and the mean difference is positive; numerical and search gates pass.'}
    print(json.dumps(protocol, indent=2), flush=True)
    if args.dry_run:
        return
    directory = new_run(c)
    print(f'Comparison directory: {directory.resolve()}', flush=True)
    write_json(directory / 'protocol.json', protocol)
    archive_sources(directory)
    x, y, manifest = dataset(c)
    if manifest != json.loads((args.snapshot_run / 'split_manifest.json').read_text()):
        raise RuntimeError('Dataset or split mismatch')
    write_json(directory / 'split_manifest.json', manifest)
    rows, diagnostics = [], []
    for batch_size in protocol['batch_sizes_in_order']:
        construction = manifest['analysis'][:batch_size]
        panels = [manifest['analysis'][i*512:i*512+batch_size] for i in range(1, 5)]
        for step in protocol['states']:
            source = args.snapshot_run / f'snapshot-{step:04d}'
            for name, expected in json.loads((source / 'artifacts.json').read_text()).items():
                if checksum(source / name) != expected:
                    raise RuntimeError(f'Baseline artifact mismatch: {source / name}')
            with np.load(source / 'arrays.npz') as arrays:
                flat = torch.from_numpy(arrays['parameters']).clone()
            commit(model, registry, flat)
            optimizer = torch.load(source / 'frozen_state.pt', map_location='cpu', weights_only=False)['pre_proposal_optimizer']
            destination = directory / f'batch-{batch_size}' / f'snapshot-{step:04d}'
            try:
                measured, info = snapshot(model, registry, flat, optimizer, x, y, construction, panels, c, step, destination)
            except Exception as error:
                write_json(directory / 'failure.json', {'batch_size': batch_size, 'step': step,
                           'type': type(error).__name__, 'message': str(error)})
                raise
            rows.extend(measured)
            diagnostics.append(info)
            report(directory, rows, diagnostics)
        print(f'Finished all fixed-state measurements at batch {batch_size}.', flush=True)
    write_json(directory / 'complete.json', {'completed': True, 'batch_sizes': [256, 512], 'snapshot_count': 6})
    print(f'Completed: {directory.resolve()}', flush=True)


if __name__ == '__main__':
    main()
