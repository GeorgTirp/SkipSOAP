"""Bounded downstream-curvature and allocation audit at saved batch-512 states."""
import argparse
import copy
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import torch

from residual_curvature.config import load_config
from residual_curvature.curvature.downstream import propagate_metric_probes
from residual_curvature.curvature.layerwise import block_action
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.sketch import categorical_probes, factors_from_probes
from residual_curvature.curvature.solves import diagonal_solve, factor_solve, relative_error
from residual_curvature.data import dataset
from residual_curvature.experiments.search import broad_direction_search
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.allocation import direction_matrix, nonnegative_qp
from residual_curvature.optim.downstream_soap import (coordinate_action, coordinate_candidate,
    exact_coordinate_diagonal, probe_coordinate_diagonal, transfer_tensor_norms)
from residual_curvature.optim.soap import SOAPMetric
from residual_curvature.provenance import checksum, new_run, write_json
from run_large_batch_oracle import measurement, state_equal


def archive(directory):
    provenance = json.loads((directory / 'provenance.json').read_text())
    sources = dict(provenance['source_checksums'])
    for p in [Path(__file__), Path('scripts/run_large_batch_oracle.py'),
              Path('docs/downstream_allocation_protocol_20261002.md')]:
        sources[str(p.relative_to(Path.cwd()) if p.is_absolute() else p)] = checksum(p)
    for name, expected in sources.items():
        if checksum(name) != expected:
            raise RuntimeError('Source changed during archiving')
        destination = directory / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    write_json(directory / 'source_archive_checksums.json', sources)


def checked_solve(solve, action, g, tolerance):
    delta = solve(g)
    for _ in range(4):
        error = action(delta) + g
        if relative_error(action(delta), -g) <= tolerance:
            break
        delta += solve(error)
    residual = relative_error(action(delta), -g)
    if residual > tolerance:
        raise RuntimeError(f'Sketch solve failed: {residual}')
    return delta, residual


def run_state(source, destination, model, registry, x, y, c, step):
    destination.mkdir(parents=True)
    start = time.perf_counter()
    artifacts = json.loads((source / 'artifacts.json').read_text())
    for name, expected in artifacts.items():
        if checksum(source / name) != expected:
            raise RuntimeError('Source exact-reference checksum mismatch')
    reference_metadata = destination / 'reference_metadata'
    reference_metadata.mkdir()
    reference_checksums = {}
    for name in ['artifacts.json', 'candidates.json', 'diagnostics.json', 'registry.json']:
        shutil.copyfile(source / name, reference_metadata / name)
        reference_checksums[name] = checksum(reference_metadata / name)
    write_json(reference_metadata / 'checksums.json', reference_checksums)
    saved = torch.load(source / 'state_and_directions.pt', map_location='cpu', weights_only=False)
    metadata = json.loads((source / 'diagnostics.json').read_text())
    previous = json.loads((source / 'candidates.json').read_text())
    tensors = torch.load(source / 'exact_blocks.pt', map_location='cpu', weights_only=False)
    blocks = [tensors[s.name] for s in registry.specs]
    arrays, contexts = saved['arrays'], saved['used_contexts']
    flat, g, native = arrays['parameters'], arrays['gradient'], arrays['native_soap']
    exact_index = next(i for i, r in enumerate(previous) if r['name'] == 'exact_tensor_ggn' and r['gamma'] == .01)
    exact = arrays[f'delta_{exact_index}']
    mu = previous[exact_index]['mu']
    commit(model, registry, flat)
    before, rng, before_context = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone(), copy.deepcopy(contexts)
    ids, panel_ids = metadata['construction_ids'], metadata['panel_ids']
    xb, yb = x[ids], y[ids]
    fn = functional_logits(model, registry, xb)
    operator = GGN(fn, flat, yb)
    tolerance = c['solver']['tolerance64']
    if relative_error(operator.gradient, g) > tolerance:
        raise RuntimeError('Saved gradient mismatch')
    objective = lambda p: loss(fn(p), yb)
    panels = [lambda v, f=functional_logits(model, registry, x[batch]), targets=y[batch]: loss(f(v), targets)
              for batch in panel_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    metric = SOAPMetric(registry, contexts)
    generator = torch.Generator().manual_seed(c['analysis_seed'] + 10000 + step)
    validation = torch.randn(registry.size, 4, generator=generator, dtype=flat.dtype)
    validation_action = block_action(blocks, registry, validation)
    exact_diag, exact_diag_metadata = exact_coordinate_diagonal(blocks, registry, contexts)
    compression_error = relative_error(coordinate_action(validation, exact_diag, registry, contexts), validation_action)
    exact_coordinate = coordinate_candidate(g, exact_diag, metric, mu)
    coordinate_residual = relative_error(coordinate_action(exact_coordinate, exact_diag, registry, contexts) + mu*metric.apply(exact_coordinate), -g)
    if coordinate_residual > tolerance:
        raise RuntimeError('Exact-coordinate solve residual failed')

    e, names, excluded = direction_matrix(native, registry)
    response = torch.stack([operator.action(e[:, i]) for i in range(e.shape[1])], 1)
    k = e.T @ response
    if relative_error(k, k.T) > tolerance:
        raise RuntimeError('Reduced GGN symmetry check failed')
    k = (k + k.T)/2
    h = k + mu * e.T @ metric.apply(e)
    phi = -e.T @ g
    amplitudes, kkt = nonnegative_qp(h, phi, tolerance)
    allocation_full = e @ amplitudes
    allocation_diag = e @ (phi / h.diag()).clamp_min(0)
    directions = [('soap', native, {}), ('frozen_minus_Pg', -metric.map(g), {}),
                  ('exact_tensor_ggn', exact, {}),
                  ('soap_directions_ggn_shares', transfer_tensor_norms(native, exact, registry), {}),
                  ('ggn_directions_soap_shares', transfer_tensor_norms(exact, native, registry), {}),
                  ('soap_allocation_diagonal', allocation_diag, {}),
                  ('soap_allocation_full', allocation_full, {'kkt': kkt}),
                  ('exact_curvature_diagonal_soap_basis', exact_coordinate,
                   {'tensor_action_error': compression_error, 'solve_residual': coordinate_residual})]
    stored = {'parameters': flat, 'gradient': g, 'native': native, 'exact_coordinate_diagonal': exact_diag,
              'validation_vectors': validation, 'exact_tensor_validation_actions': validation_action,
              'E': e, 'K': k, 'H': h, 'phi': phi, 'contexts': contexts}
    probe_results, transport_checks, probe_counts = [], [], []
    ranks = [1, 4, 8, 16]
    for repeat in range(2):
        seed = c['probe_seed'] + 100000 * step + repeat
        xi = categorical_probes(operator.logits, 16, seed)
        downstream = propagate_metric_probes(model, registry, flat, xb, xi, microbatch=64,
                     batch_id=ids, state_step=step, probe_seed=seed,
                     loss_normalization='mean over complete construction batch; rank normalization 1/sqrt(k)',
                     max_factor_bytes=16 * 2**20)
        factor = downstream.factors
        direct = factors_from_probes(fn, flat, xi)
        transport_error = relative_error(factor, direct)
        if transport_error > tolerance:
            raise RuntimeError('Explicit downstream transport differs from direct VJP')
        transport_checks.append(transport_error)
        probe_counts.append({'metadata': downstream.metadata, 'counts': downstream.counts})
        stored[f'probes_{repeat}'], stored[f'factors_{repeat}'] = xi, factor
        for rank in ranks:
            nested = factor[:, :rank] * (16/rank)**.5
            diagonal = probe_coordinate_diagonal(nested, registry, contexts)
            diagonal_error = relative_error(diagonal, exact_diag)
            action_error = relative_error(coordinate_action(validation, diagonal, registry, contexts), validation_action)
            delta = coordinate_candidate(g, diagonal, metric, mu)
            residual = relative_error(coordinate_action(delta, diagonal, registry, contexts) + mu*metric.apply(delta), -g)
            if residual > tolerance:
                raise RuntimeError('Probe-coordinate solve residual failed')
            info = {'rank': rank, 'repeat': repeat, 'probe_seed': seed, 'diagonal_error': diagonal_error,
                    'tensor_action_error': action_error, 'solve_residual': residual,
                    'rank_repeat_eligible': diagonal_error <= .25 and action_error <= .50}
            probe_results.append(info)
            directions.append(('downstream_curvature_diagonal', delta, info))
            stored[f'diagonal_{repeat}_{rank}'] = diagonal
        for coupling in ['diagonal', 'coupled']:
            if coupling == 'diagonal':
                curvature = lambda v: torch.cat([factor[s.slice] @ (factor[s.slice].T @ v[s.slice]) for s in registry.specs])
                solve = lambda rhs: diagonal_solve(rhs, factor, mu, metric.inverse, registry.groups())
            else:
                curvature = lambda v: factor @ (factor.T @ v)
                solve = lambda rhs: factor_solve(rhs, factor, mu, metric.inverse)
            action = lambda v: curvature(v) + mu * metric.apply(v)
            delta, residual = checked_solve(solve, action, g, tolerance)
            directions.append((f'shared_lowrank_{coupling}', delta,
                               {'rank': 16, 'repeat': repeat, 'probe_seed': seed, 'solve_residual': residual}))
    eligible = [rank for rank in ranks if all(r['rank_repeat_eligible'] for r in probe_results if r['rank'] == rank)]
    chosen_rank = min(eligible) if eligible else None
    print(f'State {step}: exact basis-compression action error={compression_error:.4f}; chosen rank={chosen_rank}', flush=True)
    rows = []
    for index, (name, delta, extra) in enumerate(directions):
        selected, search = broad_direction_search(flat, delta, float(native.norm()), objective)
        raw = measurement(delta, operator, metric, mu, objective, base, panels, panel_base, registry)
        measured = measurement(selected, operator, metric, mu, objective, base, panels, panel_base, registry)
        measured['within_tensor_quadratic'] = float(selected @ block_action(blocks, registry, selected))
        measured['cross_tensor_quadratic'] = measured['ggn_quadratic'] - measured['within_tensor_quadratic']
        row = {'name': name, 'step': step, 'batch_size': len(ids), 'gamma': .01, 'mu': mu, 'metric': 'normalized_inverse_SOAP',
               'scope': 'all registered tensors', 'partition': 'tensor', **extra,
               'raw': raw, 'selected': measured, 'search': search}
        rows.append(row)
        stored[f'delta_{index}'], stored[f'selected_{index}'] = delta, selected
        print(f"  {name} rank={extra.get('rank')} repeat={extra.get('repeat')}: independent decrease={measured['independent_mean_decrease']:.6f}", flush=True)
        write_json(destination / 'candidates.json', rows)
    if not state_equal(before, model.state_dict()) or not state_equal(before_context, contexts) or not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError('Analysis mutated state or RNG')
    torch.save(stored, destination / 'arrays.pt')
    diagnostics = {'step': step, 'construction_ids': ids, 'panel_ids': panel_ids, 'source_directory': str(source.resolve()),
                   'source_artifacts': artifacts, 'exact_diagonal_metadata': exact_diag_metadata,
                   'exact_coordinate_compression_error': compression_error, 'transport_errors': transport_checks,
                   'probe_errors': probe_results, 'probe_construction': probe_counts, 'selected_rank': chosen_rank,
                   'rank_selection': 'smallest passing both diagonal<=.25 and tensor action<=.50 in both repeats; no loss selection',
                   'allocation_names': names, 'allocation_excluded': excluded, 'counts': operator.counts,
                   'nonmutation_passed': True, 'search_gate_passed': all(r['search']['bracketed'] or r['search']['rejected'] for r in rows),
                   'seconds': time.perf_counter()-start,
                   'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
    write_json(destination / 'diagnostics.json', diagnostics)
    write_json(destination / 'registry.json', registry.json())
    write_json(destination / 'artifacts.json', {'arrays.pt': checksum(destination / 'arrays.pt')})
    return rows, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=Path('runs/20261002T121124-74f10aae'))
    args = parser.parse_args()
    c = load_config(args.reference_run / 'resolved.yaml')
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    model = make_model(c)
    registry = Registry(model)
    # Largest dense tensor exists only as an already saved oracle. Coordinate
    # transforms do not form the D x D network matrix or full output factor.
    sizes = [s.count for s in registry.specs]
    estimate = (sum(n*n for n in sizes) + 7*max(sizes)**2 + 10*registry.size*16)*8
    if estimate > c['resources']['budget_mib']*2**20:
        raise MemoryError('Downstream audit exceeds declared workspace budget')
    protocol = {'family': 'downstream_SOAP_coordinate_curvature_and_allocation',
                'reference_run': str(args.reference_run.resolve()), 'states': [10, 50, 100], 'batch_size': 512,
                'gamma': .01, 'ranks': [1, 4, 8, 16], 'repeats': 2, 'probe_microbatch': 64,
                'accuracy_thresholds': {'relative_diagonal': .25, 'relative_tensor_action': .50},
                'workspace_estimate_bytes': estimate, 'workspace_budget_mib': c['resources']['budget_mib'],
                'loss_role': 'construction selects only scalar; independent panels never select settings',
                'scope': 'fixed states only, no new training',
                'expected_probe_stage_vjps_per_state': 2*8*16*(len(model.blocks)+2),
                'protocol_document': 'docs/downstream_allocation_protocol_20261002.md'}
    print(json.dumps(protocol, indent=2), flush=True)
    directory = new_run(c)
    print(f'Downstream comparison directory: {directory.resolve()}', flush=True)
    write_json(directory / 'protocol.json', protocol)
    archive(directory)
    x, y, manifest = dataset(c)
    if manifest != json.loads((args.reference_run / 'split_manifest.json').read_text()):
        raise RuntimeError('Dataset or split mismatch')
    write_json(directory / 'split_manifest.json', manifest)
    rows, diagnostics = [], []
    for step in [10, 50, 100]:
        try:
            new_rows, info = run_state(args.reference_run / 'batch-512' / f'snapshot-{step:04d}',
                                      directory / f'snapshot-{step:04d}', model, registry, x, y, c, step)
        except Exception as error:
            write_json(directory / 'failure.json', {'step': step, 'type': type(error).__name__, 'message': str(error)})
            raise
        rows.extend(new_rows)
        diagnostics.append(info)
        write_json(directory / 'all_candidates.json', rows)
        write_json(directory / 'all_diagnostics.json', diagnostics)
    write_json(directory / 'complete.json', {'completed': True, 'states': [10, 50, 100], 'candidate_count': len(rows)})
    print(f'Completed: {directory.resolve()}', flush=True)


if __name__ == '__main__':
    main()
