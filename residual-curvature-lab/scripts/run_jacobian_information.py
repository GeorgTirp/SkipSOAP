"""Selected six-state information/representation diagnostic; no training."""
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
from residual_curvature.curvature.information import information_blocks, match_traces, coordinate_diagonal_blocks
from residual_curvature.curvature.layerwise import solve_blocks
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import DataOrder, dataset
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.downstream_soap import exact_coordinate_diagonal, transfer_tensor_norms
from residual_curvature.optim.soap import SOAPMetric, soap_propose
from residual_curvature.provenance import checksum, new_run, write_json
from run_large_batch_oracle import measurement, state_equal
from run_mini_vit_allocation import search_candidate


def matrix_discrepancy(blocks, reference):
    denominator = sum(float(c.square().sum()) for c in reference)
    return (sum(float((b-c).square().sum()) for b, c in zip(blocks, reference))/denominator)**.5


def audit(model, registry, state, x, y, construction, panels_ids, c, trajectory, step, directory):
    directory.mkdir()
    start = time.perf_counter()
    before, optimizer, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(state), torch.get_rng_state().clone()
    flat = registry.flatten(dict(model.named_parameters())).detach()
    xb, yb = x[construction], y[construction]
    fn = functional_logits(model, registry, xb)
    op = GGN(fn, flat, yb)
    g = op.gradient
    proposal, _, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), state, c['soap'])
    native = registry.flatten(proposal)
    metric = SOAPMetric(registry, contexts)
    matrices, checked_gradient, labels, construction_info = information_blocks(
        model, registry, flat, xb, yb, seed=c['analysis_seed']+7919*step,
        budget_mib=c['resources']['budget_mib'])
    exact = matrices['ggn']
    mu = max(.01*construction_info['traces']['ggn']/registry.size, c['damping']['floor'])
    tolerance = c['solver']['tolerance64']
    gradient_error = relative_error(checked_gradient, g)
    assert gradient_error <= tolerance
    gen = torch.Generator().manual_seed(c['analysis_seed']+step)
    actions = {}
    for spec, block in zip(registry.specs, exact):
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=flat.dtype)
        actions[spec.name] = relative_error(block@v[spec.slice], op.action(v)[spec.slice])
    assert max(actions.values()) <= tolerance
    momentum = -metric.map(native, inverse=True)
    momentum *= g.norm()/momentum.norm()
    recovered = -metric.map(momentum)
    momentum_check = relative_error(recovered/recovered.norm(), native/native.norm())
    assert momentum_check <= tolerance
    candidates, solves, trace_controls = [('soap', native), ('soap_current_gradient', -metric.map(g))], {}, {}

    def add_solve(name, blocks, rhs):
        values, info = solve_blocks(blocks, registry, rhs, metric, [mu], tolerance)
        candidates.append((name, values[mu]))
        solves[name] = info[mu]
        return values[mu]

    ggn = add_solve('ggn_gradient', exact, g)
    add_solve('ggn_momentum', exact, momentum)
    discrepancies = {'ggn': 0.}
    for target in ['ggn', 'empirical_fisher']:
        diagonal, _ = exact_coordinate_diagonal(matrices[target], registry, contexts)
        compressed = coordinate_diagonal_blocks(diagonal, registry, contexts)
        compressed, trace_controls[f'{target}_diagonal'] = match_traces(compressed, exact, registry)
        name = 'ggn_diagonal_gradient' if target == 'ggn' else 'empirical_fisher_diagonal_matched'
        add_solve(name, compressed, g)
        discrepancies[name] = matrix_discrepancy(compressed, exact)
        if target == 'ggn':
            add_solve('ggn_diagonal_momentum', compressed, momentum)
    add_solve('empirical_fisher_raw', matrices['empirical_fisher'], g)
    matched, trace_controls['empirical_fisher_dense'] = match_traces(matrices['empirical_fisher'], exact, registry)
    add_solve('empirical_fisher_matched', matched, g)
    discrepancies['empirical_fisher_raw'] = matrix_discrepancy(matrices['empirical_fisher'], exact)
    discrepancies['empirical_fisher_matched'] = matrix_discrepancy(matched, exact)
    candidates.extend([
        ('ggn_directions_soap_norms', transfer_tensor_norms(ggn, native, registry)),
        ('soap_directions_ggn_norms', transfer_tensor_norms(native, ggn, registry)),
    ])
    for name, blocks in matrices.items():
        if name.startswith('sampled_'):
            add_solve(name, blocks, g)
            discrepancies[name] = matrix_discrepancy(blocks, exact)
    assert len(candidates) == 17
    print(f'{trajectory} step {step}: curvature built in {construction_info["seconds"]:.2f}s; all pre-search checks passed', flush=True)
    objective = lambda v: loss(fn(v), yb)
    panels = [lambda v, f=functional_logits(model, registry, x[ids]), targets=y[ids]: loss(f(v), targets)
              for ids in panels_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    arrays = {'parameters': flat, 'gradient': g, 'momentum_rhs': momentum, 'contexts': contexts,
              'optimizer': optimizer, 'sampled_labels': labels,
              'ggn_blocks': exact, 'empirical_fisher_blocks': matrices['empirical_fisher']}
    rows = []
    for name, delta in candidates:
        selected, search = search_candidate(flat, delta, native, objective)
        row = {'trajectory': trajectory, 'step': step, 'name': name, 'mu': mu,
               'selected': measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry),
               'raw_norm': float(delta.norm()), 'search': search,
               'cosine_to_ggn': float(delta@ggn/(delta.norm()*ggn.norm())) if delta.norm() else None,
               'solve': solves.get(name)}
        rows.append(row)
        arrays['raw_'+name], arrays['selected_'+name] = delta, selected
        write_json(directory/'candidates.json', rows)
        print(f'  {name}: independent={row["selected"]["independent_mean_decrease"]:.6f}', flush=True)
    assert state_equal(before, model.state_dict()) and state_equal(optimizer, state) and torch.equal(rng, torch.get_rng_state())
    info = {'trajectory': trajectory, 'step': step, 'construction': construction_info,
            'gradient_error': gradient_error, 'exact_action_checks': actions, 'momentum_ray_error': momentum_check,
            'trace_controls': trace_controls, 'block_frobenius_relative_errors': discrepancies,
            'construction_ids': construction, 'panel_ids': panels_ids, 'construction_base_loss': base,
            'independent_base_losses': panel_base, 'nonmutation_passed': True,
            'numerical_gate_passed': True, 'search_gate_passed': True, 'seconds': time.perf_counter()-start,
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    torch.save(arrays, directory/'arrays.pt')
    write_json(directory/'diagnostics.json', info)
    write_json(directory/'artifacts.json', {'arrays.pt': checksum(directory/'arrays.pt')})
    return rows, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('runs/20261005T135920-d7ce29b2'))
    args = parser.parse_args()
    assert json.loads((args.parent/'complete.json').read_text())['completed']
    c = load_config(args.parent/'resolved.yaml')
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    directory = new_run(c)
    print(f'Information diagnostic: {directory.resolve()}', flush=True)
    sources = json.loads((directory/'provenance.json').read_text())['source_checksums']
    for name in ['scripts/run_jacobian_information.py', 'scripts/run_large_batch_oracle.py',
                 'scripts/run_mini_vit_allocation.py', 'scripts/run_exact_training_followup.py',
                 'docs/jacobian_information_protocol_20261005.md']:
        sources[name] = checksum(name)
    for name, digest in sources.items():
        assert checksum(name) == digest
        target = directory/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    write_json(directory/'source_archive_checksums.json', sources)
    x, y, manifest = dataset(c)
    assert manifest == json.loads((args.parent/'split_manifest.json').read_text())
    construction = manifest['analysis'][:512]
    panels = [manifest['analysis'][2560+512*i:2560+512*(i+1)] for i in range(4)]
    assert all(len(ids) == 512 for ids in panels)
    assert not set(sum(panels, [])) & set(manifest['analysis'][:2560])
    write_json(directory/'protocol.json', {'parent': str(args.parent.resolve()),
               'document': 'docs/jacobian_information_protocol_20261005.md',
               'trajectories': ['searched_soap', 'exact_tensor_ggn'], 'steps': [10, 25, 50],
               'construction_ids': construction, 'fresh_panel_ids': panels, 'training_launched': False})
    rows, diagnostics, parent_hashes = [], [], {}
    for trajectory in ['searched_soap', 'exact_tensor_ggn']:
        for step in [10, 25, 50]:
            path = args.parent/trajectory/f'checkpoint-{step:04d}.pt'
            parent_hashes[str(path)] = checksum(path)
            write_json(directory/'parent_checkpoint_checksums.json', parent_hashes)
            arm_config = json.loads((args.parent/trajectory/'resolved_arm.json').read_text())
            model = make_model(arm_config)
            registry = Registry(model)
            order = DataOrder(manifest['training'], c['data_seed']+1)
            state = checkpoint.load(path, model, order, arm_config)
            try:
                new_rows, info = audit(model, registry, state['optimizer'], x, y, construction, panels,
                                       c, trajectory, step, directory/f'{trajectory}-{step:04d}')
            except Exception as error:
                write_json(directory/'failure.json', {'trajectory': trajectory, 'step': step, 'error': repr(error)})
                raise
            rows.extend(new_rows)
            diagnostics.append(info)
            write_json(directory/'all_candidates.json', rows)
            write_json(directory/'all_diagnostics.json', diagnostics)
    write_json(directory/'complete.json', {'completed': True, 'states': len(diagnostics), 'candidates': len(rows)})
    print(f'Completed: {directory.resolve()}', flush=True)


if __name__ == '__main__':
    main()
