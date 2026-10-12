"""Bounded paired batch-512 training after the prespecified snapshot gate."""
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
from residual_curvature.config import load_config, validate
from residual_curvature.curvature.layerwise import streamed_blocks, solve_blocks, workspace_estimate
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.search import broad_direction_search
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import SOAPMetric, soap_propose
from residual_curvature.provenance import checksum, new_run, write_json


@torch.inference_mode()
def evaluate(model, x, y, ids):
    total, correct = 0.0, 0
    for start in range(0, len(ids), 256):
        batch = ids[start:start+256]
        z, targets = model(x[batch]), y[batch]
        total += float(torch.nn.functional.cross_entropy(z, targets, reduction='sum'))
        correct += int((z.argmax(-1) == targets).sum())
    return {'validation_loss': total / len(ids), 'validation_accuracy': correct / len(ids),
            'validation_examples': len(ids)}


def check_gate(oracle_run, batch_size):
    if not (oracle_run / 'complete.json').exists() or (oracle_run / 'failure.json').exists():
        raise RuntimeError('Oracle comparison is incomplete or failed')
    gaps = [r['independent_difference'] for r in json.loads((oracle_run / 'primary_comparison.json').read_text())
            if r['batch_size'] == batch_size]
    diagnostics = [r for r in json.loads((oracle_run / 'all_diagnostics.json').read_text()) if r['batch_size'] == batch_size]
    if len(gaps) != 3 or sum(v > 0 for v in gaps) < 2 or sum(gaps) <= 0:
        raise RuntimeError('Prespecified primary-damping opportunity gate did not pass')
    if len(diagnostics) != 3 or not all(d['numerical_gate_passed'] and d['search_gate_passed'] for d in diagnostics):
        raise RuntimeError('Numerical/search gate did not pass')


def propose(method, model, registry, flat, state, x, y, c):
    fn = functional_logits(model, registry, x)
    objective = lambda v: loss(fn(v), y)
    operator = GGN(fn, flat, y)
    g = operator.gradient
    proposal, next_state, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), state, c['soap'])
    native = registry.flatten(proposal)
    delta = native
    metadata = {'method': method, 'gradient_norm': float(g.norm()), 'native_norm': float(native.norm()),
                'curvature': None, 'solve': None, 'search': None}
    if method == 'exact_tensor_ggn' and all(context is not None for context in contexts.values()):
        metric = SOAPMetric(registry, contexts)
        blocks, curvature = streamed_blocks(model, registry, flat, x, 8, c['resources']['budget_mib'])
        mu = max(.01 * curvature['trace'] / registry.size, c['damping']['floor'])
        solutions, solve_info = solve_blocks(blocks, registry, g, metric, [mu], c['solver']['tolerance64'])
        delta = solutions[mu]
        metadata.update(curvature=curvature, solve=solve_info[mu], mu=mu, raw_oracle_norm=float(delta.norm()))
    if method != 'native_soap' and float(native.norm()) > 0:
        delta, search = broad_direction_search(flat, delta, float(native.norm()), objective)
        metadata['search'] = search
        if not search['bracketed'] and not search['rejected']:
            raise RuntimeError('Training search exceeded its prespecified radius window')
    metadata.update(loss_before=float(objective(flat)), loss_after=float(objective(flat + delta)),
                    norm=float(delta.norm()), slope=float(g @ delta))
    return delta, next_state, contexts, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle-run', type=Path, required=True)
    args = parser.parse_args()
    check_gate(args.oracle_run, 512)
    c = load_config(args.oracle_run / 'resolved.yaml')
    c['training'].update(steps=50, snapshots=[], checkpoint_every=25)
    c['data']['batch_size'] = 512
    validate(c)
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    directory = new_run(c)
    methods = ['native_soap', 'searched_soap', 'exact_tensor_ggn']
    protocol = {'family': 'paired_exact_tensor_training_followup', 'oracle_run': str(args.oracle_run.resolve()),
                'batch_size': 512, 'methods': methods, 'seed': c['seed'], 'iterations': 50,
                'initialization': 'same seed-0 random weights; first consumed iteration initializes SOAP without updating parameters',
                'gamma': .01, 'damping_metric': 'normalized inverse SOAP from each arm own single statistics update',
                'damping_trace': 'exact current-batch GGN trace',
                'search': 'zero and 2^j native-radius multipliers, j=-16..16, then 18 construction-only golden-section refinements',
                'native_soap_policy': 'unmodified SOAP learning rate 0.003, no search',
                'searched_soap_policy': 'same SOAP proposal with common search',
                'exact_tensor_policy': 'true-gradient exact tensor GGN solve, common search; SOAP statistics advance once per batch',
                'validation_steps': [0, 10, 25, 50], 'validation_role': 'measurement only; no tuning or early stopping',
                'workspace_budget_mib': c['resources']['budget_mib'],
                'comparison_axes': ['consumed examples / iterations', 'measured training-step time'],
                'limits': 'one seed and 49 parameter updates; exact curvature with SOAP damping is not a reproduction of the paper'}
    write_json(directory / 'protocol.json', protocol)
    provenance = json.loads((directory / 'provenance.json').read_text())
    for name, expected in provenance['source_checksums'].items():
        destination = directory / 'source' / name
        if checksum(name) != expected:
            raise RuntimeError('Source changed before archive')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    shutil.copyfile(__file__, directory / 'runner.py')
    write_json(directory / 'runner_checksum.json', {'runner.py': checksum(directory / 'runner.py')})
    print(f'Training comparison directory: {directory.resolve()}', flush=True)
    print(json.dumps(protocol, indent=2), flush=True)
    x, y, manifest = dataset(c)
    if manifest != json.loads((args.oracle_run / 'split_manifest.json').read_text()):
        raise RuntimeError('Dataset/split mismatch')
    write_json(directory / 'split_manifest.json', manifest)
    rows, paired_batches, initial_parameters = [], None, None
    for method in methods:
        arm = directory / method
        arm.mkdir()
        # Explicit custom method/policy prevents the generic training runner from
        # accidentally resuming an oracle checkpoint as native SOAP.
        arm_config = copy.deepcopy(c)
        arm_config['training'].update(method=method, policy='native' if method == 'native_soap' else 'broad_direction_search')
        write_json(arm / 'resolved_arm.json', arm_config)
        model = make_model(c)
        registry = Registry(model)
        initial = registry.flatten(dict(model.named_parameters())).detach()
        if initial_parameters is None:
            initial_parameters = initial.clone()
        elif not torch.equal(initial_parameters, initial):
            raise RuntimeError('Paired initialization mismatch')
        write_json(arm / 'workspace_estimate.json', workspace_estimate(registry, c['model']['classes'], 8))
        order = DataOrder(manifest['training'], c['data_seed'] + 1)
        state, accepted, history, total_step_seconds = None, 0, [], 0.0
        rows.append(dict(method=method, step=0, accepted_updates=0, training_seconds=0., **evaluate(model, x, y, manifest['validation'])))
        print(f'Starting {method}', flush=True)
        for step in range(1, 51):
            ids = order.take(512)
            started = time.perf_counter()
            flat = registry.flatten(dict(model.named_parameters())).detach()
            try:
                delta, next_state, contexts, metadata = propose(method, model, registry, flat, state, x[ids], y[ids], c)
            except Exception as error:
                write_json(directory / 'failure.json', {'method': method, 'step': step,
                           'type': type(error).__name__, 'message': str(error)})
                raise
            commit(model, registry, flat + delta)
            state = next_state
            accepted += int(float(delta.norm()) > 0)
            seconds = time.perf_counter() - started
            total_step_seconds += seconds
            metadata.update(step=step, batch_ids=ids, accepted_updates=accepted, step_seconds=seconds)
            history.append(metadata)
            write_json(arm / 'training.json', history)
            if step in [10, 25, 50]:
                measured = dict(method=method, step=step, accepted_updates=accepted,
                                training_seconds=total_step_seconds, **evaluate(model, x, y, manifest['validation']))
                rows.append(measured)
                checkpoint.save(arm / f'checkpoint-{step:04d}.pt', model, state, contexts, step, accepted, order, arm_config)
                write_json(directory / 'validation.json', rows)
                print(f"{method} step={step}: validation loss={measured['validation_loss']:.6f}, "
                      f"accuracy={measured['validation_accuracy']:.2%}, training_seconds={total_step_seconds:.2f}", flush=True)
        batches = [row['batch_ids'] for row in history]
        if paired_batches is None:
            paired_batches = batches
        elif paired_batches != batches:
            raise RuntimeError('Paired training order mismatch')
        write_json(arm / 'runtime.json', {'training_seconds': total_step_seconds,
                   'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)})
    write_json(directory / 'complete.json', {'completed': True, 'paired_initialization': True, 'paired_data_order': True,
               'methods': methods, 'iterations': 50})
    print(f'Completed training comparison: {directory.resolve()}', flush=True)


if __name__ == '__main__':
    main()
