"""Selected paired transport/PARSEC/NI-SOAP pilot from a prospective protocol."""
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
from residual_curvature.curvature import wasserstein
from residual_curvature.curvature.layerwise import streamed_blocks, solve_blocks
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.experiments.search import noise_learning_rate_search
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import parsec, ni_soap
from residual_curvature.optim.soap import SOAPMetric, soap_propose
from residual_curvature.provenance import checksum, new_run, write_json
from run_exact_training_followup import evaluate
from run_large_batch_oracle import state_equal
from run_mini_vit_allocation import search_candidate


METHODS = ['adam_native', 'adam_searched', 'parsec_native', 'parsec_searched',
           'ni_soap_alpha0', 'ni_soap_alpha1', 'ggn_exact_trace', 'wasserstein_tensor']
NI_LR_METHODS = ['ni_soap_alpha0_lr_searched', 'ni_soap_alpha1_lr_searched']


def optimizer_settings(method):
    if method.startswith(('adam_', 'parsec_')):
        return {'lr': .003, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.,
                'rank': 16, 'lanczos_iters': 32, 'refresh_every': 10 if method.startswith('parsec') else 0,
                'damping': 1e-4, 'seed': 0, 'split_spaces': False}
    if method.startswith('ni_soap_'):
        return {'lr': .003, 'betas': [.95, .95], 'eps': 1e-8, 'shampoo_beta': .95,
                'refresh_every': 10, 'batch_size': 512, 'seed': 0,
                'alpha': 0. if 'alpha0' in method else 1., 'weight_decay': 0.}
    return {'gamma': .01, 'trace': 'exact current metric trace', 'partition': 'tensor',
            'metric': 'trace-normalized inverse shadow SOAP', 'microbatch': 8}


def propose(method, model, registry, flat, state, x, y, c, step, settings, routes, audit=False):
    fn = functional_logits(model, registry, x)
    objective = lambda v: loss(fn(v), y)
    op = GGN(fn, flat, y)
    contexts, algorithm, audit_seconds, search = None, {}, 0., None
    if method.startswith(('adam_', 'parsec_')):
        delta, next_state, algorithm = parsec.propose(op.gradient, flat, state, settings, op)
        native = delta
    elif method.startswith('ni_soap_'):
        drift, noise, next_state, algorithm = ni_soap.decompose(op.gradient, flat, registry, routes, state, settings)
        native = delta = drift+noise
        if method.endswith('_lr_searched'):
            delta, search = noise_learning_rate_search(flat, drift, noise, objective, settings['lr'])
            a = search['selected_alpha']
            base = copy.deepcopy(algorithm)
            for key in ['noise_energy', 'expected_noise_energy', 'expected_matrix_noise_energy', 'expected_fallback_noise_energy']:
                algorithm[key] *= a
            algorithm['drift_energy'] *= a*a
            algorithm['drift_noise_cross_term'] *= a**1.5
            algorithm['update_energy'] = float(delta.square().sum())
            for tensor in algorithm['per_tensor'].values():
                tensor['drift_energy'] = tensor.get('drift_energy', 0.)*a*a
                tensor['noise_energy'] *= a
                tensor['expected_noise_energy'] *= a
            algorithm.update(base_lr_accounting=base, selected_learning_rate=search['selected_lr'],
                             learning_rate_multiplier=a,
                             expectation_note='Scaled pre-selection fixed-eta expectation; selection depends on the noise draw')
    else:
        proposal, next_state, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(op.gradient), state, c['soap'])
        native = registry.flatten(proposal)
        delta = native
        if all(value is not None for value in contexts.values()):
            metric = SOAPMetric(registry, contexts)
            if method == 'ggn_exact_trace':
                blocks, construction = streamed_blocks(model, registry, flat, x, 8, c['resources']['budget_mib'])
            elif method == 'wasserstein_tensor':
                blocks, construction = wasserstein.streamed_wasserstein_blocks(model, registry, flat, x, 8, c['resources']['budget_mib'])
            else:
                raise ValueError(method)
            mu = max(.01*construction['trace']/registry.size, c['damping']['floor'])
            values, solve = solve_blocks(blocks, registry, op.gradient, metric, [mu], c['solver']['tolerance64'])
            delta = values[mu]
            algorithm = {'curvature': construction, 'mu': mu, 'solve': solve[mu]}
            if audit:
                started = time.perf_counter()
                gen = torch.Generator().manual_seed(c['analysis_seed']+step)
                checks = {}
                for spec, block in zip(registry.specs, blocks):
                    v = torch.zeros_like(flat)
                    v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=flat.dtype)
                    product = op.action(v) if method == 'ggn_exact_trace' else wasserstein.action(model, registry, flat, x, v)
                    checks[spec.name] = relative_error(block@v[spec.slice], product[spec.slice])
                if max(checks.values()) > c['solver']['tolerance64']:
                    raise RuntimeError('Curvature action audit failed')
                algorithm['action_checks'] = checks
                audit_seconds = time.perf_counter()-started
    raw_norm = float(native.norm() if method.endswith('_lr_searched') else delta.norm())
    if search is None and (method.endswith('_searched') or method in ['ggn_exact_trace', 'wasserstein_tensor']):
        if float(native.norm()) > 0:
            delta, search = search_candidate(flat, delta, native, objective)
    row = {'method': method, 'algorithm': algorithm, 'search': search, 'audit_seconds': audit_seconds,
           'native_radius': float(native.norm()), 'raw_norm': raw_norm, 'norm': float(delta.norm()),
           'gradient_norm': float(op.gradient.norm()), 'slope': float(op.gradient@delta),
           'loss_before': float(objective(flat)), 'loss_after': float(objective(flat+delta)),
           'operator_counts': op.counts}
    if not torch.isfinite(delta).all():
        raise RuntimeError('Nonfinite update')
    return delta, next_state, contexts, row


def main(methods=None, protocol_document='docs/transport_parsec_noise_protocol_20261006.md'):
    methods = METHODS if methods is None else methods
    c = load_config('configs/mini_vit_allocation.yaml')
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    directory = new_run(c)
    print(f'Transport/PARSEC/NI-SOAP run: {directory.resolve()}', flush=True)
    sources = json.loads((directory/'provenance.json').read_text())['source_checksums']
    files = ['scripts/run_transport_parsec_noise.py', 'scripts/run_exact_training_followup.py',
             'scripts/run_large_batch_oracle.py', 'scripts/run_mini_vit_allocation.py',
             'docs/transport_parsec_noise_protocol_20261006.md', protocol_document]
    files.extend(str(p) for p in Path('references/user_optimizers').glob('*') if p.is_file())
    for name in files:
        sources[name] = checksum(name)
    for name, digest in sources.items():
        assert checksum(name) == digest
        target = directory/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    write_json(directory/'source_archive_checksums.json', sources)
    write_json(directory/'protocol.json', {'methods': methods, 'steps': 50, 'batch': 512,
               'document': protocol_document,
               'port_precision': 'float64 Torch translation, not bitwise JAX parity',
               'noise_search': any(m.endswith('_lr_searched') for m in methods), 'curvature_scope': 'GGN/Wasserstein tensor blocks; PARSEC global Ritz vectors'})
    x, y, manifest = dataset(c)
    write_json(directory/'split_manifest.json', manifest)
    initial_model = make_model(c)
    registry = Registry(initial_model)
    initial = registry.flatten(dict(initial_model.named_parameters())).detach()
    routes = ni_soap.matrix_routes(initial_model)
    write_json(directory/'ni_soap_routes.json', routes)
    write_json(directory/'registry.json', registry.json())
    validation, verification, paired_batches = [], [], None
    for method in methods:
        arm = directory/method
        arm.mkdir()
        settings = optimizer_settings(method)
        write_json(arm/'optimizer_settings.json', settings)
        arm_config = copy.deepcopy(c)
        policy = 'fixed_draw_learning_rate_search' if method.endswith('_lr_searched') else ('native' if method.endswith('_native') or method.startswith('ni_') else 'broad_direction_search')
        arm_config['training'].update(method=method, policy=policy)
        write_json(arm/'resolved_arm.json', arm_config)
        model = make_model(c)
        assert torch.equal(initial, registry.flatten(dict(model.named_parameters())).detach())
        order = DataOrder(manifest['training'], c['data_seed']+1)
        state, history, accepted, seconds, audit_seconds = None, [], 0, 0., 0.
        validation.append(dict(method=method, step=0, training_seconds=0., **evaluate(model, x, y, manifest['validation'])))
        print(f'Starting {method}', flush=True)
        for step in range(1, 51):
            ids = order.take(512)
            flat = registry.flatten(dict(model.named_parameters())).detach()
            started = time.perf_counter()
            try:
                delta, state, contexts, row = propose(method, model, registry, flat, state, x[ids], y[ids], c, step,
                                                      settings, routes, audit=step in [2, 50])
            except Exception as error:
                write_json(directory/'failure.json', {'method': method, 'step': step, 'error': repr(error)})
                raise
            commit(model, registry, flat+delta)
            elapsed = time.perf_counter()-started-row['audit_seconds']
            seconds += elapsed
            audit_seconds += row['audit_seconds']
            accepted += int(float(delta.norm()) > 0)
            row.update(step=step, batch_ids=ids, step_seconds=elapsed, accepted_updates=accepted)
            history.append(row)
            write_json(arm/'training.json', history)
            if step in [10, 25, 50]:
                measured = dict(method=method, step=step, training_seconds=seconds,
                                **evaluate(model, x, y, manifest['validation']))
                validation.append(measured)
                write_json(directory/'validation.json', validation)
                checkpoint.save(arm/f'checkpoint-{step:04d}.pt', model, state, contexts, step, accepted, order, arm_config)
                print(f"{method} step {step}: val={measured['validation_loss']:.6f}, acc={measured['validation_accuracy']:.2%}, seconds={seconds:.2f}", flush=True)
        batches = [r['batch_ids'] for r in history]
        if paired_batches is None:
            paired_batches = batches
        assert paired_batches == batches
        write_json(arm/'runtime.json', {'training_seconds': seconds, 'numerical_audit_seconds': audit_seconds,
                   'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)})
        # Verify final restoration and one identical proposal from its next batch.
        before_model, before_state = copy.deepcopy(model.state_dict()), copy.deepcopy(state)
        before_order, before_rng = copy.deepcopy(order.state_dict()), torch.get_rng_state().clone()
        replay_order = DataOrder(manifest['training'], c['data_seed']+1)
        replica = make_model(c)
        restored = checkpoint.load(arm/'checkpoint-0050.pt', replica, replay_order, arm_config)
        assert state_equal(before_order, replay_order.state_dict()) and state_equal(before_state, restored['optimizer'])
        reproduced = evaluate(replica, x, y, manifest['validation'])
        for key in ['validation_loss', 'validation_accuracy']:
            assert reproduced[key] == measured[key]
        ids = replay_order.take(512)
        flat = registry.flatten(dict(model.named_parameters())).detach()
        a, sa, _, _ = propose(method, model, registry, flat, state, x[ids], y[ids], c, 51, settings, routes)
        b, sb, _, _ = propose(method, replica, registry, flat, restored['optimizer'], x[ids], y[ids], c, 51, settings, routes)
        assert torch.equal(a, b) and state_equal(sa, sb)
        assert state_equal(before_model, model.state_dict()) and state_equal(before_state, state)
        assert state_equal(before_order, order.state_dict()) and torch.equal(before_rng, torch.get_rng_state())
        verification.append({'method': method, 'restored_validation': reproduced,
                             'next_proposal_replay_passed': True, 'nonmutation_passed': True})
        write_json(directory/'checkpoint_verification.json', verification)
    write_json(directory/'complete.json', {'completed': True, 'methods': methods, 'steps': 50,
               'paired_initialization': True, 'paired_data_order': True, 'checkpoint_replay_passed': True})
    print(f'Completed: {directory.resolve()}', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ni-lr-search', action='store_true')
    args = parser.parse_args()
    main(NI_LR_METHODS, 'docs/ni_soap_lr_search_protocol_20261006.md') if args.ni_lr_search else main()
