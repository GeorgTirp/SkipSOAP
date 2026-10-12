"""Section 4.5 cost dry run: three steps of every training arm, one fixed-state
candidate evaluation, and the per-snapshot curvature costs used by checks 3 and 4.
Writes measured seconds and projected CPU-hours per check to the run directory."""
import json
import time
from pathlib import Path

import torch

import common
from common import read
import arms
from residual_curvature.curvature import wasserstein
from residual_curvature.curvature.layerwise import streamed_blocks
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import SOAPMetric
from residual_curvature.provenance import write_json
from run_exact_training_followup import evaluate
from run_large_batch_oracle import measurement
from run_mini_vit_allocation import search_candidate

PARENT = Path('runs/20261005T143029-9b582ed5')
ARM_SETTINGS = {'native_soap': {'lr': .003}, 'adam_native': {'lr': .003}, 'searched_soap': {},
                'searched_soap_tuned': {'betas': [.9, .99], 'precondition_frequency': 1},
                'ggn_exact_trace': {'gamma': .01}, 'dense_ef': {'gamma': .01}, 'soap_gradient_map': {}}
ARCHITECTURES = [(b, w) for b in [1, 2, 3] for w in [8, 16, 24]]


def three_steps(arm, settings, c, x, y, manifest):
    c = arms.arm_config(c, arm, settings)
    model = make_model(c)
    registry = Registry(model)
    order = DataOrder(manifest['training'], common.data_order_seed(c['seed']))
    state, rows = None, []
    for step in range(1, 4):
        ids = order.take(512)
        flat = registry.flatten(dict(model.named_parameters())).detach()
        started = time.perf_counter()
        delta, state, _, row = arms.propose_step(arm, settings, 'study', model, registry, flat, state, x[ids], y[ids], c, step)
        commit(model, registry, flat+delta)
        rows.append(time.perf_counter()-started-row.get('audit_seconds', 0.))
    return {'step_seconds': rows, 'per_step_after_init': sum(rows[1:])/2, 'parameters': registry.size}, model, registry, state


def fixed_state_evaluation(c, x, y):
    """One SOAP-map candidate at saved SOAP state 10, rebuilt as run_learnable_capacity.audit does."""
    source = PARENT/'searched_soap-0010'
    arrays = torch.load(source/'arrays.pt', map_location='cpu', weights_only=False)
    info = read(source/'diagnostics.json')
    model = make_model(c)
    registry = Registry(model)
    flat, g, contexts = arrays['parameters'], arrays['gradient'], arrays['contexts']
    commit(model, registry, flat)
    construction = info['construction_ids']
    fn = functional_logits(model, registry, x[construction])
    objective = lambda w: loss(fn(w), y[construction])
    started = time.perf_counter()
    op = GGN(fn, flat, y[construction])
    metric = SOAPMetric(registry, contexts)
    mu = max(.01*sum(float(b.trace()) for b in arrays['ggn_blocks'])/registry.size, c['damping']['floor'])
    panels = [lambda w, f=functional_logits(model, registry, x[ids]), t=y[ids]: loss(f(w), t) for ids in info['panel_ids']]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    setup = time.perf_counter()-started
    started = time.perf_counter()
    selected, search = search_candidate(flat, -metric.map(g), arrays['raw_soap'], objective)
    values = measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry)
    evaluation = time.perf_counter()-started
    started = time.perf_counter()
    for _ in range(10):
        op.action(g)
    action = (time.perf_counter()-started)/10
    started = time.perf_counter()
    wasserstein.streamed_wasserstein_blocks(model, registry, flat, x[construction], 8, c['resources']['budget_mib'])
    ot = time.perf_counter()-started
    return {'state': str(source), 'setup_seconds': setup, 'candidate_evaluation_seconds': evaluation,
            'search_evaluations': search['evaluations'], 'ggn_action_seconds': action, 'ot_block_seconds': ot,
            'soap_map_independent_decrease': values['independent_mean_decrease'],
            'capacity_screen_reference': .033191}


def main():
    common.guard()
    common.setup_torch()
    c = common.study_config()
    directory = common.create_run(c, {'study': 'gap_audit_20261007', 'section': '4.5 cost dry run',
                                      'document': str(common.PROTOCOL)})
    print(directory, flush=True)
    x, y, manifest = dataset(c)
    started = time.perf_counter()
    validation = evaluate(make_model(c), x, y, manifest['validation'])
    result = {'validation_seconds': time.perf_counter()-started, 'arms': {}, 'architectures': {}}
    for arm in arms.ARMS:
        result['arms'][arm], *_ = three_steps(arm, ARM_SETTINGS[arm], c, x, y, manifest)
        result['arms'][arm]['settings'] = ARM_SETTINGS[arm]
        print(arm, json.dumps(result['arms'][arm]), flush=True)
        write_json(directory/'dryrun.json', result)
    result['arms']['searched_soap_pf10_b9-99'], *_ = three_steps(
        'searched_soap_tuned', {'betas': [.9, .99], 'precondition_frequency': 10}, c, x, y, manifest)
    for blocks, width in ARCHITECTURES:
        arch = common.study_config(model={'width': width, 'expansion': 2*width, 'blocks': blocks})
        timing, model, registry, _ = three_steps('searched_soap', {}, arch, x, y, manifest)
        flat = registry.flatten(dict(model.named_parameters())).detach()
        ids = manifest['analysis'][:512]
        begun = time.perf_counter()
        streamed_blocks(model, registry, flat, x[ids], 8, arch['resources']['budget_mib'])
        timing['ggn_block_seconds'] = time.perf_counter()-begun
        op = GGN(functional_logits(model, registry, x[ids]), flat, y[ids])
        begun = time.perf_counter()
        for _ in range(5):
            op.action(op.gradient)
        timing['ggn_action_seconds'] = (time.perf_counter()-begun)/5
        timing['largest_tensor'] = max(s.count for s in registry.specs)
        result['architectures'][f'b{blocks}_w{width}'] = timing
        print(f'b{blocks}_w{width}', json.dumps(timing), flush=True)
        write_json(directory/'dryrun.json', result)
    result['fixed_state'] = fixed_state_evaluation(c, x, y)
    print('fixed_state', json.dumps(result['fixed_state']), flush=True)
    result['initial_validation'] = validation
    write_json(directory/'dryrun.json', result)
    write_json(directory/'complete.json', {'completed': True})
    print(f'mirrored to {common.mirror(directory)}', flush=True)


if __name__ == '__main__':
    main()
