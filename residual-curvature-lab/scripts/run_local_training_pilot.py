"""Explicitly selected, bounded local pilot; no downloading or parameter sweep."""
import argparse
import copy
import csv
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from residual_curvature.analysis.report import summarize
from residual_curvature.config import load_config, validate
from residual_curvature.data import dataset
from residual_curvature.experiments.training import commit, run
from residual_curvature.models import Registry, make_model
from residual_curvature.provenance import checksum, new_run, write_json


@torch.inference_mode()
def evaluate(model, x, y, ids, device):
    loss_sum, correct = 0.0, 0
    for start in range(0, len(ids), 256):
        batch = ids[start:start + 256]
        logits, target = model(x[batch].to(device)), y[batch].to(device)
        loss_sum += float(torch.nn.functional.cross_entropy(logits, target, reduction='sum'))
        correct += int((logits.argmax(-1) == target).sum())
    return {'validation_loss': loss_sum / len(ids), 'validation_accuracy': correct / len(ids),
            'validation_examples': len(ids)}


def check_snapshot_gate(directory):
    c = load_config(directory / 'resolved.yaml')
    if (directory / 'failure.json').exists():
        raise RuntimeError('Snapshot run has a recorded failure; inspect it before training comparisons')
    rows = json.loads((directory / 'training.json').read_text())
    if rows[-1]['step'] != c['training']['steps']:
        raise RuntimeError('Baseline pilot is incomplete')
    tol = c['solver']['tolerance64' if c['dtype'] == 'float64' else 'tolerance32']
    for step in c['training']['snapshots']:
        snap = directory / f'snapshot-{step:04d}'
        diagnostics = json.loads((snap / 'diagnostics.json').read_text())
        candidates = json.loads((snap / 'candidates.json').read_text())
        if any(value > tol for value in diagnostics['checks'].values()):
            raise RuntimeError(f'Numerical gate failed at snapshot {step}')
        if any(row['full_solve_residual'] > tol for row in candidates):
            raise RuntimeError(f'Exact solve gate failed at snapshot {step}')
        for name, expected in json.loads((snap / 'artifacts.json').read_text()).items():
            if checksum(snap / name) != expected:
                raise RuntimeError(f'Artifact checksum mismatch: {snap / name}')
    return c, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-run', type=Path, required=True)
    parser.add_argument('--config', default='configs/mnist_training_pilot.yaml')
    args = parser.parse_args()
    c = load_config(args.config)
    baseline_c, baseline_training = check_snapshot_gate(args.snapshot_run)
    for key in ['seed', 'data_seed', 'probe_seed', 'analysis_seed', 'model', 'data', 'soap', 'device', 'dtype', 'partition']:
        if c[key] != baseline_c[key]:
            raise ValueError(f'Baseline/comparison mismatch: {key}')
    if c['training']['steps'] != baseline_c['training']['steps']:
        raise ValueError('All arms require the same consumed-iteration budget')
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    group = new_run(c)
    print(f'Comparison directory: {group.resolve()}', flush=True)
    shutil.copyfile(__file__, group / 'runner.py')
    write_json(group / 'runner_checksum.json', {'runner.py': checksum(group / 'runner.py')})
    x, y, manifest = dataset(c)
    if not manifest['validation']:
        raise ValueError('This pilot requires the reserved MNIST validation pool')
    if manifest != json.loads((args.snapshot_run / 'split_manifest.json').read_text()):
        raise ValueError('Data fingerprint/split differs from the snapshot run')
    write_json(group / 'split_manifest.json', manifest)
    steps = baseline_c['training']['snapshots']
    baseline_model = make_model(c)
    registry = Registry(baseline_model)
    initial = evaluate(baseline_model, x, y, manifest['validation'], c['device'])
    rows = [dict(method='soap', step=0, accepted_updates=0, training_seconds=0.0, **initial)]
    for step in steps:
        with np.load(args.snapshot_run / f'snapshot-{step:04d}' / 'arrays.npz') as arrays:
            commit(baseline_model, registry, torch.from_numpy(arrays['parameters']).to(c['device']))
        rows.append(dict(method='soap', step=step, accepted_updates=baseline_training[step-1]['accepted_updates'],
                         training_seconds=sum(r['step_seconds'] for r in baseline_training[:step]),
                         **evaluate(baseline_model, x, y, manifest['validation'], c['device'])))
    write_json(group / 'validation.json', rows)
    runs = {'soap': str(args.snapshot_run.resolve())}
    for method in ['sketch_diagonal', 'sketch_coupled']:
        config = copy.deepcopy(c)
        config['training'].update(method=method, policy='training_backtracking')
        validate(config)
        (group / f'{method}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        method_rows = [dict(method=method, step=0, accepted_updates=0, training_seconds=0.0, **initial)]
        def monitor(model, registry, state, data_x, data_y, config, step):
            if step in steps:
                measurement = dict(method=method, step=step,
                                   **evaluate(model, data_x, data_y, manifest['validation'], config['device']))
                method_rows.append(measurement)
                write_json(group / f'{method}_validation_progress.json', method_rows)
        started = time.perf_counter()
        directory, _, _ = run(config, analyze=False, monitor=monitor)
        training = json.loads((directory / 'training.json').read_text())
        if [r['batch_ids'] for r in training] != [r['batch_ids'] for r in baseline_training]:
            raise RuntimeError('Paired data-order check failed')
        for row in method_rows[1:]:
            row.update(accepted_updates=training[row['step']-1]['accepted_updates'],
                       training_seconds=sum(r['step_seconds'] for r in training[:row['step']]))
        rows.extend(method_rows)
        runs[method] = str(directory.resolve())
        write_json(directory / 'validation.json', method_rows)
        write_json(directory / 'pilot_runtime.json', {'including_load_and_validation_seconds': time.perf_counter()-started,
                   'process_lifetime_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)})
        write_json(group / 'validation.json', rows)
        write_json(group / 'runs.json', runs)
        summarize(directory)
    write_json(group / 'runs.json', runs)
    with (group / 'validation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_and_report(group, rows, runs)
    print(group.resolve(), flush=True)


def plot_and_report(group, rows, runs):
    os.environ.setdefault('MPLCONFIGDIR', str((group / '.matplotlib').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((group / '.cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    final = []
    for method, directory in runs.items():
        subset = [r for r in rows if r['method'] == method]
        axes[0].plot([r['step'] for r in subset], [r['validation_loss'] for r in subset], marker='o', label=method)
        axes[1].plot([r['training_seconds'] for r in subset], [r['validation_loss'] for r in subset], marker='o', label=method)
        axes[2].plot([r['step'] for r in subset], [100*r['validation_accuracy'] for r in subset], marker='o', label=method)
        r = subset[-1]
        training = json.loads((Path(directory) / 'training.json').read_text())
        final.append(f"| {method} | {r['validation_loss']:.6f} | {100*r['validation_accuracy']:.2f}% | "
                     f"{r['accepted_updates']} | {r['training_seconds']:.3f} | "
                     f"{sum(not t['accepted'] for t in training)} |")
    axes[0].set(xlabel='Consumed iterations', ylabel='Validation cross-entropy')
    axes[1].set(xlabel='Cumulative measured training-step seconds', ylabel='Validation cross-entropy')
    axes[2].set(xlabel='Consumed iterations', ylabel='Validation accuracy (%)')
    axes[0].legend()
    fig.suptitle('Downsampled 14×14 MNIST — seed-0 feasibility pilot, 5,000 validation examples')
    fig.savefig(group / 'comparison.png', dpi=160)
    plt.close(fig)
    (group / 'REPORT.md').write_text(
        '# Local MNIST training pilot\n\n'
        'Three arms use the same seed-0 initialization, split, and 100-batch order. '
        'The first SOAP iteration initializes statistics without a parameter update. '
        'The official test set was not evaluated. Validation measurements do not select settings.\n\n'
        '| Method | Final validation loss | Accuracy | Accepted updates | Training-step seconds | Non-updates including initialization |\n'
        '| --- | --- | --- | --- | --- | --- |\n' + '\n'.join(final) + '\n\n'
        'Sketch arms use the prespecified rank 8, gamma 0.01 and current estimated-trace damping. '
        'They share the true-gradient quadratic and Armijo policy, while native SOAP uses momentum '
        'and its configured learning rate. Coupling is isolated only at matched fixed states; '
        'independent training also changes subsequent gradients and optimizer statistics.\n\n'
        'Training-step time excludes dataset loading, snapshot diagnostics, validation, checkpoints '
        'and plotting; it includes candidate construction and line-search evaluations. It is a '
        'measured prototype cost, not an end-to-end speed claim. See each run for elapsed logs. '
        'One paired seed and un-tuned starting settings do not support replicated superiority claims. '
        'The two methods that reject a step still advance statistics exactly once.\n\n'
        'Per-state values: validation.csv and validation.json. Run locations: runs.json. '
        'Plot: comparison.png. Failed/rejected searches remain in each training.json.\n')


if __name__ == '__main__':
    main()
