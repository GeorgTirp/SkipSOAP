"""Gates G1-G3, readings R0-R3 (PROTOCOL.md, sections 5-6) and the cost table C1 of a gpu_port run.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  summarize.py --run DIR [--cost DIR]
Writes studies/gpu_port_20261012/results/summary.json and prints the tables.
"""
import argparse
import re
import sys
from pathlib import Path

import common
from common import read

import torch  # noqa: E402

import train  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

LAPTOP_LOGS = [Path('studies/topfree_confirm_20261011/logs')/run for run in
               ('20261011T135636-50234c98', '20261011T135638-221c0b6a', '20261011T140034-192dd041')]
LAPTOP_NAMES = {'soap': 'soap__constant__lr0.01__s{}', 'topfree': 'topfree__constant__lr0.01_c64__s{}'}
R1_LIMIT, R3_LIMIT, EVAL_SEEDS = 1e-8, 1e-3, (4, 5, 6)


def validation(directory, arm, configuration, seed):
    return {r['step']: r for r in read(directory/train.name_of(arm, configuration, seed)/'validation.json')}


def ema(record):
    return record['ema0.98']['validation_loss']


def final_parameters(directory, arm, configuration, seed):
    return torch.load(directory/train.name_of(arm, configuration, seed)/'final_parameters.pt', weights_only=False)


def same_run(directory, arm, a, b, seed=0):
    """Bitwise equality of two units: every validation value (raw and all EMAs) and the final parameters."""
    va, vb = validation(directory, arm, a, seed), validation(directory, arm, b, seed)
    keys = ['validation_loss', 'validation_accuracy']
    values = all(va[s][k] == vb[s][k] for s in va for k in keys) and all(
        va[s][f'ema{d:g}'][k] == vb[s][f'ema{d:g}'][k] for s in va if s > 0 for d in train.EMA_DECAYS for k in keys)
    return va.keys() == vb.keys() and values and torch.equal(final_parameters(directory, arm, a, seed),
                                                             final_parameters(directory, arm, b, seed))


def laptop_log(arm, seed):
    """{step: (validation loss, EMA 0.98 loss)} at 4 decimals from the laptop's stdout logs."""
    name = LAPTOP_NAMES[arm].format(seed)+'.log'
    path = next(d/name for d in LAPTOP_LOGS if (d/name).exists())
    out = {}
    for line in path.read_text().splitlines():
        match = re.search(r'step (\d+): val=([\d.]+) ema\.98=([\d.]+)', line)
        if match:
            out[int(match[1])] = (float(match[2]), float(match[3]))
    return out


def summarize(directory):
    specs = read(directory/'protocol.json')['units']
    out = {'run': str(directory), 'units': len(specs)}
    out['G1_copy_equals_original'] = {arm: same_run(directory, arm, 'cpu64orig', 'cpu64') for arm in train.ARMS}
    out['G2_gpu_repeatable'] = {f'{arm}_{c}': same_run(directory, arm, c, f'{c}rep')
                                for arm in train.ARMS for c in ('gpu64', 'gpu32')}
    g3, r1 = {}, {}
    for arm in train.ARMS:
        unit = directory/train.name_of(arm, 'cpu64', 0)
        cpu, gpu = read(unit/'replay_cpu.json')['rows'], read(unit/'replay_cuda.json')['rows']
        g3[arm] = all(r['relative_update_error'] == 0. and all(v == 0. for k, v in r['state_max_abs_difference'].items()
                                                               if not k.endswith('_tensor')) for r in cpu)
        r1[arm] = {
            'steps': len(gpu),
            'max_relative_update_error_plain': max((r['relative_update_error'] for r in gpu if not r['probe']), default=None),
            'max_relative_update_error_probe': max((r['relative_update_error'] for r in gpu if r['probe']), default=None),
            'refresh_steps': [{'step': r['step'], **r['state_max_abs_difference']} for r in gpu if r['soap_refresh']],
            'max_state_difference_non_refresh': {k: max(r['state_max_abs_difference'].get(k, 0.) for r in gpu if not r['soap_refresh'])
                                                 for k in ('Q', 'exp_avg', 'exp_avg_sq', 'held_operator_abs')},
            'pass': all(r['relative_update_error'] <= R1_LIMIT for r in gpu)}
    out['G3_cpu_replay_bitwise'], out['R1_one_step'] = g3, r1
    out['R0_laptop'] = {}
    for arm in train.ARMS:
        for seed in (0, *EVAL_SEEDS):
            here, there = validation(directory, arm, 'cpu64', seed), laptop_log(arm, seed)
            common_steps = sorted(set(here) & set(there))
            out['R0_laptop'][f'{arm}_s{seed}'] = {
                'steps_compared': len(common_steps),
                'max_abs_validation': max(abs(here[s]['validation_loss']-there[s][0]) for s in common_steps),
                'max_abs_ema0.98': max(abs(ema(here[s])-there[s][1]) for s in common_steps if s > 0),
                'step500_here': [here[500]['validation_loss'], ema(here[500])], 'step500_laptop': list(there[500])}
    r2 = {}
    for arm in train.ARMS:
        cpu, gpu = validation(directory, arm, 'cpu64', 0), validation(directory, arm, 'gpu64', 0)
        r2[arm] = {'by_step': {s: {'raw': gpu[s]['validation_loss']-cpu[s]['validation_loss'],
                                   'ema0.98': (ema(gpu[s])-ema(cpu[s])) if s > 0 else 0.} for s in cpu},
                   'step500_ema0.98_abs': abs(ema(gpu[500])-ema(cpu[500]))}
    floor = max(v['step500_ema0.98_abs'] for v in r2.values())
    out['R2_trajectory'], out['device_floor'] = r2, floor
    differences = {f'{arm}_s{seed}': abs(ema(validation(directory, arm, 'gpu32', seed)[500])
                                         - ema(validation(directory, arm, 'cpu64', seed)[500]))
                   for arm in train.ARMS for seed in (0, *EVAL_SEEDS)}
    a_holds, excused = all(v <= R3_LIMIT for v in differences.values()), floor > R3_LIMIT
    order = {}
    for configuration in ('gpu32', 'cpu64'):
        order[configuration] = {seed: [ema(validation(directory, arm, configuration, seed)[500]) for arm in train.ARMS]
                                for seed in EVAL_SEEDS}
    b_holds = all(top < soap for soap, top in order['gpu32'].values())
    out['R3_float32'] = {'a_step500_ema0.98_abs_difference': differences, 'a_holds': a_holds, 'a_excused_by_floor': excused,
                         'b_soap_topfree_ema0.98_step500': order, 'b_holds': b_holds,
                         'adequate': b_holds and (a_holds or excused)}
    out['gates_pass'] = all(out['G1_copy_equals_original'].values()) and all(out['G2_gpu_repeatable'].values()) and all(g3.values())
    out['R1_pass'] = all(v['pass'] for v in r1.values())
    return out


def cost_table(directory):
    rows = read(directory/'cost_summary.json')
    lines = ['| Model | Optimizer | Parameters | s/step plain | s/step probe | mean s/step | peak GiB | status |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    fmt = lambda v: '—' if v is None else f'{v:.4f}'
    for r in rows:
        lines.append(f"| {r['spec']['model']} | {r['spec']['optimizer']} | {r['parameters']:,} | "
                     f"{fmt(r['median_seconds_plain_step'])} | {fmt(r['median_seconds_probe_step'])} | "
                     f"{fmt(r['mean_seconds_per_step'])} | {r['peak_allocated_bytes']/2**30:.2f} | {r['outcome']['status']} |")
    return rows, '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cost', type=Path)
    args = parser.parse_args()
    common.setup_torch()
    out = summarize(args.run)
    if args.cost:
        out['C1_cost'], table = cost_table(args.cost)
        print(table)
    results = common.STUDY/'results'
    results.mkdir(exist_ok=True)
    write_json(results/'summary.json', out)
    for key in ['G1_copy_equals_original', 'G2_gpu_repeatable', 'G3_cpu_replay_bitwise', 'R1_pass', 'device_floor']:
        print(key, out[key])
    print('R1', {arm: {k: v for k, v in r.items() if k != 'refresh_steps'} for arm, r in out['R1_one_step'].items()})
    print('R0', {k: (round(v['max_abs_validation'], 5), round(v['max_abs_ema0.98'], 5)) for k, v in out['R0_laptop'].items()})
    print('R3', {k: v for k, v in out['R3_float32'].items()})


if __name__ == '__main__':
    sys.exit(main())
