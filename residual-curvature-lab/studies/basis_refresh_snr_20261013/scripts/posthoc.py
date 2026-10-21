"""Post-hoc transfer check of Part A (DEVIATIONS.md D2; no verdict).

Reads results/ladder (pre-declared grid) and results/ladder_posthoc (grid alpha0 * 2^k, k in -14..0). Checks that the
recomputed quadratic results equal the first run's bitwise, then reports per candidate the chosen k, whether it is
interior to the grid, and the evaluation-batch decrease; and the early-step counts of probe_exact > soap_exact and
probe_probe > soap_map. Writes results/posthoc_summary.json.
"""
import json
import math
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1]/'results'
CANDIDATES = ['soap_map', 'soap_exact', 'soapeigh_exact', 'probe_exact', 'probe_probe', 'kron_exact', 'ggn']
EARLY = [10, 25, 50]
LOW, HIGH = -14, 0


def main():
    out = {'states': {}, 'reproduced': True}
    for path in sorted((RESULTS/'ladder_posthoc').glob('*.json')):
        new = json.loads(path.read_text())
        old = json.loads((RESULTS/'ladder'/path.name).read_text())
        same = new['tensors'] == old['tensors'] and new['mu'] == old['mu']
        out['reproduced'] &= same
        row = {'step': new['step'], 'seed': new['seed'], 'quadratic_reproduced': same, 'candidates': {}}
        for c in CANDIDATES:
            t = new['transfer'][c]
            k = round(math.log2(t['multiplier']))
            row['candidates'][c] = {'k': k, 'interior': LOW < k < HIGH, 'evaluation_decrease': t['evaluation_decrease'],
                                    'selection_decrease': t['selection_decrease']}
        out['states'][path.stem] = row
    early = [r for r in out['states'].values() if r['step'] in EARLY]
    count = lambda a, b: sum(r['candidates'][a]['evaluation_decrease'] > r['candidates'][b]['evaluation_decrease'] for r in early)
    out['early_counts'] = {'probe_exact_vs_soap_exact': count('probe_exact', 'soap_exact'),
                           'probe_probe_vs_soap_map': count('probe_probe', 'soap_map'), 'of': len(early)}
    out['interior_fraction'] = statistics.mean(c['interior'] for r in out['states'].values() for c in r['candidates'].values())
    out['by_step'] = {}
    for step in sorted({r['step'] for r in out['states'].values()}):
        rows = sorted((r for r in out['states'].values() if r['step'] == step), key=lambda r: r['seed'])
        out['by_step'][step] = {c: {'decrease': [r['candidates'][c]['evaluation_decrease'] for r in rows],
                                    'k': [r['candidates'][c]['k'] for r in rows]} for c in CANDIDATES}
    (RESULTS/'posthoc_summary.json').write_text(json.dumps(out, indent=1, allow_nan=False)+'\n')
    print(json.dumps({k: out[k] for k in ('reproduced', 'early_counts', 'interior_fraction')}, indent=1))
    for step, row in out['by_step'].items():
        print(step, {c: ('/'.join(f'{d:.4f}' for d in v['decrease']), v['k']) for c, v in row.items()})


if __name__ == '__main__':
    sys.exit(main())
