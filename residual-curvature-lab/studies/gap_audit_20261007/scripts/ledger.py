"""Actual CPU-hours per check: summed single-threaded process wall time, sleep excluded.

Check-1 units record their own wall time (perf_counter, which stops during sleep). Other
units use driver-log start/done stamps or run-file modification times, minus any overlap
with the recorded sleep window (DEVIATIONS #10). Interrupted attempts are counted.
Usage: ledger.py RUN_ID ... (dry run, gate, tune, eval, 2c, 3, 4a, 4b in that order)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import common
from common import read
from residual_curvature.provenance import write_json

SLEEP = (datetime(2026, 10, 7, 13, 0, 5, tzinfo=timezone.utc), datetime(2026, 10, 7, 13, 56, 23, tzinfo=timezone.utc))


def awake_seconds(start, end):
    overlap = max(0., (min(end, SLEEP[1])-max(start, SLEEP[0])).total_seconds())
    return (end-start).total_seconds()-overlap


def stamp(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00'))


def mtime(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)


def check1_seconds(run):
    total = 0.
    for runtime in run.glob('*/runtime.json'):
        total += read(runtime)['unit_wall_seconds']
    for partial in run.glob('*.interrupted-*'):
        rows = read(partial/'validation.json')
        total += rows[-1]['training_seconds']  # lower bound: excludes its validations and startup
    return total


def driver_seconds(run):
    log = common.STUDY/'logs'/run.name/'driver.log'
    starts, total = {}, 0.
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == 'start':
            starts[parts[2]] = stamp(parts[0])
        elif len(parts) == 3 and parts[1] in ['done', 'FAILED']:
            total += awake_seconds(starts.pop(parts[2]), stamp(parts[0]))
    return total


def sequential_seconds(run):
    return awake_seconds(mtime(run/'provenance.json'), mtime(run/'complete.json'))


def main():
    common.guard()
    ids = sys.argv[1:]
    runs = dict(zip(['dryrun', 'gate', 'tune', 'eval', 'check2c', 'check3', 'check4a', 'check4b'], [common.RUNS/i for i in ids]))
    seconds = {'dryrun': sequential_seconds(runs['dryrun']), 'gate': check1_seconds(runs['gate']),
               'tune': check1_seconds(runs['tune']), 'eval': check1_seconds(runs['eval']),
               'check2c': sequential_seconds(runs['check2c']), 'check3': sequential_seconds(runs['check3']),
               'check4a': driver_seconds(runs['check4a']), 'check4b': driver_seconds(runs['check4b'])}
    hours = {'section_4_dry_run': seconds['dryrun']/3600,
             'check1_with_2a_2b': (seconds['gate']+seconds['tune']+seconds['eval'])/3600,
             'check2c': seconds['check2c']/3600, 'check3': seconds['check3']/3600,
             'check4': (seconds['check4a']+seconds['check4b'])/3600}
    result = {'seconds': seconds, 'cpu_hours': hours, 'sleep_window_utc': [t.isoformat() for t in SLEEP],
              'method': __doc__.strip()}
    write_json(common.STUDY/'logs'/'cost_actual.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
