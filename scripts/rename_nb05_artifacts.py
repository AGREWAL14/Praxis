"""Rename the NB05 artifacts from browser-collision names into run_id folders.

Downloading seven runs' worth of config.json and metrics.json out of Drive gives
"config.json", "config (1).json" ... "config (6).json" and the same for metrics.
The numbers carry no meaning, so this script does not trust them to pair a config
with its metrics. It pairs by content: for each numbered config it reads the
run_id, then checks the same-numbered metrics file's macro_f1 against the row for
that run_id in PROJECT_RECORD.md Section 5's NB05 table. Model, task and split are
checked against the same row.

Nothing is renamed unless all seven pairs pass. Files are moved, never deleted.

    python scripts/rename_nb05_artifacts.py [--dry-run]
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB05 = os.path.join(REPO, 'results', 'NB05')
RECORD = os.path.join(REPO, 'PROJECT_RECORD.md')

# The table prints macro-F1 to four decimal places.
TOLERANCE = 5e-5


def read_table(path):
    """Return {run_id: {...}} from the NB05 table under Section 5's Baseline models."""
    with open(path) as fh:
        lines = fh.read().splitlines()

    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == '### Baseline models')
    except StopIteration:
        sys.exit(f'No "### Baseline models" heading in {path}')

    rows = {}
    for ln in lines[start:]:
        if not ln.startswith('|'):
            if rows:
                break
            continue
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        if cells[0] in ('run', '---') or set(cells[0]) <= {'-', ':'}:
            continue
        if len(cells) != 7:
            sys.exit(f'Unexpected column count in NB05 table row: {ln}')
        rows[cells[0]] = {
            'model': cells[1],
            'task': cells[2],
            'split': cells[3],
            'macro_f1': float(cells[6]),
        }

    if not rows:
        sys.exit(f'Found the heading but no table rows in {path}')
    return rows


def suffix(filename):
    """'config (3).json' -> '3'; 'config.json' -> '0'."""
    match = re.search(r'\((\d+)\)', filename)
    return match.group(1) if match else '0'


def collect(prefix):
    """Return {suffix: absolute path} for config* or metrics* files."""
    found = {}
    for name in os.listdir(NB05):
        if not name.startswith(prefix) or not name.endswith('.json'):
            continue
        if not re.fullmatch(rf'{prefix}(?: \(\d+\))?\.json', name):
            continue
        key = suffix(name)
        if key in found:
            sys.exit(f'Two {prefix} files share suffix {key}: {found[key]} and {name}')
        found[key] = os.path.join(NB05, name)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='run every check and print the mapping, but do not rename')
    args = parser.parse_args()

    table = read_table(RECORD)
    configs = collect('config')
    metrics = collect('metrics')

    print(f'Read {len(table)} rows from PROJECT_RECORD.md Section 5, NB05 table.')
    print(f'Found {len(configs)} config files and {len(metrics)} metrics files in {NB05}.\n')

    if set(configs) != set(metrics):
        sys.exit(f'Config and metrics suffixes differ: {sorted(configs)} vs {sorted(metrics)}')

    plan = []
    failures = []

    print('Checking each pair against the table.\n')
    for key in sorted(configs, key=int):
        config_path, metrics_path = configs[key], metrics[key]
        config = json.load(open(config_path))
        measured = json.load(open(metrics_path))

        run_id = config['run_id']
        problems = []

        row = table.get(run_id)
        if row is None:
            problems.append(f'run_id {run_id!r} is not a row in the table')
        else:
            observed = float(measured['macro_f1'])
            if abs(observed - row['macro_f1']) > TOLERANCE:
                problems.append(f'macro_f1 {observed:.4f} != table {row["macro_f1"]:.4f}')
            for field in ('model', 'task', 'split'):
                if config[field] != row[field]:
                    problems.append(
                        f'{field} {config[field]!r} != table {row[field]!r}')

        status = 'FAIL' if problems else 'ok'
        table_f1 = f'{row["macro_f1"]:.4f}' if row else '     -'
        print(f'  [{status:4}] {os.path.basename(config_path):18} + '
              f'{os.path.basename(metrics_path):19} -> {run_id:22} '
              f'macro_f1 {float(measured["macro_f1"]):.4f} vs table {table_f1}')
        for problem in problems:
            print(f'           {problem}')

        if problems:
            failures.append(run_id)
        else:
            plan.append((run_id, config_path, metrics_path))

    print()

    if failures:
        sys.exit(f'{len(failures)} pair(s) failed the check. Nothing renamed.')

    unseen = sorted(set(table) - {run_id for run_id, _, _ in plan})
    if unseen:
        sys.exit(f'Table rows with no artifact on disk: {unseen}. Nothing renamed.')

    print('All pairs verified. Mapping:\n')
    for run_id, config_path, metrics_path in plan:
        target = os.path.join(NB05, run_id)
        print(f'  {os.path.basename(config_path):18} -> {run_id}/config.json')
        print(f'  {os.path.basename(metrics_path):18} -> {run_id}/metrics.json')

    # Refuse to clobber anything that is already there.
    collisions = []
    for run_id, _, _ in plan:
        for name in ('config.json', 'metrics.json'):
            target = os.path.join(NB05, run_id, name)
            if os.path.exists(target):
                collisions.append(os.path.relpath(target, REPO))
    if collisions:
        sys.exit(f'\nDestination files already exist: {collisions}. Nothing renamed.')

    if args.dry_run:
        print('\n--dry-run: nothing renamed.')
        return

    print()
    for run_id, config_path, metrics_path in plan:
        folder = os.path.join(NB05, run_id)
        os.makedirs(folder, exist_ok=True)
        os.rename(config_path, os.path.join(folder, 'config.json'))
        os.rename(metrics_path, os.path.join(folder, 'metrics.json'))
        print(f'  moved {run_id}')

    print(f'\nRenamed {len(plan)} pairs. No file was deleted.')


if __name__ == '__main__':
    main()
