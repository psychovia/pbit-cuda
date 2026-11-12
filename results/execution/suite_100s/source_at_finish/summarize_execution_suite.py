"""Audit completed experiment CSVs and combine them for pivot-table analysis."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

CONFIG = 'experiment dataset coloring_algorithm num_pbits num_colors threads_per_block rng_threads_per_block threads_per_spin launch_mode history_mode math_mode rng_algorithm register_cap cache_policy sweeps beta seed'.split()
METRICS = ['coloring_ms', 'transfer_ms', 'simulation_ms', 'time_per_sweep_ms', 'graph_setup_ms']


def read(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--dataset', help='Optional dataset name, e.g. 10000')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    output = a.output or a.directory
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((a.directory / 'status.json').read_text())
    skipped = json.loads((a.directory / 'skip_experiments.json').read_text()) if (a.directory / 'skip_experiments.json').exists() else []
    assert state['status'] == 'complete', state
    all_raw, all_stats, best, audit = [], [], [], []
    for exp in ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3']:
        if exp in skipped or exp not in state['completed']:
            continue
        fields, raw = read(a.directory / f'{exp}.csv')
        stat_fields, stats = read(a.directory / f'{exp}_aggregated.csv')
        if a.dataset:
            raw = [r for r in raw if Path(r['dataset']).stem == 'J_' + a.dataset]
            stats = [r for r in stats if Path(r['dataset']).stem == 'J_' + a.dataset]
        groups = {}
        for row in raw:
            key = tuple(row[k] for k in CONFIG)
            groups.setdefault(key, []).append(row)
            assert row['correctness_passed'] == 'True'
            assert int(row['sweeps']) == state['sweeps']
            for m in METRICS:
                assert math.isfinite(float(row[m])) and float(row[m]) >= 0
            assert float(row['simulation_ms']) > 0
            assert abs(float(row['simulation_ms']) / int(row['sweeps']) - float(row['time_per_sweep_ms'])) < 1e-5
        for rows in groups.values():
            warmups = [r for r in rows if r['phase'] == 'warmup']
            measurements = [r for r in rows if r['phase'] == 'measurement']
            assert len(warmups) == state['warmup']
            assert len(measurements) == state['repeats']
            assert sorted(int(r['run_id']) for r in measurements) == list(range(1, state['repeats'] + 1))
        assert len(stats) == len(groups) * len(METRICS)
        for row in stats:
            values = [float(r[row['metric']]) for r in groups[tuple(row[k] for k in CONFIG)] if r['phase'] == 'measurement']
            for name, expected in [('mean', statistics.mean(values)), ('median', statistics.median(values)),
                                   ('std', statistics.stdev(values) if len(values) > 1 else 0),
                                   ('min', min(values)), ('max', max(values))]:
                assert math.isclose(float(row[name]), expected, rel_tol=1e-10, abs_tol=1e-10), (exp, name)
        # Check expected matrix size, including explicitly unsupported partial-warp blocks.
        blocks = state['threads']
        factor = {'A1': 1, 'A2': 2, 'A3': 2, 'A4': 2, 'A5': 5, 'B1': 2, 'B2': 3, 'B3': 3}[exp]
        count = len(blocks) * factor
        if exp == 'B1':
            count = len(blocks) + sum(t >= 32 and t % 32 == 0 for t in blocks)
        assert len(groups) == (1 if a.dataset else len(state['datasets'])) * count, (exp, len(groups))
        for dataset in sorted({r['dataset'] for r in stats}):
            candidates = [r for r in stats if r['dataset'] == dataset and r['metric'] == 'time_per_sweep_ms']
            best.append(min(candidates, key=lambda r: float(r['mean'])))
        if output != a.directory:
            write(output / f'{exp}.csv', fields, raw)
            write(output / f'{exp}_aggregated.csv', stat_fields, stats)
        all_raw.extend(raw); all_stats.extend(stats)
        audit.append({'experiment': exp, 'configurations': len(groups), 'raw_rows': len(raw), 'aggregate_rows': len(stats), 'passed': True})
    write(output / 'all_raw.csv', fields, all_raw)
    write(output / 'all_aggregated.csv', stat_fields, all_stats)
    write(output / 'best_configurations.csv', stat_fields, best)
    (output / 'csv_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    for row in best:
        print(row['experiment'], Path(row['dataset']).stem, 'tpb='+row['threads_per_block'],
              'tps='+row['threads_per_spin'], row['history_mode'], row['launch_mode'], row['math_mode'],
              'cap='+row['register_cap'], f"{float(row['mean']):.6f} ms/sweep")


if __name__ == '__main__':
    main()
