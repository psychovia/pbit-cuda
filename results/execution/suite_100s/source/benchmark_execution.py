"""A1–B3 execution benchmarks: one raw row per process, including warmups."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import numpy as np

CONFIG = 'experiment dataset coloring_algorithm num_pbits num_colors threads_per_block rng_threads_per_block threads_per_spin launch_mode history_mode math_mode rng_algorithm register_cap cache_policy sweeps beta seed'.split()
METRICS = 'coloring_ms transfer_ms simulation_ms time_per_sweep_ms graph_setup_ms'.split()
FIELDS = CONFIG + ['run_id', 'phase'] + METRICS + ['correctness_passed']


def file_hash(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_variants(args):
    base = (args.history, args.launch, args.threads_per_spin, args.cache, args.executable)
    directory = args.executable.parent
    if args.experiment == 'A2':
        return [(h, *base[1:]) for h in ['on', 'off']]
    if args.experiment == 'A3':
        return [(base[0], mode, *base[2:]) for mode in ['normal', 'graph']]
    if args.experiment == 'A4':
        return [base, (*base[:4], directory / 'pbit_execution_fast')]
    if args.experiment == 'A5':
        return [base] + [(*base[:4], directory / f'pbit_execution_reg{cap}') for cap in [32, 48, 64, 96]]
    if args.experiment == 'B1':
        return [(base[0], base[1], mapping, base[3], base[4]) for mapping in [1, 32]]
    if args.experiment == 'B2':
        binaries = {'xorwow': args.executable, 'philox': directory / 'pbit_execution_philox',
                    'mrg32k3a': directory / 'pbit_execution_mrg32k3a'}
        return [(*base[:4], binaries[rng]) for rng in args.rngs]
    if args.experiment == 'B3':
        return [(*base[:3], cache, base[4]) for cache in args.caches]
    return [base]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3'], default='A1')
    p.add_argument('--datasets', nargs='+', default=['sparse', 'dense'], help='Existing J_<name>.npy or J .npy paths')
    p.add_argument('--h', type=Path, help='Explicit bias file; only with one dataset')
    p.add_argument('--threads', nargs='+', type=int, default=[32, 64, 128, 256, 512, 1024])
    p.add_argument('--rng-threads', type=int, default=128)
    p.add_argument('--history', choices=['on', 'off'], default='on', help='A1 only; A2 always measures both')
    p.add_argument('--sweeps', type=int, default=1000)
    p.add_argument('--coloring', choices=['ldf', 'greedy', 'random', 'smallest_last', 'dsatur'], default='ldf')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--executable', type=Path, default=Path('executable/pbit_execution_benchmark'))
    p.add_argument('--launch', choices=['normal', 'graph'], default='normal')
    p.add_argument('--threads-per-spin', type=int, choices=[1, 32], default=1)
    p.add_argument('--cache', choices=['default', 'ldg', 'cg'], default='default')
    p.add_argument('--rngs', nargs='+', choices=['xorwow', 'philox', 'mrg32k3a'], default=['xorwow', 'philox', 'mrg32k3a'])
    p.add_argument('--caches', nargs='+', choices=['default', 'ldg', 'cg'], default=['default', 'ldg', 'cg'])
    p.add_argument('--validation', type=Path, help='Optional external validation manifest')
    p.add_argument('--csv', type=Path, required=True)
    args = p.parse_args()
    schedule = args.csv.parent / 'skip_experiments.json'
    if schedule.exists() and args.experiment in json.loads(schedule.read_text()):
        print(f'Skipped {args.experiment}: excluded from the suite schedule', flush=True)
        return
    validation = json.loads(args.validation.read_text()) if args.validation else None
    if args.sweeps <= 0 or args.repeats <= 0 or args.warmup < 0 or args.seed < 0 or not np.isfinite(args.beta):
        p.error('Invalid sweeps/repeats/warmup/seed/beta')
    if args.h and len(args.datasets) != 1:
        p.error('--h requires exactly one dataset')
    if any(t <= 0 or t > 1024 for t in args.threads + [args.rng_threads]):
        p.error('Block sizes must be in 1..1024')
    summary = args.csv.with_name(args.csv.stem + '_aggregated.csv')
    if args.csv.exists() or summary.exists():
        p.error('Output exists; choose a new --csv path')
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pbit-benchmark-') as tmp, args.csv.open('x', newline='') as raw, summary.open('x', newline='') as agg:
        writer = csv.DictWriter(raw, fieldnames=FIELDS)
        writer.writeheader()
        sw = csv.DictWriter(agg, fieldnames=CONFIG + ['runs', 'metric', 'mean', 'median', 'std', 'min', 'max', 'correctness_passed'])
        sw.writeheader()
        for dataset in dict.fromkeys(args.datasets):
            jp = Path(dataset) if dataset.endswith('.npy') else Path('pbit-graphs') / f'J_{dataset}.npy'
            hp = args.h or jp.with_name('h_' + jp.name.removeprefix('J_'))
            J = np.asarray(np.load(jp, allow_pickle=False), dtype=np.float32, order='C')
            if J.ndim != 2 or J.shape[0] != J.shape[1] or not J.shape[0]:
                raise ValueError(f'{jp}: J must be nonempty and square')
            h = np.asarray(np.load(hp, allow_pickle=False), dtype=np.float32) if hp.exists() else np.zeros(J.shape[0], dtype=np.float32)
            if args.h and not args.h.exists():
                raise ValueError(f'Missing explicit h: {args.h}')
            if h.shape != (J.shape[0],) or not np.isfinite(J).all() or not np.isfinite(h).all():
                raise ValueError('Invalid J/h shape or nonfinite values')
            data = Path(tmp) / 'graph.bin'
            with data.open('wb') as f:
                J.tofile(f)
                h.tofile(f)
            print(f'Dataset {jp.resolve()}, h={hp.resolve() if hp.exists() else "zeros"}', flush=True)
            variants = experiment_variants(args)
            for threads, variant in itertools.product(dict.fromkeys(args.threads), variants):
                history, launch, mapping, cache, binary = variant
                if mapping == 32 and (threads < 32 or threads % 32):
                    print(f'Skip tpb={threads}, threads/spin=32: complete warps required', flush=True)
                    continue
                rows = []
                config = dict(experiment=args.experiment, dataset=str(jp.resolve()), coloring_algorithm=args.coloring,
                              threads_per_block=threads, rng_threads_per_block=args.rng_threads, threads_per_spin=mapping,
                              launch_mode=launch, history_mode=history, math_mode='default', rng_algorithm='xorwow',
                              register_cap='', cache_policy=cache, sweeps=args.sweeps, beta=args.beta, seed=args.seed)
                for index in range(args.warmup + args.repeats):
                    cmd = [str(binary.resolve()), '--data', str(data), '--n', str(len(h)), '--sweeps', str(args.sweeps),
                           '--threads', str(threads), '--rng-threads', str(args.rng_threads), '--coloring', args.coloring,
                           '--history', history, '--beta', str(args.beta), '--seed', str(args.seed),
                           '--launch', launch, '--threads-per-spin', str(mapping), '--cache', cache]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode:
                        raise RuntimeError(f"Benchmark failed ({result.returncode}): {result.stderr}\n{result.stdout}")
                    metrics = json.loads(result.stdout)
                    if metrics['simulation_ms'] <= 0 or any(not np.isfinite(metrics[k]) or metrics[k] < 0 for k in METRICS):
                        raise ValueError(f'Invalid timing: {metrics}')
                    config.update(num_pbits=metrics['num_pbits'], num_colors=metrics['num_colors'],
                                  math_mode=metrics['math_mode'], rng_algorithm=metrics['rng_algorithm'],
                                  register_cap=metrics['register_cap'] or '')
                    passed = ''
                    if validation:
                        entry = validation['experiments'][args.experiment]
                        if entry['passed'] and validation['binaries'].get(binary.name) == file_hash(binary):
                            passed = True
                    phase = 'warmup' if index < args.warmup else 'measurement'
                    row = dict(config, **{k: metrics[k] for k in METRICS}, correctness_passed=passed,
                               run_id=index + 1 if phase == 'warmup' else index - args.warmup + 1, phase=phase)
                    writer.writerow(row)
                    raw.flush()
                    if phase == 'measurement':
                        rows.append(row)
                    print(f'{dataset} tpb={threads} history={history} launch={launch} mapping={mapping} binary={binary.name} cache={cache} {phase} {row["run_id"]}: {metrics["time_per_sweep_ms"]:.6f} ms/sweep', flush=True)
                for metric in METRICS:
                    values = [r[metric] for r in rows]
                    sw.writerow(dict(config, runs=len(rows), metric=metric, mean=statistics.mean(values), median=statistics.median(values),
                                     std=statistics.stdev(values) if len(values) > 1 else 0, min=min(values), max=max(values), correctness_passed=all(r['correctness_passed'] is True for r in rows) if validation else ''))
                agg.flush()
    print(f'Raw: {args.csv}\nAggregated: {summary}')


if __name__ == '__main__':
    main()
