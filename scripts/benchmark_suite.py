"""Run A1–B3 sequentially; keep each experiment's raw, aggregate and log files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=['10000'])
    p.add_argument('--threads', nargs='+', type=int, default=[8, 16, 32, 64, 128, 256, 512, 1024])
    p.add_argument('--experiments', nargs='+', default=['A1', 'A2', 'A3', 'A4', 'A5', 'B1'])
    p.add_argument('--sweeps', type=int, default=100)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    state_file = a.output / 'status.json'
    if state_file.exists():
        p.error('Suite status already exists; choose a new output directory')
    state = dict(started=datetime.now(timezone.utc).isoformat(), datasets=a.datasets, threads=a.threads,
                 sweeps=a.sweeps, warmup=a.warmup, repeats=a.repeats, validation=str(a.validation.resolve()),
                 completed=[], current=None, status='running', commands={})
    state['build_arch'] = 'sm_120'
    state['source_sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path('cuda/pbit_execution_benchmark.cu'), Path('cuda/pbit_execution_kernels.cuh'), Path('scripts/benchmark_execution.py')]}
    state['dataset_sha256'] = {d: hashlib.sha256((Path('pbit-graphs') / f'J_{d}.npy').read_bytes()).hexdigest() for d in a.datasets}
    gpu = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True)
    (a.output / 'gpu_environment.txt').write_text(gpu.stdout + gpu.stderr)
    state['skipped'] = []
    for exp in a.experiments:
        schedule = a.output / 'skip_experiments.json'
        if schedule.exists() and exp in json.loads(schedule.read_text()):
            state['skipped'].append(exp)
            continue
        state['current'] = exp
        selection = a.output / 'dataset_selection.json'
        datasets = json.loads(selection.read_text()) if selection.exists() else a.datasets
        cmd = [sys.executable, 'scripts/benchmark_execution.py', '--experiment', exp,
               '--datasets', *datasets, '--threads', *map(str, a.threads), '--sweeps', str(a.sweeps),
               '--warmup', str(a.warmup), '--repeats', str(a.repeats), '--validation', str(a.validation),
               '--csv', str(a.output / f'{exp}.csv')]
        state['commands'][exp] = cmd
        state_file.write_text(json.dumps(state, indent=2))
        print(f'Starting {exp}', flush=True)
        with (a.output / f'{exp}.log').open('x') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='failed', returncode=result.returncode)
            state_file.write_text(json.dumps(state, indent=2))
            raise SystemExit(f'{exp} failed; see {a.output / (exp + ".log")}')
        state['completed'].append(exp)
        print(f'Completed {exp}', flush=True)
    state.update(status='complete', current=None, finished=datetime.now(timezone.utc).isoformat())
    state_file.write_text(json.dumps(state, indent=2))
    print(f'All experiments completed: {a.output}', flush=True)


if __name__ == '__main__':
    main()
