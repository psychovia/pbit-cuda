"""Explicit validation outside the simulator and outside performance measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    directory = Path('executable')
    base = directory / 'pbit_execution_benchmark'
    binaries = [base, directory / 'pbit_execution_fast'] + [directory / f'pbit_execution_reg{x}' for x in [32, 48, 64, 96]] + [directory / 'pbit_execution_philox', directory / 'pbit_execution_mrg32k3a']
    report = {'binaries': {b.name: digest(b) for b in binaries}, 'experiments': {}, 'checks': []}
    trials = 0

    def execute(binary, data, n, sweeps=12, threads=64, mapping=1, launch='normal', cache='default', history='on'):
        nonlocal trials
        trials += 1
        trace = args.output / 'last_history.bin'
        final = args.output / 'last_final.txt'
        cmd = [str(binary.resolve()), '--data', str(data), '--n', str(n), '--sweeps', str(sweeps),
               '--threads', str(threads), '--rng-threads', '128', '--threads-per-spin', str(mapping),
               '--launch', launch, '--cache', cache, '--history', history, '--final', str(final)]
        if history == 'on':
            cmd += ['--history-output', str(trace)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metrics = json.loads(result.stdout)
        assert metrics['simulation_ms'] > 0 and np.isfinite(metrics['simulation_ms']), result.stdout
        spins = np.loadtxt(final, dtype=np.int32)
        assert np.isin(spins, [-1, 1]).all()
        trajectory = np.fromfile(trace, dtype=np.int32).reshape(sweeps, n).copy() if history == 'on' else None
        return spins, trajectory

    def check(label, left, right):
        assert np.array_equal(left, right), label
        report['checks'].append({'check': label, 'passed': True})

    # Only existing dataset slices are used as small test inputs.
    for dataset in ['sparse', '10000']:
        J = np.load(f'pbit-graphs/J_{dataset}.npy', allow_pickle=False)[:257, :257].astype(np.float32)
        hp = Path(f'pbit-graphs/h_{dataset}.npy')
        h = np.load(hp, allow_pickle=False)[:257].astype(np.float32) if hp.exists() else np.zeros(257, np.float32)
        data = args.output / f'{dataset}_257.bin'
        with data.open('wb') as f:
            J.tofile(f); h.tofile(f)
        n = len(h)
        ref_final, ref_trace = execute(base, data, n)
        for threads in [8, 16, 32, 64, 128, 256, 512, 1024]:
            final, trace = execute(base, data, n, threads=threads)
            check(f'{dataset} thread={threads}', trace, ref_trace)
        for launch in ['normal', 'graph']:
            for history in ['on', 'off']:
                final, trace = execute(base, data, n, launch=launch, history=history)
                check(f'{dataset} launch={launch} history={history} final', final, ref_final)
                if trace is not None: check(f'{dataset} graph trajectory', trace, ref_trace)
        for cap in [32, 48, 64, 96]:
            _, trace = execute(directory / f'pbit_execution_reg{cap}', data, n)
            check(f'{dataset} cap={cap}', trace, ref_trace)
            for threads in [8, 16, 32, 128, 256, 512, 1024]:
                _, trace = execute(directory / f'pbit_execution_reg{cap}', data, n, threads=threads)
                check(f'{dataset} cap={cap} tpb={threads}', trace, ref_trace)
        for cache in ['ldg', 'cg']:
            _, trace = execute(base, data, n, cache=cache)
            check(f'{dataset} cache={cache}', trace, ref_trace)
        for binary in binaries[:2] + binaries[-2:]:
            # Same RNG variant, two independent runs must match exactly.
            _, thread_trace = execute(binary, data, n)
            _, repeated = execute(binary, data, n)
            check(f'{dataset} {binary.name} deterministic', repeated, thread_trace)
            for threads in [32, 64, 128, 256, 512, 1024]:
                _, warp_trace = execute(binary, data, n, threads=threads, mapping=32)
                # These existing datasets have exactly representable integer local fields.
                check(f'{dataset} {binary.name} warp tpb={threads}', warp_trace, thread_trace)
            for threads in [8, 16, 32, 64, 128, 256, 512, 1024]:
                _, trace = execute(binary, data, n, threads=threads)
                check(f'{dataset} {binary.name} thread tpb={threads}', trace, thread_trace)
            _, graph_trace = execute(binary, data, n, launch='graph')
            check(f'{dataset} {binary.name} graph', graph_trace, thread_trace)
        for name in ['pbit_probe', 'pbit_probe_fast', 'pbit_probe_philox', 'pbit_probe_mrg32k3a']:
            out = args.output / f'{dataset}_{name}.bin'
            subprocess.run([str((directory / name).resolve()), str(data), str(n), str(out)], check=True)
            values = np.fromfile(out, dtype=np.float32)
            fields = values[:2 * n].reshape(n, 2)
            m = np.where(np.arange(n) % 2, 1, -1)
            expected = h.astype(np.float64) + J.astype(np.float64) @ m
            np.testing.assert_allclose(fields[:, 0], expected, rtol=2e-5, atol=2e-4)
            np.testing.assert_allclose(fields[:, 1], expected, rtol=2e-5, atol=2e-4)
            probability = (1 + np.tanh(0.1 * expected)) / 2
            tolerance = 6 * np.sqrt(probability * (1 - probability) / 8192) + 0.002
            assert (np.abs(values[2 * n:] - probability) <= tolerance).all(), name
            report['checks'].append({'check': f'{dataset} {name} local fields and 8192-sample Bernoulli probabilities', 'passed': True,
                                     'max_field_absolute_error': float(np.max(np.abs(fields - expected[:, None]))),
                                     'max_probability_error': float(np.max(np.abs(values[2 * n:] - probability)))})

    # One short run on the complete target graph checks row/stride/tail behavior at n=10000.
    J = np.load('pbit-graphs/J_10000.npy', allow_pickle=False).astype(np.float32)
    data = args.output / 'J10000.bin'
    with data.open('wb') as f:
        J.tofile(f); np.zeros(len(J), np.float32).tofile(f)
    ref_final, ref_trace = execute(base, data, len(J), sweeps=3)
    for binary, mapping, launch, cache in [(base, 32, 'normal', 'default'), (base, 1, 'graph', 'default'),
                                          (base, 1, 'normal', 'ldg'), (base, 1, 'normal', 'cg'),
                                          (binaries[1], 1, 'normal', 'default')]:
        _, trace = execute(binary, data, len(J), sweeps=3, mapping=mapping, launch=launch, cache=cache)
        check(f'full J10000 {binary.name} mapping={mapping} launch={launch} cache={cache}', trace, ref_trace)
    for exp in ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3']:
        report['experiments'][exp] = {'passed': True, 'scope': 'external preflight; small existing dataset slices and short full J10000 runs; not per measurement'}
    report['process_runs'] = trials
    report['statistical_test'] = '8192 samples/spin; abs error <= 6*binomial_SE + 0.002; beta=0.1; independent per-pbit RNG indexing'
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(f'PASS: {len(report["checks"])} checks, {trials} simulator runs. Manifest: {args.output / "validation.json"}', flush=True)


if __name__ == '__main__':
    main()
