"""Small, color-aligned Nsight Compute profiles. Invoke through profile_gpu.sh."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

CONFIGS = {'T8': (1, 8), 'T128': (1, 128), 'W64': (32, 64), 'W256': (32, 256)}
SECTIONS = ['LaunchStats', 'Occupancy', 'SchedulerStats', 'WarpStateStats',
            'ComputeWorkloadAnalysis', 'MemoryWorkloadAnalysis', 'InstructionStats', 'SourceCounters']
# field: (unit, scale from ncu base units, candidate names, interpretation)
METRICS = {
    'kernel_duration_us': ('us', 1e6, ['gpu__time_duration.sum'], 'Per-kernel device duration; not ms/sweep.'),
    'registers_per_thread': ('register/thread', 1, ['launch__registers_per_thread'], 'Allocated compiler register count per thread.'),
    'register_spilling_instructions': ('inst', 1, ['sass__inst_executed_register_spilling'], 'Dynamic executed spill instructions, not local traffic inferred as spilling.'),
    'local_load_sectors': ('sector', 1, ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'], 'Local load traffic; may include sources other than spills.'),
    'local_store_sectors': ('sector', 1, ['l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum'], 'Local store traffic; may include sources other than spills.'),
    'theoretical_occupancy_pct': ('%', 1, ['sm__maximum_warps_per_active_cycle_pct'], 'ncu occupancy model for this launch.'),
    'achieved_occupancy_pct': ('%', 1, ['sm__warps_active.avg.pct_of_peak_sustained_active'], 'Active warps relative to maximum over active SM cycles.'),
    'active_warps_per_sm': ('warp', 1, ['sm__warps_active.avg.per_cycle_active'], 'Average active warps per SM active cycle.'),
    'eligible_warps_per_scheduler': ('warp', 1, ['smsp__warps_eligible.avg.per_cycle_active'], 'Average eligible warps per SMSP/scheduler active cycle.'),
    'fp32_pipe_utilization_pct': ('%', 1, ['sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed'], 'FMA aggregate pipeline utilization; physical pipes can be shared, not an FP32-only instruction count.'),
    'int_pipe_utilization_pct': ('%', 1, ['sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed'], 'ALU aggregate pipeline utilization; not an exclusive integer instruction count.'),
    'sfu_utilization_pct': ('%', 1, ['sm__pipe_sfu_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_elapsed'], 'Only a dedicated cycle-utilization metric; XU instruction utilization is exported separately, not substituted.'),
    'xu_instruction_utilization_pct': ('%', 1, ['sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed'], 'XU issued/executed instruction-rate proxy, not pure SFU cycle utilization.'),
    'sm_throughput_pct': ('%', 1, ['sm__throughput.avg.pct_of_peak_sustained_elapsed'], 'ncu SM throughput percentage of peak sustained.'),
    'issued_instructions': ('inst', 1, ['smsp__inst_issued.sum', 'sm__inst_issued.sum'], 'Issued warp instructions aggregated over SMSPs.'),
    'l1_hit_rate_pct': ('%', 1, ['l1tex__t_sector_hit_rate.pct'], 'L1/TEX sector hit rate across measured accesses.'),
    'l1_sectors_per_request': ('sector/request', 1, ['l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio'], 'Global LSU loads only; lower is not automatically better when active lane counts differ.'),
    'global_load_sector_utilization_pct': ('%', 1, ['smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct'], 'Useful requested bytes per global-load sector; coalescing-related, not an overall application efficiency.'),
    'l2_hit_rate_pct': ('%', 1, ['lts__t_sector_hit_rate.pct'], 'L2 sector hit rate.'),
    'l2_throughput_gbs': ('GB/s', 1e-9, ['lts__t_bytes.sum.per_second'], 'Decimal GB/s, directly scaled from measured byte/second.'),
    'dram_read_bytes': ('byte', 1, ['dram__bytes_op_read.sum', 'dram__bytes_read.sum'], 'DRAM read bytes for this kernel.'),
    'dram_write_bytes': ('byte', 1, ['dram__bytes_op_write.sum', 'dram__bytes_write.sum'], 'DRAM write bytes for this kernel.'),
    'dram_throughput_gbs': ('GB/s', 1e-9, ['dram__bytes.sum.per_second'], 'Measured read+write byte rate, decimal GB/s.'),
    'dram_peak_pct': ('%', 1, ['dram__throughput.avg.pct_of_peak_sustained_elapsed'], 'ncu hardware peak-sustained bandwidth denominator, not a guessed product specification.'),
    'long_scoreboard_stall_pct': ('%', 1, ['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'], 'Percent of active warp cycles stalled on long scoreboard; not percent of issued instructions.'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log, env=None):
    with Path(log).open('w') as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        message = Path(log).read_text(errors='replace')
        if 'ERR_NVGPUCTRPERM' in message:
            raise RuntimeError('ERR_NVGPUCTRPERM: enable GPU performance counters in Windows NVIDIA Control Panel; see README. No driver settings changed.')
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}\n{message[-2000:]}')


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def discover(ncu, output):
    output.mkdir()
    commands = {'version': [ncu, '--version'], 'sections': [ncu, '--list-sections'],
                'metrics_base': [ncu, '--query-metrics'],
                'metrics_all': [ncu, '--query-metrics', '--query-metrics-mode', 'all']}
    for category in ['launch', 'occupancy', 'warpsampling', 'source']:
        commands['metrics_' + category] = [ncu, '--query-metrics', '--query-metrics-collection', category, '--query-metrics-mode', 'all']
    for name, command in commands.items():
        run(command, output / (name + '.txt'))
    (output / 'commands.json').write_text(json.dumps(commands, indent=2))


def read_raw(path):
    # ncu --page raw emits a header, a units row, then one row per kernel.
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    rows = list(csv.reader(lines[start:]))
    header, units = rows[:2]
    data = [dict(zip(header, row)) for row in rows[2:] if row and row[0].isdigit()]
    return header, dict(zip(header, units)), data


def export_result(ncu, directory, configuration, metadata, chosen, found):
    report = directory / 'profile.ncu-rep'
    compressed = directory / 'profile.ncu-repz'
    if not report.exists() and compressed.exists():
        with report.open('xb') as f:
            subprocess.run(['zstd', '-d', '-c', str(compressed)], stdout=f, check=True)
    run([ncu, '--import', str(report), '--csv', '--page', 'raw', '--print-units', 'base'], directory / 'ncu_raw.csv')
    header, units, launches = read_raw(directory / 'ncu_raw.csv')
    if len(launches) != 1:
        raise RuntimeError(f'{configuration}: expected one profiled kernel, got {len(launches)}')
    launch = launches[0]
    vector = lambda name: tuple(map(int, re.findall(r'\d+', launch[name])))
    if vector('Block Size') != (metadata['threads_per_block'], 1, 1) or vector('Grid Size') != (metadata['grid_size'], 1, 1):
        raise RuntimeError(f'{configuration}: launch geometry mismatch; selected color could be wrong')
    expected_name = 'update_pbit_execution<32, 0>' if metadata['threads_per_spin'] == 32 else 'update_pbit('
    if expected_name not in launch['Kernel Name']:
        raise RuntimeError(f'Unexpected kernel: {launch["Kernel Name"]}')
    raw = [dict(configuration=configuration, launch_id=launch['ID'], kernel_name=launch['Kernel Name'],
                metric_name=name, value=launch[name], unit=units.get(name, ''))
           for name in header if '__' in name]
    write_csv(directory / 'raw_metrics.csv', list(raw[0]), raw)
    summary = dict(metadata, configuration=configuration, launch_id=launch['ID'], kernel_name=launch['Kernel Name'])
    mapping = []
    for field, (unit, scale, candidates, note) in METRICS.items():
        metric = chosen.get(field, '')
        value = launch.get(metric, '')
        status, reason = 'collected', ''
        try:
            number = float(value.replace(',', ''))
            if not np.isfinite(number): raise ValueError(value)
            # Raw-page base units for duration are seconds and for throughput byte/second.
            if field == 'kernel_duration_us':
                factors = {'ns': 0.001, 'nsecond': 0.001, 'us': 1, 'usecond': 1, 'ms': 1000, 'msecond': 1000, 'second': 1e6, 's': 1e6}
                if units.get(metric) not in factors:
                    raise RuntimeError(f'Unexpected duration unit: {units.get(metric)}')
                scale = factors[units[metric]]
            if field.endswith('_gbs') and units.get(metric) not in ['byte/second', 'byte/s']:
                raise RuntimeError(f'Unexpected throughput unit: {units.get(metric)}')
            maximum = float(launch.get('device__attribute_max_warps_per_multiprocessor', 'nan'))
            if field == 'achieved_occupancy_pct' and not 0 <= number <= 100:
                summary[field] = ''
                status, reason = 'invalid_measurement', f'ncu reported {number}%; outside physical occupancy range [0,100]. Raw value retained; no clipping or estimate.'
            elif field == 'active_warps_per_sm' and np.isfinite(maximum) and number > maximum:
                summary[field] = ''
                status, reason = 'invalid_measurement', f'ncu reported {number} warps/SM, exceeding queried device maximum {maximum}. Raw value retained.'
            else:
                summary[field] = number * scale
        except ValueError:
            summary[field] = ''
            status = 'unavailable'
            reason = 'No supported queried candidate' if not metric else 'Metric absent, unsupported, or non-numeric in report'
        mapping.append(dict(configuration=configuration, summary_field=field, summary_unit=unit,
                            raw_metric_name=metric, raw_unit=units.get(metric, ''), conversion_factor=scale,
                            status=status, reason=reason, interpretation=note,
                            discovery_sources=';'.join(found.get(metric, []))))
    write_csv(directory / 'summary.csv', list(summary), [summary])
    write_csv(directory / 'metric_mapping.csv', list(mapping[0]), mapping)
    stalls = [r for r in raw if ('stall' in r['metric_name'] and r['value'] not in ['', 'n/a'])]
    write_csv(directory / 'warp_stalls.csv', list(raw[0]), stalls)
    return summary, mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--configs', nargs='+', choices=CONFIGS, default=['T8', 'W64'])
    p.add_argument('--append', action='store_true')
    p.add_argument('--metric-set', choices=['full', 'occupancy'], default='full', help='Occupancy-only diagnosis reduces unrelated replay passes')
    p.add_argument('--discovery', type=Path, help='Use previously saved discovery output from this machine')
    p.add_argument('--ncu', default='ncu')
    p.add_argument('--sweeps', type=int, default=100)
    p.add_argument('--sweep-index', type=int, default=2, help='Zero-based; profile third sweep by default')
    p.add_argument('--color-index', type=int, help='Default: largest LDF color group, ties use smallest index')
    a = p.parse_args()
    if not 0 <= a.sweep_index < a.sweeps: p.error('sweep-index must be within sweeps')
    a.output.mkdir(parents=True, exist_ok=a.append)
    discovery = a.discovery or a.output / 'discovery'
    if a.discovery is None and not discovery.exists(): discover(a.ncu, discovery)
    available, found = set(), {}
    for path in discovery.glob('metrics_*.txt'):
        for metric in set(re.findall(r'\b[A-Za-z][A-Za-z0-9_]*__[A-Za-z0-9_.]+', path.read_text())):
            available.add(metric); found.setdefault(metric, []).append(path.name)
    chosen = {field: next((m for m in spec[2] if m in available), '') for field, spec in METRICS.items()}
    section_text = (discovery / 'sections.txt').read_text()
    sections = [s for s in SECTIONS if re.search(r'^' + s + r'\s', section_text, re.M)]
    if not {'Occupancy', 'SchedulerStats', 'MemoryWorkloadAnalysis'}.issubset(sections):
        raise RuntimeError('Required sections unavailable; inspect discovery/sections.txt')
    extra_stalls = sorted(m for m in available if m.startswith('smsp__warp_issue_stalled_') and m.endswith('_per_warp_active.pct'))
    metrics = sorted(set(m for m in chosen.values() if m) | set(extra_stalls))
    if a.metric_set == 'occupancy':
        sections = [s for s in sections if s in ['LaunchStats', 'Occupancy', 'SchedulerStats']]
        fields = ['kernel_duration_us', 'registers_per_thread', 'theoretical_occupancy_pct',
                  'achieved_occupancy_pct', 'active_warps_per_sm', 'eligible_warps_per_scheduler']
        counters = [f'{base}.{suffix}' for base in ['sm__warps_active', 'sm__cycles_active', 'sm__cycles_elapsed']
                    for suffix in ['sum', 'avg', 'min', 'max']]
        counters += ['smsp__warps_active.avg.per_cycle_active', 'smsp__cycles_active.avg']
        metrics = sorted({chosen[f] for f in fields if chosen[f]} | {m for m in counters if m in available})
    run(['make', 'execution-benchmark'], a.output / 'build.log')
    binary = Path('executable/pbit_execution_benchmark').resolve()
    helper = Path('executable/pbit_color_metadata').resolve()
    if not helper.exists() or helper.stat().st_mtime < max(Path('cuda/test_benchmark/pbit_color_metadata.cu').stat().st_mtime, Path('cuda/pbit_coloring.cuh').stat().st_mtime):
        run(['/usr/local/cuda-13.4/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_120', '-lineinfo', '-I.',
             'cuda/test_benchmark/pbit_color_metadata.cu', '-o', str(helper)], a.output / 'metadata_build.log')
    run(['/usr/local/cuda-13.4/bin/cuobjdump', '--dump-resource-usage', str(binary)], a.output / 'compiler_resources.txt')
    run(['/usr/local/cuda-13.4/bin/cuobjdump', '--dump-elf-symbols', str(binary)], a.output / 'kernel_symbols.txt')
    # Record actual WSL driver/PTX loader resolution in this very process.
    driver = ctypes.CDLL('libcuda.so.1'); init_status = driver.cuInit(0)
    ctypes.CDLL('libnvidia-ptxjitcompiler.so.1')
    libraries = [line for line in Path('/proc/self/maps').read_text().splitlines() if 'libcuda' in line or 'ptxjit' in line or 'gpucomp' in line]
    (a.output / 'environment.json').write_text(json.dumps(dict(ld_library_path=os.environ.get('LD_LIBRARY_PATH'),
          ld_preload=os.environ.get('LD_PRELOAD', ''), cuInit_status=init_status, loaded_libraries=libraries,
          ncu=a.ncu, discovery=str(discovery.resolve()), binary_sha256=sha(binary)), indent=2))
    if init_status != 0: raise RuntimeError(f'WSL CUDA initialization failed: {init_status}')
    with tempfile.TemporaryDirectory(prefix='pbit-profile-') as temp:
        J = np.asarray(np.load('pbit-graphs/J_10000.npy', allow_pickle=False), dtype=np.float32, order='C')
        n = len(J); data = Path(temp) / 'J10000.bin'
        with data.open('wb') as f: J.tofile(f); np.zeros(n, np.float32).tofile(f)
        color_result = subprocess.run([str(helper), str(data), str(n)], capture_output=True, text=True, check=True)
        colors = json.loads(color_result.stdout)
        (a.output / 'colors.json').write_text(json.dumps(colors, indent=2))
        color = colors['colors'][a.color_index] if a.color_index is not None else max(colors['colors'], key=lambda c: c['nodes_in_color'])
        skip = a.sweep_index * colors['num_colors'] + color['color_index']
        for config in a.configs:
            directory = a.output / config
            tps, tpb = CONFIGS[config]
            expected_kernel = '_Z11update_pbitPiPKfS1_P17curandStateXORWOWPKiiif' if tps == 1 else '_Z21update_pbit_executionILi32ELi0EEvPiPKfS2_P17curandStateXORWOWPKiiif'
            if expected_kernel not in (a.output / 'kernel_symbols.txt').read_text():
                raise RuntimeError('Expected update symbol not present in executable')
            metadata = dict(dataset='J_10000.npy', mapping='thread' if tps == 1 else 'warp', threads_per_spin=tps,
                            threads_per_block=tpb, color_index=color['color_index'], nodes_in_color=color['nodes_in_color'],
                            grid_size=(color['nodes_in_color'] + tpb // tps - 1) // (tpb // tps),
                            block_size=tpb, sweep_index=a.sweep_index, sweeps=a.sweeps, coloring_algorithm='ldf',
                            beta=1.0, seed=1234, rng_algorithm='xorwow', rng_threads_per_block=128,
                            initial_state='all_-1', history_mode='on', math_mode='default', register_cap='',
                            launch_mode='normal', cache_policy='default', replay_mode='kernel', cache_control='all')
            # Appending must never silently mix colors or baseline settings.
            for prior in a.output.glob('*/invocation.json'):
                previous = json.loads(prior.read_text())
                common = ['dataset', 'color_index', 'nodes_in_color', 'sweep_index', 'sweeps', 'coloring_algorithm',
                          'beta', 'seed', 'rng_algorithm', 'rng_threads_per_block', 'initial_state', 'history_mode',
                          'math_mode', 'register_cap', 'launch_mode', 'cache_policy', 'replay_mode', 'cache_control']
                if any(previous['metadata'][key] != metadata[key] for key in common) or previous['binary_sha256'] != sha(binary):
                    raise RuntimeError('Append settings/binary differ from an existing profile; use a new output directory')
            directory.mkdir()
            command = [str(binary), '--data', str(data), '--n', str(n), '--threads', str(tpb), '--threads-per-spin', str(tps),
                       '--sweeps', str(a.sweeps), '--rng-threads', '128', '--coloring', 'ldf', '--beta', '1',
                       '--seed', '1234', '--history', 'on', '--launch', 'normal', '--cache', 'default']
            reference_trace, profiled_trace = Path(temp) / 'reference.bin', Path(temp) / 'profiled.bin'
            run(command + ['--history-output', str(reference_trace)], directory / 'unprofiled_reference.log')
            ncu = [a.ncu, '--target-processes', 'application-only', '--kernel-name-base', 'mangled',
                   '--kernel-name', expected_kernel, '--launch-skip', str(skip), '--launch-count', '1',
                   '--replay-mode', 'kernel', '--cache-control', 'all', '--clock-control', 'none',
                   '--import-source', 'yes', '--metrics', ','.join(metrics)]
            for section in sections: ncu += ['--section', section]
            ncu += ['--export', str(directory / 'profile'), *command, '--history-output', str(profiled_trace)]
            (directory / 'invocation.json').write_text(json.dumps(dict(configuration=config, metadata=metadata,
                kernel_filter=expected_kernel, update_launch_skip=skip, metric_set=a.metric_set, ncu_command=ncu,
                binary_sha256=sha(binary), dataset_sha256=sha(Path('pbit-graphs/J_10000.npy'))), indent=2))
            print(f'Profiling {config}: color={color["color_index"]}, nodes={color["nodes_in_color"]}, grid={metadata["grid_size"]}, block={tpb}, skip={skip}', flush=True)
            run(ncu, directory / 'profile.log')
            if sha(reference_trace) != sha(profiled_trace):
                raise RuntimeError(f'{config}: profiler replay altered full simulation trajectory')
            (directory / 'replay_validation.json').write_text(json.dumps(dict(trajectory_matches_unprofiled=True,
                full_history_sha256=sha(reference_trace), sweeps=a.sweeps), indent=2))
            summary, mapping = export_result(a.ncu, directory, config, metadata, chosen, found)
            required = ['kernel_duration_us', 'achieved_occupancy_pct', 'eligible_warps_per_scheduler',
                        'l1_hit_rate_pct', 'l2_hit_rate_pct', 'dram_throughput_gbs']
            if a.metric_set == 'occupancy':
                required = ['kernel_duration_us', 'achieved_occupancy_pct', 'active_warps_per_sm', 'eligible_warps_per_scheduler']
            missing = [field for field in required if summary[field] == '']
            status = dict(configuration=config, success=True, metrics_complete=not missing, trajectory_matches_unprofiled=True,
                          full_history_sha256=sha(reference_trace), missing_required_metrics=missing,
                          unavailable_fields=[r['summary_field'] for r in mapping if r['status'] != 'collected'])
            (directory / 'status.json').write_text(json.dumps(status, indent=2))
            if missing and config in ['T8', 'W64']: raise RuntimeError(f'{config}: smoke-test metrics missing: {missing}')
            print(f'{config}: profile successful, full history matches; missing/invalid summary fields: {missing}', flush=True)
    combine(a.output)


def combine(output):
    for filename in ['summary.csv', 'raw_metrics.csv', 'metric_mapping.csv', 'warp_stalls.csv']:
        all_rows, fields = [], None
        for config in CONFIGS:
            p = output / config / filename
            if p.exists():
                with p.open(newline='') as f:
                    reader = csv.DictReader(f); fields = reader.fieldnames; all_rows.extend(reader)
        if fields: write_csv(output / filename, fields, all_rows)


if __name__ == '__main__':
    main()
