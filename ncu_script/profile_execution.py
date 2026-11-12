"""Query local Nsight metrics, then optionally profile one explicit command."""
import argparse
import csv
from pathlib import Path
import subprocess

# Exact names are selected only when advertised by this machine's ncu query.
METRICS = {
    'kernel_duration': 'gpu__time_duration.sum',
    'registers_per_thread': 'launch__registers_per_thread',
    'spill_load_sectors': 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
    'spill_store_sectors': 'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
    'achieved_occupancy': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'active_warps': 'smsp__warps_active.avg.per_cycle_active',
    'eligible_warps': 'smsp__warps_eligible.avg.per_cycle_active',
    'fp32': 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active',
    'int': 'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active',
    'sfu': 'sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_active',
    'l1_hit_rate': 'l1tex__t_sector_hit_rate.pct',
    'l2_hit_rate': 'lts__t_sector_hit_rate.pct',
    'global_load_sectors_per_request': 'l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio',
    'l2_throughput': 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram_throughput_bytes': 'dram__bytes.sum.per_second',
    'dram_utilization': 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--ncu', default='ncu')
    p.add_argument('command', nargs=argparse.REMAINDER, help='Optional -- executable arguments')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    query = subprocess.run([a.ncu, '--query-metrics', '--query-metrics-mode', 'all'], capture_output=True, text=True)
    (a.output / 'query.txt').write_text(query.stdout + query.stderr)
    # Metric tokens are emitted with surrounding whitespace or punctuation.
    import re
    available = set(re.findall(r'[A-Za-z][A-Za-z0-9_]*__[A-Za-z0-9_.]+', query.stdout)) if query.returncode == 0 else set()
    selected = {label: metric for label, metric in METRICS.items() if metric in available}
    stalls = sorted(m for m in available if m.startswith('smsp__warp_issue_stalled_') and m.endswith('.per_warp_active.pct'))
    with (a.output / 'availability.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['item', 'metric', 'status'])
        for label, metric in METRICS.items():
            w.writerow([label, metric, 'available' if label in selected else 'unavailable'])
        for metric in stalls:
            w.writerow(['warp_stall_reason', metric, 'available'])
        if not stalls:
            w.writerow(['warp_stall_reasons', '', 'unavailable'])
    if query.returncode:
        raise SystemExit('ncu query failed; see query.txt. All items marked unavailable; no profiling performed.')
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not cmd:
        return
    metrics = list(selected.values()) + stalls
    if not metrics:
        raise SystemExit('No requested metrics available')
    report = a.output / 'profile'
    # Restrict to update kernels and a small sample; excludes RNG and validation replay.
    subprocess.run([a.ncu, '--metrics', ','.join(metrics), '--kernel-name', 'regex:update_pbit',
                    '--launch-count', '10', '--export', str(report), *cmd], check=True)
    # Recent ncu versions export zstd-compressed .ncu-repz by default.
    uncompressed = Path(str(report) + '.ncu-rep')
    compressed = Path(str(report) + '.ncu-repz')
    if not uncompressed.exists() and compressed.exists():
        with uncompressed.open('xb') as f:
            subprocess.run(['zstd', '-d', '-c', str(compressed)], stdout=f, check=True)
    with (a.output / 'profile.csv').open('w') as f:
        subprocess.run([a.ncu, '--import', str(report)+'.ncu-rep', '--csv', '--page', 'raw'], stdout=f, check=True)


if __name__ == '__main__':
    main()
