# J10000 color-aligned GPU profiling

Use `scripts/profile_gpu.sh`. This is independent of A/B benchmark timing. It never writes A/B CSVs and does not modify simulator kernels or update ordering.

## Reproduce the staged run

Run from the repository root. Python needs NumPy; the wrapper chooses the existing pbit-cugraph environment, or set `PBIT_PYTHON`.

```bash
# A NEW output directory. Queries actual version, sections, metrics and metric collections.
bash scripts/profile_gpu.sh --configs T8 W64 --output results/profiling/my_color_test

# Continue only after both status.json files report success and required metrics exist.
bash scripts/profile_gpu.sh --configs T128 W256 --append \
  --output results/profiling/my_color_test
```

Default invocation profiles only T8/W64. No large profiling sweep is scheduled. Each configuration profiles exactly one update kernel using `--launch-count 1`; other launches execute normally. No RNG-init kernel is profiled.

The completed run is `results/profiling/J10000_color/`. Its machine discovery is in `results/profiling/discovery/`; use `--discovery results/profiling/discovery` to reuse that saved discovery on this same machine. Without this option the script runs `ncu --version`, `--list-sections`, `--query-metrics`, full metric names, and separate launch/occupancy/warpsampling/source collections into the output's discovery directory.

## Fixed settings and selected launch

All four runs use existing `J_10000.npy`, zero h, CPU Largest Degree First, beta=1, XORWOW seed=1234, RNG initialization block=128, all initial spins=-1, history on, default math, no register cap, default J loads, normal launch, and 100 sweeps. Build remains `-O3 -std=c++17 -arch=sm_120 -lineinfo`.

A separate CPU-only metadata helper, `cuda/pbit_color_metadata.cu`, reuses the original coloring functions. It does not run a simulation. All four LDF groups contain 2500 nodes. The default selects the largest group (smallest color index on ties), hence **color 0**. Colors are zero-based; `colors.json` records all groups.

The selected kernel is color 0 in **sweep index 2** (third sweep). After filtering only the exact mangled update symbol, `--launch-skip 8` skips two complete sweeps of four colors. `--launch-count 1` captures the next update. Previous colors/sweeps execute normally; none are removed or reordered.

| Configuration | Threads/spin | Block (x,y,z) | Grid (x,y,z) |
|---|---:|---|---|
| T8 | 1 | (8,1,1) | (313,1,1) |
| T128 | 1 | (128,1,1) | (20,1,1) |
| W64 | 32 | (64,1,1) | (1250,1,1) |
| W256 | 32 | (256,1,1) | (313,1,1) |

T uses original `update_pbit(...)`; W uses `update_pbit_execution<32,0>(...)`. The script verifies each raw report has exactly one launch and that its kernel name, actual block and grid match the selected configuration. No colors are averaged together.

## Replay, caches and correctness

Runs are isolated profiling processes, using **kernel replay**. ncu saves/restores modified device memory between passes, including spin/RNG writes. The script also compares the complete 100-sweep history against an independent unprofiled run with identical parameters. All four comparisons passed. Each report required 23 passes; these are profiler replays, not 23 normal sweeps.

Use `cache-control=all` for consistent cold caches across replay passes, and `clock-control=none` so the script does not change GPU clock settings. Therefore the reported kernel duration is a cold-cache profiling sample; clocks/background display load may still vary. It is not a normal simulation ms/sweep measurement. The executable's JSON simulation times inside `profile.log` include profiler overhead and must be ignored for benchmark analysis.

T8's first export used an option not accepted on the raw page; it was re-exported from the saved report, without rerunning its GPU profile. Its history comparison had passed before that export error, but its temporary history digest was not retained. This is explicitly recorded in T8/status.json. The other three retain full-history SHA256 evidence.

See NVIDIA's [kernel replay and cache-control documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) for these replay semantics.

## Outputs and metric meanings

Each configuration directory contains:

- `profile.ncu-repz`: original compressed profiler report; `profile.ncu-rep`: uncompressed report for GUI. Both retained.
- `ncu_raw.csv`: verbatim ncu raw-page export, including native names and units row.
- `raw_metrics.csv`: long-format raw metric/value/unit, with configuration, native launch ID and kernel name.
- `summary.csv`: one row for this kernel/configuration, units explicit in column names.
- `metric_mapping.csv`: summary field to exact native name, raw unit, output unit, conversion factor, discovery source and availability reason.
- `warp_stalls.csv`: all collected stall metrics in long format, preserving native names and denominators.
- `invocation.json`: exact filter, launch skip, command, settings and binary/dataset hashes.
- `status.json`, replay evidence and original process logs.

At the top level, `summary.csv`, `raw_metrics.csv`, `metric_mapping.csv`, and `warp_stalls.csv` combine all successful configurations. `compiler_resources.txt` provides static register/stack/local-memory usage; actual dynamic spill instructions and local load/store sectors are separately measured.

Key distinctions:

- Kernel duration is **microseconds**; conversion from ncu's actual time unit is recorded. It is not per-sweep time.
- Registers/thread and theoretical occupancy are ncu launch/occupancy attributes. Achieved occupancy and active warps/SM use active cycles. Eligible warps use scheduler/SMSP active cycles.
- FP32 field uses the supported aggregate FMA cycle-utilization metric; INT field uses aggregate ALU cycles. Blackwell physical pipelines can be shared, so these are not exclusive instruction-type counts. Exact interpretations are in metric_mapping.csv.
- **Pure SFU cycle utilization is unavailable** in this device/version's queried metrics, so `sfu_utilization_pct` is empty. `xu_instruction_utilization_pct` is a separately measured XU instruction-rate proxy; it is not substituted for SFU cycles or estimated from instruction counts.
- L1/TEX hit rate covers measured sectors; L1 sectors/request is restricted to global LSU loads. Global-load sector utilization measures useful requested bytes per sector. Active lane counts differ between mappings, so sectors/request alone is not a universal coalescing score.
- L2/DRAM GB/s uses decimal 1e9 bytes/sec, scaled only from the corresponding measured byte/sec metrics. DRAM peak percent uses ncu's hardware peak-sustained metric, not a product-spec estimate.
- Long scoreboard uses `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`: percent of active warp cycles waiting on long-latency scoreboard dependencies. Other stall denominators remain explicit in raw native names; do not mix cycles/instruction and percent of active cycles.
- Zero measured spill instructions is a collected zero. Local-memory sectors are not assumed to be spills. Unknown or unsupported metrics remain empty with reasons.

## Inspect a kernel in Nsight Compute

Open each `profile.ncu-rep` in the GUI and select its only result. Check LaunchStats: color 0's expected grid/block above. Occupancy, SchedulerStats, WarpStateStats, ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, InstructionStats and SourceCounters are included. Source import is enabled; use the Source page to correlate CUDA/PTX/SASS with the optimized `-lineinfo` build.

Offline re-export:

```bash
ncu --import results/profiling/J10000_color/T8/profile.ncu-rep \
  --csv --page raw --print-units base > /tmp/T8_raw.csv
```

## WSL library handling and permissions

The wrapper sets libraries **only for its process and children**. It puts `/usr/lib/wsl/lib` first, then the NVIDIA WSL driver directory whose `libcuda_loader.so` matches the injected WSL loader, then CUDA 13.4 runtime libraries. It does not delete Linux NVIDIA libraries, reinstall CUDA, replace drivers, or change system loader configuration. `environment.json` records actual loaded libcuda and PTX JIT library paths. On this machine both resolve to the matching `nvami.inf_amd64_b259952749ea1a39` WSL driver package, not `/usr/lib/x86_64-linux-gnu` Linux NVIDIA libraries.

If automatic matching fails, set `PBIT_WSL_DRIVER_DIR` to the independently verified WSL driver directory. Do not point it at Linux driver libraries. Existing `LD_PRELOAD` is recorded and not silently overwritten.

If `ERR_NVGPUCTRPERM` occurs, the script stops. On Windows, open NVIDIA Control Panel with administrator privileges; Desktop → Enable Developer Settings, then Developer → Manage GPU Performance Counters → allow access for all users, and Apply. This is a manual host setting; the script never changes it. See [NVIDIA's official permission instructions](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters).

## Observed W256 counter-quality warning

In this capture ncu returned achieved occupancy 105.93% and active warps/SM 50.85 while its queried device maximum is 48 warps/SM. These are physically out of range. Both remain untouched in the original report and raw CSV, but summary fields are empty and metric_mapping marks them `invalid_measurement`. Capture/replay verification succeeded; this does not make the invalid counters usable. No root cause, replacement value, or clipping is assumed. SFU unavailability is a separate issue.
