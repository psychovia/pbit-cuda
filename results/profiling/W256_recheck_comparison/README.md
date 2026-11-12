# W256 isolated recheck

Same J10000, LDF, beta=1, seed=1234, initial=-1, history on, default math, XORWOW, 100 sweeps. Selected color 0, 2500 nodes, third sweep, grid=313, block=256, threads/spin=32. Same executable SHA256 as original.

| Run | Replay passes | Average achieved occupancy | Average active warps/SM |
|---|---:|---:|---:|
| Original full collection | 23 | 105.93% (invalid) | 50.85 (above device limit 48) |
| Isolated occupancy collection | 3 | 95.96% | 46.06 |
| Isolated full collection | 23 | 99.64% | 47.83 |

Both rechecks compared the complete simulation history with isolated unprofiled runs and matched. No clocks/driver/system libraries were changed. Original reports and CSVs were not overwritten.

`comparison.csv` preserves native reported averages/max-related metrics with explicit raw labels. `summary_rechecked.csv` is a NEW four-configuration convenience export replacing only W256 with its full recheck, with source/quality columns; original four-configuration summary remains unchanged.

The original out-of-range averages are present in ncu's raw report, so the CSV exporter did not introduce them. Average active warps/SM and occupancy share the same active-warp accounting; they are not two independent measurements. 50.85/48*100 is approximately 105.94%, allowing for displayed rounding.

NVIDIA documents that out-of-range metrics can result from counter/replay/work-distribution inconsistencies and recommends reducing metric count. The occupancy-only run reduced passes from 23 to 3 and returned in-range averages. However, the full 23-pass isolated rerun also returned in-range averages, and raw .max occupancy-related metrics still exceed normal bounds in both rechecks. Therefore this does NOT establish replay count as the sole cause, nor prove a particular Blackwell/WSL/driver bug. No value is clipped, estimated, or treated as evidence of physically exceeding GPU capacity. The .max fields are preserved as diagnostic profiler outputs, not interpreted as a reliable physical peak occupancy.

Reference: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-and-precision

Reproduce in NEW output directories:

```bash
bash scripts/profile_gpu.sh --configs W256 --metric-set occupancy \
  --output results/profiling/another_occupancy_check \
  --discovery results/profiling/discovery
bash scripts/profile_gpu.sh --configs W256 \
  --output results/profiling/another_full_check \
  --discovery results/profiling/discovery
```

Reports/raw CSVs: `../W256_occupancy_recheck/W256/` and `../W256_full_recheck/W256/`.
