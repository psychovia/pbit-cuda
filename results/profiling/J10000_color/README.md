# J10000 same-color profiles

All four captures succeeded, exactly one update kernel each, color 0 (2500 nodes), sweep index 2 (third sweep). All full-history comparisons against isolated unprofiled runs passed. Every capture used 23 kernel-replay passes.

Fixed settings: existing J_10000, zero h, LDF, beta=1, XORWOW seed=1234, initial=-1, RNG block=128, history on, default math, no register cap, 100 sweeps, normal launch. Native sm_120, -O3 -lineinfo. Cache-control=all, clock-control=none.

| Configuration | Threads/spin | Block | Grid | Capture status |
|---|---:|---:|---:|---|
| T8 | 1 | 8 | 313 | success |
| T128 | 1 | 128 | 20 | success |
| W64 | 32 | 64 | 1250 | success |
| W256 | 32 | 256 | 313 | success, occupancy-counter warning |

- `summary.csv`: one row per configuration. Time is microseconds/kernel, throughput decimal GB/s.
- `raw_metrics.csv`: long-format native metric/value/unit, kernel name and profiler launch ID.
- `metric_mapping.csv`: exact metric names, units, conversion factors, meaning, unavailable/invalid reasons.
- `warp_stalls.csv`: collected native stall metrics; keep their different denominators distinct.
- Each config has the original `.ncu-repz`, uncompressed `.ncu-rep`, verbatim `ncu_raw.csv`, command and validation/status logs.

Pure SFU cycle utilization is unavailable in the queried metrics and left blank. XU instruction utilization is separately measured and explicitly not substituted for pure SFU utilization.

W256 ncu reported achieved occupancy=105.93% and active warps/SM=50.85 vs queried device maximum 48. These two values remain verbatim in raw CSV/report but are blank in summary, with reasons. No correction or estimate was applied. Other memory/scheduler metrics were collected. Root cause is not established; this run does not justify an occupancy conclusion for W256.

T8's CLI export option was corrected offline after successful capture/history comparison; its transient history digest was not retained. No extra T8 GPU profile was run. Other configurations retain history SHA256.

Reproduce in a NEW directory, then append only after smoke checks:

```bash
bash scripts/profile_gpu.sh --configs T8 W64 --output results/profiling/new_run
bash scripts/profile_gpu.sh --configs T128 W256 --append --output results/profiling/new_run
```

WSL library resolution is recorded in `environment.json`. No system libraries or drivers were changed. Original A/B benchmark CSV hashes were verified unchanged. Do not use the executable's profiler-instrumented JSON time/sweep from profile.log as ordinary benchmark data.

Full instructions and metric definitions: [docs/gpu_profiling.md](../../../docs/gpu_profiling.md).
