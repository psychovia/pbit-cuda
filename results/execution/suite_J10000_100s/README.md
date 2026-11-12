# J10000 A1–A5 and B1, 100 sweeps

Completed: A1, A2, A3, A4, A5, B1. B2/B3 were deferred and have no measurement records.
Only existing J_10000.npy, zero h, CPU LDF (4 colors), beta=1, seed=1234, initial spins=-1.
Update threads/block: 8,16,32,64,128,256,512,1024. Warp/spin uses complete warps, so 8/16 are skipped for that mapping.
Each configuration has 3 warmups and 10 measurements, 100 sweeps each, RNG threads/block=128.
110 configurations; 1430 raw rows = 330 warmups + 1100 measurements.
All aggregate statistics independently recomputed from raw measurement rows.

Upload all_aggregated.csv for pivot tables: filter metric=time_per_sweep_ms; mean/median/std/min/max are milliseconds per sweep.
Upload all_raw.csv to inspect individual runs or recompute statistics; filter phase=measurement for formal results.
Individual experiment CSVs are also retained. best_configurations.csv selects the lowest sample mean within each experiment; small differences may be noise.

All variants were preflight-validated outside benchmark timing. correctness_passed=True means the binary SHA256 matches the validated build; timed runs do not execute replay.
See validation.json for the actual test scope and tolerances. This is not a proof of numerical equivalence on arbitrary graphs.
Native sm_120 builds; -O3 -std=c++17; A4 additionally --use_fast_math, A5 additionally --maxrregcount.
Registers and local-memory resource reports are in kernel_resources.csv; requested cap is an upper bound, not actual usage.
Graph capture, instantiation and upload are excluded from simulation_ms and separately recorded as graph_setup_ms.
One variable is changed within each experiment; fastest A3/A4/A5 settings are not combined with warp mapping in B1.

A1/A2 were originally run on three datasets before the user narrowed scope. This export filters their existing raw rows to J10000 without altering values; A3 onward only ran J10000.
Original logs and source snapshots are in ../suite_100s/. Previous 1000-sweep/PTX-JIT results must not be pooled with this native-sm_120/100-sweep suite.
GPU temperature, clocks and other system load were not controlled; interpret small timing differences cautiously.
