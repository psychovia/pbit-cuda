# 100-sweep execution benchmark

Scope: A1, A2, A3, A4, A5, B1. B2/B3 explicitly deferred by user.
Datasets: existing J_10000, sparse, dense, corresponding h files when present; J_10000 uses zero h.
Threads/block: 8,16,32,64,128,256,512,1024. Warp/spin only uses 32..1024.
Each configuration: 100 sweeps, 3 warmups, 10 measurements, fresh process each run.
Fixed beta=1, seed=1234, all initial spins=-1, RNG block size=128, CPU LDF.
Compile: -O3 -std=c++17 -arch=sm_120; A4 adds --use_fast_math, A5 adds --maxrregcount.

Individual A*.csv/B1.csv are raw; *_aggregated.csv excludes warmup and uses sample std.
After completion: all_raw.csv/all_aggregated.csv combine the six experiments without merging independent repeats.
correctness_passed=True means the executable SHA256 matches a binary that passed standalone preflight; it does not mean a replay ran during each timing.
See validation/validation.json for actual checks/scope; generated temporary J/h binaries are removed after preflight.
Graph setup/upload is recorded as graph_setup_ms and excluded from simulation time.
Only one variable is changed within each experiment; this is not a full cross-product of all optimizations.

The earlier 1000-sweep experiment used default compiler targeting/PTX JIT. Do not pool its samples with this native-sm_120, 100-sweep suite.
GPU temperature/clock/background load are not controlled; inspect std and raw records, especially for small differences.
No Nsight profiling runs concurrently with this suite.

Scope change during execution: after A1/A2, user selected only J_10000 for all remaining experiments.
Sparse/dense records in A1/A2 are historical and will be excluded from the final J10000-only export.
The final J10000-only CSV set will be in ../suite_J10000_100s/.
