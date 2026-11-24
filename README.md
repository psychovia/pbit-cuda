PBIT:
    - memory / compute bound
    - tanh / sigmoid

RNG:
    - xorwow
    - philox
    - MRG32k3a


COLORING:
    color provided j
    (1) number of color generated using implementation
    (2) time taken for color
        let user decide - over xx time then maybe use a suboptimal one
        accept user provided function / color

    run p circuit
    (annealing 0.5 - 3)
    (1) for each coloring algo
    (2) time dependent on num of color
    (3) solution / correctness of final state

    coloring algo / libraries:
    - sequantial greedy
    - cuGraph
    TODO:
    - largest degree first
    - DSATUR
    - random order
    - Jones-Plassman / Luby

## Files

- `cuda/`: shared CUDA code, included by every program:
  - `pbit_rng.cuh`: RNG state type (`EXEC_RNG`) and `init_rng`.
  - `pbit_kernels.cuh`: `update_pbit`, `update_pbit_execution<p>`, `update_pbit_multiwarp<p>`.
  - `pbit_coloring.cuh`: CPU colorings (greedy, ldf, random, smallest_last, dsatur), `color_graph`, and `cugraph_coloring` (with `USE_CUGRAPH`).
  - `pbit_simulation.cuh`: shared benchmark flow (options, J/h loading, GPU buffers, timed sweeps, JSON/output).
  - `pbit_launch.cuh`: wave-minimizing choice of threads/spin and threads/block.
- `cuda/test_benchmark/`: the programs (`main`), e.g. `pbit_coloring_benchmark.cu`, `pbit_execution_benchmark.cu`, `pbit_wave_benchmark.cu`.
- `scripts/`: graph preparation, plotting, and benchmark runners.
- `pbit-graphs/`: coupling matrices `J` and bias vectors `h`.
- `executable/`: compiled programs and generated graph inputs.
- `results/annealing/0.1-0.4/`, `0.5-3.0/`: final spins and images by beta range.

## Run J_10000: LDF, beta = 1.5

Run from the repository root:

```bash
conda activate pbit-cugraph
# Install once if needed:
python -m pip install numpy matplotlib

# Prepare J and zero biases. Repeat only when inputs change.
python scripts/prepare_graph.py \
  --J pbit-graphs/J_10000.npy \
  --binary-data \
  --output executable/graph_10000.h

# Compile with CPU LDF coloring and fixed beta = 1.5.
/usr/local/cuda-13.4/bin/nvcc \
  -O3 -std=c++17 -I . \
  '-DGRAPH_HEADER="executable/graph_10000.h"' \
  -DCOLORING_FUNCTION=largest_degree_first_coloring \
  -DBETA=1.5f \
  cuda/test_benchmark/pbit_coloring_benchmark.cu \
  -o executable/pbit_ldf_J10000_b_1.5

# Save final spins: one -1/+1 value per line.
./executable/pbit_ldf_J10000_b_1.5 \
  results/annealing/0.5-3.0/J_10000_b_1.5_final_spins.txt

# Save the image and display it. Omit --show to only save.
python scripts/plot_circuit.py \
  results/annealing/0.5-3.0/J_10000_b_1.5_final_spins.txt \
  --width 50 --height 50 \
  --output results/annealing/0.5-3.0/J_10000_b_1.5_final.png \
  --show
```

Keep the generated `.h` and `.bin` under `executable/`. Create the output folder
first if using a new path. Existing TXT and PNG files are overwritten.

## State and parameters

- **Initial state:** all 10,000 spins are **-1**, not random. `prepare_graph.py`
  generates `graph_m(n_pbits, -1)`; CUDA copies this array from `h_m` to `d_m`.
- **Random seed:** 1234; each launch resets both the initial state and RNG.
- **Sweeps:** `NT = 1000` in `main()`. Each sweep updates every pbit once.
- **Beta:** `-DBETA=1.5f` fixes beta for the entire run; default is 1.0.
  Current experiments compare fixed betas, rather than ramping beta in one run.
- **Inputs:** `J_10000.npy` contains only the coupling matrix. Omitting `--h`
  generates zero biases; use `--h path/to/h.npy` to supply a bias vector.
- **Updates:** tanh with -1/+1 spins. In the plot, -1 is black and +1 is white.

The mentor's `indices_to_grid` places four consecutive spins in each 2×2 cell.
`--width 50 --height 50` means 50×50 cells, producing 100×100 pixels.
The plotting script reads final TXT/NPY data; it does not run CUDA or compute energy.

## Choose coloring

Replace `-DCOLORING_FUNCTION=...` when compiling:

| Algorithm | Function |
|---|---|
| LDF | `largest_degree_first_coloring` |
| Greedy | `greedy_coloring` |
| Random | `random_coloring` |
| Smallest last | `smallest_last_coloring` |
| DSATUR | `dsatur_coloring` |

Recompile after changing coloring, beta, or NT. The command above does not use cuGraph.

## Other graph benchmarks

```bash
make executable/pbit_coloring_ldf_sparse
./executable/pbit_coloring_ldf_sparse

# Repeat an existing executable and save timing CSVs.
python scripts/benchmark_runs.py executable/pbit_coloring_ldf_sparse --repeats 10
```

## GPU execution benchmark: phase 1 (A1 / A2)

The old simulator and runner are kept. `cuda/test_benchmark/pbit_execution_benchmark.cu`
is a new entry point that reuses the existing `init_rng`, `update_pbit` and CPU coloring
functions. Dense row-major float32 J/h and the colored-Gibbs order (one launch per
color, in sequence) are unchanged. cuGraph is not used yet.

```bash
conda activate pbit-cugraph  # Python needs numpy
make execution-benchmark

# Small smoke test: 4 configurations, 1 warmup + 2 measured runs each
python scripts/benchmark_execution.py --experiment A2 --datasets sparse \
  --threads 32 64 --sweeps 2 --warmup 1 --repeats 2 \
  --csv results/execution/smoke.csv
```

Full experiment commands (run when you choose; default is 3 warmup and 10 measured
runs per configuration):

```bash
python scripts/benchmark_execution.py --experiment A1 \
  --datasets sparse dense 10000 --threads 32 64 128 256 512 1024 \
  --sweeps 1000 --coloring ldf --csv results/execution/A1.csv
python scripts/benchmark_execution.py --experiment A2 \
  --datasets sparse dense --threads 32 64 128 256 512 1024 \
  --sweeps 1000 --coloring ldf --csv results/execution/A2.csv
```

`--datasets` accepts names (mapping to `pbit-graphs/J_<name>.npy`) or a path to a J `.npy`.
`h_<name>.npy` in the same directory is used automatically; if it does not exist, biases
are zero, matching the existing preparation behavior. For a single dataset, `--h path/to/h.npy`
can be given. The runner converts inputs to the same float32 "J followed by h" file the
existing binary loader reads, so the graph representation is unchanged and no per-dataset
compile is needed.
Also supported: `--coloring ldf|greedy|random|smallest_last|dsatur`, `--rng-threads 128`,
`--beta 1.0`, `--seed 1234`. Initial spins are always all -1.
A1 records history by default (`--history off` disables it); A2 always measures both on and off.
Every run starts a new process and reloads the data, recomputes the coloring and resets the RNG.

Timing definitions:

- `coloring_ms`: time spent in the CPU coloring function, excluding input file reads.
- `transfer_ms`: CPU wall time of the H2D copies of J/h/initial spins/color nodes plus the
  synchronization that completes them; excludes allocation, RNG initialization and the final D2H.
- `simulation_ms`: CUDA events on one stream bracket all sweeps and are read after synchronizing
  on the end event. With history on, it includes the per-sweep D2D history copy; with history off,
  nothing is recorded. It includes real gaps between launch submissions. RNG initialization
  finishes and is synchronized before the start event.
- `time_per_sweep_ms = simulation_ms / sweeps` is the main performance metric.

Each actual run (including warmup) writes one raw CSV row; `phase=warmup|measurement`, and
`run_id` starts at 1 within each phase. RNG block size, beta and seed are also recorded.
`*_aggregated.csv` aggregates measurement runs only, grouped by full configuration and metric,
with mean, median, sample std (0 for a single run), min and max. The runner refuses to overwrite
an existing output path.
In Google Sheets, filter on phase or metric and then pivot; do not include warmup in statistics.

The current entry point follows the sequential structure of the original CUDA files and does not
yet include defensive checks or correctness replay.
Without `--validation`, the CSV column `correctness_passed` is left empty; see below for
standalone pre-validation. With history off, no history memory is allocated.

### Standalone Nsight Compute profiling

```bash
# Only query what this machine supports; does not run a simulation. Directory must not exist.
python ncu_script/profile_execution.py --output results/profiling/query

# Example: first prepare the same binary sidecar with the existing tool
python scripts/prepare_graph.py --J pbit-graphs/J_sparse.npy \
  --h pbit-graphs/h_sparse.npy --binary-data --output executable/profile_sparse.h
# n for sparse is 1188; for other datasets use the n printed by preparation
python ncu_script/profile_execution.py --output results/profiling/sparse64 -- \
  ./executable/pbit_execution_benchmark --data executable/profile_sparse.bin \
  --n 1188 --threads 64 --rng-threads 128 --sweeps 2 --coloring ldf --history off
```

The script first saves `query.txt` and `availability.csv`, and selects only metrics that the
local ncu explicitly reports; unmatched items are marked unavailable. Local-memory sectors are a
proxy for spill traffic and must not be read as the compiler's spill count. Profiling saves a
`.ncu-rep` and an exported raw `profile.csv`, capturing at most the first 10 update kernels.
Profiling perturbs timing, so its results should not go into the regular benchmark CSVs.

The later A3–B3 implementation and standalone validation are described below. Carveout, multiple
streams, two-warps-per-spin and fusion remain future work.

Validation records and CSVs from the earlier version with correctness replay are in
`results/execution/phase1_smoke/`. They are historical and do not mean the current simplified
entry point validates on every run.
If a newer ncu writes `.ncu-repz` by default, the profiling script decompresses it to `.ncu-rep`
with `zstd` before exporting CSV.

## A1–A5 / B1 extensions and short experiments

`cuda/pbit_kernels.cuh` (formerly `pbit_execution_kernels.cuh`) keeps the original one-thread
kernel and adds warp-per-spin: 32 lanes accumulate the J row over j = lane, lane+32, ...,
and after a shuffle reduction only lane 0 updates that p-bit's RNG state and spin.
Threads/block must be a multiple of 32; warp configurations with 8/16 are skipped.
All colors still run in sequence on a single stream, with no concurrent streams or kernel fusion.

```bash
# Local RTX 5060: native sm_120, so default sm_75/PTX JIT does not affect compile-time experiments
make -j2 execution-variants execution-probes
# For other GPUs override EXEC_ARCH, e.g. EXEC_ARCH=sm_89

# Standalone validation, not added to the simulator's per-run path
python scripts/validate_execution.py --output results/execution/preflight

# Current short experiments: B2/B3 not run yet
python scripts/benchmark_suite.py --experiments A1 A2 A3 A4 A5 B1 \
  --datasets 10000 sparse dense --threads 8 16 32 64 128 256 512 1024 \
  --sweeps 100 --warmup 3 --repeats 10 \
  --validation results/execution/preflight/validation.json \
  --output results/execution/short_suite
```

`benchmark_execution.py --experiment A3|A4|A5|B1` can also be run on its own, with the same
other arguments as A1/A2. Each experiment changes one variable at a time; everything else stays
at the default: history on, normal launch, one thread per spin, XORWOW, default math, no register
cap, default J load. This is not a cross product of all optimizations.

- A3: normal / graph launch. The full NT sweeps are captured; capture/instantiate/upload are not
  counted in simulation time, and `graph_setup_ms` is recorded separately. Both modes use the same
  single stream and original color order.
- A4: default / `--use_fast_math`, each as its own executable.
- A5: no cap / `--maxrregcount=32/48/64/96`. The cap is an upper limit, not actual usage; if it is
  above the default usage, the same kernel may be generated. See the resource report for actual
  registers/local memory.
- B1: 1 / 32 threads per spin. The grid is sized by spins/block = threads/block ÷ threads/spin.
- B2/B3: XORWOW/Philox/MRG32k3a and J default/`__ldg`/`__ldcg` variants are provided, but formal
  experiments are on hold for now. Later, use `--experiment B2|B3` with `--rngs` / `--caches`.

`--validation` points to a standalone validation manifest; only after checking the binary's SHA256
does it write `correctness_passed=True` to the CSV. This means the preflight passed, not that every
timed run did a correctness replay. Without a manifest the column is empty. Different RNGs are not
required to produce the same trajectory.
Preflight checks threads/warps, Graph, cache, register cap and reproducibility on submatrices of
the existing datasets and on short trajectories of the full J10000, and compares the local field
against CPU float64. Each spin is sampled 8192 times to check the Bernoulli probability
(tolerance 6× standard error + 0.002). This does not prove numerical equivalence on arbitrary graphs.

The suite saves `<experiment>.csv`, `<experiment>_aggregated.csv` and `.log` per experiment, plus
`status.json`, the GPU environment and input hashes. `skip_experiments.json` can list experiments
that have not started, e.g. `["B2", "B3"]`. Profiling should still be run separately.
The old 1000-sweep CSVs and these 100-sweep sm_120 results are stored separately and must not be
merged as repeated measurements of the same configuration.

For this run the scope was narrowed to J10000 only, 100 sweeps, experiments A1–A5/B1; B2/B3 are
deferred. Future reruns should use `--datasets 10000 --experiments A1 A2 A3 A4 A5 B1`.

Completed J10000 CSVs from this run: `results/execution/suite_J10000_100s/`.
`all_raw.csv` holds 1430 actual run records (including warmup) and `all_aggregated.csv` holds
550 rows of statistics; A1–A5/B1 are done, B2/B3 deferred. Both merged CSVs can be used directly
for pivoting and analysis.

## Same-color Nsight profiling: T8 / T128 / W64 / W256

Use the independent `scripts/profile_gpu.sh`; default smoke test captures one kernel each for T8/W64.
Then append T128/W256 only after checking smoke success. The completed reports, raw CSVs and summary
are under `results/profiling/J10000_color/`. See [GPU profiling instructions](docs/gpu_profiling.md)
for exact commands, color/launch selection, metric meanings, replay checks, WSL libraries and unavailable fields.
The original A/B timing CSVs are unchanged.


### Choosing CUDA launch parameters by minimum waves

`pbit_wave_benchmark` reuses the execution benchmark's graph input, CPU coloring and simulation
flow. By default it uses the actual color class sizes `n_c` to search for one `B` (block size)
and `p` (threads/spin) shared across the whole sweep.

```bash
make wave-benchmark
./executable/pbit_wave_benchmark --data executable/graph_10000.bin \
  --n 10000 --sweeps 100 --history off
```

Following equations (1)–(3) in [pbit_cuda.pdf](pbit_cuda.pdf), each color class needs
`G_c = ceil(n_c / (B/p))` blocks and `W_c = ceil(G_c / (S*R))` waves. `S` is the GPU's SM
count, and `R` is the number of blocks of that kernel, at that block size, that can be resident
on each SM at once, queried via the CUDA occupancy API.

The search covers `p = 1, 2, 4, …, 1024`. With `--whole-warps-only on`, only 1, 2, 4, 8, 16, 32
warps/spin are searched. Above one warp, the kernel sums per-warp partial sums in shared memory,
and the model's tie-breaking extrapolates `1/p`. Auto-searched block sizes are powers of 2, must
satisfy `B % p == 0` and allow at least one resident block. When `--threads B` is given manually,
other block sizes that satisfy CUDA limits and the divisibility condition can also be used.
First `sum(W_c)` is minimized. Ties are broken by minimizing
`sum(min(n_c, ceil(G_c/S)*(B/p)) / p)` per equations (5)–(6), then by preferring the smaller `B`.
These are model predictions; the actual sweep time is `time_per_sweep_ms` in the output.

The output JSON includes `block_size`, `threads_per_spin`, `sm_count`, `resident_blocks_per_sm`,
`waves_per_sweep`, and `color_launches` in execution order (spins, blocks and waves for each
color class). Use `--threads B` or `--threads-per-spin p` to fix one of them while the other is
still searched automatically.

Running `scripts/benchmark_execution.py --experiment A1 --auto-threads
--threads-per-spin 0 --executable executable/pbit_wave_benchmark` writes the automatic
configuration and predicted wave count within the whole-warp range to CSV.
