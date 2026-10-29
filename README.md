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

## Use NumPy graphs without changing the CUDA program structure

`pbit_coloring_benchmark.cu` retains the original kernels, cuGraph branch,
color groups, per-color allocations, timing, history copies and state printing.
Additional CPU algorithms are functions: `largest_degree_first_coloring`,
`random_coloring`, `smallest_last_coloring`, and `dsatur_coloring`.
Change `COLORING_FUNCTION` at the top of the main section, or pass it to nvcc.
The original four-node example remains the default.

Read `.npy` files in a separate Python script (requires NumPy):

```bash
python scripts/prepare_graph.py --J pbit-graphs/J_sparse.npy --h pbit-graphs/h_sparse.npy --output executable/graph_data.h
nvcc -O3 -std=c++17 '-DGRAPH_HEADER="executable/graph_data.h"' -DCOLORING_FUNCTION=dsatur_coloring pbit_coloring_benchmark.cu -o executable/pbit_coloring_benchmark
./executable/pbit_coloring_benchmark
```

The generated header only supplies `n_pbits`, `h_J`, `h_h`, and `h_m` in place
of the example arrays. It uses float32 row-major J and all-minus-one spins.
The script validates dimensions, finite values, float32 range, symmetry and
zero diagonal. Header files are placed under the ignored `executable/` folder.

An external runner can compile each function and record repeated measurements:

```bash
python scripts/run_coloring.py --graph sparse --coloring all --repeats 5 --csv sparse_function_results.csv
python scripts/run_coloring.py --graph dense --coloring dsatur --repeats 5 --csv dense_function_results.csv
```

Each selected method needs its own compilation. The runner parses the original
printed timing into CSV; it does not add CLI parsing, CSV writing or NumPy
reading to the CUDA source. Set `NT` and `beta` in `main()` as before. The
original fixed seed and initial state are retained across repetitions, and
history copying is included in simulation time. There is no added warmup.
`Memory transfer time` still measures color arrays only. Short 10-sweep runs
are useful for checking execution, but increase NT for steadier measurements.
The original CUDA error-handling limitations and kernel behavior are retained.

The existing `sparse_results.csv`, `dense_results.csv`, and
`scripts/summarize_coloring.py` belong to the previous, now-reverted benchmark
implementation; their schema/timing definitions differ from the external
runner above. They are retained as historical measurements, not new results.

### Inspect performance for each coloring function

```bash
python scripts/run_coloring.py --graph sparse --coloring all --repeats 5 --csv sparse_function_results.csv
python scripts/summarize_coloring.py sparse_function_results.csv
```

The summary supports both the current runner and the earlier benchmark CSV.
It reports median/min/max across repetitions for each graph, sweep count and
method. Keep different beta values, input revisions and builds in separate CSVs;
the current runner does not record those settings.

`coloring_ms` uses `std::chrono::steady_clock` around the coloring call (and
method print). For the added functions it includes adjacency construction and
grouping. `simulation_ms` uses CUDA events around all NT sweeps and their history
copies. `ms_per_sweep = simulation_ms / NT`: a sweep updates every p-bit once,
through all color groups. It is not the latency of a single p-bit update.
Amortized time per individual p-bit update is `simulation_ms / (NT * n_pbits)`;
it includes launch/history overhead and is not an isolated thread timing.

### Separate cuGraph executables

```bash
conda activate pbit-cugraph
make sparse          # J_sparse.npy + h_sparse.npy -> executable/sparse
make dense           # J_dense.npy + h_dense.npy -> executable/dense
# Or build both:
make all

./executable/sparse
./executable/dense
```

These Makefile targets use cuGraph and CUDA 13.4. They generate separate
`executable/graph_sparse.h` and `executable/graph_dense.h`; neither overwrites
the other graph's data. The CUDA source is shared. Changing the inputs or source
causes the corresponding executable to rebuild on the next `make`.

### All twelve algorithm/graph executables

`make -j2 all` builds `executable/pbit_coloring_<algo>_<graph>` for every
combination of `greedy`, `ldf`, `random`, `smallest_last`, `dsatur`, `cugraph`
and `sparse`, `dense`. Here `ldf` means largest degree first.

```bash
./executable/pbit_coloring_dsatur_sparse
./executable/pbit_coloring_cugraph_dense
```

Each executable embeds its own graph and algorithm at compile time. All share
the current `NT`, `beta`, kernels and timing code in `pbit_coloring_benchmark.cu`.
After changing the source or input arrays, run `make -j2 all` to rebuild.
A single version can be rebuilt using its path as the make target, for example
`make executable/pbit_coloring_dsatur_sparse`.

### Average repeated runs of an already compiled executable

```bash
python scripts/benchmark_runs.py executable/pbit_coloring_dsatur_sparse --repeats 10
```

This does not compile or modify the CUDA program. It runs the chosen executable
sequentially and reports mean, sample standard deviation, median, minimum and
maximum for each timing metric. Raw runs and a separate summary CSV are saved
under `results/` with timestamped filenames. No third-party Python modules are
required. Use `--csv results/my_runs.csv` to choose a new filename; existing
files are never overwritten or mixed with new runs. `--warmup 1` optionally
runs one entire process first and discards it. Every process still initializes
its own CUDA context; this is not in-process kernel warmup. Repetitions retain
the executable's compiled seed, inputs, NT and beta. Reported simulation time
is the CUDA program's timing, not Python process startup time. Incomplete raw
CSVs can remain after failures, but no successful summary is produced.

`benchmark_runs.py` allows color counts to vary across repetitions. Each raw
row records its actual count; the summary includes a `colors` metric with
mean, sample standard deviation, median, minimum and maximum. The summary's
`colors` metadata column is the mean color count across runs. Color counts
are unitless; timing metrics remain in milliseconds. The script does not
change random seeds: fixed-seed deterministic algorithms may return the same
coloring every run.
