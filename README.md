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

## File layout

- `cuda/`: CUDA source files, including `pbit_coloring_benchmark.cu`.
- `scripts/`: graph preparation, benchmark runners, and plotting scripts.
- `pbit-graphs/`: existing sparse/dense graph inputs.
- `executable/`: compiled programs and generated graph headers/binary inputs.
- `results/`: final spins and images.

Run the commands below from the repository root. `-I .` lets CUDA source files
find generated headers under `executable/`.

## Use NumPy graphs without changing the CUDA program structure

### Render the 10,000-pbit image circuit

`J_10000.npy` uses zero biases for this experiment. Omit `--h` to create
zeros, and use `--binary-data` to avoid compiling a 100-million-element text
array. NumPy is required for preparation; plotting also requires Matplotlib.

```bash
python scripts/prepare_graph.py --J pbit-graphs/J_10000.npy --binary-data --output executable/graph_10000.h
/usr/local/cuda-13.4/bin/nvcc -O3 -std=c++17 -I . '-DGRAPH_HEADER="executable/graph_10000.h"' -DCOLORING_FUNCTION=largest_degree_first_coloring cuda/pbit_coloring_benchmark.cu -o executable/pbit_ldf_J10000
mkdir -p results
./executable/pbit_ldf_J10000 results/J_10000_final_spins.txt
python scripts/plot_circuit.py results/J_10000_final_spins.txt --shape 100 100 --block-shape 2 2 --output results/J_10000.png
```

This command uses CPU largest-degree-first (LDF) coloring, without cuGraph.
The benchmark keeps its existing beta=1, 1,000 sweeps, seed=1234, and initial
spins=-1. An optional output filename saves only the final spins as text,
one -1/+1 value per line. No sweep history is exported. The plotting script reads
this text (or an existing final `.npy` vector) and saves one final PNG.
The binary input sidecar must remain at the absolute path in the generated header.

For this matrix, placing consecutive groups of four spins into 2×2 pixel blocks
produces a chessboard image; this layout is inferred from connectivity and the
rendered result, not supplied metadata.
Omit `--block-shape` for ordinary row-major pixels. Existing output paths are
overwritten. A single image/run does not establish ground-state correctness.

### Existing graph benchmarks

To compare fixed beta values with LDF, compile and run one program per beta:

```bash
output_dir=results/annealing/0.1-0.4
mkdir -p "$output_dir"

for beta in 0.1 0.2 0.3 0.4; do
  /usr/local/cuda-13.4/bin/nvcc \
    -O3 -std=c++17 -I . \
    '-DGRAPH_HEADER="executable/graph_10000.h"' \
    -DCOLORING_FUNCTION=largest_degree_first_coloring \
    -DBETA=${beta}f \
    cuda/pbit_coloring_benchmark.cu \
    -o executable/pbit_ldf_J10000_b_${beta}

  ./executable/pbit_ldf_J10000_b_${beta} \
    "$output_dir/J_10000_b_${beta}_final_spins.txt" \
    > "$output_dir/J_10000_b_${beta}.log"

  python scripts/plot_circuit.py "$output_dir/J_10000_b_${beta}_final_spins.txt" \
    --shape 100 100 --block-shape 2 2 \
    --output "$output_dir/J_10000_b_${beta}_final.png"
done
```

For the higher-beta group, set `output_dir=results/annealing/0.5-3.0` and use
`for beta in 0.5 1.0 1.5 2.0 2.5 3.0; do` in the same loop.

Each run starts again from all-minus-one spins with seed 1234 and holds beta
constant for 1,000 sweeps. This compares fixed temperatures; it does not ramp
beta during a single simulation. Omitting `-DBETA` keeps the default beta=1.

`cuda/pbit_coloring_benchmark.cu` retains the original kernels, cuGraph branch,
color groups, per-color allocations, timing, history copies and state printing.
Additional CPU algorithms are functions: `largest_degree_first_coloring`,
`random_coloring`, `smallest_last_coloring`, and `dsatur_coloring`.
Change `COLORING_FUNCTION` at the top of the main section, or pass it to nvcc.
The original four-node example remains the default.

Read `.npy` files in a separate Python script (requires NumPy):

```bash
python scripts/prepare_graph.py --J pbit-graphs/J_sparse.npy --h pbit-graphs/h_sparse.npy --output executable/graph_data.h
nvcc -O3 -std=c++17 -I . '-DGRAPH_HEADER="executable/graph_data.h"' -DCOLORING_FUNCTION=dsatur_coloring cuda/pbit_coloring_benchmark.cu -o executable/pbit_coloring_benchmark
./executable/pbit_coloring_benchmark
```

The generated header only supplies `n_pbits`, `h_J`, `h_h`, and `h_m` in place
of the example arrays. It uses float32 row-major J and all-minus-one spins.
Header files are placed under the ignored `executable/` folder.

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
the current `NT`, `beta`, kernels and timing code in `cuda/pbit_coloring_benchmark.cu`.
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
