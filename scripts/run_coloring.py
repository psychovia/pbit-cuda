"""Compile and repeat coloring functions without restructuring the CUDA program."""
import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys

from prepare_graph import prepare_graph

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = {
    "greedy": "greedy_coloring",
    "largest-first": "largest_degree_first_coloring",
    "random": "random_coloring",
    "smallest-last": "smallest_last_coloring",
    "dsatur": "dsatur_coloring",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=("dense", "sparse"), default="sparse")
    parser.add_argument("--coloring", choices=["all", *FUNCTIONS], default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=Path("function_results.csv"))
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    build = ROOT / "executable"
    build.mkdir(exist_ok=True)
    header = build / f"graph_{args.graph}.h"
    prepare_graph(ROOT / f"pbit-graphs/J_{args.graph}.npy",
                  ROOT / f"pbit-graphs/h_{args.graph}.npy", header)
    fields = ["graph", "method", "repeat", "colors", "sweeps", "coloring_ms",
              "color_transfer_ms", "simulation_ms", "ms_per_sweep", "total_measured_ms"]
    if args.csv.exists() and args.csv.stat().st_size:
        with args.csv.open(newline="") as f:
            if next(csv.reader(f)) != fields:
                raise ValueError("CSV header mismatch; choose a new output file")
    selected = FUNCTIONS if args.coloring == "all" else {args.coloring: FUNCTIONS[args.coloring]}
    executables = {}
    for method, function in selected.items():
        executable = build / f"benchmark_{args.graph}_{method}"
        subprocess.run(["nvcc", "-O3", "-std=c++17", f'-DGRAPH_HEADER="{header}"',
                        f"-DCOLORING_FUNCTION={function}",
                        str(ROOT / "pbit_coloring_benchmark.cu"), "-o", str(executable)], check=True)
        executables[method] = executable
    patterns = {
        "colors": r"Number of colors: (\d+)",
        "sweeps": r"Simulation time \((\d+) sweeps\)",
        "coloring_ms": r"Coloring time: ([\d.]+) ms",
        "color_transfer_ms": r"Memory transfer time: ([\d.]+) ms",
        "simulation_ms": r"Simulation time \(\d+ sweeps\): ([\d.]+) ms",
        "ms_per_sweep": r"Time per sweep: ([\d.]+) ms",
        "total_measured_ms": r"Total measured time: ([\d.]+) ms",
    }
    empty = not args.csv.exists() or args.csv.stat().st_size == 0
    with args.csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if empty:
            writer.writeheader()
        for repeat in range(args.repeats):
            for method, executable in executables.items():
                # The original program intentionally retains fixed seed=1234,
                # all-minus-one initial spins, beta=1, NT=10 and history copies.
                result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
                output = result.stdout
                if result.stderr:
                    raise RuntimeError(result.stderr)
                row = dict(graph=args.graph, method=method, repeat=repeat)
                for field, pattern in patterns.items():
                    match = re.search(pattern, output)
                    if not match:
                        raise RuntimeError(f"Missing benchmark metric: {field}")
                    row[field] = match[1]
                if float(row["simulation_ms"]) <= 0:
                    raise RuntimeError("Invalid GPU timing; check CUDA access")
                writer.writerow(row)
                f.flush()
                print(row, flush=True)
    print(f"Results appended to {args.csv}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
