"""Repeat an existing benchmark executable and save raw runs plus statistics."""
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
PATTERNS = {
    "colors": r"Number of colors: (\d+)",
    "sweeps": r"Simulation time \((\d+) sweeps\)",
    "coloring_ms": rf"Coloring time: {NUMBER} ms",
    "color_transfer_ms": rf"Memory transfer time: {NUMBER} ms",
    "simulation_ms": rf"Simulation time \(\d+ sweeps\): {NUMBER} ms",
    "ms_per_sweep": rf"Time per sweep: {NUMBER} ms",
    "total_measured_ms": rf"Total measured time: {NUMBER} ms",
}
METRICS = ("colors",) + tuple(
    name for name in PATTERNS if name.endswith("_ms") or name == "ms_per_sweep"
)


def parse_output(output):
    row = {}
    for name, pattern in PATTERNS.items():
        match = re.search(pattern, output)
        if not match:
            raise ValueError(f"Missing metric: {name}")
        value = int(match[1]) if name in ("colors", "sweeps") else float(match[1])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid metric: {name}={value}")
        row[name] = value
    if row["colors"] == 0 or row["sweeps"] == 0 or row["simulation_ms"] == 0:
        raise ValueError("Invalid colors/sweeps/GPU timing; check CUDA access")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of complete process runs to discard (default: 0)")
    parser.add_argument("--csv", type=Path,
                        help="Raw output CSV; must not exist. A summary CSV is also created.")
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup nonnegative")
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"Executable not found: {executable}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.csv or Path("results") / f"{executable.name}_{stamp}.csv"
    summary = output.with_name(output.stem + "_summary.csv")
    if output.resolve() == summary.resolve() or output.exists() or summary.exists():
        parser.error("Output already exists; choose a new --csv filename")
    output.parent.mkdir(parents=True, exist_ok=True)

    def run():
        result = subprocess.run([str(executable)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Process failed ({result.returncode}):\n{result.stderr}\n{result.stdout}")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return parse_output(result.stdout)

    for index in range(args.warmup):
        run()
        print(f"Warmup {index + 1}/{args.warmup} discarded", flush=True)

    rows = []
    fields = ["executable", "run", *PATTERNS]
    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index in range(args.repeats):
            row = run()
            if rows and row["sweeps"] != rows[0]["sweeps"]:
                raise ValueError("Sweep count changed between runs")
            rows.append(row)
            writer.writerow(dict(executable=str(executable), run=index + 1, **row))
            f.flush()
            print(f"Run {index + 1}/{args.repeats}: colors={row['colors']}, "
                  f"coloring={row['coloring_ms']:.6f} ms, "
                  f"simulation={row['simulation_ms']:.6f} ms, "
                  f"sweep={row['ms_per_sweep']:.6f} ms", flush=True)

    print(f"\n{executable.name}: {len(rows)} runs, "
          f"sweeps={rows[0]['sweeps']}")
    print(f"{'metric':<22} {'mean':>12} {'stddev':>12} {'median':>12} {'min':>12} {'max':>12}")
    print("colors is a count; all timing metrics are in milliseconds.")
    mean_colors = statistics.mean(row["colors"] for row in rows)
    with summary.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["executable", "runs", "colors", "sweeps",
                                              "metric", "mean", "stddev", "median", "min", "max"])
        writer.writeheader()
        for metric in METRICS:
            values = [row[metric] for row in rows]
            stats = dict(mean=statistics.mean(values),
                         stddev=statistics.stdev(values) if len(values) > 1 else 0,
                         median=statistics.median(values), min=min(values), max=max(values))
            writer.writerow(dict(executable=str(executable), runs=len(rows),
                                 colors=mean_colors, sweeps=rows[0]["sweeps"],
                                 metric=metric, **stats))
            print(f"{metric:<22}" + "".join(f"{value:>12.6f}" for value in stats.values()))
    print(f"\nRaw runs: {output}\nSummary:  {summary}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"Error: {error}")
