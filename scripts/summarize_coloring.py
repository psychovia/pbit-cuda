"""Summarize repeated measurements from either benchmark CSV format."""
import argparse
import csv
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV produced by run_coloring.py or the earlier benchmark")
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if {"graph", "method", "sweeps"} <= fields:
            keys = ("graph", "sweeps", "method")
            metrics = ("colors", "coloring_ms", "color_transfer_ms", "simulation_ms",
                       "ms_per_sweep", "total_measured_ms")
        elif {"J_path", "h_path", "mode", "sweeps", "beta", "method"} <= fields:
            keys = ("J_path", "h_path", "mode", "sweeps", "beta", "method")
            metrics = ("colors", "coloring_ms", "setup_ms", "simulation_ms",
                       "ms_per_sweep", "final_energy")
        else:
            parser.error("Unrecognized benchmark CSV header")

        groups = defaultdict(list)
        for row in reader:
            groups[tuple(row[k] for k in keys)].append(row)

    print("Timings are in milliseconds; median/min/max are across repetitions.")
    for key, group in sorted(groups.items()):
        print("\n" + " ".join(f"{name}={value}" for name, value in zip(keys, key))
              + f" runs={len(group)}")
        print(f"  {'metric':<22} {'median':>12} {'min':>12} {'max':>12}")
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric)]
            if values:
                print(f"  {metric:<22} {statistics.median(values):>12.6g} "
                      f"{min(values):>12.6g} {max(values):>12.6g}")


if __name__ == "__main__":
    main()
