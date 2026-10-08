import csv
from collections import defaultdict

CSV_FILE = "profile.csv"

kernel_times = defaultdict(list)

# 去掉 ==PROF== 这些不是 CSV 的行
with open(CSV_FILE, newline="") as f:
    lines = [
        line for line in f
        if not line.startswith("==")
    ]

reader = csv.DictReader(lines)

if reader.fieldnames is None:
    print("Could not find CSV header.")
    exit(1)

# 自动找 duration 那一列
duration_column = None

for name in reader.fieldnames:
    if name and "gpu__time_duration" in name:
        duration_column = name
        break

if duration_column is None:
    print("Could not find gpu__time_duration column.")
    print()
    print("Available timing-related columns:")

    for name in reader.fieldnames:
        if name and "time" in name.lower():
            print("  ", name)

    exit(1)

print(f"Using metric: {duration_column}")
print()

for row in reader:
    kernel = row.get("Kernel Name", "").strip()
    value = row.get(duration_column, "").strip()

    if not kernel or not value:
        continue

    # ncu 数字有时带逗号
    value = value.replace(",", "")

    try:
        duration_ns = float(value)
    except ValueError:
        # 有些 ncu CSV 可能存在 units row，直接跳过
        continue

    # gpu__time_duration.sum raw metric 按 ns 转 ms
    duration_ms = duration_ns / 1_000_000.0

    kernel_times[kernel].append(duration_ms)


if not kernel_times:
    print("No kernel timing information found.")
    exit(1)


print(
    f"{'Kernel':<55}"
    f"{'Calls':>8}"
    f"{'Total (ms)':>15}"
    f"{'Avg (ms)':>15}"
)

print("-" * 93)

grand_total = 0.0

for kernel, times in kernel_times.items():

    calls = len(times)
    total = sum(times)
    avg = total / calls

    grand_total += total

    display_name = kernel

    if len(display_name) > 52:
        display_name = display_name[:49] + "..."

    print(
        f"{display_name:<55}"
        f"{calls:>8}"
        f"{total:>15.6f}"
        f"{avg:>15.6f}"
    )

print("-" * 93)

print(
    f"{'TOTAL GPU KERNEL TIME':<63}"
    f"{grand_total:>15.6f} ms"
)