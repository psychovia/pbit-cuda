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

- `cuda/`: CUDA source; main benchmark: `pbit_coloring_benchmark.cu`.
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
  cuda/pbit_coloring_benchmark.cu \
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

## GPU execution benchmark：第一阶段 A1 / A2

保留旧 simulator 和旧 runner，新增 `cuda/pbit_execution_benchmark.cu`，直接复用
原有 `init_rng`、`update_pbit` 和 CPU coloring 函数。dense row-major float32 J/h
及按 color 串行启动的 colored-Gibbs 顺序不变。暂不使用 cuGraph。

```bash
conda activate pbit-cugraph  # Python 需要 numpy
make execution-benchmark

# 小规模验证：4 个配置，每配置 1 warmup + 2 measured runs
python scripts/benchmark_execution.py --experiment A2 --datasets sparse \
  --threads 32 64 --sweeps 2 --warmup 1 --repeats 2 \
  --csv results/execution/smoke.csv
```

以下是完整实验命令，需自行决定何时运行（默认每配置 warmup 3 次、正式 10 次）：

```bash
python scripts/benchmark_execution.py --experiment A1 \
  --datasets sparse dense 10000 --threads 32 64 128 256 512 1024 \
  --sweeps 1000 --coloring ldf --csv results/execution/A1.csv
python scripts/benchmark_execution.py --experiment A2 \
  --datasets sparse dense --threads 32 64 128 256 512 1024 \
  --sweeps 1000 --coloring ldf --csv results/execution/A2.csv
```

`--datasets` 接受名称（对应 `pbit-graphs/J_<name>.npy`）或 J 的 `.npy` 路径。
自动使用同目录的 `h_<name>.npy`；不存在时用零 bias，与现有 preparation 行为一致。
单 dataset 可指定 `--h path/to/h.npy`。runner 将输入转换为与现有 binary loader
相同的 J 后接 h 的 float32 文件，不改变 graph representation，不需要按 dataset 编译。
支持 `--coloring ldf|greedy|random|smallest_last|dsatur`、`--rng-threads 128`、
`--beta 1.0`、`--seed 1234`。初始 spins 固定全 -1。
A1 默认 history on，可用 `--history off`；A2 总是测量 on 和 off。
每一次运行均重新创建 process、加载数据、coloring 和重置 RNG。

计时定义：

- `coloring_ms`：CPU coloring 函数耗时，不含输入文件读取。
- `transfer_ms`：J/h/初始 spins/color nodes 的 H2D copy 和完成同步的 CPU wall time；
  不含分配、RNG 初始化、最终结果的 D2H。
- `simulation_ms`：同一 CUDA stream 上 CUDA events 包围所有 sweep；结束 event 同步后读取。
  history on 包含每 sweep D2D history copy；off 不录制。
  包含实际 launch 提交间隙，RNG 初始化在 start event 前同步完成。
- `time_per_sweep_ms = simulation_ms / sweeps`，为主要性能指标。

每次实际运行（含 warmup）写一行 raw CSV；`phase=warmup|measurement`，
`run_id` 在各 phase 内从 1 起。额外记录 RNG block size、beta、seed。
`*_aggregated.csv` 只聚合 measurement，按完整配置、metric 输出 mean、median、
样本 std（单次为 0）、min、max。输出路径已存在时拒绝覆盖。
可在 Google Sheets 中过滤 phase 或 metric 后 pivot；不要把 warmup 纳入统计。

当前入口按原有 CUDA 文件的顺序结构编写，暂不包含防御性检查和 correctness replay。
未提供 `--validation` 时，CSV 的 `correctness_passed` 留空；独立预验证用法见下方。
history off 不分配 history 内存。

### 独立 Nsight Compute profiling

```bash
# 只查询本机支持项，不运行 simulation；目录必须不存在
python ncu_script/profile_execution.py --output results/profiling/query

# 示例：先用现有工具准备相同 binary sidecar
python scripts/prepare_graph.py --J pbit-graphs/J_sparse.npy \
  --h pbit-graphs/h_sparse.npy --binary-data --output executable/profile_sparse.h
# sparse 的 n 为 1188；其他 dataset 请使用 preparation 输出的 n
python ncu_script/profile_execution.py --output results/profiling/sparse64 -- \
  ./executable/pbit_execution_benchmark --data executable/profile_sparse.bin \
  --n 1188 --threads 64 --rng-threads 128 --sweeps 2 --coloring ldf --history off
```

脚本先保存 `query.txt` 和 `availability.csv`，只选择本机 ncu 明确返回的指标；
未匹配的项目标记 unavailable。local-memory sectors 作为 spill 流量代理，不能直接
当作编译器 spill 数量。profile 保存 `.ncu-rep` 和导出的 raw `profile.csv`，
最多抓前 10 次 update kernel。
profiling 会扰动计时，其结果不应写入常规 benchmark CSV。

A3–B3 的后续实现和独立验证见下方扩展说明。carveout、多 streams、
two-warps-per-spin、fusion 保留为 future work。

此前带 correctness replay 版本的验证记录及 CSV：`results/execution/phase1_smoke/`；
这些是历史记录，不代表当前简化入口每次运行执行了验证。
新版本 ncu 若默认输出 `.ncu-repz`，profiling 脚本使用 `zstd` 解压为 `.ncu-rep` 后导出 CSV。

## A1–A5 / B1 扩展与短实验

新增 `pbit_execution_kernels.cuh` 保留原 one-thread kernel，并提供 warp-per-spin：
32 lanes 按 j=lane, lane+32, ... 累加 J row，shuffle reduction 后仅 lane 0 更新
该 p-bit 的 RNG state 和 spin。threads/block 必须是 32 的倍数；8/16 的 warp 配置跳过。
所有 color 仍在单个 stream 上依次执行，没有并发 streams 或 kernel fusion。

```bash
# 本机 RTX 5060：原生 sm_120，避免默认 sm_75/PTX JIT 干扰编译期实验
make -j2 execution-variants execution-probes
# 其他 GPU 请覆盖 EXEC_ARCH，例如 EXEC_ARCH=sm_89

# 独立验证，不添加到 simulator 的每次执行路径
python scripts/validate_execution.py --output results/execution/preflight

# 当前短实验：B2/B3 暂不跑
python scripts/benchmark_suite.py --experiments A1 A2 A3 A4 A5 B1 \
  --datasets 10000 sparse dense --threads 8 16 32 64 128 256 512 1024 \
  --sweeps 100 --warmup 3 --repeats 10 \
  --validation results/execution/preflight/validation.json \
  --output results/execution/short_suite
```

也可单独运行 `benchmark_execution.py --experiment A3|A4|A5|B1`，其他参数与 A1/A2 相同。
每项逐个改变一个变量，其余默认：history on、normal launch、one-thread/spin、
XORWOW、default math、无 register cap、默认 J load。不是所有优化的交叉组合。

- A3：normal / graph。捕获完整 NT sweeps，capture/instantiate/upload 不计入 simulation；
  `graph_setup_ms` 单独记录。两种模式使用相同单 stream 和原 color 顺序。
- A4：default / `--use_fast_math`，各自 executable。
- A5：无 cap / `--maxrregcount=32/48/64/96`。cap 是上限，不是实际使用量；
  高于默认使用量时可以生成相同 kernel。实际 registers/local memory 见资源报告。
- B1：1 / 32 threads per spin。grid 按 spins/block=threads/block÷threads/spin 计算。
- B2/B3：已提供 XORWOW/Philox/MRG32k3a、J default/`__ldg`/`__ldcg` 变体，
  当前用户要求暂不跑正式实验。后续用 `--experiment B2|B3`，分别支持 `--rngs` / `--caches`。

`--validation` 引用独立验证 manifest，并核对 binary SHA256 后才将
`correctness_passed=True` 写入 CSV；它表示 preflight 已通过，不是每个 timed run
执行 correctness replay。无 manifest 则留空。不同 RNG 不要求轨迹相同。
preflight 对现有 dataset 子矩阵及完整 J10000 短轨迹检查线程/warp、Graph、cache、
register cap、可重复性，并将 local field 与 CPU float64 比较；每个 spin 8192 次
采样检查 Bernoulli 概率（6×标准误+0.002 容差）。这不等同于证明任意图上的数值等价。

Suite 每项保存 `<experiment>.csv`、`<experiment>_aggregated.csv`、`.log`，
还有 `status.json`、GPU 环境和输入 hash。`skip_experiments.json` 可列出尚未开始的
实验，例如 `["B2", "B3"]`。profiling 仍应独立执行。
旧 1000-sweep CSV 和本次 100-sweep、sm_120 结果分开保存，不能合并作为同一配置的重复测量。

本次运行中用户将范围缩为仅 J10000，100 sweeps，实验 A1–A5/B1；B2/B3 暂缓。
后续重跑应使用 `--datasets 10000 --experiments A1 A2 A3 A4 A5 B1`。

本次已完成的 J10000 专用 CSV：`results/execution/suite_J10000_100s/`。
其中 `all_raw.csv` 为 1430 条实际运行记录（含 warmup），`all_aggregated.csv` 为 550 行统计；
已完成 A1–A5/B1，B2/B3 暂缓。两份合并 CSV 可直接用于后续 pivot 和分析。

## Same-color Nsight profiling: T8 / T128 / W64 / W256

Use the independent `scripts/profile_gpu.sh`; default smoke test captures one kernel each for T8/W64.
Then append T128/W256 only after checking smoke success. The completed reports, raw CSVs and summary
are under `results/profiling/J10000_color/`. See [GPU profiling instructions](docs/gpu_profiling.md)
for exact commands, color/launch selection, metric meanings, replay checks, WSL libraries and unavailable fields.
The original A/B timing CSVs are unchanged.
