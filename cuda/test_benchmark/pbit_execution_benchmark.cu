// Runtime-configurable execution benchmark: fixed --threads and --threads-per-spin (1 or 32),
// with --cache picking how J is read.
#include "../pbit_simulation.cuh"
#include "../pbit_kernels.cuh"


void launch_pbit_update(int* m, const float* J, const float* h, RngState* states,
                        const int* nodes, int count, int n, float beta, int threads,
                        int threads_per_spin, const std::string& cache, cudaStream_t stream) {
    int spins_per_block = threads / threads_per_spin;
    int blocks = (count + spins_per_block - 1) / spins_per_block;
#if EXEC_RNG == 0
    if (threads_per_spin == 1 && cache == "default") {
        update_pbit<<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        return;
    }
#endif
    if (threads_per_spin == 32) {
        if (cache == "ldg") update_pbit_execution<32, 1><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        else if (cache == "cg") update_pbit_execution<32, 2><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        else update_pbit_execution<32, 0><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
    } else {
        if (cache == "ldg") update_pbit_execution<1, 1><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        else if (cache == "cg") update_pbit_execution<1, 2><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        else update_pbit_execution<1, 0><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
    }
}


int main(int argc, char** argv) {
    BenchmarkOptions opt;
    parse_options(argc, argv, opt);

    HostGraph graph = load_graph(opt.data_path, opt.n_pbits);

    // record coloring time
    auto coloring_start = std::chrono::steady_clock::now();
    auto color_nodes = color_graph(opt.coloring_algorithm, graph.J.data(), opt.n_pbits);
    auto coloring_end = std::chrono::steady_clock::now();
    double coloring_ms = std::chrono::duration<double, std::milli>(
        coloring_end - coloring_start).count();
    int n_colors = static_cast<int>(color_nodes.size());

    DeviceGraph d = allocate_device(opt, color_nodes);
    double transfer_ms = copy_to_device(graph, color_nodes, d);

    SimulationTimes times = run_simulation(opt, d, color_nodes,
        [&](int c, int count, cudaStream_t stream) {
            launch_pbit_update(d.m, d.J, d.h, d.states, d.color_nodes[c], count, opt.n_pbits,
                               opt.beta, opt.threads, opt.threads_per_spin, opt.cache_policy, stream);
        });

    print_result_json(opt, n_colors, coloring_ms, transfer_ms, times);
    puts("}");

    save_outputs(opt, d, graph);
    free_device(d);
    return 0;
}
