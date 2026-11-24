// Graph-colored p-bit benchmark with automatic CUDA launch selection.
// threads per spin (p) and threads per block (B) minimize waves per sweep, see pbit_launch.cuh.
// --threads 0 / --threads-per-spin 0 mean "search"; a positive value locks that one.
#include "../pbit_simulation.cuh"
#include "../pbit_launch.cuh"


int main(int argc, char** argv) {
    BenchmarkOptions opt;
    opt.threads = 0;
    opt.threads_per_spin = 0;
    parse_options(argc, argv, opt);

    HostGraph graph = load_graph(opt.data_path, opt.n_pbits);

    // record coloring time
    auto coloring_start = std::chrono::steady_clock::now();
    auto color_nodes = color_graph(opt.coloring_algorithm, graph.J.data(), opt.n_pbits);
    auto coloring_end = std::chrono::steady_clock::now();
    double coloring_ms = std::chrono::duration<double, std::milli>(
        coloring_end - coloring_start).count();
    int n_colors = static_cast<int>(color_nodes.size());

    LaunchConfig launch = choose_launch(color_nodes, opt.threads, opt.threads_per_spin,
                                        opt.whole_warps_only == "on");

    DeviceGraph d = allocate_device(opt, color_nodes);
    double transfer_ms = copy_to_device(graph, color_nodes, d);

    SimulationTimes times = run_simulation(opt, d, color_nodes,
        [&](int c, int count, cudaStream_t stream) {
            launch_update(launch, d.m, d.J, d.h, d.states, d.color_nodes[c],
                          count, opt.n_pbits, opt.beta, stream);
        });

    print_result_json(opt, n_colors, coloring_ms, transfer_ms, times);
    print_launch_json(launch, color_nodes);
    puts("}");

    save_outputs(opt, d, graph);
    free_device(d);
    return 0;
}
