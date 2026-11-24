// Shared benchmark flow for the runtime-configurable benchmarks:
// options -> load J/h -> coloring -> copy to GPU -> run NT sweeps -> report / save.
// Each benchmark only decides how one color step is launched.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "pbit_rng.cuh"
#include "pbit_coloring.cuh"


// runtime parameters, filled from "--option value" pairs
struct BenchmarkOptions {
    std::string data_path;
    std::string coloring_algorithm = "ldf";
    std::string history_mode = "on";
    std::string output_path;
    std::string launch_mode = "normal";
    std::string cache_policy = "default";
    std::string history_output;
    std::string whole_warps_only = "off";
    int threads_per_spin = 1;
    int n_pbits = 0;
    int NT = 1000;
    int threads = 64;
    int rng_threads = 128;
    unsigned long seed = 1234;
    float beta = 1.0f;
};

void parse_options(int argc, char** argv, BenchmarkOptions& opt) {
    for (int a = 1; a < argc; a += 2) {
        std::string option = argv[a];
        const char* value = argv[a + 1];
        if (option == "--data") opt.data_path = value;
        else if (option == "--n") opt.n_pbits = atoi(value);
        else if (option == "--sweeps") opt.NT = atoi(value);
        else if (option == "--threads") opt.threads = atoi(value);
        else if (option == "--rng-threads") opt.rng_threads = atoi(value);
        else if (option == "--seed") opt.seed = strtoul(value, nullptr, 10);
        else if (option == "--beta") opt.beta = atof(value);
        else if (option == "--coloring") opt.coloring_algorithm = value;
        else if (option == "--history") opt.history_mode = value;
        else if (option == "--final") opt.output_path = value;
        else if (option == "--launch") opt.launch_mode = value;
        else if (option == "--cache") opt.cache_policy = value;
        else if (option == "--threads-per-spin") opt.threads_per_spin = atoi(value);
        else if (option == "--whole-warps-only") opt.whole_warps_only = value;
        else if (option == "--history-output") opt.history_output = value;
    }
}


// host graph: float32 J matrix followed by h vector; every spin starts at -1
struct HostGraph {
    std::vector<float> J;
    std::vector<float> h;
    std::vector<int> m;
};

HostGraph load_graph(const std::string& path, int n_pbits) {
    HostGraph graph;
    graph.J.resize(size_t(n_pbits) * n_pbits);
    graph.h.resize(n_pbits);
    graph.m.assign(n_pbits, -1);

    FILE* file = fopen(path.c_str(), "rb");
    fread(graph.J.data(), sizeof(float), graph.J.size(), file);
    fread(graph.h.data(), sizeof(float), graph.h.size(), file);
    fclose(file);
    return graph;
}


// everything the sweeps need on the GPU
struct DeviceGraph {
    float* J = nullptr;
    float* h = nullptr;
    int* m = nullptr;
    int* history = nullptr;  // NT x n_pbits, only with --history on
    RngState* states = nullptr;
    std::vector<int*> color_nodes;
};

DeviceGraph allocate_device(const BenchmarkOptions& opt,
                            const std::vector<std::vector<int>>& color_nodes) {
    int n = opt.n_pbits;
    DeviceGraph d;
    cudaMalloc(&d.J, size_t(n) * n * sizeof(float));
    cudaMalloc(&d.h, n * sizeof(float));
    cudaMalloc(&d.m, n * sizeof(int));
    cudaMalloc(&d.states, n * sizeof(RngState));
    if (opt.history_mode == "on") {
        cudaMalloc(&d.history, size_t(opt.NT) * n * sizeof(int));
    }

    d.color_nodes.resize(color_nodes.size(), nullptr);
    for (size_t c = 0; c < color_nodes.size(); c++) {
        cudaMalloc(&d.color_nodes[c], color_nodes[c].size() * sizeof(int));
    }
    return d;
}

// copy host -> device once, before any sweep; returns the transfer time in ms
double copy_to_device(const HostGraph& graph, const std::vector<std::vector<int>>& color_nodes,
                      DeviceGraph& d) {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    cudaMemcpy(d.J, graph.J.data(), graph.J.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d.h, graph.h.data(), graph.h.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d.m, graph.m.data(), graph.m.size() * sizeof(int), cudaMemcpyHostToDevice);
    for (size_t c = 0; c < color_nodes.size(); c++) {
        cudaMemcpy(d.color_nodes[c], color_nodes[c].data(),
                   color_nodes[c].size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void free_device(DeviceGraph& d) {
    for (int* ptr : d.color_nodes) cudaFree(ptr);
    cudaFree(d.J);
    cudaFree(d.h);
    cudaFree(d.m);
    cudaFree(d.states);
    cudaFree(d.history);
}


struct SimulationTimes {
    float simulation_ms = 0.0f;
    double graph_setup_ms = 0.0;
};

// Run NT sweeps. launch_color(c, count, stream) enqueues the update of color c.
// One stream preserves color dependencies in both "normal" and "graph" launch modes.
template<typename LaunchColor>
SimulationTimes run_simulation(const BenchmarkOptions& opt, DeviceGraph& d,
                               const std::vector<std::vector<int>>& color_nodes,
                               LaunchColor launch_color) {
    int n_pbits = opt.n_pbits;
    int n_colors = static_cast<int>(color_nodes.size());
    SimulationTimes times;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // initialize RNG separately from simulation timing
    int rng_blocks = (n_pbits + opt.rng_threads - 1) / opt.rng_threads;
    init_rng<<<rng_blocks, opt.rng_threads, 0, stream>>>(d.states, opt.seed, n_pbits);
    cudaDeviceSynchronize();

    auto enqueue_simulation = [&]() {
        for (int i = 0; i < opt.NT; i++) {
            for (int c = 0; c < n_colors; c++) {
                int count = static_cast<int>(color_nodes[c].size());
                if (count == 0) continue;
                launch_color(c, count, stream);
            }
            if (opt.history_mode == "on") {
                cudaMemcpyAsync(d.history + size_t(i) * n_pbits, d.m,
                                n_pbits * sizeof(int), cudaMemcpyDeviceToDevice, stream);
            }
        }
    };

    // Capture all sweeps before timing; capture and instantiation do not execute updates.
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    if (opt.launch_mode == "graph") {
        auto graph_start = std::chrono::steady_clock::now();
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        enqueue_simulation();
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graph_exec, graph, 0);
        cudaGraphUpload(graph_exec, stream);
        cudaStreamSynchronize(stream);
        times.graph_setup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - graph_start).count();
    }

    // record simulation time on the same stream as every update/copy
    cudaEvent_t sim_start, sim_end;
    cudaEventCreate(&sim_start);
    cudaEventCreate(&sim_end);
    cudaEventRecord(sim_start, stream);
    if (opt.launch_mode == "graph") cudaGraphLaunch(graph_exec, stream);
    else enqueue_simulation();
    cudaEventRecord(sim_end, stream);
    cudaEventSynchronize(sim_end);
    cudaEventElapsedTime(&times.simulation_ms, sim_start, sim_end);
    cudaEventDestroy(sim_start);
    cudaEventDestroy(sim_end);

    if (opt.launch_mode == "graph") {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(stream);
    return times;
}


// Print the common JSON fields without the closing brace, so a benchmark can append its own.
// The executable reports its build options, so CSV metadata follows the actual binary.
void print_result_json(const BenchmarkOptions& opt, int n_colors, double coloring_ms,
                       double transfer_ms, const SimulationTimes& times) {
    printf("{\"num_pbits\":%d,\"num_colors\":%d,"
           "\"coloring_ms\":%.9f,\"transfer_ms\":%.9f,"
           "\"simulation_ms\":%.9f,\"time_per_sweep_ms\":%.9f,"
           "\"graph_setup_ms\":%.9f,\"math_mode\":\"%s\","
           "\"rng_algorithm\":\"%s\",\"register_cap\":%d",
           opt.n_pbits, n_colors, coloring_ms, transfer_ms,
           times.simulation_ms, times.simulation_ms / opt.NT, times.graph_setup_ms,
           EXEC_MATH, EXEC_RNG_NAME, EXEC_REGISTER_CAP);
}

// optionally save the whole history (binary) and the final spins (text)
void save_outputs(const BenchmarkOptions& opt, const DeviceGraph& d, HostGraph& graph) {
    int n_pbits = opt.n_pbits;
    if (!opt.history_output.empty()) {
        std::vector<int> h_history(size_t(opt.NT) * n_pbits);
        cudaMemcpy(h_history.data(), d.history, h_history.size() * sizeof(int), cudaMemcpyDeviceToHost);
        FILE* output = fopen(opt.history_output.c_str(), "wb");
        fwrite(h_history.data(), sizeof(int), h_history.size(), output);
        fclose(output);
    }

    if (!opt.output_path.empty()) {
        cudaMemcpy(graph.m.data(), d.m, n_pbits * sizeof(int), cudaMemcpyDeviceToHost);
        FILE* output = fopen(opt.output_path.c_str(), "w");
        for (int i = 0; i < n_pbits; i++) {
            fprintf(output, "%d\n", graph.m[i]);
        }
        fclose(output);
    }
}
