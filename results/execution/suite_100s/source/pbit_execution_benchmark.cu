// reuse the existing kernels and CPU coloring functions
#define main legacy_simulator_main
#include "pbit_coloring_benchmark.cu"
#undef main
#include "pbit_execution_kernels.cuh"


int main(int argc, char** argv) {
    // runtime parameters
    std::string data_path;
    std::string coloring_algorithm = "ldf";
    std::string history_mode = "on";
    std::string output_path;
    std::string launch_mode = "normal";
    std::string cache_policy = "default";
    std::string history_output;
    int threads_per_spin = 1;
    int n_pbits = 0;
    int NT = 1000;
    int threads = 64;
    int rng_threads = 128;
    unsigned long seed = 1234;
    float beta = 1.0f;

    for (int a = 1; a < argc; a += 2) {
        std::string option = argv[a];
        const char* value = argv[a + 1];
        if (option == "--data") data_path = value;
        else if (option == "--n") n_pbits = atoi(value);
        else if (option == "--sweeps") NT = atoi(value);
        else if (option == "--threads") threads = atoi(value);
        else if (option == "--rng-threads") rng_threads = atoi(value);
        else if (option == "--seed") seed = strtoul(value, nullptr, 10);
        else if (option == "--beta") beta = atof(value);
        else if (option == "--coloring") coloring_algorithm = value;
        else if (option == "--history") history_mode = value;
        else if (option == "--final") output_path = value;
        else if (option == "--launch") launch_mode = value;
        else if (option == "--cache") cache_policy = value;
        else if (option == "--threads-per-spin") threads_per_spin = atoi(value);
        else if (option == "--history-output") history_output = value;
    }

    // load J/h: float32 J matrix followed by h vector
    std::vector<float> graph_J(size_t(n_pbits) * n_pbits);
    std::vector<float> graph_h(n_pbits);
    std::vector<int> graph_m(n_pbits, -1);
    float* h_J = graph_J.data();
    float* h_h = graph_h.data();
    int* h_m = graph_m.data();

    FILE* graph_file = fopen(data_path.c_str(), "rb");
    fread(h_J, sizeof(float), graph_J.size(), graph_file);
    fread(h_h, sizeof(float), graph_h.size(), graph_file);
    fclose(graph_file);

    // record coloring time
    auto coloring_start = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> color_nodes;
    if (coloring_algorithm == "ldf") {
        color_nodes = largest_degree_first_coloring(h_J, n_pbits);
    } else if (coloring_algorithm == "greedy") {
        color_nodes = greedy_coloring(h_J, n_pbits);
    } else if (coloring_algorithm == "random") {
        color_nodes = random_coloring(h_J, n_pbits);
    } else if (coloring_algorithm == "smallest_last") {
        color_nodes = smallest_last_coloring(h_J, n_pbits);
    } else if (coloring_algorithm == "dsatur") {
        color_nodes = dsatur_coloring(h_J, n_pbits);
    }
    auto coloring_end = std::chrono::steady_clock::now();
    double coloring_ms = std::chrono::duration<double, std::milli>(
        coloring_end - coloring_start).count();
    int n_colors = static_cast<int>(color_nodes.size());

    // allocate GPU memory
    float *d_J, *d_h;
    int *d_m, *d_history = nullptr;
    ExecutionRngState* d_states;
    cudaMalloc(&d_J, size_t(n_pbits) * n_pbits * sizeof(float));
    cudaMalloc(&d_h, n_pbits * sizeof(float));
    cudaMalloc(&d_m, n_pbits * sizeof(int));
    cudaMalloc(&d_states, n_pbits * sizeof(ExecutionRngState));
    if (history_mode == "on") {
        cudaMalloc(&d_history, size_t(NT) * n_pbits * sizeof(int));
    }

    std::vector<int*> d_color_nodes(n_colors, nullptr);
    for (int c = 0; c < n_colors; c++) {
        cudaMalloc(&d_color_nodes[c], color_nodes[c].size() * sizeof(int));
    }

    // copy host -> device and record transfer time
    cudaDeviceSynchronize();
    auto transfer_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_J, h_J, size_t(n_pbits) * n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, n_pbits * sizeof(int), cudaMemcpyHostToDevice);
    for (int c = 0; c < n_colors; c++) {
        cudaMemcpy(d_color_nodes[c], color_nodes[c].data(),
                   color_nodes[c].size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    auto transfer_end = std::chrono::steady_clock::now();
    double transfer_ms = std::chrono::duration<double, std::milli>(
        transfer_end - transfer_start).count();

    // One stream preserves color dependencies in both launch modes.
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // initialize RNG separately from simulation timing
    int rng_blocks = (n_pbits + rng_threads - 1) / rng_threads;
    init_execution_rng<<<rng_blocks, rng_threads, 0, stream>>>(d_states, seed, n_pbits);
    cudaDeviceSynchronize();

    // Capture all sweeps before timing; capture and instantiation do not execute updates.
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    auto enqueue_simulation = [&]() {
        for (int i = 0; i < NT; i++) {
            for (int c = 0; c < n_colors; c++) {
                int count = static_cast<int>(color_nodes[c].size());
                launch_pbit_update(d_m, d_J, d_h, d_states, d_color_nodes[c],
                                   count, n_pbits, beta, threads, threads_per_spin,
                                   cache_policy, stream);
            }
            if (history_mode == "on") {
                cudaMemcpyAsync(d_history + size_t(i) * n_pbits, d_m,
                                n_pbits * sizeof(int), cudaMemcpyDeviceToDevice, stream);
            }
        }
    };

    double graph_setup_ms = 0.0;
    if (launch_mode == "graph") {
        auto graph_start = std::chrono::steady_clock::now();
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        enqueue_simulation();
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graph_exec, graph, 0);
        cudaGraphUpload(graph_exec, stream);
        cudaStreamSynchronize(stream);
        graph_setup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - graph_start).count();
    }

    // record simulation time on the same stream as every update/copy
    cudaEvent_t sim_start, sim_end;
    cudaEventCreate(&sim_start);
    cudaEventCreate(&sim_end);
    cudaEventRecord(sim_start, stream);
    if (launch_mode == "graph") cudaGraphLaunch(graph_exec, stream);
    else enqueue_simulation();
    cudaEventRecord(sim_end, stream);
    cudaEventSynchronize(sim_end);
    float simulation_ms;
    cudaEventElapsedTime(&simulation_ms, sim_start, sim_end);
    cudaEventDestroy(sim_start);
    cudaEventDestroy(sim_end);

    // executable reports its build options, so CSV metadata follows the actual binary
    printf("{\"num_pbits\":%d,\"num_colors\":%d,"
           "\"coloring_ms\":%.9f,\"transfer_ms\":%.9f,"
           "\"simulation_ms\":%.9f,\"time_per_sweep_ms\":%.9f,"
           "\"graph_setup_ms\":%.9f,\"math_mode\":\"%s\","
           "\"rng_algorithm\":\"%s\",\"register_cap\":%d}\n",
           n_pbits, n_colors, coloring_ms, transfer_ms,
           simulation_ms, simulation_ms / NT, graph_setup_ms,
           EXEC_MATH, EXEC_RNG_NAME, EXEC_REGISTER_CAP);

    if (!history_output.empty()) {
        std::vector<int> h_history(size_t(NT) * n_pbits);
        cudaMemcpy(h_history.data(), d_history, h_history.size() * sizeof(int), cudaMemcpyDeviceToHost);
        FILE* output = fopen(history_output.c_str(), "wb");
        fwrite(h_history.data(), sizeof(int), h_history.size(), output);
        fclose(output);
    }

    // optionally save final spins
    if (!output_path.empty()) {
        cudaMemcpy(h_m, d_m, n_pbits * sizeof(int), cudaMemcpyDeviceToHost);
        FILE* output = fopen(output_path.c_str(), "w");
        for (int i = 0; i < n_pbits; i++) {
            fprintf(output, "%d\n", h_m[i]);
        }
        fclose(output);
    }

    if (launch_mode == "graph") {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(stream);

    // free memory
    for (int* ptr : d_color_nodes) cudaFree(ptr);
    cudaFree(d_J);
    cudaFree(d_h);
    cudaFree(d_m);
    cudaFree(d_states);
    cudaFree(d_history);
    return 0;
}
