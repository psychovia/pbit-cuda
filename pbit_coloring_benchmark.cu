// based on pbit_coloring_test.cu
// keeps track of coloring, transfer and simulation time and color group sizes

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// uncomment to use cuGraph coloring instead of CPU greedy coloring
// #define USE_CUGRAPH

#ifdef USE_CUGRAPH
#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>
#include <rmm/device_uvector.hpp>
#include <optional>
#include <tuple>
#include <cstdint>
#include <utility>
#endif


__global__ void init_rng(curandState* states, unsigned long seed, int n_pbits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_pbits) curand_init(seed, i, 0, &states[i]);
}


__global__ void update_pbit(int* m, const float* J, const float* h,
                            curandState* states, const int* color_nodes,
                            int n_nodes_in_color, int n_pbits, float beta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nodes_in_color) return;

    int i = color_nodes[tid];
    float I = h[i];
    for (int j = 0; j < n_pbits; j++) {
        I += J[i * n_pbits + j] * m[j];
    }

    float r = curand_uniform(&states[i]);
    float value = tanhf(beta * I) - (1.0f - 2.0f * r);
    m[i] = (value >= 0.0f) ? 1 : -1;
}


// CPU greedy: color vertices in index order, choosing the first available color.
std::vector<std::vector<int>> greedy_coloring(const float* J, int n) {
    std::vector<int> colors(n, -1);
    std::vector<std::vector<int>> color_nodes;
    for (int i = 0; i < n; i++) {
        std::vector<bool> used(n, false);
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            if ((J[i * n + j] != 0.0f || J[j * n + i] != 0.0f)
                && colors[j] != -1) used[colors[j]] = true;
        }
        int c = 0;
        while (used[c]) c++;
        colors[i] = c;
        if (c == static_cast<int>(color_nodes.size())) color_nodes.push_back({});
        color_nodes[c].push_back(i);
    }
    return color_nodes;
}

// Shared implementation: choose a vertex order, then assign available colors.
std::vector<std::vector<int>> ordered_coloring(
    const float* J, int n, const std::string& method, unsigned long seed) {
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (J[i * n + j] != 0.0f || J[j * n + i] != 0.0f) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }

    std::vector<int> order(n);
    std::vector<int> color(n, -1);
    std::iota(order.begin(), order.end(), 0);

    if (method == "largest-first") {
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return adj[a].size() > adj[b].size();
        });
    } else if (method == "random") {
        std::mt19937_64 rng(seed);
        std::shuffle(order.begin(), order.end(), rng);
    } else if (method == "smallest-last") {
        std::vector<int> degree(n);
        std::set<std::pair<int, int>> queue;
        std::vector<bool> removed(n, false);

        for (int v = 0; v < n; ++v) {
            degree[v] = adj[v].size();
            queue.insert({degree[v], v});
        }

        // Remove minimum-degree vertices; color them in reverse removal order.
        for (int k = n - 1; k >= 0; --k) {
            int v = queue.begin()->second;
            queue.erase(queue.begin());
            removed[v] = true;
            order[k] = v;

            for (int u : adj[v]) {
                if (!removed[u]) {
                    queue.erase({degree[u], u});
                    --degree[u];
                    queue.insert({degree[u], u});
                }
            }
        }
    } else if (method != "greedy" && method != "dsatur") {
        throw std::runtime_error("Unknown coloring: " + method);
    }

    // DSATUR tracks the distinct colors already assigned to each neighbor set.
    std::vector<std::set<int>> saturation(method == "dsatur" ? n : 0);
    std::vector<int> used(n, -1);
    int n_colors = 0;

    for (int step = 0; step < n; ++step) {
        int v = order[step];
        if (method == "dsatur") {
            v = -1;
            for (int u = 0; u < n; ++u) {
                if (color[u] < 0 &&
                    (v < 0 || saturation[u].size() > saturation[v].size() ||
                     (saturation[u].size() == saturation[v].size() &&
                      adj[u].size() > adj[v].size()))) {
                    v = u;
                }
            }
        }

        // Mark unavailable colors with this step number to reuse the buffer.
        for (int u : adj[v]) {
            if (color[u] >= 0) {
                used[color[u]] = step;
            }
        }

        int c = 0;
        while (c < n && used[c] == step) {
            ++c;
        }
        color[v] = c;
        n_colors = std::max(n_colors, c + 1);

        if (method == "dsatur") {
            for (int u : adj[v]) {
                if (color[u] < 0) {
                    saturation[u].insert(c);
                }
            }
        }
    }

    std::vector<std::vector<int>> groups(n_colors);
    for (int v = 0; v < n; ++v) {
        groups[color[v]].push_back(v);
    }
    return groups;
}

std::vector<std::vector<int>> largest_degree_first_coloring(const float* J, int n) {
    return ordered_coloring(J, n, "largest-first", 1234);
}

std::vector<std::vector<int>> random_coloring(const float* J, int n) {
    return ordered_coloring(J, n, "random", 1234);
}

std::vector<std::vector<int>> smallest_last_coloring(const float* J, int n) {
    return ordered_coloring(J, n, "smallest-last", 1234);
}

std::vector<std::vector<int>> dsatur_coloring(const float* J, int n) {
    return ordered_coloring(J, n, "dsatur", 1234);
}

#ifdef USE_CUGRAPH
std::vector<std::vector<int>> cugraph_coloring(const float* J, int n) {
    using vertex_t = int32_t;
    using edge_t = int32_t;

    // convert J matrix to edge list
    // edge = src[i], dst[i]
    std::vector<vertex_t> src, dst;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (J[i*n+j] != 0.0f || J[j*n+i] != 0.0f) {
                // undirectioned
                src.push_back(i);
                dst.push_back(j);
                src.push_back(j);
                dst.push_back(i);
            }
        }
    }


    // raft - keep track of resource context
    // rmm - manage dram
    raft::handle_t handle;
    auto stream = handle.get_stream().get();

    // alloc memory
    rmm::device_uvector<vertex_t> d_vertices(n, stream);
    rmm::device_uvector<vertex_t> d_src(src.size(), stream);
    rmm::device_uvector<vertex_t> d_dst(dst.size(), stream);

    std::vector<vertex_t> vertices(n);
    std::iota(vertices.begin(), vertices.end(), 0);



    // copy to device
    cudaMemcpyAsync(d_vertices.data(), vertices.data(), n * sizeof(vertex_t), cudaMemcpyHostToDevice, stream);

    if (!src.empty()) {
        cudaMemcpyAsync(d_src.data(), src.data(), src.size() * sizeof(vertex_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_dst.data(), dst.data(), dst.size() * sizeof(vertex_t), cudaMemcpyHostToDevice, stream);
    }

    cudaStreamSynchronize(stream);



    // construct graph
    std::vector<cugraph::arithmetic_device_uvector_t> edge_properties;

    auto [graph, edge_props, renumber_map] =
        cugraph::create_graph_from_edgelist<
            vertex_t, edge_t, false, false
        >(
            handle,
            std::make_optional(std::move(d_vertices)),
            std::move(d_src),
            std::move(d_dst),
            std::move(edge_properties),
            cugraph::graph_properties_t{true, false},
            false, std::nullopt, std::nullopt, false
        );

    // graph coloring
    raft::random::RngState rng_state(1234);
    auto d_colors = cugraph::vertex_coloring(handle, graph.view(), rng_state);


    // copy colors to CPU
    std::vector<vertex_t> colors(n);

    cudaMemcpyAsync(colors.data(), d_colors.data(), n * sizeof(vertex_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);


    // group vertices by color
    int n_colors = *std::max_element(colors.begin(), colors.end()) + 1;

    std::vector<std::vector<int>> color_nodes(n_colors);

    for (int i = 0; i < n; i++) {
        color_nodes[colors[i]].push_back(i);
    }

    return color_nodes;
}
#endif



// Switch CPU algorithms by changing this function name, or compiling with
// -DCOLORING_FUNCTION=dsatur_coloring (cuGraph still uses USE_CUGRAPH).
#ifndef COLORING_FUNCTION
#define COLORING_FUNCTION greedy_coloring
#endif
#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)

int main() {
    const float beta = 1.0f;
    const int NT = 1000;

#ifdef GRAPH_HEADER
    // Generated by scripts/prepare_graph.py; defines n_pbits, h_J, h_h, h_m.
    #include GRAPH_HEADER
#else
    const int n_pbits = 4;
    float h_J[16] = {
         0, -1,  0,  1,
        -1,  0,  2,  0,
         0,  2,  0, -1,
         1,  0, -1,  0
    };
    float h_h[4] = {1, 1, 1, 0};
    int h_m[4] = {-1, -1, -1, -1};
#endif

    auto coloring_start = std::chrono::steady_clock::now();
#ifdef USE_CUGRAPH
    printf("Coloring method: cuGraph (GPU)\n");
    auto color_nodes = cugraph_coloring(h_J, n_pbits);
#else
    printf("Coloring method: %s (CPU)\n", STRINGIFY(COLORING_FUNCTION));
    auto color_nodes = COLORING_FUNCTION(h_J, n_pbits);
#endif
    auto coloring_end = std::chrono::steady_clock::now();
    double coloring_ms = std::chrono::duration<double, std::milli>(
        coloring_end - coloring_start).count();

    int n_colors = static_cast<int>(color_nodes.size());
    printf("Number of colors: %d\n", n_colors);
    for (int c = 0; c < n_colors; c++) {
        printf("Color %d (%zu nodes)", c, color_nodes[c].size());
    //     for (int i : color_nodes[c]) printf("%d ", i);
        printf("\n");
    }

    float *d_J, *d_h;
    int *d_m, *d_history;
    curandState* d_states;
    cudaMalloc(&d_J, n_pbits * n_pbits * sizeof(float));
    cudaMalloc(&d_h, n_pbits * sizeof(float));
    cudaMalloc(&d_m, n_pbits * sizeof(int));
    cudaMalloc(&d_states, n_pbits * sizeof(curandState));
    cudaMalloc(&d_history, NT * n_pbits * sizeof(int));

    cudaMemcpy(d_J, h_J, n_pbits * n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, n_pbits * sizeof(int), cudaMemcpyHostToDevice);

    auto transfer_start = std::chrono::steady_clock::now();
    std::vector<int*> d_color_nodes(n_colors, nullptr);
    for (int c = 0; c < n_colors; c++) {
        int size = static_cast<int>(color_nodes[c].size());
        cudaMalloc(&d_color_nodes[c], size * sizeof(int));
        cudaMemcpy(d_color_nodes[c], color_nodes[c].data(),
                   size * sizeof(int), cudaMemcpyHostToDevice);
    }
    auto transfer_end = std::chrono::steady_clock::now();
    double transfer_ms = std::chrono::duration<double, std::milli>(
        transfer_end - transfer_start).count();

    const int threads = 256;
    int blocks = (n_pbits + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(d_states, 1234, n_pbits);

    cudaEvent_t sim_start, sim_end;
    cudaEventCreate(&sim_start);
    cudaEventCreate(&sim_end);
    cudaEventRecord(sim_start);
    for (int i = 0; i < NT; i++) {
        for (int c = 0; c < n_colors; c++) {
            int n_nodes_in_color = static_cast<int>(color_nodes[c].size());
            if (n_nodes_in_color == 0) continue;
            int num_blocks = (n_nodes_in_color + threads - 1) / threads;
            update_pbit<<<num_blocks, threads>>>(
                d_m, d_J, d_h, d_states, d_color_nodes[c],
                n_nodes_in_color, n_pbits, beta);
        }
        cudaMemcpyAsync(d_history + i * n_pbits, d_m,
                        n_pbits * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(sim_end);
    cudaEventSynchronize(sim_end);
    float simulation_ms;
    cudaEventElapsedTime(&simulation_ms, sim_start, sim_end);
    cudaEventDestroy(sim_start);
    cudaEventDestroy(sim_end);

    printf("Coloring time: %.6f ms\n", coloring_ms);
    printf("Memory transfer time: %.6f ms\n", transfer_ms);
    printf("Simulation time (%d sweeps): %.6f ms\n", NT, simulation_ms);
    printf("Time per sweep: %.6f ms\n", simulation_ms / NT);
    printf("Total measured time: %.6f ms\n", coloring_ms + transfer_ms + simulation_ms);

    std::vector<int> h_history(NT * n_pbits);
    cudaMemcpy(h_history.data(), d_history,
               NT * n_pbits * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < NT; i++) {
    //     printf("state %d: [", i);
    //     for (int j = 0; j < n_pbits; j++) {
    //         printf("%d%s", h_history[i * n_pbits + j],
    //                (j == n_pbits - 1) ? "" : ", ");
    //     }
    //     printf("]\n");
    // }
    for (int* ptr : d_color_nodes) cudaFree(ptr);
    cudaFree(d_J);
    cudaFree(d_h);
    cudaFree(d_m);
    cudaFree(d_states);
    cudaFree(d_history);
    return 0;
}
