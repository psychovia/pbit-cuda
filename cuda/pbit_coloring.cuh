// Graph coloring: every function takes a dense row-major J (n x n) and
// returns color_nodes[c] = the spins in color c.
// Define USE_CUGRAPH before including this file to also get cugraph_coloring.
#pragma once
#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <cuda_runtime.h>

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

// pick a CPU coloring by its command-line name
std::vector<std::vector<int>> color_graph(const std::string& algorithm, const float* J, int n) {
    if (algorithm == "ldf") return largest_degree_first_coloring(J, n);
    if (algorithm == "greedy") return greedy_coloring(J, n);
    if (algorithm == "random") return random_coloring(J, n);
    if (algorithm == "smallest_last") return smallest_last_coloring(J, n);
    if (algorithm == "dsatur") return dsatur_coloring(J, n);
    return {};
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
