// color provided j
// (1) number of color generated using implementation
// (2) time taken for color
    // let user decide - over xx time then maybe use a suboptimal one
    // accept user provided function / color

// run p circuit
// * annealing 0.5 - 3
// (1) for each coloring algo
// (2) time dependent on num of color
// (3) solution / correctness of final state

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <map>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#define USE_CUGRAPH

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


std::vector<std::vector<int>> greedy_coloring(const float* J, int n) {
    // color_nodes[i] = vertices in color i
    std::vector<std::vector<int>> color_nodes;

    for (int i = 0; i < n; i++) {
        bool assigned = false;

        // try add in existing colors
        for (int c = 0; c < color_nodes.size(); c++) {
            bool conflict = false;

            // check if current vtx is connected to any other vtx in this color
            for (int j : color_nodes[c]) {
                if (J[i*n+j] != 0.0f || J[j*n+i] != 0.0f) {
                    conflict = true;
                    break;
                }
            }

            if (!conflict) {
                color_nodes[c].push_back(i);
                assigned = true;
                break;
            }
        }

        // create new color
        if (!assigned) {
            color_nodes.push_back({i});
        }
    }

    return color_nodes;
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



int main() {
    const int n_pbits = 4;
    const float beta = 1.0f;
    const int NT = 10;


    //      0 ---(-1)--- 1
    //      |            |
    //     (+1)         (+2)
    //      |            |
    //      3 ---(-1)--- 2

    float h_J[16] = {
         0, -1,  0,  1,
        -1,  0,  2,  0,
         0,  2,  0, -1,
         1,  0, -1,  0
    };

    // bias
    float h_h[4] = {1, 1, 1, 0};
    // init state
    int h_m[4] = {-1, -1, -1, -1};



#ifdef USE_CUGRAPH
    printf("Coloring method: cuGraph (GPU)\n");
    auto color_nodes = cugraph_coloring(h_J, n_pbits);
#else
    printf("Coloring method: sequential greedy (CPU)\n");
    auto color_nodes = greedy_coloring(h_J, n_pbits);
#endif


    int n_colors = static_cast<int>(color_nodes.size());

    printf("Number of colors: %d\n", n_colors);
    for (int c = 0; c < n_colors; c++) {
        printf("Color %d: ", c);
        for (int i : color_nodes[c]) printf("%d ", i);
        printf("\n");
    }

    float *d_J, *d_h;
    int *d_m, *d_history;
    
    // alloc memory
    curandState* d_states;
    cudaMalloc(&d_J, n_pbits * n_pbits * sizeof(float));
    cudaMalloc(&d_h, n_pbits * sizeof(float));
    cudaMalloc(&d_m, n_pbits * sizeof(int));
    cudaMalloc(&d_states, n_pbits * sizeof(curandState));
    cudaMalloc(&d_history, NT * n_pbits * sizeof(int));

    // copy to device
    cudaMemcpy(d_J, h_J, n_pbits * n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, n_pbits * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int*> d_color_nodes(n_colors, nullptr);

    for (int c = 0; c < n_colors; c++) {
        int size = static_cast<int>(color_nodes[c].size());
        cudaMalloc(&d_color_nodes[c], size * sizeof(int));
        cudaMemcpy(d_color_nodes[c], color_nodes[c].data(), size * sizeof(int), cudaMemcpyHostToDevice);
    }


    const int threads = 256;
    int blocks = (n_pbits + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(d_states, 1234, n_pbits);

    // update
    for (int i = 0; i < NT; i++) {
        for (int c = 0; c < n_colors; c++) {
            int n_nodes_in_color = static_cast<int>(color_nodes[c].size());
            int num_blocks = (n_nodes_in_color + threads - 1) / threads;
            update_pbit<<<num_blocks, threads>>>(d_m, d_J, d_h, d_states, d_color_nodes[c], n_nodes_in_color, n_pbits, beta);
        }
        cudaMemcpyAsync(d_history + i * n_pbits, d_m, n_pbits * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
    std::vector<int> h_history(NT * n_pbits);
    cudaMemcpy(h_history.data(), d_history, NT * n_pbits * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NT; i++) {
        printf("state %d: [", i);
        for (int j = 0; j < n_pbits; j++) {
            printf("%d%s", h_history[i * n_pbits + j],
                   (j == n_pbits - 1) ? "" : ", ");
        }
        printf("]\n");
    }

    for (int* ptr : d_color_nodes) {
        cudaFree(ptr);
    }
    cudaFree(d_J);
    cudaFree(d_h);
    cudaFree(d_m);
    cudaFree(d_states);
    cudaFree(d_history);
    return 0;
}
