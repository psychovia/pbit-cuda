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

#define USE_CUGRAPH

#include "../pbit_kernels.cuh"
#include "../pbit_coloring.cuh"



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
