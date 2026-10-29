#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_rng(curandState* states, unsigned long seed, int n_pbits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_pbits) {
        curand_init(seed, i, 0, &states[i]);
    }
}


__global__ void update_pbit(int* m, const float* J, const float* h, curandState* states,
                                const int* color_nodes, int n_nodes_in_color,
                                int n_pbits, float beta) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_nodes_in_color) {
        return;
    }

    int i = color_nodes[tid];

    float I = h[i];

    for (int j = 0; j < n_pbits; j++) {
        I += J[i * n_pbits + j] * m[j];
    }

    // range = (0, 1]
    float r = curand_uniform(&states[i]);

    
    float value = tanhf(beta * I) - (1.0f - 2.0f * r); // (0, -1] -> (-1, 1]

    // sign
    m[i] = (value >= 0.0f) ? 1 : -1;
}


std::vector<int> greedy_coloring(const float* J, int n) {
    std::vector<int> colors(n, -1);

    for (int i = 0; i < n; i++) {
        std::vector<bool> used(n, false);

        // Check colors of already-colored neighbors
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            if (J[i * n + j] != 0.0f ||
                J[j * n + i] != 0.0f) {
                if (colors[j] != -1) {
                    used[colors[j]] = true;
                }
            }
        }

        // Find the smallest available color
        int c = 0;
        while (used[c]) {
            c++;
        }

        colors[i] = c;
    }

    return colors;
}



int main() {
    const int n_pbits = 4;
    const int n_colors = 3;
    const float beta = 1.0f;
    int NT = 10;

    // host - cpu / device = gpu
    
    // data in host

    // // AND gate
    // float h_J[9] = {
    //      0, -1,  2,
    //     -1,  0,  2,
    //      2,  2,  0
    // };
    // float h_h[3] = {1, 1, -2};
    // int h_m[3] = {-1, -1, -1};
    // int n_colors = 1;

    // 1 - 2
    //   /
    // 3 - 4

    float h_J[16] = {
         0, -1,  0, 1,
        -1,  0,  2, 0,
         0,  2,  0, -1,
         1,  0, -1, 0
    };
    
    float h_h[4] = {1, 1, 1, 0};
    int h_m[4] = {-1, -1, -1, -1};
    int h_color_nodes[4] = {0, 2, 1, 3};
    int color_offsets[3] = {0, 1, 2};
    int color_sizes[3] = {1, 1, 2}; // number of nodes in each color

    // device pointers
    float* d_J;
    float* d_h;
    int* d_m;
    int* d_color_nodes;
    int* d_history; // store all states seen so far, print together by the end
    curandState* d_states;

    // alloc gpu memory
    cudaMalloc(&d_J, n_pbits * n_pbits * sizeof(float));
    cudaMalloc(&d_h, n_pbits * sizeof(float));
    cudaMalloc(&d_m, n_pbits * sizeof(int));
    cudaMalloc(&d_states, n_pbits * sizeof(curandState));
    cudaMalloc(&d_color_nodes, n_pbits * sizeof(int));
    cudaMalloc(&d_history, NT * n_pbits * sizeof(int));

    // copy host -> device
    cudaMemcpy(d_J, h_J, n_pbits * n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, n_pbits * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_color_nodes, h_color_nodes, n_pbits * sizeof(int), cudaMemcpyHostToDevice);

    // initialize rng with seed
    init_rng<<<1, 3>>>(d_states, 1234, n_pbits);

    cudaDeviceSynchronize();

    for (int i = 0; i < NT; i++) { // num of states
        for (int j = 0; j < n_colors; j++) { // within each state, update each pbit in each color
            int n_nodes_in_color = color_sizes[j]; // # of nodes in current color
            update_pbit<<<1, n_nodes_in_color>>>(d_m, d_J, d_h, d_states,
                                               d_color_nodes + color_offsets[j],
                                               n_nodes_in_color, n_pbits, beta);
        }
        // add current state to history
        cudaMemcpyAsync(d_history + i * n_pbits, d_m, n_pbits * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // copy result back GPU -> CPU

    // cudaMemcpy(h_m, d_m, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("m = [%d, %d, %d]\n", h_m[0], h_m[1], h_m[2]);

    cudaDeviceSynchronize();

    int* h_history = (int*)malloc(NT * n_pbits * sizeof(int));

    cudaMemcpy(h_history, d_history, NT * n_pbits * sizeof(int), cudaMemcpyDeviceToHost);

    for (int ii = 0; ii < NT; ii++) {
        printf("state %d: [", ii);

        for (int jj = 0; jj < n_pbits; jj++) {
            printf("%d", h_history[ii * n_pbits + jj]);
            if (jj != n_pbits - 1) {
                printf(", ");
            }
        }

        printf("]\n");
    }

    free(h_history);

    cudaFree(d_J);
    cudaFree(d_h);
    cudaFree(d_m);
    cudaFree(d_states);
    cudaFree(d_color_nodes);
    cudaFree(d_history);

    return 0;
}
