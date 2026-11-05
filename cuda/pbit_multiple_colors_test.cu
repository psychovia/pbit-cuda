
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "simulation_config.h"


// Initialize RNG for each p-bit
__device__ void init_rng(curandState* state, unsigned long seed, int i) {
    curand_init(seed, i, 0, state);
}


// Update all p-bits across all sweeps on the GPU
__global__ void update_pbit(int* m, const float* J, const float* h,
                            const int* colors, const float* beta) {

    int i = threadIdx.x;

    curandState state;
    init_rng(&state, SEED, i);

    __syncthreads();

    // All iterations run on GPU
    for (int t = 0; t < NT; t++) {

        // Update each color group
        for (int c = 0; c < N_COLORS; c++) {

            int new_m = 0;

            if (colors[i] == c) {

                float I = h[i];

                for (int j = 0; j < N_PBITS; j++) {
                    I += J[i * N_PBITS + j] * m[j];
                }

                // Original p-bit update equation
                float r = curand_uniform(&state);

                float value = tanhf(beta[t] * I)
                            - (1.0f - 2.0f * r);

                new_m = (value >= 0.0f) ? 1 : -1;
            }

            // All threads finish reading before updating
            __syncthreads();

            if (colors[i] == c) {
                m[i] = new_m;
            }

            // Finish this color before next color
            __syncthreads();
        }
    }
}


// Original greedy coloring function
std::vector<int> greedy_coloring(const float* J, int n) {

    std::vector<int> colors(n, -1);

    for (int i = 0; i < n; i++) {

        std::vector<bool> used(n, false);

        for (int j = 0; j < n; j++) {

            if (i == j) continue;

            if (J[i * n + j] != 0.0f ||
                J[j * n + i] != 0.0f) {

                if (colors[j] != -1) {
                    used[colors[j]] = true;
                }
            }
        }

        int c = 0;

        while (used[c]) {
            c++;
        }

        colors[i] = c;
    }

    return colors;
}

int main() {

    // Values supplied through Makefile-generated header
    const int n_pbits = N_PBITS;
    const int n_colors = N_COLORS;

    // Host data
    const float* h_J = HOST_J;
    const float* h_h = HOST_H;
    const int* h_m = HOST_M;

    // Device pointers
    float* d_J;
    float* d_h;
    float* d_beta;

    int* d_m;
    int* d_colors;

    // Check GPU block size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (n_pbits > prop.maxThreadsPerBlock) {
        printf("Too many p-bits for one GPU block\n");
        return 1;
    }

    // Allocate GPU memory
    cudaMalloc(&d_J, n_pbits * n_pbits * sizeof(float));
    cudaMalloc(&d_h, n_pbits * sizeof(float));
    cudaMalloc(&d_m, n_pbits * sizeof(int));
    cudaMalloc(&d_colors, n_pbits * sizeof(int));
    cudaMalloc(&d_beta, NT * sizeof(float));

    // Copy host -> device
    cudaMemcpy(d_J, h_J,
               n_pbits * n_pbits * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_h, h_h,
               n_pbits * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_m, h_m,
               n_pbits * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_colors, HOST_COLORS,
               n_pbits * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_beta, HOST_BETA,
               NT * sizeof(float),
               cudaMemcpyHostToDevice);

    // One GPU kernel launch for all iterations
    update_pbit<<<1, n_pbits>>>(
        d_m, d_J, d_h, d_colors, d_beta
    );


    // Copy final states to CPU only once
    std::vector<int> final_m(n_pbits);

    cudaMemcpy(final_m.data(), d_m,
               n_pbits * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Print final states
    for (int i = 0; i < n_pbits; i++) {
        printf("%d%s", final_m[i],
               i == n_pbits - 1 ? "\n" : " ");
    }

    // Free GPU memory
    cudaFree(d_J);
    cudaFree(d_h);
    cudaFree(d_m);
    cudaFree(d_colors);
    cudaFree(d_beta);

    return 0;
}
