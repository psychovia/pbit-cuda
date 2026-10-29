#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_rng(curandState* states, unsigned long seed, int n_pbits ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_pbits) {
        curand_init(seed, i, 0, &states[i]);
    }
}


__global__ void update_one_pbit(int* m, const float* J, const float* h, curandState* states,
                                int n_pbits, float beta, int i) {
    float I = h[i];

    for (int j = 0; j < n_pbits; j++) {
        I += J[i * n_pbits + j] * m[j];
    }

    // range = (0, 1]
    float r = curand_uniform(&states[i]);

    
    float value = tanhf(beta * I) - (1.0f - 2.0f * r); // (0, 1] -> (-1, 1]

    // sign
    m[i] = (value >= 0.0f) ? 1 : -1;
}

int main() {
    const int n_pbits = 3;
    const float beta = 1.0f;

    // host - cpu / device = gpu
    
    // data in host

    // AND gate
    float h_J[9] = {
         0, -1,  2,
        -1,  0,  2,
         2,  2,  0
    };

    float h_h[3] = {1, 1, -2};

    int h_m[3] = {-1, -1, -1};

    // device pointers
    float* d_J;
    float* d_h;
    int* d_m;
    curandState* d_states;

    // allocate GPU memory
    cudaMalloc(&d_J, n_pbits * n_pbits * sizeof(float));
    cudaMalloc(&d_h, n_pbits * sizeof(float));
    cudaMalloc(&d_m, n_pbits * sizeof(int));
    cudaMalloc(&d_states, n_pbits * sizeof(curandState));

    // copy host -> device
    cudaMemcpy(d_J, h_J, n_pbits * n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, n_pbits * sizeof(int), cudaMemcpyHostToDevice);

    // initialize rng with seed
    init_rng<<<1, 3>>>(d_states, 1234, n_pbits);

    cudaDeviceSynchronize();

    
    int NT = 10;

    for (int i = 0; i < NT; i++) { // num of states
        for (int j = 0; j < n_pbits; j++) { // within each state, update each pbit
            update_one_pbit<<<1, 1>>>(d_m, d_J, d_h, d_states, n_pbits, beta, j);
        }
        // print each state - copy back to cpu - slow!
        cudaMemcpy(h_m, d_m, n_pbits * sizeof(int), cudaMemcpyDeviceToHost);
        printf("state %d: [%d, %d, %d]\n", i, h_m[0], h_m[1], h_m[2]);
    }

    // copy result back GPU -> CPU

    // cudaMemcpy(h_m, d_m, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("m = [%d, %d, %d]\n", h_m[0], h_m[1], h_m[2]);


    cudaFree(d_J);
    cudaFree(d_h);
    cudaFree(d_m);
    cudaFree(d_states);

    return 0;
}

