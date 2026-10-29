// color data structure
// figure out indx - tid/bid
// tanh / sigmoid

// rng - true/pseudo? - different libraries?
// nsight - check performance - script

// ds for J/h

// COLORING ALGORITHM
// J/h - color - thread / block
// optimal color vs automation performance
    // automate coloring

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>



__global__ void init_rng(curandState* states, unsigned long seed, int n_pbits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_pbits) {
        curand_init(seed, i, 0, &states[i]);
    }
}


__global__ void update_pbit(int* m, const float* J, const float* h, curandState* states,
                            const int* colors, int color, int n_nodes_in_color, int max_color_size,
                            int n_pbits, float beta) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_nodes_in_color) {
        return;
    }

    int i = colors[color * max_color_size + tid];

    float I = h[i];

    for (int j = 0; j < n_pbits; j++) {
        I += J[i * n_pbits + j] * m[j];
    }

    // range = (0, 1]
    float r = curand_uniform(&states[i]);

    // tanh
    // float value = tanhf(beta * I) - (1.0f - 2.0f * r);
    // m[i] = (value >= 0.0f) ? 1 : -1;

    // sigmoid
    float p_plus = 1.0f / (1.0f + expf(-2.0f * beta * I));
    m[i] = (r < p_plus) ? 1 : -1;
}


int main() {
    const int width = 4;
    const int height = 4;
    const int n_pbits = width * height;
    const int n_colors = 2;
    const int max_color_size = 8;

    const float beta = 0.2f;
    int NT = 50;


    // host - cpu / device = gpu

    // calloc?
    float* h_J = (float*)calloc(n_pbits * n_pbits, sizeof(float));
    float* h_h = (float*)calloc(n_pbits, sizeof(float));

    //  0 -- 1 -- 2 -- 3
    //  |    |    |    |
    //  4 -- 5 -- 6 -- 7
    //  |    |    |    |
    //  8 -- 9 --10 --11
    //  |    |    |    |
    // 12 --13 --14 --15

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int i = row * width + col;

            if (row > 0) {
                int j = (row - 1) * width + col;
                h_J[i * n_pbits + j] = 1.0f;
            }

            if (row < height - 1) {
                int j = (row + 1) * width + col;
                h_J[i * n_pbits + j] = 1.0f;
            }

            if (col > 0) {
                int j = row * width + (col - 1);
                h_J[i * n_pbits + j] = 1.0f;
            }

            if (col < width - 1) {
                int j = row * width + (col + 1);
                h_J[i * n_pbits + j] = 1.0f;
            }
        }
    }

    int h_m[16] = {
         1, -1,  1, -1,
        -1,  1, -1,  1,
         1,  1, -1, -1,
        -1, -1,  1,  1
    };


    // color 0:
    //  X . X .
    //  . X . X
    //  X . X .
    //  . X . X
    //
    // color 1:
    //  . X . X
    //  X . X .
    //  . X . X
    //  X . X .

    int h_colors[2][8] = {
        {0, 2, 5, 7, 8, 10, 13, 15},
        {1, 3, 4, 6, 9, 11, 12, 14}
    };

    int color_sizes[2] = {8, 8};


    // device pointers

    float* d_J;
    float* d_h;
    int* d_m;
    int* d_colors; 
    int* d_history;
    curandState* d_states;
    // curandStatePhilox4_32_10_t* d_states;

    cudaMalloc(&d_J, n_pbits * n_pbits * sizeof(float));
    cudaMalloc(&d_h, n_pbits * sizeof(float));
    cudaMalloc(&d_m, n_pbits * sizeof(int));
    cudaMalloc(&d_states, n_pbits * sizeof(curandState));
    cudaMalloc(&d_colors, n_colors * max_color_size * sizeof(int));
    cudaMalloc(&d_history, NT * n_pbits * sizeof(int));

    cudaMemcpy(d_J, h_J, n_pbits * n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, n_pbits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, n_pbits * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, h_colors, n_colors * max_color_size * sizeof(int), cudaMemcpyHostToDevice);


    // -------------------------
    // initialize rng
    // -------------------------

    // if workload / n_pbits is very large
    // want to make sure N = n_pbits//threads + 1
    
    // int threads = xxx;
    // int blocks = (n_pbits + threads - 1) / threads;

    init_rng<<<1, n_pbits>>>(d_states, 1234, n_pbits);

    cudaDeviceSynchronize();



    for (int i = 0; i < NT; i++) {
        for (int color = 0; color < n_colors; color++) {

            int n_nodes_in_color = color_sizes[color];

            // similar as above

            // int color_threads = 256;
            // int color_blocks = (n_nodes_in_color + color_threads - 1) / color_threads;

            update_pbit<<<1, n_nodes_in_color>>>(d_m, d_J, d_h, d_states,
                                                         d_colors, color, n_nodes_in_color, max_color_size,
                                                         n_pbits, beta);
        }

        // add current state to history
        cudaMemcpyAsync(d_history + i * n_pbits, d_m, n_pbits * sizeof(int), cudaMemcpyDeviceToDevice);
    }


    cudaDeviceSynchronize();


    int* h_history = (int*)malloc(NT * n_pbits * sizeof(int)); // on cpu

    cudaMemcpy(h_history, d_history, NT * n_pbits * sizeof(int), cudaMemcpyDeviceToHost);


    // print

    for (int ii = 0; ii < NT; ii++) {
        printf("state %d:\n", ii);

        // CHANGED: print as 4x4 grid instead of flat vector
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int j = row * width + col;
                printf("%2d ", h_history[ii * n_pbits + j]);
            }

            printf("\n");
        }

        printf("\n");
    }


    free(h_J);
    free(h_h);
    free(h_history);

    cudaFree(d_J);
    cudaFree(d_h);
    cudaFree(d_m);
    cudaFree(d_states);
    cudaFree(d_colors);
    cudaFree(d_history);

    return 0;
}