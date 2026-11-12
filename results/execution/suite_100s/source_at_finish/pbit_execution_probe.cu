// Standalone validation probe, never called during a benchmark.
#define main legacy_simulator_main
#include "pbit_coloring_benchmark.cu"
#undef main
#include "pbit_execution_kernels.cuh"

__global__ void probe_fields(const float* J, const float* h, const int* m, float* fields, int n) {
    int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (i >= n) return;
    float warp = warp_local_field<0>(J, h, m, i, n);
    if (threadIdx.x % 32 == 0) {
        fields[2 * i] = thread_local_field<0>(J, h, m, i, n);
        fields[2 * i + 1] = warp;
    }
}

__global__ void probe_distribution(const float* fields, ExecutionRngState* states, float* probabilities, int n, int samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int positive = 0;
    for (int s = 0; s < samples; s++) {
        float r = curand_uniform(&states[i]);
        float value = tanhf(0.1f * fields[2 * i + 1]) - (1.0f - 2.0f * r);
        positive += value >= 0.0f;
    }
    probabilities[i] = float(positive) / samples;
}

int main(int argc, char** argv) {
    int n = atoi(argv[2]);
    std::vector<float> J(size_t(n) * n), h(n), fields(2 * n), probabilities(n);
    std::vector<int> m(n);
    for (int i = 0; i < n; i++) m[i] = i % 2 ? 1 : -1;
    FILE* input = fopen(argv[1], "rb");
    fread(J.data(), sizeof(float), J.size(), input);
    fread(h.data(), sizeof(float), h.size(), input);
    fclose(input);
    float *d_J, *d_h, *d_fields, *d_probabilities;
    int* d_m;
    ExecutionRngState* states;
    cudaMalloc(&d_J, J.size() * sizeof(float));
    cudaMalloc(&d_h, h.size() * sizeof(float));
    cudaMalloc(&d_m, n * sizeof(int));
    cudaMalloc(&d_fields, fields.size() * sizeof(float));
    cudaMalloc(&d_probabilities, probabilities.size() * sizeof(float));
    cudaMalloc(&states, n * sizeof(ExecutionRngState));
    cudaMemcpy(d_J, J.data(), J.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    init_execution_rng<<<(n + 127) / 128, 128>>>(states, 1234, n);
    probe_fields<<<(n + 3) / 4, 128>>>(d_J, d_h, d_m, d_fields, n);
    probe_distribution<<<(n + 127) / 128, 128>>>(d_fields, states, d_probabilities, n, 8192);
    cudaMemcpy(fields.data(), d_fields, fields.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(probabilities.data(), d_probabilities, probabilities.size() * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* output = fopen(argv[3], "wb");
    fwrite(fields.data(), sizeof(float), fields.size(), output);
    fwrite(probabilities.data(), sizeof(float), probabilities.size(), output);
    fclose(output);
    cudaFree(d_J); cudaFree(d_h); cudaFree(d_m); cudaFree(d_fields); cudaFree(d_probabilities); cudaFree(states);
    return 0;
}
