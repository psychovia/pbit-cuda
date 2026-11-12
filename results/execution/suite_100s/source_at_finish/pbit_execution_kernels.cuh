// Execution variants; the original one-thread kernel stays in pbit_coloring_benchmark.cu.
#ifndef EXEC_RNG
#define EXEC_RNG 0
#endif
#ifndef EXEC_MATH
#define EXEC_MATH "default"
#endif
#ifndef EXEC_REGISTER_CAP
#define EXEC_REGISTER_CAP 0
#endif

#if EXEC_RNG == 1
using ExecutionRngState = curandStatePhilox4_32_10_t;
#define EXEC_RNG_NAME "philox"
#elif EXEC_RNG == 2
using ExecutionRngState = curandStateMRG32k3a_t;
#define EXEC_RNG_NAME "mrg32k3a"
#else
using ExecutionRngState = curandState;
#define EXEC_RNG_NAME "xorwow"
#endif

__global__ void init_execution_rng(ExecutionRngState* states, unsigned long seed, int n_pbits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_pbits) curand_init(seed, i, 0, &states[i]);
}

// Only J reads change; spins must always observe preceding color updates.
template<int cache_policy>
__device__ float read_coupling(const float* address) {
    if constexpr (cache_policy == 1) return __ldg(address);
    else if constexpr (cache_policy == 2) return __ldcg(address);
    else return *address;
}

template<int cache_policy>
__device__ float thread_local_field(const float* J, const float* h, const int* m, int i, int n) {
    float I = h[i];
    for (int j = 0; j < n; j++) {
        I += read_coupling<cache_policy>(J + size_t(i) * n + j) * m[j];
    }
    return I;
}

template<int cache_policy>
__device__ float warp_local_field(const float* J, const float* h, const int* m, int i, int n) {
    int lane = threadIdx.x % 32;
    float partial = 0.0f;
    for (int j = lane; j < n; j += 32) {
        partial += read_coupling<cache_policy>(J + size_t(i) * n + j) * m[j];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    return h[i] + partial;
}

template<int threads_per_spin, int cache_policy>
__global__ void update_pbit_execution(int* m, const float* J, const float* h,
                                     ExecutionRngState* states, const int* color_nodes,
                                     int n_nodes_in_color, int n_pbits, float beta) {
    int spin_in_color = blockIdx.x * (blockDim.x / threads_per_spin) + threadIdx.x / threads_per_spin;
    if (spin_in_color >= n_nodes_in_color) return;
    int i = color_nodes[spin_in_color];
    float I;
    if constexpr (threads_per_spin == 32) I = warp_local_field<cache_policy>(J, h, m, i, n_pbits);
    else I = thread_local_field<cache_policy>(J, h, m, i, n_pbits);

    if (threadIdx.x % threads_per_spin == 0) {
        float r = curand_uniform(&states[i]);
        float value = tanhf(beta * I) - (1.0f - 2.0f * r);
        m[i] = (value >= 0.0f) ? 1 : -1;
    }
}

void launch_pbit_update(int* m, const float* J, const float* h, ExecutionRngState* states,
                        const int* nodes, int count, int n, float beta, int threads,
                        int threads_per_spin, const std::string& cache, cudaStream_t stream) {
    int spins_per_block = threads / threads_per_spin;
    int blocks = (count + spins_per_block - 1) / spins_per_block;
#if EXEC_RNG == 0
    if (threads_per_spin == 1 && cache == "default") {
        update_pbit<<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        return;
    }
#endif
    if (threads_per_spin == 32) {
        if (cache == "ldg") update_pbit_execution<32, 1><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        else if (cache == "cg") update_pbit_execution<32, 2><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        else update_pbit_execution<32, 0><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
    } else {
        if (cache == "ldg") update_pbit_execution<1, 1><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        else if (cache == "cg") update_pbit_execution<1, 2><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
        else update_pbit_execution<1, 0><<<blocks, threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
    }
}
