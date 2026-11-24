// p-bit update kernels.
//   update_pbit                   : original one thread per spin
//   update_pbit_execution<p>      : p <= 32 threads per spin, inside one warp
//   update_pbit_multiwarp<p>      : p > 32 threads per spin, across warps
// cache_policy only changes how J is read: 0 = plain load, 1 = __ldg, 2 = __ldcg.
#pragma once
#include "pbit_rng.cuh"

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


// Only J reads change; spins must always observe preceding color updates.
template<int cache_policy>
__device__ float read_coupling(const float* address) {
    if constexpr (cache_policy == 1) return __ldg(address);
    else if constexpr (cache_policy == 2) return __ldcg(address);
    else return *address;
}

// p = 1: one thread sums the whole row
template<int cache_policy = 0>
__device__ float thread_local_field(const float* J, const float* h, const int* m, int i, int n) {
    float I = h[i];
    for (int j = 0; j < n; j++) {
        I += read_coupling<cache_policy>(J + size_t(i) * n + j) * m[j];
    }
    return I;
}

// p = 32: one warp sums the row, then shuffle-reduces
template<int cache_policy = 0>
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

// 1 < p < 32: p lanes inside one warp share the row
template<int threads_per_spin, int cache_policy = 0>
__device__ float subwarp_local_field(const float* J, const float* h, const int* m,
                                     int i, int n) {
    int warp_lane = threadIdx.x % 32;
    int lane = warp_lane % threads_per_spin;
    unsigned mask = ((1u << threads_per_spin) - 1u) << (warp_lane - lane);
    float partial = 0.0f;
    for (int j = lane; j < n; j += threads_per_spin) {
        partial += read_coupling<cache_policy>(J + size_t(i) * n + j) * m[j];
    }
    for (int offset = threads_per_spin / 2; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(mask, partial, offset, threads_per_spin);
    }
    return h[i] + partial;
}

// p <= 32: each spin lives inside one warp
template<int threads_per_spin, int cache_policy = 0>
__global__ void update_pbit_execution(int* m, const float* J, const float* h,
                                      RngState* states, const int* color_nodes,
                                      int n_nodes_in_color, int n_pbits, float beta) {
    int spin_in_color = blockIdx.x * (blockDim.x / threads_per_spin) + threadIdx.x / threads_per_spin;
    if (spin_in_color >= n_nodes_in_color) return;
    int i = color_nodes[spin_in_color];
    float I;
    if constexpr (threads_per_spin == 32) I = warp_local_field<cache_policy>(J, h, m, i, n_pbits);
    else if constexpr (threads_per_spin == 1) I = thread_local_field<cache_policy>(J, h, m, i, n_pbits);
    else I = subwarp_local_field<threads_per_spin, cache_policy>(J, h, m, i, n_pbits);

    if (threadIdx.x % threads_per_spin == 0) {
        float r = curand_uniform(&states[i]);
        float value = tanhf(beta * I) - (1.0f - 2.0f * r);
        m[i] = (value >= 0.0f) ? 1 : -1;
    }
}

// p > 32: reduce within each warp, then across that spin's warp sums
template<int threads_per_spin, int cache_policy = 0>
__global__ void update_pbit_multiwarp(int* m, const float* J, const float* h,
                                      RngState* states, const int* color_nodes,
                                      int n_nodes_in_color, int n_pbits, float beta) {
    __shared__ float warp_sums[32];  // At most 32 warps in a block.
    constexpr int warps_per_spin = threads_per_spin / 32;
    int thread = threadIdx.x;
    int warp = thread / 32;
    int lane = thread % 32;
    int lane_in_spin = thread % threads_per_spin;
    int spin_in_color = blockIdx.x * (blockDim.x / threads_per_spin) + thread / threads_per_spin;
    bool active = spin_in_color < n_nodes_in_color;
    int i = active ? color_nodes[spin_in_color] : 0;

    float partial = 0.0f;
    if (active) {
        for (int j = lane_in_spin; j < n_pbits; j += threads_per_spin) {
            partial += read_coupling<cache_policy>(J + size_t(i) * n_pbits + j) * m[j];
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    if (lane == 0) warp_sums[warp] = partial;
    // The last block can contain inactive groups, which must also reach this barrier.
    __syncthreads();

    if (active && warp % warps_per_spin == 0) {
        partial = lane < warps_per_spin ? warp_sums[warp + lane] : 0.0f;
        for (int offset = warps_per_spin / 2; offset > 0; offset /= 2) {
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }
        if (lane == 0) {
            float r = curand_uniform(&states[i]);
            float value = tanhf(beta * (h[i] + partial)) - (1.0f - 2.0f * r);
            m[i] = (value >= 0.0f) ? 1 : -1;
        }
    }
}
