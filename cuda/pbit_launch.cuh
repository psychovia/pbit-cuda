// Automatic choice of threads per spin (p) and threads per block (B):
// minimize the number of waves per sweep, following pbit_cuda.pdf.
#pragma once
#include <algorithm>
#include <cstdio>
#include <vector>
#include "pbit_kernels.cuh"

// Every update kernel has the same signature, so one pointer type covers all of them.
using UpdateKernel = void (*)(int*, const float*, const float*, RngState*,
                              const int*, int, int, float);

UpdateKernel get_update_kernel(int threads_per_spin) {
    if (threads_per_spin == 1) return update_pbit_execution<1>;
    if (threads_per_spin == 2) return update_pbit_execution<2>;
    if (threads_per_spin == 4) return update_pbit_execution<4>;
    if (threads_per_spin == 8) return update_pbit_execution<8>;
    if (threads_per_spin == 16) return update_pbit_execution<16>;
    if (threads_per_spin == 32) return update_pbit_execution<32>;
    if (threads_per_spin == 64) return update_pbit_multiwarp<64>;
    if (threads_per_spin == 128) return update_pbit_multiwarp<128>;
    if (threads_per_spin == 256) return update_pbit_multiwarp<256>;
    if (threads_per_spin == 512) return update_pbit_multiwarp<512>;
    return update_pbit_multiwarp<1024>;
}

struct LaunchConfig {
    UpdateKernel kernel = nullptr;
    int threads = 0;                 // B: threads per block
    int threads_per_spin = 0;        // p: threads per spin
    int sm_count = 0;                // S: number of SMs
    int resident_blocks_per_sm = 0;  // R: blocks one SM can hold at once
    long long waves_per_sweep = 0;   // sum_c W_c, the quantity we minimize
    double sm_time = 0.0;            // sum_c n_SM / p, tie-break only (Eq. (6) without d/rho)
};

int ceil_div(long long n, long long d) {
    return (n + d - 1) / d;
}

// Eq. (1): G_c = ceil(n_c * p / B)
int color_blocks(const LaunchConfig& config, int n_spins) {
    return ceil_div(1LL * n_spins * config.threads_per_spin, config.threads);
}

// Eq. (3): W_c = ceil(G_c / (S*R))
int color_waves(const LaunchConfig& config, int n_blocks) {
    return ceil_div(n_blocks, config.sm_count * config.resident_blocks_per_sm);
}

void launch_update(const LaunchConfig& config, int* m, const float* J, const float* h,
                   RngState* states, const int* nodes, int count, int n,
                   float beta, cudaStream_t stream) {
    int blocks = color_blocks(config, count);
    config.kernel<<<blocks, config.threads, 0, stream>>>(m, J, h, states, nodes, count, n, beta);
}

// Try every (p, B) pair and keep the one with the fewest waves per sweep (pbit_cuda.pdf).
// Ties: smaller busiest-SM time (Eqs. (5)-(6)), then smaller B.
LaunchConfig choose_launch(const std::vector<std::vector<int>>& color_nodes,
                           int fixed_threads, int fixed_threads_per_spin,
                           bool whole_warps_only) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    LaunchConfig best;
    for (int p = 1; p <= 1024; p *= 2) {
        if (whole_warps_only && p < 32) continue;
        if (fixed_threads_per_spin > 0 && p != fixed_threads_per_spin) continue;

        UpdateKernel kernel = get_update_kernel(p);
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, kernel);
        int max_threads = std::min(prop.maxThreadsPerBlock, attr.maxThreadsPerBlock);

        // auto mode searches powers of two; --threads tries just that one block size
        std::vector<int> block_sizes;
        if (fixed_threads > 0) block_sizes.push_back(fixed_threads);
        else for (int B = p; B <= max_threads; B *= 2) block_sizes.push_back(B);

        for (int B : block_sizes) {
            if (B > max_threads || B % p != 0) continue;

            // Eq. (2): ask CUDA how many blocks of this kernel fit on one SM
            int resident;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, B, 0);
            if (resident == 0) continue;

            LaunchConfig config;
            config.kernel = kernel;
            config.threads = B;
            config.threads_per_spin = p;
            config.sm_count = prop.multiProcessorCount;
            config.resident_blocks_per_sm = resident;

            for (const auto& nodes : color_nodes) {
                int n_spins = nodes.size();
                int n_blocks = color_blocks(config, n_spins);
                config.waves_per_sweep += color_waves(config, n_blocks);

                // Eq. (5): spins on the busiest SM; Eq. (6): its time is proportional to n_SM / p
                int n_sm = std::min(n_spins, ceil_div(n_blocks, config.sm_count) * (B / p));
                config.sm_time += double(n_sm) / p;
            }

            bool better = !best.kernel
                || config.waves_per_sweep < best.waves_per_sweep
                || (config.waves_per_sweep == best.waves_per_sweep && config.sm_time < best.sm_time)
                || (config.waves_per_sweep == best.waves_per_sweep && config.sm_time == best.sm_time
                    && config.threads < best.threads);
            if (better) best = config;
        }
    }
    return best;
}

void print_launch_json(const LaunchConfig& config,
                       const std::vector<std::vector<int>>& color_nodes) {
    printf(",\"block_size\":%d,\"threads_per_spin\":%d,\"sm_count\":%d,"
           "\"resident_blocks_per_sm\":%d,\"waves_per_sweep\":%lld,"
           "\"color_launches\":[",
           config.threads, config.threads_per_spin, config.sm_count,
           config.resident_blocks_per_sm, config.waves_per_sweep);
    for (size_t c = 0; c < color_nodes.size(); c++) {
        int n_spins = color_nodes[c].size();
        int n_blocks = color_blocks(config, n_spins);
        printf("%s{\"spins\":%d,\"blocks\":%d,\"waves\":%d}",
               c == 0 ? "" : ",", n_spins, n_blocks, color_waves(config, n_blocks));
    }
    printf("]");
}
