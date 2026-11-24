// RNG state type and initialization, shared by every p-bit program.
// Build options (from the Makefile): -DEXEC_RNG=0/1/2 picks xorwow/philox/mrg32k3a.
#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

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
using RngState = curandStatePhilox4_32_10_t;
#define EXEC_RNG_NAME "philox"
#elif EXEC_RNG == 2
using RngState = curandStateMRG32k3a_t;
#define EXEC_RNG_NAME "mrg32k3a"
#else
using RngState = curandState;
#define EXEC_RNG_NAME "xorwow"
#endif

// one RNG state per p-bit
__global__ void init_rng(RngState* states, unsigned long seed, int n_pbits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_pbits) curand_init(seed, i, 0, &states[i]);
}
