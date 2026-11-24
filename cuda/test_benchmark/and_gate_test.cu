#include <stdio.h>

__global__ void and_gate() {
    int i = threadIdx.x;

    int a = i / 2;
    int b = i % 2;

    printf("thread %d: %d AND %d = %d\n",
           i, a, b, a & b);
}

int main() {
    and_gate<<<1, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}