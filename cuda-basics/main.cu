//
// These headers should not be added since NVCC takes care, but
// for VSCode Intellisense we need them so it recognizes CUDA functions
//
#ifdef __INTELLISENSE__
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#endif

#include <stdio.h>

__global__ void hello_from_gpu()
{
    int id_thread = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from GPU thread %d\n", id_thread);
}

int main(void)
{
    int n_blocks = 2;
    int n_threads_per_block = 3;

    hello_from_gpu<<<n_blocks, n_threads_per_block>>>();

    cudaDeviceSynchronize();

    return 0;
}