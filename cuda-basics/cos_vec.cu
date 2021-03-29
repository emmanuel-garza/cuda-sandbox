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

#include <omp.h>
#include <vector>
#include <cmath>
#include <iostream>

// For Single Precision
using cu_prec = float;
#define cu_cos cosf

#define N_REP 200

// // For Double Precision
// using cu_prec = double;
// #define cu_cos cos

// CUDA Kernel
__global__ void fill_vec(cu_prec *f, cu_prec *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // If the tread number is larger than the problem, do nothing
    if (i >= n)
        return;

    cu_prec x_local = x[i]; // Read from global memory
    cu_prec f_local = (cu_prec)0.0;

#pragma unroll
    for (int j = 0; j < N_REP; j++)
        f_local += cu_cos(j * x_local) / (j + 1);

    f[i] = f_local; // Fill the output in global memory
}

int main(void)
{
    int n = 10000000;

    double t1, t2;

    std::vector<cu_prec> f(n, 0.0), h_f(n), h_x(n);

    for (int i = 0; i < n; i++)
        h_x[i] = (cu_prec)i;

    t1 = omp_get_wtime();

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < N_REP; j++)
            f[i] += std::cos(j * h_x[i]) / (j + 1);
    }

    t2 = omp_get_wtime();
    std::cout << "Time CPU = " << t2 - t1 << " sec" << std::endl;

    // Execute the CUDA Implementation
    t1 = omp_get_wtime();

    cu_prec *d_f, *d_x; // Functions on the device

    int bytes = n * sizeof(cu_prec); // Size in memory of the arrays

    cudaMalloc(&d_f, bytes);
    cudaMalloc(&d_x, bytes);

    cudaMemcpy(d_x, &(h_x[0]), bytes, cudaMemcpyHostToDevice);

    int n_threads_per_block = 32;
    int n_blocks = std::ceil(((double)n) / ((double)n_threads_per_block));

    fill_vec<<<n_blocks, n_threads_per_block>>>(d_f, d_x, n);

    cudaMemcpy(&(h_f[0]), d_f, bytes, cudaMemcpyDeviceToHost);

    t2 = omp_get_wtime();
    std::cout << "Time GPU = " << t2 - t1 << " sec" << std::endl;

    cudaFree(d_f);
    cudaFree(d_x);

    //
    // Check for the error
    //
    cu_prec max_err;
    for (int i = 0; i < n; i++)
    {
        cu_prec err_tmp = std::abs(h_f[i] - f[i]);

        if (i == 0 || (err_tmp > max_err))
            max_err = err_tmp;
    }

    std::cout << "Error = " << max_err << std::endl;

    return 0;
}