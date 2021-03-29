//
// For intellisense
//
#ifdef __INTELLISENSE__
#define __CUDACC__
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#endif

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>

#define N_PER_B 8

using cu_prec = double;
// using cu_prec = float;

//
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
//
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

//
// Test function in 2D
// cos(x) * exp( y );
//
void f_xyz(std::vector<cu_prec> &x,
           std::vector<cu_prec> &y,
           std::vector<cu_prec> &z,
           std::vector<cu_prec> &f)
{
    for (std::size_t i = 0; i < x.size(); i++)
    {
        f[i] = std::cos(x[i]) * std::exp(y[i]) * std::exp(z[i]);
    }

    return;
}

cu_prec f_laplacian(cu_prec x, cu_prec y, cu_prec z)
{
    return std::exp(y + z) * std::cos(x);
}

//
// Finite Difference in the CPU
//
void fd_3d_cpu(int nx)
{
    int ny = nx;
    int nz = nx;
    int n_tot = nx * ny * nz;

    std::vector<cu_prec> x(n_tot);
    std::vector<cu_prec> y(n_tot);
    std::vector<cu_prec> z(n_tot);
    std::vector<cu_prec> f(n_tot);
    std::vector<cu_prec> f_d2(n_tot);

    cu_prec h2 = std::pow(2.0 / (nx - 1.0), 2);

    //
    // Initialize the values of x and y
    //
    for (int k = 0; k < nz; k++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                int ell = k * nx * ny + j * nx + i;

                x[ell] = -1.0 + 2.0 * i / (nx - 1.0);
                y[ell] = -1.0 + 2.0 * j / (ny - 1.0);
                z[ell] = -1.0 + 2.0 * k / (nz - 1.0);
            }
        }
    }

    f_xyz(x, y, z, f);

    double t1 = omp_get_wtime();

    for (int k = 1; k < nz - 1; k++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = 1; i < nx - 1; i++)
            {
                int ell = k * nx * ny + j * nx + i;

                int ell_i_p1 = k * nx * ny + j * nx + (i + 1); // i+1
                int ell_i_m1 = k * nx * ny + j * nx + (i - 1); // i-1

                int ell_j_p1 = k * nx * ny + (j + 1) * nx + i; // j+1
                int ell_j_m1 = k * nx * ny + (j - 1) * nx + i; // j-1

                int ell_k_p1 = (k + 1) * nx * ny + j * nx + i; // k+1
                int ell_k_m1 = (k - 1) * nx * ny + j * nx + i; // k-1

                f_d2[ell] = ((f[ell_i_p1] + f[ell_i_m1]) +
                             (f[ell_j_p1] + f[ell_j_m1]) +
                             (f[ell_k_p1] + f[ell_k_m1]) -
                             6.0 * f[ell]) /
                            h2;
            }
        }
    }

    double t2 = omp_get_wtime();
    std::cout << "Time for 3D Laplacian in CPU = " << t2 - t1 << " sec" << std::endl;

    //
    // Check for the maximum error
    //
    cu_prec max_err = 0.0;

    for (int k = 1; k < nz - 1; k++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = 1; i < nx - 1; i++)
            {
                int ell = k * nx * ny + j * nx + i;

                cu_prec lap_exact = f_laplacian(x[ell], y[ell], z[ell]);

                cu_prec err_tmp = std::abs(lap_exact - f_d2[ell]);

                // std::cout << f_d2[ell] - lap_exact << std::endl;

                if ((i == 1) && (j == 1) && (k == 1))
                {
                    max_err = err_tmp;
                }
                else if (err_tmp > max_err)
                {
                    max_err = err_tmp;
                }
            }
        }
    }
    std::cout << "Maximum error in CPU = " << max_err << std::endl;

    return;
}

__global__ void laplacian_shared(cu_prec *f,
                                 cu_prec *f_d2,
                                 int n,
                                 cu_prec h2_inv)
{
    __shared__ cu_prec f_shared[N_PER_B + 2][N_PER_B + 2][N_PER_B + 2];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int ell = k * n * n + j * n + i;

    // printf("Blk: (%d,%d) Thread: (%d,%d) -> Row/Col = (%d,%d)\n",
    //             blockIdx.x, blockIdx.y,
    //             threadIdx.x, threadIdx.y,
    //             i, j);

    // if (ell > nx_cu * ny_cu)
    //     return;

    //
    // Fill the local memory data
    //
    f_shared[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] = f[ell];

    // If we are on an edge on the block we add the values
    if (threadIdx.x == 0 && i != 0)
    {
        int ell_tmp = k * n * n + j * n + (i - 1);
        f_shared[threadIdx.z + 1][threadIdx.y + 1][0] = f[ell_tmp];
    }

    if (threadIdx.x == (blockDim.x - 1) && i != (n - 1))
    {
        int ell_tmp = k * n * n + j * n + (i + 1);
        f_shared[threadIdx.z + 1][threadIdx.y + 1][N_PER_B + 1] = f[ell_tmp];
    }

    if (threadIdx.y == 0 && j != 0)
    {
        int ell_tmp = k * n * n + (j - 1) * n + i;
        f_shared[threadIdx.z + 1][0][threadIdx.x + 1] = f[ell_tmp];
    }

    if (threadIdx.y == (blockDim.y - 1) && j != (n - 1))
    {
        int ell_tmp = k * n * n + (j + 1) * n + i;
        f_shared[threadIdx.z + 1][N_PER_B + 1][threadIdx.x + 1] = f[ell_tmp];
    }

    if (threadIdx.z == 0 && k != 0)
    {
        int ell_tmp = (k - 1) * n * n + j * n + i;
        f_shared[0][threadIdx.y + 1][threadIdx.x + 1] = f[ell_tmp];
    }

    if (threadIdx.z == (blockDim.z - 1) && k != (n - 1))
    {
        int ell_tmp = (k + 1) * n * n + j * n + i;
        f_shared[N_PER_B + 1][threadIdx.y + 1][threadIdx.x + 1] = f[ell_tmp];
    }

    __syncthreads();

    int i_m = threadIdx.x;
    int i_0 = threadIdx.x + 1;
    int i_p = threadIdx.x + 2;

    int j_m = threadIdx.y;
    int j_0 = threadIdx.y + 1;
    int j_p = threadIdx.y + 2;

    int k_m = threadIdx.z;
    int k_0 = threadIdx.z + 1;
    int k_p = threadIdx.z + 2;

    f_d2[ell] = f_shared[k_0][j_0][i_p] + f_shared[k_0][j_0][i_m] +
                f_shared[k_0][j_p][i_0] + f_shared[k_0][j_m][i_0] +
                f_shared[k_p][j_0][i_0] + f_shared[k_m][j_0][i_0] -
                6.0 * f_shared[k_0][j_0][i_0];

    f_d2[ell] *= h2_inv;

    // f_d2[ell] = h2_inv * ((f_shared[k_0][j_0][i_p] + f_shared[k_0][j_0][i_m]) +
    //                       (f_shared[k_0][j_p][i_0] + f_shared[k_0][j_m][i_0]) +
    //                       (f_shared[k_p][j_0][i_0] + f_shared[k_m][j_0][i_0]) -
    //                       6.0 * f_shared[k_0][j_0][i_0]);
}

__global__ void laplacian_direct(cu_prec *f,
                                 cu_prec *f_d2,
                                 int n,
                                 cu_prec h2_inv)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int ell = k * n * n + j * n + i;

    // if (ell > n * n * n)
    //     return;

    if (i == 0 || j == 0 || k == 0 || i == n - 1 || j == n - 1 || k == n - 1)
        return;

    int ell_i_p1 = k * n * n + j * n + (i + 1); // i+1
    int ell_i_m1 = k * n * n + j * n + (i - 1); // i-1

    int ell_j_p1 = k * n * n + (j + 1) * n + i; // j+1
    int ell_j_m1 = k * n * n + (j - 1) * n + i; // j-1

    int ell_k_p1 = (k + 1) * n * n + j * n + i; // k+1
    int ell_k_m1 = (k - 1) * n * n + j * n + i; // k-1

    f_d2[ell] = f[ell_i_p1] + f[ell_i_m1] +
                f[ell_j_p1] + f[ell_j_m1] +
                f[ell_k_p1] + f[ell_k_m1] -
                6.0 * f[ell];

    f_d2[ell] *= h2_inv;
}

void fd_3d_gpu(int nx)
{
    int ny = nx;
    int nz = nx;
    int n_tot = nx * ny * nz;

    std::vector<cu_prec> x(n_tot);
    std::vector<cu_prec> y(n_tot);
    std::vector<cu_prec> z(n_tot);
    std::vector<cu_prec> f(n_tot);
    std::vector<cu_prec> f_d2(n_tot);

    cu_prec h2 = std::pow(2.0 / (nx - 1.0), 2);
    cu_prec h2_inv = 1.0 / h2;

    //
    // Initialize the values of x and y
    //
    for (int k = 0; k < nz; k++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                int ell = k * nx * ny + j * nx + i;

                x[ell] = -1.0 + 2.0 * i / (nx - 1.0);
                y[ell] = -1.0 + 2.0 * j / (ny - 1.0);
                z[ell] = -1.0 + 2.0 * k / (nz - 1.0);
            }
        }
    }

    f_xyz(x, y, z, f);

    double t1 = omp_get_wtime();

    cu_prec *f_cu, *f_d2_cu;

    int bytes = nx * ny * nz * sizeof(cu_prec);

    checkCuda(cudaMalloc(&f_cu, bytes));
    checkCuda(cudaMalloc(&f_d2_cu, bytes));

    checkCuda(cudaMemset(f_d2_cu, 0, bytes));

    checkCuda(cudaMemcpy(f_cu, &(f[0]), bytes, cudaMemcpyHostToDevice));


    // int n_th_pb  = 4; // Threads per Block
    // int n_blocks = 4; // # of Blocks

    // cuda_kernel<<<n_blocks, n_th_pb>>>(); // Launch CUDA Kernel


    // dim3 block = dim3(16, 16); // 16 x 16 total threads per block
    // dim3 grid = dim3(3, 2);    // 3 x 2 grid of thread blocks

    // cuda_kernel<<<grid, block>>>(); // Launch CUDA Kernel



    dim3 block = dim3(N_PER_B, N_PER_B, N_PER_B);
    dim3 grid = dim3(nz / N_PER_B, ny / N_PER_B, nx / N_PER_B);

    const int nReps = 0;

    // Use nvproof to profile

    laplacian_direct<<<grid, block>>>(f_cu, f_d2_cu, nx, h2_inv); // Warm up

    for (int i = 0; i < nReps; i++)
        laplacian_direct<<<grid, block>>>(f_cu, f_d2_cu, nx, h2_inv);

    // laplacian_shared<<<grid, block>>>(f_cu, f_d2_cu, nx, h2_inv); // Warm up

    // for (int i = 0; i < nReps; i++)
    //     laplacian_shared<<<grid, block>>>(f_cu, f_d2_cu, nx, h2_inv);

    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(&(f_d2[0]), f_d2_cu, bytes, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(f_cu));
    checkCuda(cudaFree(f_d2_cu));

    double t2 = omp_get_wtime();
    std::cout << "Time for 3D Laplacian in GPU = " << t2 - t1 << " sec" << std::endl;

    //
    // Check for the maximum error
    //
    cu_prec max_err = 0.0;

    for (int k = 1; k < nz - 1; k++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = 1; i < nx - 1; i++)
            {
                int ell = k * nx * ny + j * nx + i;

                cu_prec lap_exact = f_laplacian(x[ell], y[ell], z[ell]);

                cu_prec err_tmp = std::abs(lap_exact - f_d2[ell]);

                // std::cout << f_d2[ell] - lap_exact << std::endl;

                if ((i == 1) && (j == 1) && (k == 1))
                {
                    max_err = err_tmp;
                }
                else if (err_tmp > max_err)
                {
                    max_err = err_tmp;
                }
            }
        }
    }
    std::cout << "Maximum error in GPU = " << max_err << std::endl;

    return;
}

int main(void)
{

    int nx = N_PER_B * 70;

    fd_3d_cpu(nx);

    fd_3d_gpu(nx);

    return 0;
}