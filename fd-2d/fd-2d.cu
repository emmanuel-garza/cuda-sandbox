#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>

using prec = double;
using cu_prec = double;

//
// For intellisense
//
#ifdef __INTELLISENSE__
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#endif

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
void f_xy(std::vector<prec> &x, std::vector<prec> &y, std::vector<prec> &f)
{
    for (std::size_t i = 0; i < x.size(); i++)
    {
        f[i] = std::cos(x[i]) * std::exp(y[i]);
    }

    return;
}

prec f_laplacian(prec x, prec y)
{
    return 0.0;
}

void f_xy(std::vector<float> &x, std::vector<float> &y, std::vector<float> &f)
{
    for (std::size_t i = 0; i < x.size(); i++)
    {
        f[i] = std::cos(x[i]) * std::exp(y[i]);
    }

    return;
}

float f_laplacian(float x, float y)
{
    return (float)0.0;
}

//
// Finite Difference in the CPU
//
void fd_2d_cpu(int nx, int ny)
{
    std::vector<prec> x(nx * ny);
    std::vector<prec> y(nx * ny);
    std::vector<prec> f(nx * ny);
    std::vector<prec> f_d2(nx * ny);

    prec dx2 = std::pow(2.0 / (nx - 1.0), 2);
    prec dy2 = std::pow(2.0 / (ny - 1.0), 2);

    //
    // Initialize the values of x and y
    //
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int ell = i * ny + j;

            x[ell] = -1.0 + 2.0 * i / (nx - 1.0);
            y[ell] = -1.0 + 2.0 * j / (ny - 1.0);
        }
    }

    f_xy(x, y, f);

    double t1 = omp_get_wtime();

    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            int ell = i * ny + j;

            int ell_i_p1 = (i + 1) * ny + j; // i+1
            int ell_i_m1 = (i - 1) * ny + j; // i-1

            int ell_j_p1 = i * ny + (j + 1); // i+1
            int ell_j_m1 = i * ny + (j - 1); // i-1

            f_d2[ell] = (f[ell_i_p1] + f[ell_i_m1]) / dx2 +
                        (f[ell_j_p1] + f[ell_j_m1]) / dy2 -
                        (2.0 / dx2 + 2.0 / dy2) * f[ell];
        }
    }

    double t2 = omp_get_wtime();
    std::cout << "Time for 2D Laplacian in CPU = " << t2 - t1 << " sec" << std::endl;

    //
    // Check for the maximum error
    //
    prec max_err = 0.0;

    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            int ell = i * ny + j;

            prec lap_exact = f_laplacian(x[ell], y[ell]);

            prec err_tmp = std::abs(lap_exact - f_d2[ell]);

            // std::cout << f_d2[ell] << std::endl;

            if ((i == 1) && (j == 1))
            {
                max_err = err_tmp;
            }
            else if (err_tmp > max_err)
            {
                max_err = err_tmp;
            }
        }
    }
    std::cout << "Maximum error in CPU = " << max_err << std::endl;

    return;
}

//
// GPU Kernel the "silly" way
//
__constant__ int nx_cu, ny_cu;

__global__ void laplacian_silly(cu_prec *f,
                                cu_prec *f_d2)
{
    const int ell = blockIdx.x * blockDim.x + threadIdx.x;

    if (ell > nx_cu * ny_cu)
        return;

    int i = ell / ny_cu;
    int j = ell - i * ny_cu;

    if (i == 0 || i == nx_cu - 1 || j == 0 || j == ny_cu - 1)
        return;

    int ell_i_p1 = (i + 1) * ny_cu + j; // i+1
    int ell_i_m1 = (i - 1) * ny_cu + j; // i-1

    int ell_j_p1 = i * ny_cu + (j + 1); // i+1
    int ell_j_m1 = i * ny_cu + (j - 1); // i-1

    f_d2[ell] = (f[ell_i_p1] + f[ell_i_m1]) +
                (f[ell_j_p1] + f[ell_j_m1]) -
                4.0 * f[ell];

    // f_d2[ell] = (f[ell+1] + f[ell-1]) +
    //             (f[ell+1] + f[ell-1]) -
    //             4.0 * f[ell];

    // f_d2[ell] = 0.0;
}

//
// GPU Kernel using shared memory
//
// stencil coefficients
__constant__ cu_prec c_a, c_b, c_c;

const int n_per_b = 32;

__global__ void laplacian_shared(cu_prec *f,
                                 cu_prec *f_d2)
{
    __shared__ cu_prec f_shared[n_per_b + 2][n_per_b + 2];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ell = i * ny_cu + j;

    // printf("Blk: (%d,%d) Thread: (%d,%d) -> Row/Col = (%d,%d)\n",
    //             blockIdx.x, blockIdx.y,
    //             threadIdx.x, threadIdx.y,
    //             i, j);

    // if (ell > nx_cu * ny_cu)
    //     return;

    //
    // Fill the local memory data
    //
    f_shared[threadIdx.x + 1][threadIdx.y + 1] = f[ell];

    // If we are on an edge on the block we add the values
    if (threadIdx.x == 0 && i != 0)
    {
        int ell_tmp = (i - 1) * ny_cu + j;
        f_shared[0][threadIdx.y + 1] = f[ell_tmp];
    }

    if (threadIdx.x == (blockDim.x - 1) && i != (nx_cu - 1))
    {
        int ell_tmp = (i + 1) * ny_cu + j;
        f_shared[n_per_b + 1][threadIdx.y + 1] = f[ell_tmp];
    }

    if (threadIdx.y == 0 && j != 0)
    {
        int ell_tmp = i * ny_cu + (j - 1);
        f_shared[threadIdx.x + 1][0] = f[ell_tmp];
    }

    if (threadIdx.y == (blockDim.y - 1) && j != (ny_cu - 1))
    {
        int ell_tmp = i * ny_cu + (j + 1);
        f_shared[threadIdx.x + 1][n_per_b + 1] = f[ell_tmp];
    }

    __syncthreads();

    // int i_m = threadIdx.x;
    // int i_0 = threadIdx.x + 1;
    // int i_p = threadIdx.x + 2;

    // int j_m = threadIdx.y;
    // int j_0 = threadIdx.y + 1;
    // int j_p = threadIdx.y + 2;

    // f_d2[ell] = (f_shared[i_p][j_0] + f_shared[i_m][j_0]) +
    //             (f_shared[i_0][j_p] + f_shared[i_0][j_m]) -
    //             4.0 * f_shared[i_0][j_0];

    f_d2[ell] = (f_shared[threadIdx.x + 2][threadIdx.y + 1] + f_shared[threadIdx.x][threadIdx.y + 1]) +
                (f_shared[threadIdx.x + 1][threadIdx.y + 2] + f_shared[threadIdx.x + 1][threadIdx.y]) -
                4.0 * f_shared[threadIdx.x + 1][threadIdx.y + 1];
}

//
// Finite Difference in the GPU
//
void fd_2d_gpu_v0(int nx, int ny)
{
    std::vector<cu_prec> x(nx * ny);
    std::vector<cu_prec> y(nx * ny);
    std::vector<cu_prec> f(nx * ny);
    std::vector<cu_prec> f_d2(nx * ny);

    cu_prec dx2 = (cu_prec)std::pow(2.0 / (nx - 1.0), 2);
    cu_prec dy2 = (cu_prec)std::pow(2.0 / (ny - 1.0), 2);

    //
    // Initialize the values of x and y
    //
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int ell = i * ny + j;

            x[ell] = (cu_prec)(-1.0 + 2.0 * i / (nx - 1.0));
            y[ell] = (cu_prec)(-1.0 + 2.0 * j / (ny - 1.0));
        }
    }

    f_xy(x, y, f);

    //
    // Set the CUDA variables
    //
    double t1 = omp_get_wtime();

    cu_prec *f_cu, *f_d2_cu;

    int bytes = nx * ny * sizeof(cu_prec);

    checkCuda(cudaMemcpyToSymbol(nx_cu, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(ny_cu, &ny, sizeof(int), 0, cudaMemcpyHostToDevice));

    checkCuda(cudaMalloc(&f_cu, bytes));
    checkCuda(cudaMalloc(&f_d2_cu, bytes));

    checkCuda(cudaMemset(f_d2_cu, 0, bytes));

    checkCuda(cudaMemcpy(f_cu, &(f[0]), bytes, cudaMemcpyHostToDevice));

    int n_th_per_block = 1024;
    int n_blocks = std::ceil(((prec)(nx * ny)) /
                             ((prec)n_th_per_block));

    const int nReps = 20;

    laplacian_silly<<<n_blocks, n_th_per_block>>>(f_cu, f_d2_cu); // Warm up

    for (int i = 0; i < nReps; i++)
        laplacian_silly<<<n_blocks, n_th_per_block>>>(f_cu, f_d2_cu);

    checkCuda(cudaMemcpy(&(f_d2[0]), f_d2_cu, bytes, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(f_cu));
    checkCuda(cudaFree(f_d2_cu));

    // Add dx2 factor
    for (int i = 0; i < nx * ny; i++)
    {
        f_d2[i] /= dx2;
    }

    double t2 = omp_get_wtime();

    std::cout << "Time for 2D Laplacian in GPU (v0) = " << t2 - t1 << " sec" << std::endl;

    //
    // Check for the maximum error
    //
    cu_prec max_err = 0.0;

    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            int ell = i * ny + j;

            cu_prec lap_exact = f_laplacian(x[ell], y[ell]);

            cu_prec err_tmp = std::abs(lap_exact - f_d2[ell]);

            // std::cout << f_d2[ell] << std::endl;

            if ((i == 1) && (j == 1))
            {
                max_err = err_tmp;
            }
            else if (err_tmp > max_err)
            {
                max_err = err_tmp;
            }
        }
    }
    std::cout << "Maximum error in GPU (v0) = " << max_err << std::endl;

    return;
}

//
// Finite Difference in the GPU
//
void fd_2d_gpu_v1(int nx, int ny)
{
    std::vector<cu_prec> x(nx * ny);
    std::vector<cu_prec> y(nx * ny);
    std::vector<cu_prec> f(nx * ny);
    std::vector<cu_prec> f_d2(nx * ny);

    cu_prec dx2 = std::pow(2.0 / (nx - 1.0), 2);
    cu_prec dy2 = std::pow(2.0 / (ny - 1.0), 2);

    //
    // Initialize the values of x and y
    //
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int ell = i * ny + j;

            x[ell] = -1.0 + 2.0 * i / (nx - 1.0);
            y[ell] = -1.0 + 2.0 * j / (ny - 1.0);
        }
    }

    f_xy(x, y, f);

    //
    // Set the CUDA variables
    //
    double t1 = omp_get_wtime();

    cu_prec *f_cu, *f_d2_cu;

    int bytes = nx * ny * sizeof(cu_prec);

    checkCuda(cudaMemcpyToSymbol(nx_cu, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(ny_cu, &ny, sizeof(int), 0, cudaMemcpyHostToDevice));

    checkCuda(cudaMalloc(&f_cu, bytes));
    checkCuda(cudaMalloc(&f_d2_cu, bytes));

    checkCuda(cudaMemset(f_d2_cu, 0, bytes));

    checkCuda(cudaMemcpy(f_cu, &(f[0]), bytes, cudaMemcpyHostToDevice));

    dim3 block = dim3(n_per_b, n_per_b);
    dim3 grid = dim3(nx / n_per_b, ny / n_per_b);

    // int n_th_per_block = 64;
    // int n_blocks = std::ceil(((prec)(nx * ny)) /
    //                          ((prec)n_th_per_block));

    cu_prec a = 1.0 / dx2;
    cu_prec b = 1.0 / dy2;
    cu_prec c = 4.0 / dx2;

    // checkCuda(cudaMemcpyToSymbol(c_a, &a, sizeof(cu_prec), 0, cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpyToSymbol(c_b, &b, sizeof(cu_prec), 0, cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpyToSymbol(c_c, &c, sizeof(cu_prec), 0, cudaMemcpyHostToDevice));

    const int nReps = 20;

    laplacian_shared<<<grid, block>>>(f_cu, f_d2_cu); // Warm up

    for (int i = 0; i < nReps; i++)
        laplacian_shared<<<grid, block>>>(f_cu, f_d2_cu);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(&(f_d2[0]), f_d2_cu, bytes, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(f_cu));
    checkCuda(cudaFree(f_d2_cu));

    // Add dx2 factor
    for (int i = 0; i < nx * ny; i++)
    {
        f_d2[i] /= dx2;
    }

    double t2 = omp_get_wtime();

    std::cout << "Time for 2D Laplacian in GPU (v1) = " << t2 - t1 << " sec" << std::endl;

    //
    // Check for the maximum error
    //
    cu_prec max_err = 0.0;

    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            int ell = i * ny + j;

            cu_prec lap_exact = f_laplacian(x[ell], y[ell]);

            cu_prec err_tmp = std::abs(lap_exact - f_d2[ell]);

            // std::cout << f_d2[ell] << std::endl;

            if ((i == 1) && (j == 1))
            {
                max_err = err_tmp;
            }
            else if (err_tmp > max_err)
            {
                max_err = err_tmp;
            }
        }
    }
    std::cout << "Maximum error in GPU (v1) = " << max_err << std::endl;

    return;
}

//
// Wave equation
//
__constant__ cu_prec alpha2;

__global__ void wave_eq_propagate(cu_prec *u_n,
                                  cu_prec *u_n_p1,
                                  cu_prec *u_n_m1)
{
    const int ell = blockIdx.x * blockDim.x + threadIdx.x;

    if (ell > nx_cu * ny_cu)
        return;

    int i = ell / ny_cu;
    int j = ell - i * ny_cu;

    if (i == 0 || i == nx_cu - 1 || j == 0 || j == ny_cu - 1)
        return;

    int ell_i_p1 = (i + 1) * ny_cu + j; // i+1
    int ell_i_m1 = (i - 1) * ny_cu + j; // i-1

    int ell_j_p1 = i * ny_cu + (j + 1); // i+1
    int ell_j_m1 = i * ny_cu + (j - 1); // i-1

    //
    // Time propagation
    //
    cu_prec u = u_n[ell];

    u_n_p1[ell] = 2.0 * u - u_n_m1[ell] +
                  alpha2 * ((u_n[ell_i_p1] + u_n[ell_i_m1]) +
                            (u_n[ell_j_p1] + u_n[ell_j_m1]) -
                            4.0 * u);
    // Set u_n_m1
    u_n_m1[ell] = u;
}

__global__ void set_un(cu_prec *u_n,
                       cu_prec *u_n_p1)
{
    const int ell = blockIdx.x * blockDim.x + threadIdx.x;

    if (ell > nx_cu * ny_cu)
        return;

    int i = ell / ny_cu;
    int j = ell - i * ny_cu;

    // Boundary condition
    if (i == 0 || i == nx_cu - 1 || j == 0 || j == ny_cu - 1)
    {
        u_n[ell] = 0.0;
        return;
    }

    u_n[ell] = u_n_p1[ell];
}

//
// Solve the Wave Equation
//
void solve_wave_eq_gpu(int n, int n_steps, int n_animate)
{

    cu_prec c_wv = 1.0;
    cu_prec dx = 2.0 / (n - 1.0);
    cu_prec dt = 0.5 * dx / c_wv; // To meet the CFL condition
    cu_prec aa = std::pow(c_wv * dt / dx, 2);

    int nx = n;
    int ny = n;

    std::vector<cu_prec> x(n * n);
    std::vector<cu_prec> y(n * n);
    std::vector<cu_prec> u_cpu(n * n);

    int n_th_per_block = 1024;
    int n_blocks = std::ceil(((prec)(nx * ny)) /
                             ((prec)n_th_per_block));

    //
    // Set the initial derivative
    //
    cu_prec sigma2 = std::pow(0.2, 2);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int ell = i * ny + j;

            x[ell] = -1.0 + 2.0 * i / (nx - 1.0);
            y[ell] = -1.0 + 2.0 * j / (ny - 1.0);

            u_cpu[ell] = std::exp(-(std::pow(x[ell], 2) + std::pow(y[ell], 2)) / sigma2);

            u_cpu[ell] *= dt;
        }
    }

    int bytes = nx * ny * sizeof(cu_prec);

    checkCuda(cudaMemcpyToSymbol(nx_cu, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(ny_cu, &ny, sizeof(int), 0, cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpyToSymbol(alpha2, &aa, sizeof(cu_prec), 0, cudaMemcpyHostToDevice));

    cu_prec *u_n, *u_n_p1, *u_n_m1;

    checkCuda(cudaMalloc(&u_n, bytes));
    checkCuda(cudaMalloc(&u_n_p1, bytes));
    checkCuda(cudaMalloc(&u_n_m1, bytes));

    checkCuda(cudaMemset(u_n, 0, bytes));
    checkCuda(cudaMemset(u_n_p1, 0, bytes));

    checkCuda(cudaMemcpy(u_n, &(u_cpu[0]), bytes, cudaMemcpyHostToDevice));

    // int n_animate = 120;
    int aux = n_steps / n_animate;
    int c = 0;
    int skip = 10;

    for (int t = 0; t < n_steps; t++)
    {
        wave_eq_propagate<<<n_blocks, n_th_per_block>>>(u_n, u_n_p1, u_n_m1);

        set_un<<<n_blocks, n_th_per_block>>>(u_n, u_n_p1);

        if (t % aux == 0)
        {
            checkCuda(cudaMemcpy(&(u_cpu[0]), u_n, bytes, cudaMemcpyDeviceToHost));

            std::string filename = "data/results";
            c++;
            filename.append(std::to_string(c));
            filename.append(".m");

            std::ofstream file;

            file.open(filename, std::ios::out);

            file << "F = [";

            for (int i = 0; i < nx; i += skip)
            {
                if (i != 0)
                    file << ";" << std::endl;

                for (int j = 0; j < ny; j += skip)
                {
                    int ell = i * ny + j;

                    if (j != 0)
                        file << ", ";

                    file << u_cpu[ell];
                }
            }
            file << "];" << std::endl;
        }
    }

    checkCuda(cudaMemcpy(&(u_cpu[0]), u_n, bytes, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(u_n));
    checkCuda(cudaFree(u_n_p1));
    checkCuda(cudaFree(u_n_m1));

    //
    // Save text file for visualization on matlab
    //

    std::string filename = "data/results.m";
    std::ofstream file;

    file.open(filename, std::ios::out);

    file << "X = [";

    for (int i = 0; i < nx; i += skip)
    {
        if (i != 0)
            file << ";" << std::endl;

        for (int j = 0; j < ny; j += skip)
        {
            int ell = i * ny + j;

            if (j != 0)
                file << ", ";

            file << x[ell];
        }
    }
    file << "];" << std::endl;

    file << "Y = [";

    for (int i = 0; i < nx; i += skip)
    {
        if (i != 0)
            file << ";" << std::endl;

        for (int j = 0; j < ny; j += skip)
        {
            int ell = i * ny + j;

            if (j != 0)
                file << ", ";

            file << y[ell];
        }
    }

    file << "];" << std::endl;

    file << "F = [";

    for (int i = 0; i < nx; i += skip)
    {
        if (i != 0)
            file << ";" << std::endl;

        for (int j = 0; j < ny; j += skip)
        {
            int ell = i * ny + j;

            if (j != 0)
                file << ", ";
            file << u_cpu[ell];
        }
    }
    file << "];" << std::endl;

    return;
}

//
// Wave equation on CPU
//
void solve_wave_eq_cpu(int n, int n_steps, int n_animate)
{

    cu_prec c_wv = 1.0;
    cu_prec dx = 2.0 / (n - 1.0);
    cu_prec dt = 0.5 * dx / c_wv; // To meet the CFL condition
    cu_prec aa = std::pow(c_wv * dt / dx, 2);

    int nx = n;
    int ny = n;

    std::vector<cu_prec> x(n * n);
    std::vector<cu_prec> y(n * n);
    std::vector<cu_prec> u_n(n * n);
    std::vector<cu_prec> u_n_p1(n * n, 0.0);
    std::vector<cu_prec> u_n_m1(n * n, 0.0);

    //
    // Set the initial derivative
    //
    cu_prec sigma2 = std::pow(0.2, 2);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int ell = i * ny + j;

            x[ell] = -1.0 + 2.0 * i / (nx - 1.0);
            y[ell] = -1.0 + 2.0 * j / (ny - 1.0);

            u_n[ell] = std::exp(-(std::pow(x[ell], 2) + std::pow(y[ell], 2)) / sigma2);

            u_n[ell] *= dt;
        }
    }

    // int n_animate = 30;
    int aux = n_steps / n_animate;
    int c = 0;
    int skip = 10;

    for (int t = 0; t < n_steps; t++)
    {
        for (int ell = 0; ell < n * n; ell++)
        {
            int i = ell / ny;
            int j = ell - i * ny;

            int ell_i_p1 = (i + 1) * ny + j; // i+1
            int ell_i_m1 = (i - 1) * ny + j; // i-1

            int ell_j_p1 = i * ny + (j + 1); // i+1
            int ell_j_m1 = i * ny + (j - 1); // i-1

            //
            // Time propagation
            //
            cu_prec u = u_n[ell];

            u_n_p1[ell] = 2.0 * u - u_n_m1[ell] +
                          aa * ((u_n[ell_i_p1] + u_n[ell_i_m1]) +
                                (u_n[ell_j_p1] + u_n[ell_j_m1]) -
                                4.0 * u);
            u_n_m1[ell] = u_n[ell];
        }

        // Set updated result
        for (int ell = 0; ell < n * n; ell++)
        {
            int i = ell / ny;
            int j = ell - i * ny;

            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
            {
                u_n[ell] = 0.0;
                continue;
            }

            u_n[ell] = u_n_p1[ell];
        }

        if (t % aux == 0)
        {

            std::string filename = "data/results";
            c++;
            filename.append(std::to_string(c));
            filename.append(".m");

            std::ofstream file;

            file.open(filename, std::ios::out);

            file << "F = [";

            for (int i = 0; i < nx; i += skip)
            {
                if (i != 0)
                    file << ";" << std::endl;

                for (int j = 0; j < ny; j += skip)
                {
                    int ell = i * ny + j;

                    if (j != 0)
                        file << ", ";

                    file << u_n[ell];
                }
            }
            file << "];" << std::endl;
        }
    }

    //
    // Save text file for visualization on matlab
    //

    std::string filename = "data/results.m";
    std::ofstream file;

    file.open(filename, std::ios::out);

    file << "X = [";

    for (int i = 0; i < nx; i += skip)
    {
        if (i != 0)
            file << ";" << std::endl;

        for (int j = 0; j < ny; j += skip)
        {
            int ell = i * ny + j;

            if (j != 0)
                file << ", ";

            file << x[ell];
        }
    }
    file << "];" << std::endl;

    file << "Y = [";

    for (int i = 0; i < nx; i += skip)
    {
        if (i != 0)
            file << ";" << std::endl;

        for (int j = 0; j < ny; j += skip)
        {
            int ell = i * ny + j;

            if (j != 0)
                file << ", ";

            file << y[ell];
        }
    }

    file << "];" << std::endl;

    file << "F = [";

    for (int i = 0; i < nx; i += skip)
    {
        if (i != 0)
            file << ";" << std::endl;

        for (int j = 0; j < ny; j += skip)
        {
            int ell = i * ny + j;

            if (j != 0)
                file << ", ";
            file << u_n[ell];
        }
    }
    file << "];" << std::endl;

    return;
}

int main(void)
{

    int nx = n_per_b * 100;
    int ny = n_per_b * 100;

    int n_steps = 500;
    int n_animate = 30;

    double t1, t2;

    // t1 = omp_get_wtime();

    // solve_wave_eq_cpu(nx, n_steps, n_animate);
    
    // t2 = omp_get_wtime();

    // std::cout << "Time for CPU Wave Equation = " << t2-t1 << std::endl;


    t1 = omp_get_wtime();

    solve_wave_eq_gpu(nx, n_steps, n_animate);
    
    t2 = omp_get_wtime();

    std::cout << "Time for GPU Wave Equation = " << t2-t1 << std::endl;



    // solve_wave_eq(nx, n_steps);

    // fd_2d_cpu(nx, ny);

    // fd_2d_gpu_v0(nx, ny);

    // fd_2d_gpu_v1(nx, ny);

    return 0;
}