//
// For intellisense, it has to be on top
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

using cu_prec = double;

// Constant coefficients for the numerical laplacian
__constant__ cu_prec alpha_a, alpha_b;
__constant__ int nx_cu, ny_cu;

// Propagation of one time step for the wave equation
__global__ void wave_eq_propagate(cu_prec *u_n,
                                  cu_prec *u_n_p1,
                                  cu_prec *u_n_m1)
{
    // Global thread number
    const int ell = blockIdx.x * blockDim.x + threadIdx.x;

    if (ell > nx_cu * ny_cu)
        return;

    int i = ell / ny_cu;     // x-index
    int j = ell - i * ny_cu; // y-index

    if (i == 0 || i == nx_cu - 1 || j == 0 || j == ny_cu - 1)
        return;

    int ell_i_p1 = (i + 1) * ny_cu + j; // i+1
    int ell_i_m1 = (i - 1) * ny_cu + j; // i-1

    int ell_j_p1 = i * ny_cu + (j + 1); // i+1
    int ell_j_m1 = i * ny_cu + (j - 1); // i-1

    // Time propagation
    u_n_p1[ell] = alpha_a * u_n[ell] - u_n_m1[ell] +
                  alpha_b * ((u_n[ell_i_p1] + u_n[ell_i_m1]) +
                             (u_n[ell_j_p1] + u_n[ell_j_m1]));
    // Set previous solution
    u_n_m1[ell] = u_n[ell];
}

// Set the new solution & boundary conditions
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

void solve_wave_eq_gpu(int n, int n_steps)
{

    cu_prec c_wv = 1.0;
    cu_prec dx = 2.0 / (n - 1.0);
    cu_prec dt = 0.5 * dx / c_wv; // To meet the CFL condition
    cu_prec bb = std::pow(c_wv * dt / dx, 2);
    cu_prec aa = 2.0 - 4.0 * bb;

    int nx = n;
    int ny = n;

    std::vector<cu_prec> x(n * n);
    std::vector<cu_prec> y(n * n);
    std::vector<cu_prec> u_cpu(n * n);

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

            u_cpu[ell] = dt * std::exp(-(std::pow(x[ell], 2) +
                                         std::pow(y[ell], 2)) /
                                       sigma2);
        }
    }

    cudaMemcpyToSymbol(nx_cu, &nx, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ny_cu, &ny, sizeof(int), 0, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(alpha_a, &aa, sizeof(cu_prec), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(alpha_b, &bb, sizeof(cu_prec), 0, cudaMemcpyHostToDevice);

    int n_th_per_block = 1024;
    int n_blocks = std::ceil(((double)(nx * ny)) / ((double)n_th_per_block));

    int bytes = nx * ny * sizeof(cu_prec);

    cu_prec *u_n, *u_n_p1, *u_n_m1;

    cudaMalloc(&u_n, bytes);
    cudaMalloc(&u_n_p1, bytes);
    cudaMalloc(&u_n_m1, bytes);

    cudaMemset(u_n, 0, bytes);
    cudaMemset(u_n_p1, 0, bytes);

    cudaMemcpy(u_n, &(u_cpu[0]), bytes, cudaMemcpyHostToDevice);

    // Time Propagation
    for (int t = 0; t < n_steps; t++)
    {
        wave_eq_propagate<<<n_blocks, n_th_per_block>>>(u_n, u_n_p1, u_n_m1);

        set_un<<<n_blocks, n_th_per_block>>>(u_n, u_n_p1);
    }

    cudaMemcpy(&(u_cpu[0]), u_n, bytes, cudaMemcpyDeviceToHost);

    cudaFree(u_n);
    cudaFree(u_n_p1);
    cudaFree(u_n_m1);

    return;
}

void solve_wave_eq_cpu(int n, int n_steps)
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
    }

    return;
}

int main(void)
{

    int n = 32 * 200;
    int n_steps = 100;

    double t1, t2;

    t1 = omp_get_wtime();

    solve_wave_eq_cpu(n, n_steps);

    t2 = omp_get_wtime();
    std::cout << "Time for GPU Wave Equation = " << t2 - t1 << std::endl;

    t1 = omp_get_wtime();

    solve_wave_eq_gpu(n, n_steps);

    t2 = omp_get_wtime();
    std::cout << "Time for GPU Wave Equation = " << t2 - t1 << std::endl;

    return 0;
}