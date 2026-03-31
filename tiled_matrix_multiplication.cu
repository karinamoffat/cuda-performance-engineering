#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define TILE_WIDTH 8
#define TILE_HEIGHT 12

// CHANGE: Kernel for tiled matrix multiplication on rectangular matrices
// M: dimensions m x k, N: dimensions k x n, P: dimensions m x n
__global__ void MatrixMulKernel(float* M, float* N, float* P, int m, int k, int n) {
    __shared__ float Ms[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float Ns[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute the global row and column indices for P
    int Row = by * TILE_HEIGHT + ty;
    int Col = bx * TILE_WIDTH + tx;
    float pvalue = 0.0f;

    // The number of tiles needed to cover the k dimension
    int numTiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < numTiles; ph++) {
        // Load one element of the M tile into shared memory
        int m_col = ph * TILE_WIDTH + tx;
        if (Row < m && m_col < k) {
            Ms[ty][tx] = M[Row * k + m_col];
        }
        else {
            Ms[ty][tx] = 0.0f;
        }

        // For N, only the first TILE_WIDTH rows in the block load data
        if (ty < TILE_WIDTH) {
            int N_row = ph * TILE_WIDTH + ty;
            if (N_row < k && Col < n) {
                Ns[ty][tx] = N[N_row * n + Col];
            }
            else {
                Ns[ty][tx] = 0.0f;
            }
        }

        __syncthreads();

        // Each thread computes partial dot product using the loaded tiles
        for (int i = 0; i < TILE_WIDTH; i++) {
            pvalue += Ms[ty][i] * Ns[i][tx];
        }
        __syncthreads();
    }

    // A check if it is within the bounds
    if (Row < m && Col < n) {
        P[Row * n + Col] = pvalue;
    }
}

void MatrixMulDevice(const float* h_M, const float* h_N, float* h_P, int m, int k, int n) {
    size_t size_M = m * k * sizeof(float);
    size_t size_N = k * n * sizeof(float);
    size_t size_P = m * n * sizeof(float);
    float* d_M, * d_N, * d_P;

    cudaMalloc(&d_M, size_M);
    cudaMalloc(&d_N, size_N);
    cudaMalloc(&d_P, size_P);

    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);

    // CHANGE: Uses the new rectangular tile dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_HEIGHT - 1) / TILE_HEIGHT);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MatrixMulKernel << <dimGrid, dimBlock >> > (d_M, d_N, d_P, m, k, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("GPU kernel time: %f ms\n", kernelTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main(int argc, char** argv) {

    // CHANGE: allows matricies of differing sizes
    //M: m x k, N: k x n, P: m x n
    int m = 2000;
    int k = 1750;
    int n = 1900;

    size_t size_M = m * k * sizeof(float);
    size_t size_N = k * n * sizeof(float);
    size_t size_P = m * n * sizeof(float);
    size_t size_P_ref = m * n * sizeof(float);

    float* h_M, * h_N, * h_P, * h_P_ref;
    cudaMallocHost((void**)&h_M, size_M);
    cudaMallocHost((void**)&h_N, size_N);
    cudaMallocHost((void**)&h_P, size_P);
    cudaMallocHost((void**)&h_P_ref, size_P_ref);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            h_M[i * k + j] = (float)rand() / RAND_MAX * 100;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            h_N[i * n + j] = (float)rand() / RAND_MAX * 100;
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int t = 0; t < k; t++) {
                sum += h_M[i * k + t] * h_N[t * n + j];
            }
            h_P_ref[i * n + j] = sum;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float CPUTime = 0;
    cudaEventElapsedTime(&CPUTime, start, stop);
    printf("CPU time: %f ms\n", CPUTime);

 
    MatrixMulDevice(h_M, h_N, h_P, m, k, n);

    int if_correct = 1;
    for (int i = 0; i < m * n; i++) {
        if (fabs(h_P_ref[i] - h_P[i]) > 1) {
            if_correct = 0;
            break;
        }
    }

    if (if_correct)
        printf("Test PASSED\n");
    else if (!if_correct)
        printf("Test FAILED\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_M);
    cudaFreeHost(h_N);
    cudaFreeHost(h_P);
    cudaFreeHost(h_P_ref);
    cudaDeviceReset();
    return 0;
}



