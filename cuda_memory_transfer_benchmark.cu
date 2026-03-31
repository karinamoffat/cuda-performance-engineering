#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 256

int main() {
    // Calculate total size in bytes for a WIDTH x WIDTH matrix
    int size = WIDTH * WIDTH * sizeof(float);

    // Allocate pinned host memory for two matrices (h_M and h_N)
    float* h_M, * h_N;
    cudaMallocHost((void**)&h_M, size);
    cudaMallocHost((void**)&h_N, size);

    // Initialize the host matrices with random values
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_M[i] = (float)rand() / RAND_MAX;
        h_N[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory for the two matrices
    float* d_M, * d_N;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);

    // Transfer the two input matrices from host to device (setup; not timed)
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Allocate regular host memory to copy the data back from the device
    float* h_M_back = (float*)malloc(size);
    float* h_N_back = (float*)malloc(size);

    // Create CUDA events for timing the device-to-host transfer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Transfer the two matrices back from device to host
    cudaMemcpy(h_M_back, d_M, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_N_back, d_N, size, cudaMemcpyDeviceToHost);

    // Record the stop event and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float transferTime;
    cudaEventElapsedTime(&transferTime, start, stop);

    // Print the matrix size and device-to-host transfer time
    printf("Matrix Size: %d x %d, Device-to-Host Transfer Time: %.3f ms\n", WIDTH, WIDTH, transferTime);

    //Verify that the transferred data matches the original host data
    int correct = 1;
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        if (fabs(h_M[i] - h_M_back[i]) > 1e-6 || fabs(h_N[i] - h_N_back[i]) > 1e-6) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("TEST PASSED\n");
    else
        printf("TEST FAILED\n");

    // Clean up CUDA events, device memory, and host memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFreeHost(h_M);
    cudaFreeHost(h_N);
    free(h_M_back);
    free(h_N_back);

    cudaDeviceReset();
    return 0;
}
