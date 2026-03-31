#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <math.h>

#include <stdio.h>



#define TILE_WIDTH 16


__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; //scope is block, into shared memory
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y; //scope is thread, into registers
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty; // identify the row index and the column index
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	// loop over the M and N tiles required to compute the P element

	for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
		// collaborative loading of  M and N tiles into shared memory
		Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
		Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	}
	P[Row * Width + Col] = Pvalue; //all threads write to their P element
}

void MatrixMulDevice(const float* h_M, const float* h_N, float* h_P, int Width)
{
	// Allocate device memory
	size_t size = Width * Width * sizeof(float);
	float* d_M, * d_N, * d_P;

	cudaMalloc(&d_M, size);
	cudaMalloc(&d_N, size);
	cudaMalloc(&d_P, size);

	// Copy input data from host to device
	cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

	// Define execution configuration
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);

	// Create CUDA events to time the kernel (ignoring host-device transfer)
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record the start event
	cudaEventRecord(start, 0);

	// Launch the kernel
	MatrixMulKernel << <dimGrid, dimBlock >> > (d_M, d_N, d_P, Width);

	// Record the stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate the elapsed time
	float kernelTime = 0.0f;
	cudaEventElapsedTime(&kernelTime, start, stop);
	printf("GPU kernel time: %f ms\n", kernelTime);

	// Destroy events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy result back to host
	cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
}


int main(int argc, char** argv) {

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
	int totalThreadsAtOnce = prop.multiProcessorCount * maxThreadsPerSM;
	printf("Max Threads Per Streaming Multiprocessor: %d\n", maxThreadsPerSM);
	printf("Number of Multiprocessors: %d\n", prop.multiProcessorCount);
	printf("The Total Number of Threads that can Run Simultaenously is %d", totalThreadsAtOnce);

	int blockSize = TILE_WIDTH * TILE_WIDTH;

	int maxBlocksPerSM_kernel;

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&maxBlocksPerSM_kernel,
		MatrixMulKernel,
		blockSize,
		2048
	);

	int totalThreads = maxBlocksPerSM_kernel * blockSize;
	int totalThreadsGPU = blockSize * maxBlocksPerSM_kernel * prop.multiProcessorCount;

	printf("Max active blocks per SM: %d\n", maxBlocksPerSM_kernel);
	printf("Max total threads per SM: %d\n", totalThreadsGPU);

	int Width = 256;
	size_t size = Width * Width * sizeof(float);

	float* h_M;
	float* h_N;
	float* h_P;
	float* h_P_ref;

	cudaMallocHost((void**)&h_M, size);
	cudaMallocHost((void**)&h_N, size);
	cudaMallocHost((void**)&h_P, size);
	cudaMallocHost((void**)&h_P_ref, size);

	for (int i = 0; i < Width; i++) {
		for (int j = 0; j < Width; j++) {
			h_M[i * Width + j] = (float)rand() / RAND_MAX * 100;
			h_N[i * Width + j] = (float)rand() / RAND_MAX * 100;
			h_P[i * Width + j] = (float)rand() / RAND_MAX * 100;
			h_P_ref[i * Width + j] = 0;
		}
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int i = 0; i < Width; i++) {
		for (int j = 0; j < Width; j++) {
			float sum = 0.0f;
			for (int k = 0; k < Width; k++) {
				sum += h_M[i * Width + k] * h_N[k * Width + j];
			}
			h_P_ref[i * Width + j] = sum;
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float CPUTime = 0.0f;
	cudaEventElapsedTime(&CPUTime, start, stop);
	printf("CPU time: %f ms\n", CPUTime);

	MatrixMulDevice(h_M, h_N, h_P, Width);

	int if_correct = 1;
	for (int i = 0; i < Width * Width; i++) {
		if (fabs(h_P_ref[i] - h_P[i]) > 1e-3) {
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
