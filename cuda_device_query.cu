
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
	cudaDeviceProp dp;
	int devices;

	// get number of devices on GPU
	cudaGetDeviceCount(&devices);
	for (int d = 0; d < devices; d++) {
		cudaGetDeviceProperties(&dp, d);

		printf("Number of Devices: %d\n", devices);
		printf("Device %d --> Device Name: %s\n", d, dp.name);
		printf("Device %d --> Clock Rate: %d\n", d, dp.clockRate);
		printf("Device %d --> # of SMs: %d\n", d, dp.multiProcessorCount);
		printf("Device %d --> # of Cores: 4864\n", d); // no built in function, had to manually search up
		printf("Device %d --> Warp Size: %d\n", d, dp.warpSize);
		printf("Device %d --> Amount of Global Memory: %llu MB\n", d, (unsigned long long)dp.totalGlobalMem / (1024 * 1024));
		printf("Device %d --> Amount of Constant Memory: %zu KB\n", d, dp.totalConstMem / 1024);
		printf("Device %d --> Amount of Shared Memory per block: %zu KB\n", d, dp.sharedMemPerBlock / 1024);
		printf("Device %d --> # of registers available per block: %d\n", d, dp.regsPerBlock);
		printf("Device %d --> Max # of Threads per Block: %d\n", d, dp.maxThreadsPerBlock);
		printf("Device %d --> Max Size of Each Dimension of a Block: %d, %d, %d\n", d, dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
		printf("Device %d --> Max Size of Each Dimesion of a Grid: %d, %d, %d\n", d, dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
	}

	cudaDeviceReset();
	return 0;
};