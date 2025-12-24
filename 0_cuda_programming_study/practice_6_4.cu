#include <cstdio>
#include "error.cuh"

int main(int argc, char *argv[]) {
    int device_id = 0;
    if (argc > 1) {
        device_id = atoi(argv[1]);
    }
    CHECK_CUDA(cudaSetDevice(device_id));
    
    cudaDeviceProp device_prop;
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device_id));

    printf("Using Device %d: %s\n", device_id, device_prop.name);
    printf("  Compute Capability: %d.%d\n", device_prop.major, device_prop.minor);
    printf("  Total Global Memory: %g MB\n",
           device_prop.totalGlobalMem / (1024.0f * 1024.0f));
    printf("  constant Memory: %g KB\n", device_prop.totalConstMem / 1024.0f);
    printf("  Max Grid Size: %d x %d x %d\n", 
           device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    printf("  Max Block Size: %d x %d x %d\n", 
           device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
    printf("  SM Count: %d\n", device_prop.multiProcessorCount);
    printf("  Max shared Memory per Block: %g KB\n", device_prop.sharedMemPerBlock / 1024.0f);
    printf("  Max shared Memory per SM: %g KB\n", device_prop.sharedMemPerMultiprocessor / 1024.0f);
    printf("  Max Registers per Block: %d\n", device_prop.regsPerBlock);
    printf("  Max Registers per SM: %d\n", device_prop.regsPerMultiprocessor);
    printf("  Max Threads per Block: %d\n", device_prop.maxThreadsPerBlock);
    printf("  Max Threads per SM: %d\n", device_prop.maxThreadsPerMultiProcessor);
    printf("  Warp Size: %d\n", device_prop.warpSize);

    return 0;
}