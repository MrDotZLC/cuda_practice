#include "error.cuh"
#include <cmath>
#include <cstdio>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);
__device__ __managed__ int ret[1000];
__global__ void AplusB(int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}

int main(int argc, char *argv[]) {
    // 使用动态统一内存
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x, *y, *z;
    CHECK_CUDA(cudaMallocManaged((void **)&x, M));
    CHECK_CUDA(cudaMallocManaged((void **)&y, M));
    CHECK_CUDA(cudaMallocManaged((void **)&z, M));

    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }

    const int block_size = 128;
    const int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(x, y, z);

    CHECK_CUDA(cudaDeviceSynchronize());
    check(z, N);

    CHECK_CUDA(cudaFree(x));
    CHECK_CUDA(cudaFree(y));
    CHECK_CUDA(cudaFree(z));

    // 使用静态统一内存
    AplusB<<<1, 1000>>>(10, 100);
    cudaDeviceSynchronize();
    for (int i = 0; i < 1000; i++) {
        printf("%d: A+B = %d\n", i, ret[i]);
    }
    return 0;
}

void __global__ add(const double *x, const double *y, double *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}