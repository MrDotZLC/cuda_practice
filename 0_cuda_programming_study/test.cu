#include "error.cuh"
#include <stdio.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(float) * N;
const int BLOCK_SIZE = 128;
const int GRID_SIZE = 10240;

void timing(const float *h_x);

int main(void)
{
    float *h_x = (float *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    float *d_x;
    CHECK_CUDA(cudaMalloc(&d_x, M));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    timing(d_x);

    free(h_x);
    CHECK_CUDA(cudaFree(d_x));
    return 0;
}

void __global__ reduce_cp(const float *d_x, float *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ float s_y[];

    float y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (int n = bid * blockDim.x + tid; n < N; n += stride)
    {
        y += d_x[n];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }

    if (tid == 0)
    {
        d_y[bid] = y;
    }
}

float reduce(const float *d_x)
{
    const int ymem = sizeof(float) * GRID_SIZE;
    const int smem = sizeof(float) * BLOCK_SIZE;

    float h_y[1] = {0};
    float *d_y;
    CHECK_CUDA(cudaMalloc(&d_y, ymem));

    reduce_cp<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_x, d_y, N);
    reduce_cp<<<1, 1024, sizeof(float) * 1024>>>(d_y, d_y, GRID_SIZE);

    CHECK_CUDA(cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_y));

    return h_y[0];
}

void timing(const float *d_x)
{
    float sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x); 

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}