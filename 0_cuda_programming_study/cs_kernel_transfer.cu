#include "error.cuh"
#include <cmath>
#include <cstdio>

const int NUM_REPEATS = 20;
const int N = 1 << 22;
const int M = sizeof(float) * N;
const int MAX_NUM_STREAMS = 64;
cudaStream_t streams[MAX_NUM_STREAMS];

void timing(const float *h_x, const float *h_y, float *h_z,
            float *d_x, float *d_y, float *d_z,
            const int num_stream
           );

int main(int argc, char *argv[])
{
    float *h_x, *h_y, *h_z;
    CHECK_CUDA(cudaMallocHost(&h_x, M));
    CHECK_CUDA(cudaMallocHost(&h_y, M));
    CHECK_CUDA(cudaMallocHost(&h_z, M));
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
        h_y[n] = 2.34;
    }

    float *d_x, *d_y, *d_z;
    CHECK_CUDA(cudaMalloc(&d_x, M));
    CHECK_CUDA(cudaMalloc(&d_y, M));
    CHECK_CUDA(cudaMalloc(&d_z, M));

    for (int i = 0; i < MAX_NUM_STREAMS; i++)
    {
        CHECK_CUDA(cudaStreamCreate(&(streams[i])));
    }

    for (int num_stream = 1; num_stream <= MAX_NUM_STREAMS; num_stream *= 2)
    {
        timing(h_x, h_y, h_z, d_x, d_y, d_z, num_stream);
    }

    for (int i = 0 ; i < MAX_NUM_STREAMS; i++)
    {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    CHECK_CUDA(cudaFreeHost(h_x));
    CHECK_CUDA(cudaFreeHost(h_y));
    CHECK_CUDA(cudaFreeHost(h_z));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_z));
    
    return 0;
}

void __global__ add(const float *x, const float *y, float *z, int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        // 轻量计算，但比单纯加法重一点
        float tmp = x[n];
        for (int i = 0; i < 50; ++i)
            tmp = tmp * 1.000001f + y[n];
        z[n] = tmp;
    }
}

void timing(const float *h_x, const float *h_y, float *h_z,
            float *d_x, float *d_y, float *d_z, 
            const int num_stream)
{
    int N1 = N / num_stream;
    int M1 = M / num_stream;

    int block_size = 128;
    int grid_size = (N1 - 1) / block_size + 1;
    
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));

        for (int i = 0; i < num_stream; i++)
        {
            int offset = i * N1;
            CHECK_CUDA(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, cudaMemcpyHostToDevice, streams[i]));
            CHECK_CUDA(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, cudaMemcpyHostToDevice, streams[i]));
            
            
            add<<<grid_size, block_size, 0, streams[i]>>>
            (d_x + offset, d_y + offset, d_z + offset, N1);

            CHECK_CUDA(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, cudaMemcpyDeviceToHost, streams[i]));
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        t_sum += elapsed_time;
        t2_sum += elapsed_time * elapsed_time;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

    }
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("%d %g\n", num_stream, t_ave);
}