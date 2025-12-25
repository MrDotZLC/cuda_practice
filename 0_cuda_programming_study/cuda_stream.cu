#include "error.cuh"
#include <cstdio>

const int NUM_REPEATS = 10;
const int N1 = 128;
const int MAX_NUM_STREAMS = 100;
const int N = N1 * MAX_NUM_STREAMS;
const int M = sizeof(float) * N;
const int block_size = 128;
const int grid_size = (N1 - 1) / block_size + 1;
cudaStream_t streams[MAX_NUM_STREAMS];

void timing(const float *d_x, const float *d_y, float *d_z, const int num_stream);
void __global__ add(const float *d_x, const float *d_y, float *d_z);

int main(int argc, char *argv[])
{
    float *h_x = (float *) malloc(M);
    float *h_y = (float *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
        h_y[n] = 2.34;
    }
    float *d_x, *d_y, *d_z;
    CHECK_CUDA(cudaMalloc(&d_x, M));
    CHECK_CUDA(cudaMalloc(&d_y, M));
    CHECK_CUDA(cudaMalloc(&d_z, M));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    for (int n = 0; n < MAX_NUM_STREAMS; ++n) {
        CHECK_CUDA(cudaStreamCreate(&streams[n]));
    }
    for (int num = 1; num <= MAX_NUM_STREAMS; ++num) {
        timing(d_x, d_y, d_z, num);
    }
    for (int n = 0; n < MAX_NUM_STREAMS; ++n) {
        CHECK_CUDA(cudaStreamDestroy(streams[n]));
    }

    free(h_x);
    free(h_y);
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_z));
    return 0;
}

void timing(const float *d_x, const float *d_y, float *d_z, const int num_stream) {
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));

        for (int n = 0; n < num_stream; ++n) {
            int offset = n * N1;
            add<<<grid_size, block_size, 0, streams[n]>>>(
                d_x + offset, d_y + offset, d_z + offset
            );
        }

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

        if (repeat > 0) {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    const float t_avg = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_avg * t_avg); 
    printf("num_stearm : %d, avg : %g, err : %g\n", num_stream, t_avg, t_err);
}

void __global__ add(const float *d_x, const float *d_y, float *d_z) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N1) {
        for (int i = 0; i < 10000; ++i) {
            d_z[id] = d_x[id] + d_y[id];
        }
    }
}