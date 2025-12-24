#include "error.cuh"
#include <stdio.h>

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

void timing(const float *d_A, float *d_B, const int N, const int task);
__global__ void transpose1(const float *A, float *B, const int N);
__global__ void transpose2(const float *A, float *B, const int N);
void print_matrix(const int N, const float *A);

int main() {
    const int N = 1e4; // 矩阵边长
    const int N2 = N * N;
    const int M = sizeof(float *) * N2;
    float *h_A = (float *) malloc(M);
    float *h_B = (float *) malloc(M);
    for (int n = 0; n < N2; n++) {
        h_A[n] = n;
    }
    float *d_A, *d_B;
    CHECK_CUDA(cudaMalloc(&d_A, M));
    CHECK_CUDA(cudaMalloc(&d_B, M));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    printf("\ntranspose with shared memory bank conflict:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose without shared memory bank conflict:\n");
    timing(d_A, d_B, N, 2);

    CHECK_CUDA(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));
    if (N <= 10) {
        printf("A =\n");
        print_matrix(N, h_A);
        printf("\nB =\n");
        print_matrix(N, h_B);
    }

    free(h_A);
    free(h_B);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    return 0;
}

void timing(const float *d_A, float *d_B, const int N, const int task) {
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = grid_size_x;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);

    float t_sum = 0, t2_sum = 0; // 耗时和，耗时平方和
    for (int repeat = 0; repeat <= NUM_REPEATS; repeat++) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        cudaEventQuery(start);

        switch (task)
        {
            case 1:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("Error: wrong task\n");
                exit(1);
                break;
        }

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

void __global__ transpose1(const float *A, float *B, const int N) {
    __shared__ float S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;
    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;

    if (nx1 < N && ny1 < N) {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N) {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

void __global__ transpose2(const float *A, float *B, const int N) {
    __shared__ float S[TILE_DIM][TILE_DIM + 1]; // 内存块边长大于32，避免bank冲突
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;
    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;

    if (nx1 < N && ny1 < N) {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N) {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

void print_matrix(const int N, const float *A) {
    for (int ny = 0; ny < N; ny++)
    {
        for (int nx = 0; nx < N; nx++)
        {
            printf("%g\t", A[ny * N + nx]);
        }
        printf("\n");
    }
}