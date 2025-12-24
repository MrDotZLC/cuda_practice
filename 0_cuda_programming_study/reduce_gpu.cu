#include "error.cuh"
#include <cstdio>
#include <numeric>
#include <cooperative_groups.h>

using namespace cooperative_groups;

const int NUM_REPEATS = 10;
const int N = 1e8;
const int M = sizeof(float) * N;
const int GRID_SIZE2 = 10240;
const int BLOCK_SIZE = 128;
const unsigned FULL_MASK = 0xffffffff;
__device__ float static_y[GRID_SIZE2];

void timing(float *h_x, float *d_x, const int method);

int main() {
    float *h_x = (float *) malloc(M);
    for (int n = 0; n < N; n++) {
        h_x[n] = 1.23;
    }
    float *d_x;
    CHECK_CUDA(cudaMalloc(&d_x, M));

    printf("\nGlobal Memory:\n");
    timing(h_x, d_x, 0); // 30ms, 123633392.000000, 精度为3位
    printf("\nShared Memory:\n");
    timing(h_x, d_x, 1); // 35ms, 123633392.000000, 精度为3位
    printf("\nDynamic Shared Memory:\n");
    timing(h_x, d_x, 2); // 35ms, 123633392.000000, 精度为3位
    printf("\nDynamic Shared Memory with AtomicAdd:\n");
    timing(h_x, d_x, 3); // 37ms, 123633392.000000, 精度为3位
    printf("\nDynamic Shared Memory with AtomicAdd, syncwarp:\n");
    timing(h_x, d_x, 4); // 45ms, 123633392.000000, 精度为3位
    printf("\nDynamic Shared Memory with AtomicAdd, shuffle:\n");
    timing(h_x, d_x, 5); // 52ms, 123633392.000000, 精度为3位
    printf("\nDynamic Shared Memory with AtomicAdd, shuffle, cooperative_group:\n");
    timing(h_x, d_x, 6); // 303ms, 123633392.000000, 精度为3位
    printf("\nDynamic Shared Memory with shuffle, cooperative_group, cross-grid:\n");
    timing(h_x, d_x, 7); // 6.6ms, 123000064.000000, 精度为7位
    printf("\nDynamic Shared Memory with shuffle, cooperative_group, cross-grid, static-memory:\n");
    timing(h_x, d_x, 8); // 6.6ms, 123000064.000000, 精度为7位

    free(h_x);
    CHECK_CUDA(cudaFree(d_x));
    return 0;
}

// 仅使用全局内存
void __global__ reduce_global(float *d_x, float *d_y) {
    const int tid = threadIdx.x;
    float *x = d_x + blockDim.x * blockIdx.x; // 线程所在block

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) { // 折半规约
        if (tid < offset) {
            x[tid] += x[tid + offset];
        }
        __syncthreads(); // 每次折半都要同步所有线程，确保数据都已更新
    }

    if (tid == 0) { // 结果累加到block第一个4字节地址后，第一个线程把结果传到d_y
        d_y[blockIdx.x] = x[0];
    }
}

// 在每个block中使用共享内存
void __global__ reduce_shared(float *d_x, float *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int id = bid * blockDim.x + tid; // 线程在block中的id
    __shared__ float s_y[128]; // blocksize长度的共享内存数组
    s_y[tid] = (id < N) ? d_x[id] : 0.0; // 将全局内存中对应id的数据赋值到共享内存中，供所有线程使用
    __syncthreads(); // 赋值后同步所有线程，再规约

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) { // 折半规约
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads(); // 每次折半都要同步所有线程，确保数据都已更新
    }

    if (tid == 0) { // 结果累加到block第一个4字节地址后，第一个线程把结果传到d_y
        d_y[bid] = s_y[0];
    }
}

// 动态分配共享内存（避免共享内存不够或写错） 
void __global__ reduce_dynamic(float *d_x, float *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int id = bid * blockDim.x + tid; // 线程在block中的id
    extern __shared__ float s_y[]; // 动态长度的共享内存数组
    s_y[tid] = (id < N) ? d_x[id] : 0.0; // 将全局内存中对应id的数据赋值到共享内存中，供所有线程使用
    __syncthreads(); // 赋值后同步所有线程，再规约

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) { // 折半规约
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads(); // 每次折半都要同步所有线程，确保数据都已更新
    }

    if (tid == 0) { // 结果累加到block第一个4字节地址后，第一个线程把结果传到d_y
        d_y[bid] = s_y[0];
    }
}

// 在共享内存中原子加，代替全局累加
void __global__ reduce_atomic(float *d_x, float *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int id = bid * blockDim.x + tid;
    extern __shared__ float s_y[];
    s_y[tid] = (id < N) ? d_x[id] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}

// 在归约过程中，最后32个线程时，使用范围更小的warp同步
void __global__ reduce_warp(float *d_x, float *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int id = bid * blockDim.x + tid;
    extern __shared__ float s_y[];
    s_y[tid] = (id < N) ? d_x[id] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncwarp();
    }

    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}

// 还剩32线程时，使用洗牌函数，将高位数据同步到低位，代替warp同步锁
void __global__ reduce_shuffle(float *d_x, float *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int id = bid * blockDim.x + tid;
    extern __shared__ float s_y[];
    s_y[tid] = (id < N) ? d_x[id] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    float y = s_y[tid];
    for (int offset = 16; offset > 0; offset >>= 1) {
        y += __shfl_down_sync(FULL_MASK, y, offset);
    }
    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}

// 使用协程组，将block分成数个线程块片，块片中使用洗牌函数归约
void __global__ reduce_cp(float *d_x, float *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int id = bid * blockDim.x + tid;
    extern __shared__ float s_y[];
    s_y[tid] = (id < N) ? d_x[id] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    float y = s_y[tid];
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1) {
        y += g.shfl_down(y, i);
    }
    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}

// 将原始数据归约到1个grid中，再归约到每个block前32个线程中，
// 再利用block_tile计算每个block的元素和，需在调用一次该核函数
// 第一次调用得到每个block的元素和，第二次得到最终结果
void __global__ reduce_cp_grid(float *d_x, float *d_y, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int id = bid * blockDim.x + tid;
    extern __shared__ float s_y[];
    
    float y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (int n = id; n < N; n += stride) {
        y += d_x[n];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    y = s_y[tid];
    
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1) {
        y += g.shfl_down(y, i);
    }
    if (tid == 0) {
        d_y[bid] = y;
    }
}

float reduce(float *d_x, const int method) {
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // 动态grid数
    const int ymem = sizeof(float) * grid_size;
    const int ymem2 = sizeof(float) * GRID_SIZE2;
    const int smem = sizeof(float) * BLOCK_SIZE;
    float *d_y;
    float *d_y2;
    float *d_y3;
    CHECK_CUDA(cudaMalloc(&d_y, ymem));
    CHECK_CUDA(cudaMalloc(&d_y2, ymem2));
    CHECK_CUDA(cudaGetSymbolAddress((void**)&d_y3, static_y)); // 申请静态全局内存，避免反复创建
    float *h_y = (float *) malloc(ymem);
    float *h_y2 = (float *) malloc(ymem2);

    switch (method) {
    case 0:
        reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
        break;
    case 1:
        reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
        break;
    case 2:
        reduce_dynamic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
        break;
    case 3:
        reduce_atomic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
        break;
    case 4:
        reduce_warp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
        break;
    case 5:
        reduce_shuffle<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
        break;
    case 6:
        reduce_cp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
        break;
    case 7:
        reduce_cp_grid<<<GRID_SIZE2, BLOCK_SIZE, smem>>>(d_x, d_y2, N);
        reduce_cp_grid<<<1, 1024, sizeof(float) * 1024>>>(d_y2, d_y2, GRID_SIZE2);
        break;
    case 8:
        reduce_cp_grid<<<GRID_SIZE2, BLOCK_SIZE, smem>>>(d_x, d_y3, N);
        reduce_cp_grid<<<1, 1024, sizeof(float) * 1024>>>(d_y3, d_y3, GRID_SIZE2);
        break;
    default:
        printf("Error: wrong method\n");
        exit(1);
        break;
    }
    
    if (method < 7) {
        CHECK_CUDA(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));
    } else if (method == 7) {
        CHECK_CUDA(cudaMemcpy(h_y2, d_y2, sizeof(float), cudaMemcpyDeviceToHost));
    } else if (method == 8) {
        CHECK_CUDA(cudaMemcpy(h_y2, d_y3, sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    float result = 0.0;
    if (method < 3) {
        for (int n = 0; n < grid_size; ++n) {
            result += h_y[n];
        }
    } else if (method < 7) {
        result = h_y[0];
    } else if (method < 9) {
        result = h_y2[0];
    }
    free(h_y);
    free(h_y2);
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_y2));
    CHECK_CUDA(cudaFree(d_y3));
    return result;
}

void timing(float *h_x, float *d_x, const int method) {
    float sum = 0;
    for (int repeat = 0; repeat < NUM_REPEATS; repeat++) {
        CHECK_CUDA(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));

        sum = reduce(d_x, method);

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