#include "error.cuh"

const int NUM_REPEATS = 10;
const int N = 1 << 25;
const int M = sizeof(float) * N;
const int THREAD_PER_BLOCK = 256;

// 全局内存 baseline
__global__ void reduce0(float *d_in, float *d_out) {
    float *block_begin = d_in + blockIdx.x * blockDim.x;
    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (2 * i) == 0) {
            block_begin[threadIdx.x] += block_begin[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = block_begin[0];
    }
}

// 共享内存 shared_memory
__global__ void reduce1(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = d_in[idx];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        // warp中每次循环，只有部分线程参与计算，剩下的等待，浪费资源
        if (threadIdx.x % (2 * i) == 0) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// warp分支分歧 divergence_branch：warp利用不充分
__global__ void reduce2(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = d_in[idx];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        // 使用index，参与计算的线程集中在前几个warp，无资源浪费
        int index = threadIdx.x * 2 * i;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// 银行冲突 bank_conflict：多线程访问同一warp会串行化
__global__ void reduce3(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = d_in[idx];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// 增加线程工作量 add during load：又称idle，增加每个线程的工作量
// 每个线程加载到共享内存时，每个线程处理多个元素
// Plan 1：减少block数量，block中thread数量不变
// Plan 2: block数量不变，block中thread数量减少一半
__global__ void reduce4(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    float *block_begin = d_in + blockIdx.x * blockDim.x * 2;
    sdata[threadIdx.x] = block_begin[threadIdx.x] + block_begin[threadIdx.x + blockDim.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}
__global__ void reduce5(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    float *block_begin = d_in + blockIdx.x * blockDim.x * 2;
    sdata[threadIdx.x] = block_begin[threadIdx.x] + block_begin[threadIdx.x + blockDim.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// 展开最后一个warp Unroll Last Warp：折半到一个warp时，省略线程同步操作
// 旧架构（Volta前）：架构提供了隐式warp同步
// 新架构（Volta后）：需要手动进行warp内数据同步
// Method 1：不做warp内共享数据同步
// Method 2：使用volatile修饰共享内存
// Method 3：使用__syncwarp()进行同步
__global__ void reduce6(float *d_in, float *d_out) { // Method 1：不做warp内共享数据同步
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    float *block_begin = d_in + blockIdx.x * blockDim.x * 2;
    sdata[threadIdx.x] = block_begin[threadIdx.x] + block_begin[threadIdx.x + blockDim.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        if (i > 32) {
            __syncthreads();
        }
    }

    // for (int i = 32; i > 0; i >>= 1) {
    //     if (threadIdx.x < i) {
    //         sdata[threadIdx.x] += sdata[threadIdx.x + i];
    //     }
    // }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}
__device__ void warp_reduce(volatile float *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
// 展开最后一个warp Unroll Last Warp：折半到一个warp时，省略线程同步操作
// 旧架构（Volta前）：架构提供了隐式warp同步
// 新架构（Volta后）：需要手动进行warp内数据同步
// Method 1：不做warp内共享数据同步
// Method 2：使用volatile修饰共享内存
// Method 3：使用__syncwarp()进行同步
__global__ void reduce7(float *d_in, float *d_out) { // Method 2：使用volatile修饰共享内存
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    float *block_begin = d_in + blockIdx.x * blockDim.x * 2;
    sdata[threadIdx.x] = block_begin[threadIdx.x] + block_begin[threadIdx.x + blockDim.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 32; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warp_reduce(sdata, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}
// 展开最后一个warp Unroll Last Warp：折半到一个warp时，省略线程同步操作
// 旧架构（Volta前）：架构提供了隐式warp同步
// 新架构（Volta后）：需要手动进行warp内数据同步
// Method 1：不做warp内共享数据同步
// Method 2：使用volatile修饰共享内存
// Method 3：使用__syncwarp()进行同步
__global__ void reduce8(float *d_in, float *d_out) { // Method 3：使用__syncwarp()进行同步
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    float *block_begin = d_in + blockIdx.x * blockDim.x * 2;
    sdata[threadIdx.x] = block_begin[threadIdx.x] + block_begin[threadIdx.x + blockDim.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        if (i > 32) {
            __syncthreads();
        } else {
            __syncwarp();
        }
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// 完全展开所有循环 Completely Unroll：手动展开for循环
__global__ void reduce9(float *d_in, float *d_out) { // Method 3：使用__syncwarp()进行同步
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    float *block_begin = d_in + blockIdx.x * blockDim.x * 2;
    sdata[threadIdx.x] = block_begin[threadIdx.x] + block_begin[threadIdx.x + blockDim.x];
    __syncthreads();
    int tid = threadIdx.x;

    if (THREAD_PER_BLOCK >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warp_reduce(sdata, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// 增加每个线程的处理量 Multi Add：在复制到共享内存过程中，每个线程多累加几个数据
__global__ void reduce10(float *d_in, float *d_out, int task_num_per_block) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程将部分数据加载到共享内存中
    int tid = threadIdx.x;
    float *block_begin = d_in + task_num_per_block * blockDim.x;
    sdata[tid] = 0;
    for (int i = 0; i < task_num_per_block / THREAD_PER_BLOCK; i++) {
        sdata[tid] += block_begin[tid + i * THREAD_PER_BLOCK];
    }
    __syncthreads();
    
    if (THREAD_PER_BLOCK >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warp_reduce(sdata, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// 使用shuffle代替warp同步操作 Shuffle
__global__ void reduce11(float *d_in, float *d_out, int task_num_per_block) {
    float sum = 0.0f;

    // 每个线程将部分数据加载到共享内存中
    int tid = threadIdx.x;
    float *block_begin = d_in + task_num_per_block * blockDim.x;
    for (int i = 0; i < task_num_per_block / THREAD_PER_BLOCK; i++) {
        sum += block_begin[tid + i * THREAD_PER_BLOCK];
    }
    
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    //一个block最多有1024/2048个线程，也就是32/64个warp
    __shared__ float warp_level_sum[32]; // 这里例子用不到64那么多
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_level_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) { // 共享内存的32个元素，全都放到第一个warp中
        sum = lane_id < 32 ? warp_level_sum[lane_id] : 0.f; // 第一个warp中的每个线程存放一个元素到reg
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sum;
    }
}

float reduce(float *d_in, const int method) {
    int num_block0 = N / THREAD_PER_BLOCK;
    int num_block1 = N / THREAD_PER_BLOCK / 2;
    int num_block2 = 1024;
    int out_mem0 = num_block0 * sizeof(float);
    int out_mem1 = num_block1 * sizeof(float);
    int out_mem2 = num_block2 * sizeof(float);
    float *d_out0;
    float *d_out1;
    float *d_out2;
    CHECK_CUDA(cudaMalloc(&d_out0, out_mem0));
    CHECK_CUDA(cudaMalloc(&d_out1, out_mem1));
    CHECK_CUDA(cudaMalloc(&d_out2, out_mem2));
    float *h_out0 = (float *)malloc(out_mem0);
    float *h_out1 = (float *)malloc(out_mem1);
    float *h_out2 = (float *)malloc(out_mem2);

    dim3 grid0(num_block0, 1);
    dim3 grid1(num_block1, 1);
    dim3 grid2(num_block2, 1);
    dim3 block0(THREAD_PER_BLOCK, 1);
    dim3 block1(THREAD_PER_BLOCK / 2, 1);

    unsigned int task_num_per_block = (N - 1) / num_block2 + 1;

    switch (method) {
    case 0:
        reduce0<<<grid0, block0>>>(d_in, d_out0);
        break;
    case 1:
        reduce1<<<grid0, block0>>>(d_in, d_out0);
        break;
    case 2:
        reduce2<<<grid0, block0>>>(d_in, d_out0);
        break;
    case 3:
        reduce3<<<grid0, block0>>>(d_in, d_out0);
        break;
    case 4:
        reduce4<<<grid1, block0>>>(d_in, d_out1);
        break;
    case 5:
        reduce5<<<grid0, block1>>>(d_in, d_out0);
        break;
    case 6:
        reduce6<<<grid1, block0>>>(d_in, d_out1);
        break;
    case 7:
        reduce7<<<grid1, block0>>>(d_in, d_out1);
        break;
    case 8:
        reduce8<<<grid1, block0>>>(d_in, d_out1);
        break;
    case 9:
        reduce9<<<grid1, block0>>>(d_in, d_out1);
        break;
    case 10:
        reduce10<<<grid2, block0>>>(d_in, d_out2, task_num_per_block);
        break;
    case 11:
        reduce11<<<grid2, block0>>>(d_in, d_out2, task_num_per_block);
        break;
    default:
        printf("Error: wrong method\n");
        exit(1);
        break;
    }

    float res = 0.0;
    if (method <= 5 && method != 4) {
        CHECK_CUDA(cudaMemcpy(h_out0, d_out0, out_mem0, cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_block0; ++i) {
            res += h_out0[i];
        }
    } else if (method <= 9) {
        CHECK_CUDA(cudaMemcpy(h_out1, d_out1, out_mem1, cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_block1; ++i) {
            res += h_out1[i];
        }
    } else if (method <= 11) {
        CHECK_CUDA(cudaMemcpy(h_out2, d_out2, out_mem2, cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_block2; ++i) {
            res += h_out2[i];
        }
    }
    
    free(h_out0);
    free(h_out1);
    
    CHECK_CUDA(cudaFree(d_out0));
    CHECK_CUDA(cudaFree(d_out1));
    
    return res;
}

void timing(float *h_in, float *d_in, const int method) {
    float t_avg = 0.0, sum = 0.0;
    for (int repeat = 0; repeat < NUM_REPEATS; repeat++) {
        CHECK_CUDA(cudaMemcpy(d_in, h_in, M, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));

        sum = reduce(d_in, method);

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        t_avg += elapsed_time;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    t_avg /= NUM_REPEATS;
    printf("Average Time = %.6f ms. Sum = %.6f\n", t_avg, sum);
}

int main() {
    float *h_in = (float *)malloc(M);
    for (int n = 0; n < N; n++) {
        h_in[n] = 1.23;
    }
    float *d_in;
    CHECK_CUDA(cudaMalloc((void **)&d_in, M));

    printf("\nReduce v0 Global Memory:            "); // 使用全局内存
    timing(h_in, d_in, 0);  // 16.0ms, 41257796.000000, 精度为3位
    printf("\nReduce v1 Shared Memory:            "); // 使用共享内存
    timing(h_in, d_in, 1);  // 17.4ms, 41257796.000000, 精度为3位
    printf("\nReduce v2 Divergence Branch:        "); // 使执行归约的线程连续
    timing(h_in, d_in, 2);  // 12.7ms, 41257796.000000, 精度为3位
    printf("\nReduce v3 Bank Conflict:            "); // 处理数据来自不同warp
    timing(h_in, d_in, 3);  // 12.1ms, 41257796.000000, 精度为3位
    printf("\nReduce v4 Add During Load Plan1:    "); // block数减半
    timing(h_in, d_in, 4);  // 6.7ms, 41261476.000000, 精度为3位
    printf("\nReduce v5 Add During Load Plan2:    "); // block内线程减半
    timing(h_in, d_in, 5);  // 6.4ms, 41257796.000000, 精度为3位
    printf("\nReduce v6 Unroll Last Warp Method1: "); // 不考虑warp内同步
    timing(h_in, d_in, 6);  // 6.8ms, 41261476.000000, 精度为3位
    printf("\nReduce v7 Unroll Last Warp Method2: "); // 使用volatile保证warp内同步
    timing(h_in, d_in, 7);  // 5.2ms, 41261476.000000, 精度为3位
    printf("\nReduce v8 Unroll Last Warp Method3: "); // 使用__syncwarp()保证warp内同步
    timing(h_in, d_in, 8);  // 8.5ms, 41261476.000000, 精度为3位
    printf("\nReduce v9 Unroll Completely:        "); // 把循环手动展开，减少循环消耗
    timing(h_in, d_in, 9);  // 4.9ms, 41261476.000000, 精度为3位
    printf("\nReduce v10 Multi Add:               "); // 增加每个线程在初始化共享内存时的处理量
    timing(h_in, d_in, 10); // 2.0ms, 41271628.000000, 精度为3位
    printf("\nReduce v11 Shuffle:                 "); // 使用寄存器内存
    timing(h_in, d_in, 11); // 1.7ms, 41271628.000000, 精度为3位
    free(h_in);
    CHECK_CUDA(cudaFree(d_in));
    return 0;
}