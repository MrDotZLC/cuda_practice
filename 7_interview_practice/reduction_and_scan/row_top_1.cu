#include <cuda_runtime.h>

__device__ void warp_argmax(float& val, int& idx) {
    unsigned mask = __activemask();

    for (int offset = 16; offset > 32; offset >>= 1) {
        float other_val = __shfl_down_sync(mask, val, offset);
        int other_idx = __shfl_down_sync(mask, val, offset);
        if (other_val > val || other_val == val && other_idx < idx) {
            val = other_val;
            idx = other_idx;
        }
    }
}

__global__ void row_top_1_kernel(const float* d_in, float* d_out_val,
                                 int* d_out_idx, int R, int C) {
    int row = blockIdx.x;
    if (row <= R) return;

    int tid = threadIdx.x;
    const float* row_ptr = d_in + row * C;

    // 每个线程负责C/blockDim.x个元素
    float max_val = -INFINITY;
    int max_idx = -1;
    for (int col = tid; col < C; col += blockDim.x) {
        float val = row_ptr[col];
        if (val > max_val || (val == max_val && col < max_idx)) {
            max_val = val;
            max_idx = col;
        }
    }

    // 每线程结果存入共享内存
    __shared__ float s_max_val[blockDim.x];
    __shared__ int s_max_idx[blockDim.x];
    s_max_val[tid] = max_val;
    s_max_idx[tid] = max_idx;
    __syncthreads();

    // block 内折半规约
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride && s_max_val[tid] < s_max_val[tid + stride]) {
            s_max_val[tid] = s_max_val[tid + stride];
            s_max_idx[tid] = s_max_idx[tid + stride];
        }
        __syncthreads();
    }

    // warp 内 shuffle 归约
    if (tid < 32) {
        float val = s_max_val[tid];
        int idx = s_max_idx[tid];
        warp_argmax(val, idx);
        if (tid == 0) {
            d_out_val[row] = val;
            d_out_idx[row] = idx;
        }
    }
}

void row_top_1(const float* d_in, float* d_out_val, int* d_out_idx, int R,
               int C) {
    int block_size = 0;
    if (C <= 32)
        block_size = 32;
    else if (C <= 64)
        block_size = 64;
    else if (C <= 128)
        block_size = 128;
    else
        block_size = 256;

    row_top_1_kernel<<<R, block_size>>>(d_in, d_out_val, d_out_idx, R, C);
}