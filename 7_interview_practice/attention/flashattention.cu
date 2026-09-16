#include "attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

static constexpr int Br = 32;
static constexpr int Bc = 32;
static constexpr int THREADS_PER_BLOCK = 128; // 4 个 Warp

// 每个 Warp 需要处理的 Q 行数 = Br / (Warp 数量)
static constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32; // 4
static constexpr int ROWS_PER_WARP = Br / WARPS_PER_BLOCK;     // 8

// 设定最大支持的 head_dim (例如 128)。每个线程最多处理 128 / 32 = 4 个元素
static constexpr int MAX_HEAD_DIM_PER_THREAD = 4;

__device__ __forceinline__
float warp_reduce_sum(float val)
{
    #pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void flash_attn_v2_onepass_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    int seq_len,
    int head_dim)
{
    if (head_dim > MAX_HEAD_DIM_PER_THREAD * 32) {
        // 抛出异常或处理错误
        return;
    }
    /*
        Shared memory layout:
        Q : [Br, head_dim]
        K : [Bc, head_dim]
        V : [Bc, head_dim]
    */
    extern __shared__ __half smem[];

    __half* smem_Q = smem;
    __half* smem_K = smem_Q + Br * head_dim;
    __half* smem_V = smem_K + Bc * head_dim;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    const int block_q_start = blockIdx.x * Br;
    const float scale = rsqrtf((float)head_dim);

    /* 
       1. 协作加载 Q Tile 到 Shared Memory
    */
    const int q_total_elements = Br * head_dim;
    for (int i = tid; i < q_total_elements; i += THREADS_PER_BLOCK)
    {
        int row = i / head_dim;
        int col = i % head_dim;
        int global_q_row = block_q_start + row;

        if (global_q_row < seq_len)
            smem_Q[row * head_dim + col] = Q[global_q_row * head_dim + col];
        else
            smem_Q[row * head_dim + col] = __float2half(0.0f);
    }

    __syncthreads();

    /* 
       2. 每个 Warp 绑定 ROWS_PER_WARP 行 Q 的状态（私有寄存器数组）
    */
    float m[ROWS_PER_WARP];
    float l[ROWS_PER_WARP];
    float o[ROWS_PER_WARP][MAX_HEAD_DIM_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++)
    {
        m[r] = -FLT_MAX;
        l[r] = 0.f;
        #pragma unroll
        for (int d = 0; d < MAX_HEAD_DIM_PER_THREAD; d++) {
            o[r][d] = 0.f;
        }
    }

    /*
       3. 外层无分支差异的 KV 循环
    */
    for (int kv_start = 0; kv_start < seq_len; kv_start += Bc)
    {
        // 全体线程协作加载 K/V 到 Shared Memory (所有 Warp 必须同时到达这里)
        const int kv_total_elements = Bc * head_dim;
        for (int i = tid; i < kv_total_elements; i += THREADS_PER_BLOCK)
        {
            int row = i / head_dim;
            int col = i % head_dim;
            int global_kv_row = kv_start + row;

            if (global_kv_row < seq_len)
            {
                smem_K[row * head_dim + col] = K[global_kv_row * head_dim + col];
                smem_V[row * head_dim + col] = V[global_kv_row * head_dim + col];
            }
            else
            {
                smem_K[row * head_dim + col] = __float2half(0.0f);
                smem_V[row * head_dim + col] = __float2half(0.0f);
            }
        }

        // 统一屏障，绝不会死锁
        __syncthreads();

        // 每个 Warp 遍历其负责的 ROWS_PER_WARP 行
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++)
        {
            const int q_sub_row = warp_id * ROWS_PER_WARP + r;
            const int q_row = block_q_start + q_sub_row;

            // 只有处理有效 Q 行才执行计算
            if (q_sub_row < Br && q_row < seq_len)
            {
                float score[Bc];

                #pragma unroll
                for (int j = 0; j < Bc; j++)
                {
                    if (kv_start + j >= seq_len)
                    {
                        score[j] = -1e20f;
                        continue;
                    }

                    float dot = 0.f;
                    #pragma unroll
                    for (int d = lane_id; d < head_dim; d += 32)
                    {
                        dot += __half2float(smem_Q[q_sub_row * head_dim + d]) *
                               __half2float(smem_K[j * head_dim + d]);
                    }

                    dot = warp_reduce_sum(dot);
                    score[j] = __shfl_sync(0xffffffff, dot, 0) * scale;
                }

                // Online Softmax 更新
                float m_new = m[r];
                #pragma unroll
                for (int j = 0; j < Bc; j++)
                {
                    m_new = fmaxf(m_new, score[j]);
                }

                float alpha = expf(m[r] - m_new);
                l[r] *= alpha;
                
                #pragma unroll
                for (int d = 0; d < MAX_HEAD_DIM_PER_THREAD; d++)
                {
                    o[r][d] *= alpha;
                }

                #pragma unroll
                for (int j = 0; j < Bc; j++)
                {
                    float p = expf(score[j] - m_new);
                    l[r] += p;

                    #pragma unroll
                    for (int d = lane_id, idx = 0; d < head_dim; d += 32, idx++)
                    {
                        o[r][idx] += p * __half2float(smem_V[j * head_dim + d]);
                    }
                }
                m[r] = m_new;
            }
        }

        // 确保下一个循环重写 Shared Memory 前，计算已全部完成
        __syncthreads();
    }

    /*
       4. 写回 Output
    */
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++)
    {
        const int q_sub_row = warp_id * ROWS_PER_WARP + r;
        const int q_row = block_q_start + q_sub_row;

        if (q_sub_row < Br && q_row < seq_len)
        {
            float inv_l = (l[r] > 0.0f) ? (1.0f / l[r]) : 0.0f;

            #pragma unroll
            for (int d = lane_id, idx = 0; d < head_dim; d += 32, idx++)
            {
                O[q_row * head_dim + d] = __float2half(o[r][idx] * inv_l);
            }
        }
    }
}

void flash_attn_v2_onepass(
    const __half *Q,
    const __half *K,
    const __half *V,
    __half *O,
    int seq_len,
    int head_dim)
{
    int grid = (seq_len + Br - 1) / Br;
    int block = THREADS_PER_BLOCK; 

    size_t smem_bytes = (Br + 2 * Bc) * head_dim * sizeof(__half);

    flash_attn_v2_onepass_kernel
    <<<grid, block, smem_bytes>>>
    (
        Q, K, V, O,
        seq_len, head_dim
    );

    // 假设使用者在自己的库中定义了 CUDA_CHECK_LAST 宏
    // CUDA_CHECK_LAST();
}