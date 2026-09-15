#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int get_block_size(int hidden_dim) {
    int block_size = 1;

    while (block_size < hidden_dim && block_size < 1024) {
        block_size <<= 1;
    }

    return std::min(block_size, 1024);
}

// LayerNorm Kernel
// 一个 Block 处理一个 token
// 输入:
//     in     [tokens, hidden_dim]
//     gamma  [hidden_dim]
//     beta   [hidden_dim]
// 输出:
//     out    [tokens, hidden_dim]
//
// 统计量:
//     mean = 1/N * sum(x)
//     M2   = sum((x - mean)^2)
//     variance = M2 / N
//
// 使用 Welford + Chan Parallel Merge。
__global__ void layer_norm_welford_kernel(const float* __restrict__ in,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ out, int hidden_dim,
                                       float eps) {
    const int tid = threadIdx.x;
    const int token_id = blockIdx.x;

    // 一个 block 对应一个 token
    const size_t token_offset =
        static_cast<size_t>(token_id) * static_cast<size_t>(hidden_dim);

    const float* x = in + token_offset;
    float* y = out + token_offset;

    // Step 1: Thread Local Welford
    float mean = 0.0f;
    float M2 = 0.0f;
    int cnt = 0;

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float value = x[i];
        ++cnt;
        float delta = value - mean;
        mean += delta / static_cast<float>(cnt);
        M2 += delta * (value - mean);
    }

    // Step 2: Dynamic Shared Memory
    extern __shared__ unsigned char smem[];
    float* s_mean = reinterpret_cast<float*>(smem);
    float* s_M2 = s_mean + blockDim.x;
    int* s_cnt = reinterpret_cast<int*>(s_M2 + blockDim.x);

    s_mean[tid] = mean;
    s_M2[tid] = M2;
    s_cnt[tid] = cnt;

    __syncthreads();

    // Step 3: Block-level Welford Merge
    // State A: (na, ma, M2a)
    // State B: (nb, mb, M2b)
    // 合并:
    //     n = na + nb
    //     delta = mb - ma
    //     mean = ma + delta * nb / n
    //     M2 = M2a + M2b + delta^2 * na * nb / n
    // 特别处理:
    //     na == 0
    //     nb == 0

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const int na = s_cnt[tid];
            const int nb = s_cnt[tid + stride];
            if (nb != 0) {
                if (na == 0) {
                    s_mean[tid] = s_mean[tid + stride];

                    s_M2[tid] = s_M2[tid + stride];

                    s_cnt[tid] = nb;
                } else {
                    const float ma = s_mean[tid];
                    const float mb = s_mean[tid + stride];

                    const float M2a = s_M2[tid];
                    const float M2b = s_M2[tid + stride];
                    const float delta = mb - ma;
                    const int n = na + nb;

                    const float merged_mean = ma + delta *
                                                       static_cast<float>(nb) /
                                                       static_cast<float>(n);

                    const float merged_M2 =
                        M2a + M2b +
                        delta * delta *
                            (static_cast<float>(na) * static_cast<float>(nb) /
                             static_cast<float>(n));

                    s_mean[tid] = merged_mean;
                    s_M2[tid] = merged_M2;
                    s_cnt[tid] = n;
                }
            }
        }
        __syncthreads();
    }

    // Step 4: 得到最终 mean / variance
    const float global_mean = s_mean[0];
    const float variance = s_M2[0] / static_cast<float>(hidden_dim);
    const float inv_std = rsqrtf(variance + eps);

    // Step 5: LayerNorm
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        y[i] = (x[i] - global_mean) * inv_std * gamma[i] + beta[i];
    }
}

void layer_norm_welford(const float* __restrict__ d_in,
                      const float* __restrict__ d_gamma,
                      const float* __restrict__ d_beta,
                      float*       __restrict__ d_out,
                      int tokens, int hidden_dim, float eps = 1e-5f) {

    const int block_size = get_block_size(hidden_dim);

    // Shared Memory:
    //     float s_mean[block]
    //     float s_M2[block]
    //     int   s_cnt[block]
    //
    // 因此:
    //     2 * block * sizeof(float)
    //     +
    //     block * sizeof(int)
    const size_t shared_mem_size =
        static_cast<size_t>(block_size) * (2 * sizeof(float) + sizeof(int));

    const dim3 grid(tokens);
    const dim3 block(block_size);

    layer_norm_welford_kernel<<<grid, block, shared_mem_size>>>(
        d_in, d_gamma, d_beta, d_out, hidden_dim, eps);
}
