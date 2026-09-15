#include <cuda_runtime.h>

// ── Warp 级求和 ───────────────────────────────────────────────────────────
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ── Block 级求和 ──────────────────────────────────────────────────────────
__device__ float blockReduceSum(float val) {
    __shared__ float warp_sums[32];
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warpReduceSum(val);
    return val;
}

// ── RMSNorm Kernel ────────────────────────────────────────────────────────
// in/out: [num_tokens, hidden_dim]；gamma: [hidden_dim]
__global__ void rmsNormKernel(const float* __restrict__ in,
                               const float* __restrict__ gamma,
                               float*       __restrict__ out,
                               int hidden_dim, float eps) {
    int token_id = blockIdx.x;
    const float* x = in  + token_id * hidden_dim;
    float*       y = out + token_id * hidden_dim;

    // 1. 每线程步长遍历，累加平方和
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x)
        sum_sq += x[i] * x[i];

    // 2. Block 内归约得到总平方和
    sum_sq = blockReduceSum(sum_sq);

    // 3. 共享 inv_rms 给所有线程
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0)
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(hidden_dim) + eps);
    __syncthreads();

    // 4. 归一化 + 仿射
    float inv_rms = s_inv_rms;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x)
        y[i] = x[i] * inv_rms * gamma[i];
}