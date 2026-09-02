#include "error.cuh"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warpScanInclusive(float val) {
    constexpr unsigned MASK = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float other = __shfl_up_sync(MASK, val, offset);
        if (lane >= offset) {
            val += other;
        }
    }
    return val;
}

// 采用 Warp Shuffle + Warp-level aggregation

__device__ __forceinline__ float blockScanInclusive(float val) {
    static_assert(BLOCK_SIZE % WARP_SIZE == 0);
    constexpr int NUM_WARPS_LOCAL = BLOCK_SIZE / WARP_SIZE;

    __shared__ float warp_sums[NUM_WARPS_LOCAL];
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;

    val = warpScanInclusive(val);

    if (lane == WARP_SIZE - 1) {
        warp_sums[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float warp_sum = (lane < NUM_WARPS_LOCAL) ? warp_sums[lane] : 0.0f;
        warp_sum = warpScanInclusive(warp_sum);
        if (lane < NUM_WARPS_LOCAL) {
            warp_sums[lane] = warp_sum;
        }
    }
    __syncthreads();

    if (warp > 0) {
        val += warp_sums[warp - 1];
    }
    return val;
}

// Phase 1: 原始输入 -> 输出 local prefix，同时输出 level 0 的 block sums

__global__ void scanPhase1(const float* __restrict__ input,
                           float* __restrict__ output,
                           float* __restrict__ block_sums, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * BLOCK_SIZE + tid;

    float value = (idx < N) ? input[idx] : 0.0f;
    float prefix = blockScanInclusive(value);

    if (idx < N) {
        output[idx] = prefix;
    }

    if (tid == BLOCK_SIZE - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = prefix;
    }
}

// 向上归降 stage：从 input_sums 读取，计算 local prefix 写入
// output_prefix，并将 block sum 写入 next_block_sums

__global__ void scanStage(const float* __restrict__ input_sums,
                          float* __restrict__ output_prefix,
                          float* __restrict__ next_block_sums, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * BLOCK_SIZE + tid;

    float value = (idx < N) ? input_sums[idx] : 0.0f;
    float prefix = blockScanInclusive(value);

    if (idx < N) {
        output_prefix[idx] = prefix;
    }

    if (tid == BLOCK_SIZE - 1 && next_block_sums != nullptr) {
        next_block_sums[blockIdx.x] = prefix;
    }
}

// 向下传播：将上一层的 global block prefix 作为 offset 加到当前层
__global__ void addBlockOffsets(float* __restrict__ output,
                                const float* __restrict__ global_block_prefix,
                                int N) {
    const int tid = threadIdx.x;
    const int block = blockIdx.x;
    const int idx = block * BLOCK_SIZE + tid;

    float offset = (block == 0) ? 0.0f : global_block_prefix[block - 1];

    if (idx < N) {
        output[idx] += offset;
    }
}

std::vector<int> buildLevelSizes(int numBlocks) {
    std::vector<int> sizes;
    sizes.push_back(numBlocks);
    while (numBlocks > 1) {
        numBlocks = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sizes.push_back(numBlocks);
    }
    return sizes;
}

void prefixSum(const float* d_input, float* d_output, int N) {
    if (N <= 0) return;

    const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 单 block 特判
    if (numBlocks == 1) {
        scanPhase1<<<1, BLOCK_SIZE>>>(d_input, d_output, nullptr, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    std::vector<int> levelSizes = buildLevelSizes(numBlocks);
    const int numLevels = static_cast<int>(levelSizes.size());

    // 我们为每一层分配 2 个 buffer：
    // block_sums: 保存该层的原始 sums
    // local_prefix: 保存该层 local scan 后的结果
    std::vector<size_t> sumsOffsets(numLevels);
    std::vector<size_t> prefixOffsets(numLevels);

    size_t totalElements = 0;
    for (int i = 0; i < numLevels; ++i) {
        sumsOffsets[i] = totalElements;
        totalElements += levelSizes[i];

        prefixOffsets[i] = totalElements;
        totalElements += levelSizes[i];
    }

    float* d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scratch, totalElements * sizeof(float)));

    auto getSumsPtr = [&](int level) { return d_scratch + sumsOffsets[level]; };
    auto getPrefixPtr = [&](int level) {
        return d_scratch + prefixOffsets[level];
    };

    // Step 1: Phase 1
    // 计算 d_output 的 local prefix，并将 block sums 写入 level 0 的 sums
    // 缓冲区
    scanPhase1<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, getSumsPtr(0), N);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Upward Scan
    // 逐层向上计算，直到顶层
    for (int level = 0; level + 1 < numLevels; ++level) {
        const int currentSize = levelSizes[level];
        const int blocks = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        scanStage<<<blocks, BLOCK_SIZE>>>(getSumsPtr(level),
                                          getPrefixPtr(level),
                                          getSumsPtr(level + 1), currentSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // 最顶层 (numLevels - 1) 只有 1 个 block，直接进行 local scan，此时它的
    // local_prefix 就是 global_prefix
    {
        int topLevel = numLevels - 1;
        int topSize = levelSizes[topLevel];
        int blocks = (topSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scanStage<<<blocks, BLOCK_SIZE>>>(
            getSumsPtr(topLevel), getPrefixPtr(topLevel), nullptr, topSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: Downward Propagation
    // 从顶层向下，把上一层的 global prefix 作为 offset 加到当前层的 local
    // prefix 上
    for (int level = numLevels - 2; level >= 0; --level) {
        const int currentSize = levelSizes[level];
        const int blocks = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 用 level+1 的 global prefix 来更新 level 的 local prefix，使其变为
        // global prefix
        addBlockOffsets<<<blocks, BLOCK_SIZE>>>(
            getPrefixPtr(level), getPrefixPtr(level + 1), currentSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 4: 最终将 level 0 的 global prefix 加回 d_output
    addBlockOffsets<<<numBlocks, BLOCK_SIZE>>>(d_output, getPrefixPtr(0), N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_scratch));
}

// ============================================================
// Test
// ============================================================

int main() {
    // 可以修改 N 测试任意长度。

    constexpr int N = 1'000'003;

    // constexpr int N = 1'024;

    std::vector<float> h_input(N);

    std::vector<float> h_output(N);

    // 使用随机数据

    std::mt19937 rng(12345);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < N; ++i) {
        h_input[i] = dist(rng);
    }

    // Device memory

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // GPU Prefix Sum

    prefixSum(d_input, d_output, N);

    // Copy back

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // CPU reference

    std::vector<float> h_reference(N);

    float sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        sum += h_input[i];

        h_reference[i] = sum;
    }

    // Verify
    //
    // float prefix sum 存在浮点累加误差，
    // 因此不能使用 ==。

    bool correct = true;

    for (int i = 0; i < N; ++i) {
        float a = h_output[i];

        float b = h_reference[i];

        float diff = std::fabs(a - b);

        float tolerance = 1e-3f * std::max(1.0f, std::fabs(b));

        if (diff > tolerance) {
            std::printf(
                "Mismatch at %d: "
                "GPU=%f CPU=%f diff=%f\n",
                i, a, b, diff);

            correct = false;

            break;
        }
    }

    if (correct) {
        std::printf(
            "PASS\n"
            "N = %d\n"
            "output[N-1] = %.6f\n",
            N, h_output[N - 1]);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
