#include "error.cuh"
#include <cuda_runtime.h>
#include <assert.h>
#include <vector>
#include <iostream>

// ============================================================
// Stage 1
//
// Input:
//     [R, C]
// Output:
//     [grid_size, C]
//
// 固定 C=128：
// 一个 block = 128 threads = 4 warps
//
// thread 0   -> column 0
// thread 1   -> column 1
// ...
// thread 127 -> column 127
//
// Grid-Stride Loop 沿 row 方向执行。
// ============================================================
__global__ void col_reduce_stage1(const float* __restrict__ d_input,
                                  float* __restrict__ d_partial,
                                  int R, int C) {
    int col = threadIdx.x;

    float sum = 0.f;
    for (int row = blockIdx.x; row < R; row += gridDim.x) {
        sum += d_input[static_cast<size_t>(row) * C + col];
    }

    d_partial[static_cast<size_t>(blockIdx.x) * C + col] = sum;
}

// ============================================================
// Stage 2
//
// Input:
//     [grid_size, C]
// Output:
//     [1, C]
//
// 仍然：
// thread 0   -> column 0
// ...
// thread 127 -> column 127
//
// grid 只有 1 个 block。
// ============================================================
__global__ void col_reduce_stage2(const float* __restrict__ d_partial,
                                  float* __restrict__ d_output,
                                  int grid_size, int C) {
    int col = threadIdx.x;

    float sum = 0.f;
    for (int row = 0; row < grid_size; row += gridDim.x) {
        sum += d_partial[static_cast<size_t>(row) * C + col];
    }

    d_output[static_cast<size_t>(blockIdx.x) * C + col] = sum;
}


// ============================================================
// Host API
//
// d_input : [R, 128]
// d_output: [1, 128]
// ============================================================
void col_reduce_matrix(const float* d_input, float* d_output,
                       int R, int C) {
    constexpr int BLOCK_SIZE = 128;

    // 针对 C=128 的特殊情况
    assert(C == 128);

    // 查询 GPU SM 数量
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // 经验 benchmark 值：实际最佳值需根据 GPU 测试
    constexpr int BLOCKS_PER_SM = 4;

    int grid_size = BLOCKS_PER_SM * prop.multiProcessorCount;
    grid_size = std::min(grid_size, R); // 避免 grid_size > R

    float* d_partial = nullptr;
    size_t partial_bytes = static_cast<size_t>(grid_size) * C * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_partial, partial_bytes));

    // Stage 1
    col_reduce_stage1<<<grid_size, BLOCK_SIZE>>>(d_input, d_partial, R, C);

    CUDA_CHECK(cudaGetLastError());

    // Stage 2
    col_reduce_stage1<<<1, BLOCK_SIZE>>>(d_partial, d_output, grid_size, C);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_partial));
}

int main() {
    constexpr int R = 1'000'000;
    constexpr int C = 128;

    // --------------------------------------------------------
    // Host input
    // --------------------------------------------------------

    std::vector<float> h_input(static_cast<size_t>(R) * C);

    // 方便验证：
    // 所有元素 = 1
    //
    // 每一列最终应该得到 R。
    std::fill(h_input.begin(), h_input.end(), 1.0f);

    // --------------------------------------------------------
    // CPU reference
    // --------------------------------------------------------

    std::vector<float> h_cpu(C, 0.0f);

    for (int row = 0; row < R; ++row) {
        for (int col = 0; col < C; ++col) {
            h_cpu[col] += h_input[static_cast<size_t>(row) * C + col];
        }
    }

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(&d_input, static_cast<size_t>(R) * C * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_output, C * sizeof(float)));

    // --------------------------------------------------------
    // H2D
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          static_cast<size_t>(R) * C * sizeof(float),
                          cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // GPU reduction
    // --------------------------------------------------------

    col_reduce_matrix(d_input, d_output, R, C);

    // --------------------------------------------------------
    // D2H
    // --------------------------------------------------------

    std::vector<float> h_gpu(C);

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_output, C * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // Verify
    // --------------------------------------------------------

    bool passed = true;

    for (int col = 0; col < C; ++col) {
        float diff = std::fabs(h_cpu[col] - h_gpu[col]);

        if (diff > 1e-3f) {
            std::cerr << "Mismatch at column " << col << ": CPU=" << h_cpu[col]
                      << ", GPU=" << h_gpu[col] << ", diff=" << diff
                      << std::endl;

            passed = false;
            break;
        }
    }

    std::cout << "CPU column[0] = " << h_cpu[0] << std::endl;

    std::cout << "GPU column[0] = " << h_gpu[0] << std::endl;

    std::cout << "CPU column[127] = " << h_cpu[127] << std::endl;

    std::cout << "GPU column[127] = " << h_gpu[127] << std::endl;

    if (passed) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return passed ? 0 : 1;
}