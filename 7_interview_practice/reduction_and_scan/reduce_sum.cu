// grid-stride loop + warp shuffle + shared warp sum
#include <error.cuh>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <assert.h>

__device__ __forceinline__ 
float warp_reduce_sum_kernel(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    // val += __shfl_down_sync(mask, val, 16);
    // val += __shfl_down_sync(mask, val, 8);
    // val += __shfl_down_sync(mask, val, 4);
    // val += __shfl_down_sync(mask, val, 2);
    // val += __shfl_down_sync(mask, val, 1);
    return val;
}

__device__ __forceinline__ 
float block_reduce_sum_kernel(float val) {
    // 理论上，block 最多 32 个 warp，再多就
    static __shared__ float smem[32];

    int lane_id = threadIdx.x & 31;  // 等价于 % 32
    int warp_id = threadIdx.x >> 5;  // 等价于 / 32

    val = warp_reduce_sum_kernel(val);

    if (lane_id == 0) {
        smem[warp_id] = val;
    }

    __syncthreads();

    int warp_count = (blockDim.x + 31) / 32;
    val = (lane_id < warp_count) ? smem[lane_id] : 0.f;

    // 第一个 warp 完成最后的 warp reduce
    if (warp_id == 0) {
        val = warp_reduce_sum_kernel(val);
    }

    return val;
}

__global__ void reduce_sum_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           size_t N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // grid-stride loop
    float sum = 0.f;
    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += input[i];
    }

    // block reduction
    sum = block_reduce_sum_kernel(sum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

// Host API
float reduce_sum(const std::vector<float>& input) {
    int N = input.size();
    float* d_buf1 = nullptr;
    float* d_buf2 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_buf1, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_buf1, input.data(), sizeof(float) * N,
                          cudaMemcpyHostToDevice));

    int block_size = 256;

    // 最大需要的中间缓存
    int total_grid_size = (N + block_size - 1) / block_size;
    CUDA_CHECK(cudaMalloc(&d_buf2, sizeof(float) * total_grid_size));

    size_t current_size = N;

    while (current_size > 1) {
        int grid_size = (current_size + block_size - 1) / block_size;
        // 归约 block
        reduce_sum_kernel<<<grid_size, block_size>>>(d_buf1, d_buf2,
                                                     current_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "dot kernel error" << cudaGetErrorString(err)
                      << std::endl;
        }

        // 下一轮: dst 作为输入, src 作为输出
        std::swap(d_buf1, d_buf2);
        current_size = grid_size;
    }

    float result = 0.f;

    CUDA_CHECK(
        cudaMemcpy(&result, d_buf1, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaFree(d_buf2));

    return result;
}

int main() {
    size_t N = 1 << 20;

    std::vector<float> input(N);

    for (size_t i = 0; i < N; i++) {
        input[i] = 1.0f;
    }

    float cpu_sum = std::accumulate(input.begin(), input.end(), 0.0f);

    float gpu_sum = reduce_sum(input);

    std::cout << "CPU sum = " << cpu_sum << std::endl;

    std::cout << "GPU sum = " << gpu_sum << std::endl;

    assert(fabs(cpu_sum - gpu_sum) < 1e-3);

    std::cout << "Test passed!" << std::endl;

    return 0;
}