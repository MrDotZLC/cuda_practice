#include "error.cuh"

#include <iostream>
#include <random>

// 使用 volatile 防止编译器优化：
//     避免将数据在寄存器命中，而不是实时读取
// 不用 volatile，就要保证 warp 内同步（__syncwarp()）
//     或者用warp原语：shuffle
__device__ void block_reduce(volatile float* smem, int tid) {
    // smem reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    
    // warp 内 reduction
    // 一个 warp 只有 32 个线程，无需 __syncthreads()
    if (tid < 32)
    {
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
}

// Stage 1: Dot Product
// partial[blockIdx.x] = sum(A[i] * B[i])
__global__ void dot_product_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* partial,
    int N) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    smem[tid] = (idx < N) ? A[idx] * B[idx] : 0.f;

    __syncthreads();

    block_reduce(smem, tid);

    if (tid == 0)
        partial[blockIdx.x] = smem[0];
}

// Stage 2:Sum Reduction
// result = sum(partial)
__global__ void reduce_sum_kernel(float* data, int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float sum = 0.f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += data[i];
    }
    smem[tid] = sum;

    __syncthreads();
    
    block_reduce(smem, tid);

    if (tid == 0)
        data[0] = smem[0];
}

// Host API
float dot_product(const float* d_A, const float* d_B, int N) {
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    size_t bytes = block_size * sizeof(float);

    float* d_partial;

    CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

    // 1. block 归约到 partial[block_id]：partial[i] = block sum
    dot_product_kernel<<<grid_size, block_size, bytes>>>(d_A, d_B, d_partial, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "dot kernel error" 
            << cudaGetErrorString(err) << std::endl;
    }

    // 2. 归约 partial：result = sum(partial)
    reduce_sum_kernel<<<1, block_size, bytes>>>(d_partial, grid_size);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "reduce kernel error" 
            << cudaGetErrorString(err) << std::endl;
    }

    float result;

    CUDA_CHECK(cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_partial);

    return result;
}

int main() {
    // 测试参数
    const int N = 1000000;  // 向量大小
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    std::cout << "=== 点积测试程序 ===" << std::endl;
    std::cout << "向量大小: " << N << std::endl;
    std::cout << "Block 大小: " << block_size << std::endl;
    std::cout << "Grid 大小: " << grid_size << std::endl;
    std::cout << std::endl;

    // 分配主机内存
    float* h_A = new float[N];
    float* h_B = new float[N];
    
    // 使用随机数生成器初始化向量
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    double host_dot = 0.0;
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<double>(dist(gen));
        h_B[i] = static_cast<double>(dist(gen));
        host_dot += h_A[i] * h_B[i];
    }
    
    std::cout << "主机计算结果: " << host_dot << std::endl;
    std::cout << std::endl;

    // 分配设备内存
    float* d_A;
    float* d_B;
    size_t bytes = N * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));


    // 执行点积
    float gpu_result = dot_product(d_A, d_B, N);

    std::cout << "GPU计算结果: " << gpu_result << std::endl;
    std::cout << std::endl;
    
    // 验证结果
    float diff = std::abs(host_dot - gpu_result);
    float rel_error = diff / (std::abs(host_dot) + 1e-10f);
    
    std::cout << "=== 验证结果 ===" << std::endl;
    std::cout << "绝对误差: " << diff << std::endl;
    std::cout << "相对误差: " << rel_error << std::endl;
    
    if (rel_error < 1e-5f) {
        std::cout << "✓ 测试通过!" << std::endl;
    } else {
        std::cout << "✗ 测试失败!" << std::endl;
    }

    // 清理内存
    delete[] h_A;
    delete[] h_B;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    
    return 0;
}