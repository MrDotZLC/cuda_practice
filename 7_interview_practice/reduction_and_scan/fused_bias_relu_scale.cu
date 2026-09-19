#include "cuda_runtime.h"
#include "error.cuh"
#include <iostream>

__global__ void fused_bias_relu_scale_kernel(const float* __restrict__ X,
                                      const float* __restrict__ Bias,
                                      float* __restrict__ O, int N,
                                      float scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
        float v = X[i] * Bias[i];
        v = v > 0.f ? v : 0.f;
        O[i] = v * scale;
    }
}

void fused_bias_relu_scale(const float* __restrict__ X,
                           const float* __restrict__ Bias,
                           float* __restrict__ O, int N, float scale) {
    float* d_A;
    float* d_B;
    float* d_C;
    size_t bytes = N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, X, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, Bias, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, O, bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    fused_bias_relu_scale_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, N, 0.5f);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O, d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    // 1. 设置测试参数
    int N = 10000;  // 向量大小
    size_t bytes = N * sizeof(float);

    printf("Testing vector addition with N = %d\n", N);
    printf("Memory size: %.2f KB\n", bytes / 1024.0);

    // 2. 分配主机内存
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // 3. 初始化数据（使用简单的递增序列）
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
        h_C[i] = 0.0f;  // 初始化为0
    }

    printf("Sample data: A[0]=%.1f, B[0]=%.1f\n", h_A[0], h_B[0]);
    printf("            A[1]=%.1f, B[1]=%.1f\n", h_A[1], h_B[1]);

    // 4. 执行向量加法
    printf("\nRunning CUDA kernel...\n");
    fused_bias_relu_scale(h_A, h_B, h_C, N, 0.5f);

    // 5. 验证结果
    printf("\nVerifying results...\n");
    bool passed = true;
    int maxErrors = 5;  // 只显示前5个错误
    int errorCount = 0;
    
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        expected = (expected > 0.f) ? expected * 0.5f : 0.f;
        float actual = h_C[i];
        
        if (fabs(actual - expected) > 1e-5f) {
            if (errorCount < maxErrors) {
                printf("  Error at index %d: expected %.2f, got %.2f\n", 
                       i, expected, actual);
            }
            errorCount++;
            passed = false;
        }
    }
    
    if (errorCount > maxErrors) {
        printf("  ... and %d more errors\n", errorCount - maxErrors);
    }

    // 6. 输出结果
    printf("\n--- Results ---\n");
    if (passed) {
        printf("✓ PASSED! All %d elements correct.\n", N);
    } else {
        printf("✗ FAILED! %d errors found.\n", errorCount);
    }

    // 显示前几个结果
    printf("\nFirst 5 results:\n");
    for (int i = 0; i < (N > 5 ? 5 : N); i++) {
        printf("  C[%d] = %.2f + %.2f = %.2f\n", 
               i, h_A[i], h_B[i], h_C[i]);
    }

    // 7. 清理资源
    free(h_A);
    free(h_B);
    free(h_C);

    // 8. 重置CUDA设备（可选）
    cudaDeviceReset();

    return passed ? 0 : 1;
}