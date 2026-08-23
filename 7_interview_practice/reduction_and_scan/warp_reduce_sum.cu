#include "error.cuh"

constexpr int N = 1 << 11;
constexpr int M = N * sizeof(float);
constexpr int BLOCK_SIZE = 256;

__device__ float warp_reduce_sum(float val) {
    #pragma unroll 
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void block_reduce_sum_kernel(const float *d_in, float *d_out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (gid < n) ? d_in[gid] : 0.f;
    val = warp_reduce_sum(val);

    __shared__ float shared[32]; // 最多32个warp
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = val;
    }
}

float reduce(float* d_in) {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int out_mem = num_blocks * sizeof(float);
    float *d_out;
    CHECK_CUDA(cudaMalloc(&d_out, out_mem));
    float *h_out = (float*)malloc(out_mem);
    
    dim3 grid(num_blocks, 1);
    dim3 block(BLOCK_SIZE, 1);

    float sum = 0.f;

    block_reduce_sum_kernel<<<grid, block>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());

    float* d_final;
    CHECK_CUDA(cudaMalloc(&d_final, sizeof(float)));
    block_reduce_sum_kernel<<<1, block>>>(d_out, d_final, num_blocks);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(&sum, d_final, sizeof(float),
                          cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_final));
    return sum;
}

int main() {
    float *h_in = (float*)malloc(M);
    for (int n = 0; n < N; n++) {
        // h_in[n] = 2.0 * (float)drand48() - 1.0;
        h_in[n] = 1.0;
    }
    float *d_in;
    CHECK_CUDA(cudaMalloc((void**)&d_in, M));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, M, cudaMemcpyHostToDevice));

    printf("\nwarp_reduce_sum:            %f\n", reduce(d_in));
    CHECK_CUDA(cudaFree(d_in));
    free(h_in);
    return 0;
}