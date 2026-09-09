
static const int TILE = 32;
__global__ void cuda_core_sgemm_smem_kernel(const float* A, const float* B, float* C,
                          int M, int N, int K,
                          float alpha, float beta) {
    const int block_row = blockIdx.y * TILE;
    const int block_col = blockIdx.x * TILE;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = block_row + ty;
    const int col = block_col + tx;

    __shared__ float smem_A[TILE][TILE];
    __shared__ float smem_B[TILE][TILE];
    
    float acc = 0.f;

    #pragma unroll
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int row_a = block_row + ty;
        int col_a = t * TILE + tx;
        smem_A[ty][tx] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : 0.f;

        int row_b = t * TILE + ty;
        int col_b = block_col + tx;
        smem_B[ty][tx] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += smem_A[ty][k] * smem_B[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

void cuda_core_sgemm_smem(const float* A, const float* B, float* C, int M,
                          int N, int K, float alpha, float beta) {
    dim3 block(TILE, TILE);
    dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    cuda_core_sgemm_smem_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}
