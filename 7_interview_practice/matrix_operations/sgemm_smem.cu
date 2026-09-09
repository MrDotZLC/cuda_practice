
// 策略：block tile = 32x32，每线程计算 C 的一个元素
//       A/B tile 协作搬入 smem，每元素 Global Memory 只读一次
//       Global Memory 访问量从 O(MNK) 降至 O(MNK/T)

static constexpr int TILE = 32;

__global__ void cuda_core_sgemm_smem_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // block 起始位置索引
    const int block_row = blockIdx.y * TILE;
    const int block_col = blockIdx.x * TILE;

    // block 内线程索引
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // C 的元素索引
    const int row = block_row + ty;
    const int col = block_col + tx;
    
    __shared__ float smem_A[TILE][TILE];
    __shared__ float smem_B[TILE][TILE];

    float acc = 0.f;

    // 沿 K 轴分 tile 循环
    #pragma unroll
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // 搬运 A Tile
        int a_row = block_row + ty;
        int a_col = t * TILE + tx;
        smem_A[ty][tx] = (a_row < M && a_col < K) 
                            ? A[a_row * K + a_col] 
                            : 0.f;

        // 搬运 B Tile
        int b_row = t * TILE + ty;
        int b_col = block_col + tx;
        smem_B[ty][tx] = (b_row < K && b_col < N) 
                            ? B[b_row * N + b_col]
                            : 0.f; 

        __syncthreads();

        // 计算 C Tile 的一个元素
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += smem_A[ty][k] * smem_B[k][tx];
        }

        // 防止部分线程先执行后续迭代，导致 smem 被覆盖
        __syncthreads();
    }

    // 写回 C
    if (row < M && col < N) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

void cuda_core_sgemm_smem(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    dim3 block(TILE, TILE);   // 32x32 = 1024 线程/block
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);
    cuda_core_sgemm_smem_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}