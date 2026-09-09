// shared memory + coarsen + vec + double buffer + large tile

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int TM = 4;
static constexpr int TN = 4;

constexpr int SMEM_PER_BLOCK = BM * BK * 2 * sizeof(float);  // 32KB
constexpr int SMEM_PER_SM = 64 * 1024;       // 64KB
constexpr int MAX_BLOCKS = SMEM_PER_SM / SMEM_PER_BLOCK;  // 2

// ---------------------------------------------------------------
// 用 PTX ld.global.ca 显式加载，无分支，编译器可乱序调度
// __ldg 走 texture cache 路径（只读），SM75 上等价于
// ld.global.nc.v4.f32，warp scheduler 可在数据未就绪时切换
// ---------------------------------------------------------------
__device__ __forceinline__ float4 ldg128_safe(
    const float* ptr, int r, int c, int rows, int cols)
{
    float4 val = {0.f, 0.f, 0.f, 0.f};
    if (r >= rows) return val;
    if (c + 3 < cols) {
        val = __ldg(reinterpret_cast<const float4*>(&ptr[r * cols + c]));
    } else {
        val.x = (c     < cols) ? ptr[r * cols + c    ] : 0.f;
        val.y = (c + 1 < cols) ? ptr[r * cols + c + 1] : 0.f;
        val.z = (c + 2 < cols) ? ptr[r * cols + c + 2] : 0.f;
        val.w = (c + 3 < cols) ? ptr[r * cols + c + 3] : 0.f;
    }
    return val;
}

// ---------------------------------------------------------------
// __launch_bounds__ 限制每线程寄存器数
// 256 线程/block，目标 occupancy = 2 blocks/SM（SM75 共 64KB smem）
// double buf 占 32KB，2 blocks × 32KB = 64KB，恰好满
// 限制寄存器 ≤ 64，保证 2 blocks 并发
// ---------------------------------------------------------------
__global__ __launch_bounds__(256, MAX_BLOCKS) 
void cuda_core_sgemm_double_buf_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    // double buffer smem
    __shared__ float smem_A[2][BK][BM];
    __shared__ float smem_B[2][BK][BN];
    

    float acc[TM][TN] = {};
    float reg_A[TM], reg_B[TN];
    // 故意填充非零垃圾值
    // #pragma unroll
    // for (int i = 0; i < TM; i++) reg_A[i] = 99999.f;
    // #pragma unroll
    // for (int i = 0; i < TN; i++) reg_B[i] = 99999.f;

    // prefetch 寄存器：每线程 2 次 float4 × A/B
    // BK*BM / (256*4) = 32*64/1024 = 2
    float4 p_A[2], p_B[2];

    const int num_tiles  = (K + BK - 1) / BK;

    // ---- 预加载第 0 个 tile 到 smem[0] ----
    #pragma unroll
    for (int i = 0, idx = tid * 4; i < 2;
            i++, idx += 256 * 4) {
        int k_ = idx % BK, m = idx / BK;
        float4 v = ldg128_safe(A, block_row + m, 0 * BK + k_, M, K);
        smem_A[0][k_  ][m] = v.x;
        smem_A[0][k_+1][m] = v.y;
        smem_A[0][k_+2][m] = v.z;
        smem_A[0][k_+3][m] = v.w;
    }
    #pragma unroll
    for (int i = 0, idx = tid * 4; i < 2;
            i++, idx += 256 * 4) {
        int n = idx % BN, k_ = idx / BN;
        float4 v = ldg128_safe(B, 0 * BK + k_, block_col + n, K, N);
        *reinterpret_cast<float4*>(&smem_B[0][k_][n]) = v;
    }
    __syncthreads();

    // 主循环
    #pragma unroll
    for (int k = 0; k < num_tiles; k++) {
        const int cur     = k & 1;
        const int next    = cur ^ 1;
        const bool has_next = (k + 1 < num_tiles);

        // ---- 发射 ldg，prefetch tile_{k+1} 到寄存器 ----
        if (has_next) {
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                    i++, idx += 256 * 4) {
                int k_ = idx % BK, m = idx / BK;
                p_A[i] = ldg128_safe(
                    A, block_row + m, (k+1) * BK + k_, M, K);
            }
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                    i++, idx += 256 * 4) {
                int n = idx % BN, k_ = idx / BN;
                p_B[i] = ldg128_safe(
                    B, (k+1) * BK + k_, block_col + n, K, N);
            }
        }

        // ---- FMA（掩盖上方 ldg 的 latency）----
        #pragma unroll
        for (int k_ = 0; k_ < BK; k_++) {
            #pragma unroll
            for (int m = 0; m < TM; m++)
                reg_A[m] = smem_A[cur][k_][thread_row + m];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                reg_B[n] = smem_B[cur][k_][thread_col + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += reg_A[m] * reg_B[n];
        }

        // ---- sync + sts：此时 ldg 数据已就绪 ----
        if (has_next) {
            __syncthreads();
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                 i++, idx += 256 * 4) {
                int k_ = idx % BK, m = idx / BK;
                smem_A[next][k_  ][m] = p_A[i].x;
                smem_A[next][k_+1][m] = p_A[i].y;
                smem_A[next][k_+2][m] = p_A[i].z;
                smem_A[next][k_+3][m] = p_A[i].w;
            }
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                 i++, idx += 256 * 4) {
                int n = idx % BN, k_ = idx / BN;
                *reinterpret_cast<float4*>(&smem_B[next][k_][n]) = p_B[i];
            }
            __syncthreads();
        }
    }

    // ---- 写回 C ----
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        const int gr = block_row + thread_row + m;
        const int gc = block_col + thread_col;
        if (gr >= M) continue;
        // 前 4 列
        if (gc + 3 < N) {
            float4 out = {acc[m][0], acc[m][1], acc[m][2], acc[m][3]};
            if (beta != 0.f) {
                float4 old =
                    __ldg(reinterpret_cast<const float4 *>(&C[gr * N + gc]));
                out.x = alpha*out.x + beta*old.x;
                out.y = alpha*out.y + beta*old.y;
                out.z = alpha*out.z + beta*old.z;
                out.w = alpha*out.w + beta*old.w;
            } else {
                out.x *= alpha; out.y *= alpha;
                out.z *= alpha; out.w *= alpha;
            }
            *reinterpret_cast<float4*>(&C[gr * N + gc]) = out;
        } else {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int col = gc + n;
                if (col < N)
                    C[gr * N + col] =
                        alpha * acc[m][n] + beta * C[gr * N + col];
            }
        }
    }
}

void cuda_core_sgemm_double_buf(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    // 显式配置 smem 上限，保证 32KB 分配成功
    cudaFuncSetAttribute(
        cuda_core_sgemm_double_buf_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);

    dim3 block(BN / TN, BM / TM);  // 16×16 = 256
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cuda_core_sgemm_double_buf_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}