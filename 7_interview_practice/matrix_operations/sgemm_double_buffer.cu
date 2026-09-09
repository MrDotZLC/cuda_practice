// shared memory + coarsen + vec + double buffer + large tile

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int TM = 4;
static constexpr int TN = 4;

constexpr int SMEM_PER_BLOCK = BM * BK * 2 * sizeof(float);  // 32KB
constexpr int SMEM_PER_SM = 64 * 1024;       // 64KB
constexpr int MAX_BLOCKS = SMEM_PER_SM / SMEM_PER_BLOCK;  // 2

__device__ float4 ldg128_safe(const float* ptr, int r, int c, int rows,
                            int cols) {
    float4 val = {0.f, 0.f, 0.f, 0.f};
    if (r >= rows) return val;
    if (c + 3 < cols) {
        val = __ldg(reinterpret_cast<const float4*>(&ptr[r * cols + c]));
    } else {
        val.x = (c     < cols) ? ptr[r * cols + c] : 0.f;
        val.x = (c + 1 < cols) ? ptr[r * cols + c] : 0.f;
        val.x = (c + 2 < cols) ? ptr[r * cols + c] : 0.f;
        val.x = (c + 3 < cols) ? ptr[r * cols + c] : 0.f;
    }
    return val;
}

__global__ __launch_bounds__(256, MAX_BLOCKS) 
void cuda_core_sgemm_v6_large_tile_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = ty * blockDim.x + tx;

    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    __shared__ float smem_A[2][BK][BM];
    __shared__ float smem_B[2][BK][BN];
    

    float acc[TM][TN] = {};
    float reg_A[TM], reg_B[TN];

    float4 p_A[2], p_B[2];

    const int num_tiles = (K + BK - 1) / BK;

    // 预加载第 0 个 tile 到 ldg，再到 smem[0]
    #pragma unroll
    for (int i = 0, idx = tid * 4; i < 2; ++i, idx += 256 * 4) {
        int m = idx / BK, k = idx % BK;
        float4 v = ldg128_safe(A, block_row + m, 0 * BK + k, M, K);
        smem_A[0][k    ][m] = v.x;
        smem_A[0][k + 1][m] = v.y;
        smem_A[0][k + 2][m] = v.z;
        smem_A[0][k + 3][m] = v.w;
    }
    #pragma unroll
    for (int i = 0, idx = tid * 4; i < 2; ++i, idx += 256 * 4) {
        int k = idx / BN, n = idx % BN;
        float4 v = ldg128_safe(B, 0 * BK + k, block_col + n, K, N);
        *reinterpret_cast<float4*>(&smem_B[0][k][n]) = v;
    }
    __syncthreads();

    #pragma unroll
    for (int tk = 0; tk < num_tiles; ++tk) {
        const int cur = tk & 1;
        const int next = cur ^ 1;
        const bool has_next = (tk + 1 < num_tiles);

        if (has_next) {
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2; ++i, idx += 256 * 4) {
                int m = idx / BK, k = idx % BK;
                p_A[i] = ldg128_safe(A, block_row + m, (tk + 1) * BK + k, M, K);
            }
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2; ++i, idx += 256 * 4) {
                int k = idx / BN, n = idx % BN;
                p_B[i] = ldg128_safe(B, (tk + 1) * BK + k, block_row + n, K, N);
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_A[m] = smem_A[cur][k][thread_row + m];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                reg_B[n] = smem_B[cur][k][thread_col + n];
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] += reg_A[m] * reg_B[n];
                }
            }
        }

        if (has_next) {
            __syncthreads();
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2; ++i, idx += 256 * 4) {
                int m = idx / BK, k = idx % BK;
                smem_A[next][k    ][m] = p_A[i].x;
                smem_A[next][k + 1][m] = p_A[i].y;
                smem_A[next][k + 2][m] = p_A[i].z;
                smem_A[next][k + 3][m] = p_A[i].w;
            }
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2; ++i, idx += 256 * 4) {
                int k = idx / BN, n = idx % BN;
                *reinterpret_cast<float4*>(&smem_B[next][k][n]) = p_B[i];
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int gr = block_row + thread_row + m;
        const int gc = block_col + thread_col;
        if (gr >= M) break;
        // TN = 4
        if (gc + TN < N) {
            float4 out = {acc[m][0], acc[m][1], acc[m][2], acc[m][3]};
            if (beta != 0.f) {
                float4 old = __ldg(reinterpret_cast<const float4*>(&C[gr * N + gc]));
                out.x = alpha * out.x + beta * old.x;
                out.y = alpha * out.y + beta * old.y;
                out.z = alpha * out.z + beta * old.z;
                out.w = alpha * out.w + beta * old.w;
            } else {
                out.x *= alpha;
                out.y *= alpha;
                out.z *= alpha;
                out.w *= alpha;
            }
            *reinterpret_cast<float4*>(&C[gr * N + gc]) = out;
        } else {
            #pragma
            for (int n = 0; n < TN; ++n) {
                int col = gc + n;
                if (col < N) {
                    C[gr * N + col] = alpha * acc[m][n] + beta * C[gr * N + col];
                }
            }
        }
    }
}

void cuda_core_sgemm_v6_large_tile(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cuda_core_sgemm_v6_large_tile_kernel<<<grid, block>>>(
        A, B, C, M, N, K, alpha, beta);
}