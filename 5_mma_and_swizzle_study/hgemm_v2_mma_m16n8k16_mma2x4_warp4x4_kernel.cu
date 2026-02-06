#include <iostream>

#include "common/tester.h"
#include "common/common.h"
// #include "mma.h"
using namespace nvcuda;

// 128x128, mma2x4, warp4x4(64,32,16)
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel(half *A, half *B, half *C, int M,
                                             int N, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;  // 16*2*4=128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;  // 8*4*4=128
    constexpr int BK = MMA_K;                             // 16

    __shared__ half s_a[BM][BK + A_PAD];  // 128*16*2=4KB
    __shared__ half s_b[BK][BN + B_PAD];  // 16*128*2=4KB, 16*(128+16)*2=4.5KB

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // within block
    const int warp_id = tid / WARP_SIZE;  // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE;  // 0~31
    const int warp_m = warp_id % 2;       // 0,1
    const int warp_n = warp_id / 2;       // 0,1,2,3

    // 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=16 按行读取
    // A行主序
    // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                 // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取
    // B行主序
    // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
    int load_smem_b_k = tid / 16;        // row 0~15
    int load_smem_b_n = (tid % 16) * 8;  // col 0,8,...,120
    // 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数
    // 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m;  // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n;  // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k) {
        // gmem -> smem
        int load_gmem_a_k = k * BK + load_smem_a_k;  // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK + load_smem_b_k;  // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) =
            (LDST128BITS(B[load_gmem_b_addr]));
        LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) =
            (LDST128BITS(A[load_gmem_a_addr]));
        __syncthreads();

        // ldmatrix for s_a, ldmatrix.trans for s_b.
        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

// smem -> reg
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16;  // 0~15
            int lane_smem_a_k = (lane_id / 16) * 8;            // 0,8
            uint32_t lane_smem_a_ptr =
                __cvta_generic_to_shared(&s_a[lane_smem_a_m][lane_smem_a_k]);
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                        lane_smem_a_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_smem_b_k = lane_id % 16;   // 0~15
            int lane_smem_b_n = warp_smem_b_n;  // 0, MMA_N=8
            uint32_t lane_smem_b_ptr =
                __cvta_generic_to_shared(&s_b[lane_smem_b_k][lane_smem_b_n]);
            LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

// MMA compute
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1],
                          RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0],
                          RC[i][j][1]);
            }
        }
        __syncthreads();
    }

// reg -> gmem, MMA_MxMMA_N=16x8
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int store_warp_smem_c_m =
                warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int store_warp_smem_c_n =
                warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            // mapping lane smem index -> global index.
            // [16][8],
            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
            // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
            // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
            int store_lane_gmem_c_m =
                by * BM + store_warp_smem_c_m + lane_id / 4;
            int store_lane_gmem_c_n =
                bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
            int store_gmem_c_addr_0 =
                store_lane_gmem_c_m * N + store_lane_gmem_c_n;
            int store_gmem_c_addr_1 =
                (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
            // TODO: how to use LDST128BITS here ? reverse the loop order ?
            LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
            LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
        }
    }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                       \
    if (((T).options().dtype() != (th_type))) {                    \
        std::cout << "Tensor Info:" << (T).options() << std::endl; \
        throw std::runtime_error("values must be " #th_type);      \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {  \
        throw std::runtime_error("Tensor size mismatch!"); \
    }

// 128x128, mma2x4, warp4x4(64,32,16)
void hgemm_mma_m16n8k16_mma2x4_warp4x4(half *A, half *B, half *C, int M, int N,
                                       int K) {
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int MMA_TILE_M = 2;
    constexpr int MMA_TILE_N = 4;
    constexpr int WARP_TILE_M = 4;
    constexpr int WARP_TILE_N = 4;
    // bank conflicts free via pad = 8, reject fantasy, trust the profile.
    // ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
    // ./hgemm_mma_stage.89.debug.bin ncu --metrics
    // sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm
    // ./hgemm_mma_stage.89.debug.bin
    constexpr int A_PAD = 8;
    constexpr int B_PAD = 8;
    constexpr int NUM_THREADS =
        (MMA_TILE_M * MMA_TILE_N * WARP_SIZE);  // 2 * 4 * 32 = 256

    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, MMA_N * MMA_TILE_N * WARP_TILE_N),
              div_ceil(M, MMA_M * MMA_TILE_M * WARP_TILE_M));

    hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<MMA_M, MMA_N, MMA_K, MMA_TILE_M,
                                             MMA_TILE_N, WARP_TILE_M,
                                             WARP_TILE_N, A_PAD, B_PAD>
        <<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char *argv[]) {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_mma_m16n8k16_mma2x4_warp4x4, "hgemm_mma_m16n8k16_mma2x4_warp4x4");
}